#!/usr/bin/env python3
"""
Local HTTP & Neuroimaging API Server for the NiiVue (niivue/mono) Multimodal Brain Visualization
& Coregistration Tool.
- Cross-platform (macOS, Linux, Windows) with automatic discovery of FSL, FreeSurfer, FastSurfer, and dcm2niix
- Pure-Python / SciPy / pydicom fallbacks when external CLI binaries are not installed on the host OS
- ThreadingHTTPServer for concurrent non-blocking requests
"""
import os
import re
import io
import sys
import glob
import json
import shlex
import base64
import shutil
import platform
import tempfile
import subprocess
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

import numpy as np
import nibabel as nib
from scipy.ndimage import (
    affine_transform,
    sobel,
    gaussian_filter,
    binary_erosion,
    binary_dilation,
    binary_closing,
    binary_fill_holes,
    distance_transform_edt,
    generate_binary_structure,
    label as nd_label,
)
from scipy.optimize import minimize

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
IS_WINDOWS = os.name == "nt" or platform.system().lower().startswith("win")
IS_MAC = platform.system().lower() == "darwin"


def find_fsl_dir() -> str:
    """Locates FSL installation directory across macOS, Linux, and WSL/Windows."""
    candidates = [
        os.environ.get("FSLDIR", "").strip(),
        "/usr/local/fsl",
        os.path.expanduser("~/fsl"),
        "/opt/fsl",
        "/usr/share/fsl/6.0",
        "/usr/share/fsl/5.0",
        "/usr/lib/fsl/5.0",
    ]
    for c in candidates:
        if c and os.path.isdir(c) and os.path.isdir(os.path.join(c, "bin")):
            return c
    # Infer from flirt or bet in PATH
    for tool in ("flirt", "bet"):
        w = shutil.which(tool)
        if w:
            parent = os.path.dirname(os.path.dirname(os.path.abspath(w)))
            if os.path.isdir(os.path.join(parent, "bin")):
                return parent
    return "/usr/local/fsl"


def find_neuro_binary(name: str, fsl_dir: str = None) -> str:
    """Finds an executable binary (e.g. flirt, bet, dcm2niix) in FSLDIR/bin, PATH, or standard locations."""
    search_paths = []
    if fsl_dir:
        search_paths.append(os.path.join(fsl_dir, "bin", name))
        if IS_WINDOWS:
            search_paths.append(os.path.join(fsl_dir, "bin", f"{name}.exe"))
    w = shutil.which(name)
    if w:
        search_paths.append(w)
    for prefix in ("/usr/local/fsl/bin", "/usr/local/bin", "/opt/homebrew/bin", "/usr/bin", "/usr/lib/fsl/5.0"):
        search_paths.append(os.path.join(prefix, name))
    for p in search_paths:
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return os.path.join(fsl_dir or "/usr/local/fsl", "bin", name)


def find_freesurfer_dir() -> str:
    """Locates FreeSurfer directory across macOS and Linux."""
    candidates = [
        os.environ.get("FREESURFER_HOME", "").strip(),
        "/Applications/freesurfer/7.2.0",
    ]
    candidates.extend(sorted(glob.glob("/Applications/freesurfer/*"), reverse=True))
    candidates.extend(sorted(glob.glob("/usr/local/freesurfer/*"), reverse=True))
    candidates.extend(["/usr/local/freesurfer", "/opt/freesurfer"])
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    return "/Applications/freesurfer/7.2.0"


FSLDIR = find_fsl_dir()
FLIRT_BIN = find_neuro_binary("flirt", FSLDIR)
BET_BIN = find_neuro_binary("bet", FSLDIR)
DCM2NIIX_BIN = find_neuro_binary("dcm2niix", FSLDIR)
FREESURFER_DIR = find_freesurfer_dir()
DKT_LUT_JSON_PATH = os.path.join(ROOT_DIR, "sample_data", "dkt_lut.json")
MNI_T1_BRAIN_PATH = os.path.join(ROOT_DIR, "sample_data", "nifti", "MNI152_T1_brain_2mm.nii.gz")
MNI_DKT_ATLAS_PATH = os.path.join(ROOT_DIR, "sample_data", "nifti", "MNI152_DKT_atlas_2mm.nii.gz")
MNI_DESTRIEUX_ATLAS_PATH = os.path.join(ROOT_DIR, "sample_data", "nifti", "MNI152_Destrieux_atlas_2mm.nii.gz")
SAMPLE_T1_PATH = os.path.join(ROOT_DIR, "sample_data", "nifti", "sub-01_T1w.nii.gz")
SAMPLE_FASTSURFER_DKT_PATH = os.path.join(
    ROOT_DIR, "sample_data", "nifti", "sub-01_T1w_FastSurferVINN_DKT.nii.gz"
)
import getpass

def _get_user_cache_dir() -> str:
    """Returns a per-user writable cache directory in tempdir or ~/.cache, requiring no admin privileges."""
    try:
        uid_tag = str(os.getuid())
    except AttributeError:
        uid_tag = re.sub(r"[^a-zA-Z0-9_-]", "_", getpass.getuser() or "user")
    cand = os.path.join(tempfile.gettempdir(), f"neurocoreg_fastsurfer_cache_{uid_tag}")
    try:
        os.makedirs(cand, exist_ok=True)
        if os.access(cand, os.W_OK):
            return cand
    except Exception:
        pass
    home_cand = os.path.join(os.path.expanduser("~"), ".cache", "neurocoreg_fastsurfer_cache")
    try:
        os.makedirs(home_cand, exist_ok=True)
    except Exception:
        pass
    return home_cand


FASTSURFER_CACHE_DIR = _get_user_cache_dir()


def normalize_user_path(raw_path: str) -> str:
    """
    Normalizes local and mounted network paths across macOS (/Volumes/...), Linux (/mnt/..., /media/..., gvfs),
    and Windows (mapped drives Z:\\... and UNC network shares \\\\server\\share\\...).
    """
    if not raw_path:
        return ""
    p = str(raw_path).strip().strip('"').strip("'")
    if p.lower().startswith("file://"):
        from urllib.parse import unquote
        parsed_uri = urlparse(p)
        if IS_WINDOWS and parsed_uri.netloc:
            # file://server/share/folder -> \\server\share\folder
            p = f"\\\\{parsed_uri.netloc}{unquote(parsed_uri.path).replace('/', '\\')}"
        else:
            p = unquote(parsed_uri.path)
            if IS_WINDOWS and re.match(r"^/[a-zA-Z]:", p):
                p = p[1:]
    p = os.path.expanduser(os.path.expandvars(p))
    # Preserve Windows UNC prefix (\\server\share) during normpath/abspath
    if IS_WINDOWS and p.startswith("\\\\"):
        return os.path.normpath(p)
    return os.path.abspath(p)


def get_default_projects_dir(sub_path: str = "") -> str:
    """Returns a user-writable default directory for saving/opening projects even if ROOT_DIR is read-only."""
    base_root = ROOT_DIR if os.access(ROOT_DIR, os.W_OK) else os.path.expanduser("~")
    projects_root = os.path.join(base_root, "saved_projects" if base_root == ROOT_DIR else "NeuroCoreg_Projects")
    return os.path.join(projects_root, sub_path) if sub_path else projects_root


def strip_neuro_ext(filename: str) -> str:
    """Case-insensitively strips .nii.gz, .nii, .voi, .mgz, .mgh, or .dcm from a filename."""
    base = os.path.basename(filename or "volume")
    return re.sub(r"\.(nii\.gz|nii|voi|mgz|mgh|dcm)$", "", base, flags=re.IGNORECASE) or "volume"


def get_best_affine(img) -> np.ndarray:
    """Returns a valid 4x4 float64 RAS affine matrix from a NIfTI/MGH image, guarding against NaN or zero determinants."""
    try:
        aff = np.asarray(img.affine, dtype=np.float64)
        if aff.shape == (4, 4) and np.all(np.isfinite(aff)) and abs(np.linalg.det(aff[:3, :3])) > 1e-8:
            return aff.copy()
    except Exception:
        pass
    try:
        qaff = np.asarray(img.get_qform(), dtype=np.float64)
        if qaff.shape == (4, 4) and np.all(np.isfinite(qaff)) and abs(np.linalg.det(qaff[:3, :3])) > 1e-8:
            return qaff.copy()
    except Exception:
        pass
    return np.eye(4, dtype=np.float64)


def safe_inv4x4(mat: np.ndarray) -> np.ndarray:
    """Inverts a 4x4 matrix safely, falling back to pseudo-inverse if near-singular."""
    m = np.asarray(mat, dtype=np.float64)
    try:
        if np.all(np.isfinite(m)) and abs(np.linalg.det(m[:3, :3])) > 1e-10:
            return np.linalg.inv(m)
    except Exception:
        pass
    try:
        return np.linalg.pinv(m)
    except Exception:
        return np.eye(4, dtype=np.float64)


def get_clean_3d_data(img, dtype=np.float32) -> np.ndarray:
    """
    Extracts a finite 3D spatial array (nx, ny, nz) from a 2D, 3D, or 4D+ NIfTI/MGH image:
      - For 4D+ volumes (e.g. fMRI / DWI), extracts volume index 0 along trailing axes
      - For 2D slices, expands to (nx, ny, 1)
      - Replaces any NaN / Inf values with 0.0
    """
    raw = np.asarray(img.dataobj)
    while raw.ndim > 3:
        raw = raw[..., 0]
    while raw.ndim < 3:
        raw = np.expand_dims(raw, axis=-1)
    slope = float(getattr(img.header, "get_slope_inter", lambda: (1.0, 0.0))()[0] or 1.0)
    inter = float(getattr(img.header, "get_slope_inter", lambda: (1.0, 0.0))()[1] or 0.0)
    arr = raw.astype(np.float32)
    if np.isfinite(slope) and slope not in (0.0, 1.0):
        arr = arr * slope
    if np.isfinite(inter) and inter != 0.0:
        arr = arr + inter
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(dtype)


def build_canonical_uint8_nifti(voxels_3d: np.ndarray, affine_4x4: np.ndarray) -> nib.Nifti1Image:
    """
    Constructs a canonical uint8 NIfTI-1 image compatible with MRIcroGL/MRIcron (.voi),
    NiiVue, FSLeyes, and 3D Slicer with explicit sform=1, qform=1, scl_slope=1, scl_inter=0.
    """
    arr = np.asarray(voxels_3d, dtype=np.uint8)
    aff = np.asarray(affine_4x4, dtype=np.float64)
    if aff.shape != (4, 4) or not np.all(np.isfinite(aff)) or abs(np.linalg.det(aff[:3, :3])) < 1e-10:
        aff = np.eye(4, dtype=np.float64)
    out_nii = nib.Nifti1Image(arr, aff)
    out_nii.set_data_dtype(np.uint8)
    out_nii.set_sform(aff, code=1)
    out_nii.set_qform(aff, code=1)
    out_nii.header["scl_slope"] = 1.0
    out_nii.header["scl_inter"] = 0.0
    return out_nii


def save_nifti_bytes_for_format(nii_img: nib.Nifti1Image, fmt: str = "voi") -> bytes:
    """
    Saves a nibabel NIfTI-1 image as bytes for '.voi' (gzipped uint8 NIfTI), '.nii.gz', or '.nii'.
    Never passes '.voi' directly to nib.save() since nibabel infers format by file extension.
    """
    norm_fmt = (fmt or "voi").lower().lstrip(".")
    disk_ext = ".nii" if norm_fmt == "nii" else ".nii.gz"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, f"drawing{disk_ext}")
        nib.save(nii_img, tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()


def load_neuro_image_from_bytes(raw_bytes: bytes, source_name: str, tmpdir: str):
    """
    Writes raw neuroimaging bytes (.voi, .nii.gz, .nii, .mgz, .mgh) to a nibabel-safe extension
    inside tmpdir and returns the loaded nibabel image object.
    """
    lower = (source_name or "").lower()
    if lower.endswith(".mgz"):
        ext = ".mgz"
    elif lower.endswith(".mgh"):
        ext = ".mgh"
    elif len(raw_bytes) >= 2 and raw_bytes[:2] == b"\x1f\x8b":
        ext = ".nii.gz"
    else:
        ext = ".nii"
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", strip_neuro_ext(source_name or "vol"))
    tmp_path = os.path.join(tmpdir, f"{safe_name}{ext}")
    with open(tmp_path, "wb") as f:
        f.write(raw_bytes)
    return nib.load(tmp_path)


def _write_label_mask(out_view: np.ndarray, mask: np.ndarray, lbl: int, overwrite: bool):
    """Writes integer label `lbl` into `out_view` where `mask` is True, respecting overwrite."""
    if overwrite:
        out_view[mask] = lbl
    else:
        out_view[mask & (out_view == 0)] = lbl


def _pick_interpolation_axis(mask: np.ndarray, axis_pref="auto") -> int:
    """Selects slice axis (0, 1, or 2) with multi-slice gaps for signed-distance interpolation."""
    if str(axis_pref) in ("0", "1", "2"):
        return int(axis_pref)
    best_score = (-1, -1, -1)
    best_ax = 2
    for ax in (2, 1, 0):
        n_slices = mask.shape[ax]
        annotated = [idx for idx in range(n_slices) if np.any(np.take(mask, idx, axis=ax))]
        if len(annotated) >= 2:
            gaps = sum(1 for a, b in zip(annotated, annotated[1:]) if b - a > 1)
            span = annotated[-1] - annotated[0]
            score = (1 if gaps > 0 else 0, gaps, span + (1 if ax == 2 else 0))
            if score > best_score:
                best_score = score
                best_ax = ax
    return best_ax


def apply_drawing_morph_op(
    arr: np.ndarray,
    affine: np.ndarray,
    op: str,
    target_label: int = 1,
    overwrite: bool = True,
    params: dict = None,
) -> tuple:
    """
    Runs 3D morphological cleanup, signed-distance slice interpolation, or intensity-guided
    region growing on a uint8 3D drawing array.
    Returns (updated_uint8_arr, changed_voxel_count, status_message).
    """
    params = params or {}
    out = arr.copy()
    op_norm = (op or "").lower()

    if target_label and target_label > 0:
        labels_to_process = [int(target_label)]
    else:
        labels_to_process = [int(v) for v in np.unique(arr) if int(v) > 0]

    if op_norm == "region_grow":
        src_b64 = params.get("sourceBase64")
        seed = params.get("seedVoxel")
        if not src_b64 or not seed or len(seed) != 3:
            raise ValueError("region_grow requires sourceBase64 and seedVoxel [ix, iy, iz].")
        sx, sy, sz = (int(round(seed[0])), int(round(seed[1])), int(round(seed[2])))
        nx, ny, nz = arr.shape
        if not (0 <= sx < nx and 0 <= sy < ny and 0 <= sz < nz):
            raise ValueError("Seed voxel is outside volume bounds.")

        src_bytes = base64.b64decode(src_b64)
        src_name = params.get("sourceName") or "source.nii.gz"
        with tempfile.TemporaryDirectory() as tmpdir:
            src_img = load_neuro_image_from_bytes(src_bytes, src_name, tmpdir)
            src_3d = get_clean_3d_data(src_img, dtype=np.float32)
            src_aff_raw = params.get("sourceAffine")
            src_aff = np.asarray(src_aff_raw, dtype=np.float64) if src_aff_raw else get_best_affine(src_img)

        if src_3d.shape != arr.shape or not np.allclose(src_aff, affine, atol=1e-2):
            vox_to_vox = safe_inv4x4(src_aff) @ affine
            vol_on_grid = affine_transform(
                src_3d,
                vox_to_vox[:3, :3],
                offset=vox_to_vox[:3, 3],
                output_shape=arr.shape,
                order=1,
                mode="constant",
                cval=0.0,
            )
        else:
            vol_on_grid = src_3d

        nz_vals = vol_on_grid[vol_on_grid != 0]
        if nz_vals.size >= 16 and float(np.min(vol_on_grid)) >= 0.0:
            p2, p98 = np.percentile(nz_vals, [2, 98])
        else:
            p2, p98 = np.percentile(vol_on_grid, [2, 98])
        span = max(1e-5, float(p98 - p2))
        tol_pct = max(0.5, min(80.0, float(params.get("tolerancePct", 15.0))))
        tol = max(1e-5, span * (tol_pct / 100.0))

        max_rad_mm = max(2.0, min(150.0, float(params.get("maxRadiusMm", 35.0))))
        vx = max(0.2, float(np.linalg.norm(affine[:3, 0])))
        vy = max(0.2, float(np.linalg.norm(affine[:3, 1])))
        vz = max(0.2, float(np.linalg.norm(affine[:3, 2])))
        rx = min(nx - 1, int(np.ceil(max_rad_mm / vx)))
        ry = min(ny - 1, int(np.ceil(max_rad_mm / vy)))
        rz = min(nz - 1, int(np.ceil(max_rad_mm / vz)))

        seed_val = float(vol_on_grid[sx, sy, sz])
        scope = (params.get("scope") or "3d").lower()
        slice_ax = int(params.get("sliceAxis", 2))
        if slice_ax not in (0, 1, 2):
            slice_ax = 2

        x0, x1 = max(0, sx - rx), min(nx, sx + rx + 1)
        y0, y1 = max(0, sy - ry), min(ny, sy + ry + 1)
        z0, z1 = max(0, sz - rz), min(nz, sz + rz + 1)
        if scope == "2d":
            if slice_ax == 0:
                x0, x1 = sx, sx + 1
            elif slice_ax == 1:
                y0, y1 = sy, sy + 1
            else:
                z0, z1 = sz, sz + 1

        sub_vol = vol_on_grid[x0:x1, y0:y1, z0:z1]
        gx = (np.arange(x0, x1, dtype=np.float32)[:, None, None] - sx) * vx
        gy = (np.arange(y0, y1, dtype=np.float32)[None, :, None] - sy) * vy
        gz = (np.arange(z0, z1, dtype=np.float32)[None, None, :] - sz) * vz
        dist2 = gx * gx + gy * gy + gz * gz
        cand = (np.abs(sub_vol - seed_val) <= tol) & (dist2 <= (max_rad_mm * max_rad_mm))
        lsx, lsy, lsz = sx - x0, sy - y0, sz - z0
        cand[lsx, lsy, lsz] = True

        struct6 = generate_binary_structure(3, 1)
        labeled, _ = nd_label(cand, structure=struct6)
        seed_id = int(labeled[lsx, lsy, lsz])
        grown_sub = labeled == seed_id
        lbl = max(1, int(target_label or 1))
        _write_label_mask(out[x0:x1, y0:y1, z0:z1], grown_sub, lbl, overwrite)
        changed = int(np.count_nonzero(out != arr))
        grown_cnt = int(np.count_nonzero(grown_sub))
        return out, changed, f"Region grow ({scope.upper()}) captured {grown_cnt} voxels around seed."

    if not labels_to_process:
        return out, 0, "Layer is currently empty; paint a region first."

    if op_norm == "fill_holes":
        for lbl in labels_to_process:
            mask = arr == lbl
            if not np.any(mask):
                continue
            filled = binary_fill_holes(mask)
            _write_label_mask(out, filled, lbl, overwrite)
        changed = int(np.count_nonzero(out != arr))
        return out, changed, f"Filled 3D interior holes ({changed} voxels added)."

    if op_norm == "largest_island":
        struct26 = generate_binary_structure(3, 3)
        for lbl in labels_to_process:
            mask = arr == lbl
            if not np.any(mask):
                continue
            labeled, ncomp = nd_label(mask, structure=struct26)
            if ncomp <= 1:
                continue
            counts = np.bincount(labeled.ravel())
            counts[0] = 0
            keep_id = int(np.argmax(counts))
            out[mask & (labeled != keep_id)] = 0
        changed = int(np.count_nonzero(out != arr))
        return out, changed, f"Retained largest 3D connected component ({changed} stray voxels removed)."

    if op_norm == "dilate":
        struct6 = generate_binary_structure(3, 1)
        for lbl in labels_to_process:
            mask = arr == lbl
            if not np.any(mask):
                continue
            dil = binary_dilation(mask, structure=struct6)
            _write_label_mask(out, dil, lbl, overwrite)
        changed = int(np.count_nonzero(out != arr))
        return out, changed, f"Dilated mask by 1 voxel ({changed} voxels added)."

    if op_norm == "erode":
        struct6 = generate_binary_structure(3, 1)
        for lbl in labels_to_process:
            mask = arr == lbl
            if not np.any(mask):
                continue
            ero = binary_erosion(mask, structure=struct6)
            out[mask & (~ero)] = 0
        changed = int(np.count_nonzero(out != arr))
        return out, changed, f"Eroded mask by 1 voxel ({changed} boundary voxels removed)."

    if op_norm == "interpolate_slices":
        axis_pref = params.get("axis", "auto")
        any_interp = False
        used_ax = 2
        for lbl in labels_to_process:
            mask = arr == lbl
            ax = _pick_interpolation_axis(mask, axis_pref)
            used_ax = ax
            n_slices = mask.shape[ax]
            annotated = [idx for idx in range(n_slices) if np.any(np.take(mask, idx, axis=ax))]
            if len(annotated) < 2:
                continue
            for s1, s2 in zip(annotated, annotated[1:]):
                if s2 - s1 <= 1:
                    continue
                any_interp = True
                m1 = np.take(mask, s1, axis=ax)
                m2 = np.take(mask, s2, axis=ax)
                sd1 = (
                    np.full(m1.shape, 1e4, dtype=np.float32)
                    if np.all(m1)
                    else (distance_transform_edt(m1) - distance_transform_edt(~m1)).astype(np.float32)
                )
                sd2 = (
                    np.full(m2.shape, 1e4, dtype=np.float32)
                    if np.all(m2)
                    else (distance_transform_edt(m2) - distance_transform_edt(~m2)).astype(np.float32)
                )
                for s in range(s1 + 1, s2):
                    t = (s - s1) / float(s2 - s1)
                    interp_mask = ((1.0 - t) * sd1 + t * sd2) > 0.0
                    slc = [slice(None), slice(None), slice(None)]
                    slc[ax] = s
                    _write_label_mask(out[tuple(slc)], interp_mask, lbl, overwrite)
        changed = int(np.count_nonzero(out != arr))
        if not any_interp:
            return out, 0, "Need painted contours on at least 2 non-adjacent slices to interpolate."
        ax_name = {0: "X", 1: "Y", 2: "Z"}.get(used_ax, str(used_ax))
        return out, changed, f"Interpolated between slices along axis {ax_name} ({changed} voxels added)."

    raise ValueError(f"Unsupported drawing morphological operation: {op}")


def scipy_morphological_skull_strip(data_3d: np.ndarray, zooms=(1.0, 1.0, 1.0), frac: float = 0.5):
    """
    Pure-Python / SciPy 3D brain extraction fallback when FSL BET (/usr/local/fsl/bin/bet) is not installed
    on the host OS (e.g. Windows or Linux without FSL).
    Returns (brain_extracted_3d, binary_mask_uint8).
    """
    vol = np.nan_to_num(np.asarray(data_3d, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    pos = vol[vol > 0]
    if pos.size < 100:
        return vol.copy(), (vol > 0).astype(np.uint8)

    p2, p98 = np.percentile(pos, [2, 98])
    norm = np.clip((vol - p2) / max(p98 - p2, 1e-5), 0.0, 1.0)
    smooth = gaussian_filter(norm, sigma=1.0)

    # Threshold scaled by fractional intensity threshold `frac` (0.1 .. 0.9, default 0.5)
    thresh = 0.12 + 0.36 * float(np.clip(frac, 0.1, 0.9))
    fg = smooth > thresh

    # Morphological erosion to disconnect skull/scalp bridges, keep largest 3D connected component (brain), then dilate back
    iters = 2 if min(vol.shape[:3]) >= 64 else 1
    eroded = binary_erosion(fg, iterations=iters)
    if not np.any(eroded):
        eroded = fg

    labeled, num_features = nd_label(eroded)
    if num_features > 0:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        largest_label = int(np.argmax(counts))
        brain_core = labeled == largest_label
    else:
        brain_core = eroded

    dilated = binary_dilation(brain_core, iterations=iters) & (norm > (thresh * 0.55))
    closed = binary_closing(dilated, iterations=2)
    mask = binary_fill_holes(closed)

    out = np.where(mask, vol, 0.0).astype(np.float32)
    return out, mask.astype(np.uint8)


def convert_dicom_dir_pydicom(in_dir: str, out_dir: str) -> list:
    """
    Pure-Python pydicom + nibabel fallback for converting a directory of 2D DICOM slices into a 3D NIfTI (.nii.gz)
    when dcm2niix is not installed on the host OS.
    """
    import pydicom

    slices = []
    for fname in sorted(os.listdir(in_dir)):
        fpath = os.path.join(in_dir, fname)
        if not os.path.isfile(fpath) or fname.startswith("."):
            continue
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=False, force=True)
            if hasattr(ds, "pixel_array"):
                slices.append(ds)
        except Exception:
            continue

    if not slices:
        return []

    def slice_sort_key(s):
        if hasattr(s, "ImagePositionPatient") and len(s.ImagePositionPatient) == 3:
            return float(s.ImagePositionPatient[2])
        if hasattr(s, "SliceLocation"):
            return float(s.SliceLocation)
        if hasattr(s, "InstanceNumber"):
            return float(s.InstanceNumber)
        return 0.0

    slices.sort(key=slice_sort_key)
    first = slices[0]

    # Stack 2D slices (Rows, Columns) -> 3D RAS volume
    arr_list = []
    for s in slices:
        px = s.pixel_array.astype(np.float32)
        slope = float(getattr(s, "RescaleSlope", 1.0) or 1.0)
        inter = float(getattr(s, "RescaleIntercept", 0.0) or 0.0)
        px = px * slope + inter
        # Standard axial LPS -> RAS conversion matching create_dicom_series
        arr_list.append(np.flip(np.flip(px.T, axis=1), axis=0))

    vol_3d = np.stack(arr_list, axis=2)  # (nx, ny, nz)
    nx, ny, nz = vol_3d.shape

    ps = getattr(first, "PixelSpacing", [2.0, 2.0])
    dy, dx = float(ps[0]), float(ps[1])
    if len(slices) > 1 and hasattr(slices[0], "ImagePositionPatient") and hasattr(slices[1], "ImagePositionPatient"):
        p0 = np.array([float(x) for x in slices[0].ImagePositionPatient])
        p1 = np.array([float(x) for x in slices[1].ImagePositionPatient])
        dz = float(np.linalg.norm(p1 - p0)) or float(getattr(first, "SliceThickness", 2.0) or 2.0)
    else:
        dz = float(getattr(first, "SliceThickness", 2.0) or 2.0)

    affine = np.array([
        [-dx, 0.0, 0.0, (nx - 1) * dx / 2.0],
        [0.0,  dy, 0.0, -(ny - 1) * dy / 2.0],
        [0.0, 0.0,  dz, -(nz - 1) * dz / 2.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    series_desc = re.sub(r"[^a-zA-Z0-9._-]", "_", str(getattr(first, "SeriesDescription", "DICOM_Series") or "DICOM_Series"))
    series_num = str(getattr(first, "SeriesNumber", 1) or 1)
    out_fname = f"{series_desc}_{series_num}.nii.gz"
    out_fpath = os.path.join(out_dir, out_fname)

    nii = nib.Nifti1Image(vol_3d.astype(np.float32), affine)
    nii.set_sform(affine, code=1)
    nii.set_qform(affine, code=1)
    nib.save(nii, out_fpath)

    meta = {
        "SeriesDescription": str(getattr(first, "SeriesDescription", "DICOM Series")),
        "Modality": str(getattr(first, "Modality", "MR")),
        "PatientID": str(getattr(first, "PatientID", "")),
    }
    with open(os.path.join(out_dir, f"{series_desc}_{series_num}.json"), "w", encoding="utf-8") as jf:
        json.dump(meta, jf)

    return [out_fname]


def load_dkt_lut() -> dict:
    """Loads the 402-entry FreeSurfer / FastSurfer DKT (aparc.DKTatlas+aseg) lookup table."""
    if os.path.exists(DKT_LUT_JSON_PATH):
        try:
            with open(DKT_LUT_JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def find_fastsurfer_installation():
    """
    Searches /Applications (e.g. /Applications/FastSurfer2.7.0.dev0) and standard paths for a
    macOS FastSurfer installation as documented in:
      https://github.com/Deep-MI/FastSurfer/blob/dev/doc/overview/install/MACOS.md
    Returns a dict with fastsurferHome, setupScript, cliPath, pythonBin, freesurferHome, version.
    """
    search_dirs = []
    env_home = os.environ.get("FASTSURFER_HOME", "").strip()
    if env_home and os.path.isdir(env_home):
        search_dirs.append(env_home)

    # Search /Applications/FastSurfer* directories (excluding .app launcher bundles)
    for match in sorted(glob.glob("/Applications/FastSurfer*"), reverse=True):
        if os.path.isdir(match) and not match.endswith(".app"):
            search_dirs.append(match)
    for match in sorted(glob.glob("/Applications/fastsurfer*"), reverse=True):
        if os.path.isdir(match) and not match.endswith(".app"):
            search_dirs.append(match)

    search_dirs.extend([
        "/opt/fastsurfer",
        "/usr/local/fastsurfer",
        os.path.expanduser("~/fastsurfer"),
        os.path.expanduser("~/FastSurfer"),
    ])

    for d in search_dirs:
        run_sh = os.path.join(d, "run_fastsurfer.sh")
        if os.path.exists(run_sh) and os.access(run_sh, os.X_OK):
            setup_sh = os.path.join(d, "macos_setup_fastsurfer.sh")
            py_bin = os.path.join(d, "python", "bin", "python3")
            fs_pruned = os.path.join(d, "fs-pruned")
            build_info = os.path.join(d, "BUILD.info")
            version_str = os.path.basename(d)
            if os.path.exists(build_info):
                try:
                    with open(build_info, "r", encoding="utf-8") as bf:
                        first_line = bf.readline().strip()
                        if first_line:
                            version_str = first_line
                except Exception:
                    pass
            return {
                "fastsurferHome": d,
                "setupScript": setup_sh if os.path.exists(setup_sh) else None,
                "cliPath": run_sh,
                "pythonBin": py_bin if os.path.exists(py_bin) else None,
                "freesurferHome": fs_pruned if os.path.isdir(fs_pruned) else (FREESURFER_DIR if os.path.isdir(FREESURFER_DIR) else None),
                "version": version_str,
            }

    # Also check PATH directly
    which_cli = shutil.which("run_fastsurfer.sh") or shutil.which("fastsurfer")
    if which_cli:
        d = os.path.dirname(os.path.abspath(which_cli))
        setup_sh = os.path.join(d, "macos_setup_fastsurfer.sh")
        return {
            "fastsurferHome": d,
            "setupScript": setup_sh if os.path.exists(setup_sh) else None,
            "cliPath": which_cli,
            "pythonBin": os.path.join(d, "python", "bin", "python3") if os.path.exists(os.path.join(d, "python", "bin", "python3")) else None,
            "freesurferHome": os.path.join(d, "fs-pruned") if os.path.isdir(os.path.join(d, "fs-pruned")) else FREESURFER_DIR,
            "version": os.path.basename(d),
        }
    return None


CACHED_FS_LICENSE_PATH = os.path.join(FASTSURFER_CACHE_DIR, "fs_license.txt")
CACHED_FS_LICENSE_META_PATH = os.path.join(FASTSURFER_CACHE_DIR, "fs_license_meta.json")


def validate_freesurfer_license_content(raw_text: str) -> tuple:
    """
    Validates that a text string conforms to the FreeSurfer license file format:
      - 3 to 5 non-empty lines:
        * Line 1: registered email address (contains '@')
        * Line 2: numeric license ID (digits)
        * Line 3 (and optional Line 4): cryptographic key tokens
    Returns (is_valid: bool, message: str).
    """
    if not raw_text or not isinstance(raw_text, str):
        return False, "License file is empty."
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if len(lines) < 3 or len(lines) > 6:
        return (
            False,
            f"Expected 3–4 non-empty lines in FreeSurfer license (email, numeric ID, key lines), found {len(lines)}.",
        )
    email_line = lines[0]
    id_line = lines[1]
    key_lines = lines[2:]
    if "@" not in email_line or len(email_line) > 128 or " " in email_line:
        return False, "Line 1 of FreeSurfer license must be a valid registered email address."
    if not re.match(r"^\d+$", id_line):
        return False, "Line 2 of FreeSurfer license must be a numeric license ID."
    if any(len(k) < 4 or len(k) > 128 for k in key_lines):
        return False, "Lines 3–4 of FreeSurfer license must contain valid license key strings."
    return True, "Valid FreeSurfer license"


def validate_freesurfer_license_file(path: str) -> tuple:
    """Checks that `path` exists on disk, is a regular file <= 16 KB, and contains a valid FreeSurfer license."""
    if not path or not os.path.isfile(path):
        return False, "File does not exist."
    try:
        if os.path.getsize(path) > 16384:
            return False, "File is too large to be a FreeSurfer license file."
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return validate_freesurfer_license_content(content)
    except Exception as e:
        return False, f"Cannot read license file: {e}"


def find_freesurfer_license(fs_install: dict = None) -> str:
    """
    Locates and validates a FreeSurfer license file required when running the FastSurfer surface pipeline (--fsaparc).
    Checks user-selected cached license first, then $FS_LICENSE, ~/license.txt, ~/.license, and $FREESURFER_HOME.
    """
    candidates = []
    # 1. Check if the user previously selected a license file in the UI
    if os.path.isfile(CACHED_FS_LICENSE_META_PATH):
        try:
            with open(CACHED_FS_LICENSE_META_PATH, "r", encoding="utf-8") as mf:
                meta = json.load(mf)
            if meta.get("originalPath") and os.path.isfile(meta["originalPath"]):
                candidates.append(meta["originalPath"])
        except Exception:
            pass
    candidates.append(CACHED_FS_LICENSE_PATH)

    # 2. Standard environment & home/FreeSurfer locations
    candidates.extend([
        os.environ.get("FS_LICENSE", "").strip(),
        os.path.expanduser("~/license.txt"),
        os.path.expanduser("~/.license"),
        os.path.expanduser("~/Downloads/license.txt"),
        os.path.expanduser("~/Desktop/license.txt"),
    ])
    if fs_install and fs_install.get("freesurferHome"):
        fsh = fs_install["freesurferHome"]
        candidates.extend([
            os.path.join(fsh, "license.txt"),
            os.path.join(fsh, ".license"),
            os.path.join(fsh, "license.dat"),
        ])
    if FREESURFER_DIR:
        candidates.extend([
            os.path.join(FREESURFER_DIR, "license.txt"),
            os.path.join(FREESURFER_DIR, ".license"),
            os.path.join(FREESURFER_DIR, "license.dat"),
            os.path.join(os.path.dirname(FREESURFER_DIR), "license.txt"),
        ])
    for c in candidates:
        if c:
            ok, _ = validate_freesurfer_license_file(c)
            if ok:
                return c
    return None


def build_fastsurfer_macos_env(fs_install: dict) -> dict:
    """
    Constructs the environment dictionary for FastSurfer across macOS and Linux following
    macos_setup_fastsurfer.sh / MACOS.md:
      - FASTSURFER_HOME=/Applications/FastSurfer... (or /opt/fastsurfer)
      - PYTHONPATH=$FASTSURFER_HOME:$PYTHONPATH
      - FREESURFER_HOME=$FASTSURFER_HOME/fs-pruned (or $FREESURFER_HOME)
      - FS_LICENSE=<path to license.txt> (if found, needed for surface pipeline / --fsaparc)
      - PATH=$FASTSURFER_HOME:$FASTSURFER_HOME/python/bin:$FREESURFER_HOME/bin:$PATH
    """
    env = os.environ.copy()
    if not fs_install:
        return env

    sep = os.pathsep
    fs_home = fs_install["fastsurferHome"]
    env["FASTSURFER_HOME"] = fs_home

    existing_py = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{fs_home}{sep}{existing_py}" if existing_py else fs_home

    freesurfer_home = fs_install.get("freesurferHome")
    if freesurfer_home and os.path.isdir(freesurfer_home):
        env["FREESURFER_HOME"] = freesurfer_home

    fs_lic = find_freesurfer_license(fs_install)
    if fs_lic:
        env["FS_LICENSE"] = fs_lic

    path_parts = []
    for p in [
        fs_home,
        os.path.join(fs_home, "python", "bin"),
        os.path.join(freesurfer_home, "bin") if freesurfer_home else "",
        os.path.join(freesurfer_home, "fsfast", "bin") if freesurfer_home else "",
        os.path.join(freesurfer_home, "mni", "bin") if freesurfer_home else "",
        "/opt/homebrew/opt/grep/libexec/gnubin",
        "/usr/local/opt/grep/libexec/gnubin",
    ]:
        if p and os.path.isdir(p) and p not in path_parts:
            path_parts.append(p)

    env["PATH"] = sep.join(path_parts + [env.get("PATH", "")])
    return env


def detect_neuro_tools() -> dict:
    """
    Checks availability of local neuroimaging tools across macOS, Linux, and Windows:
      1. FSL BET (or pure-Python SciPy 3D morphological skull stripping fallback)
      2. FastSurfer in /Applications, /opt/fastsurfer, PATH, or Docker (deepmi/fastsurfer)
      3. FSL FLIRT / SciPy + FreeSurfer DKT Atlas template fallback (MNI152_DKT_atlas_2mm.nii.gz)
    """
    bet_cli_ok = os.path.exists(BET_BIN) and os.access(BET_BIN, os.X_OK)
    fs_install = find_fastsurfer_installation()

    # Check Docker image deepmi/fastsurfer if native installation is absent
    docker_bin = shutil.which("docker") or (
        "/Applications/Docker.app/Contents/Resources/bin/docker"
        if os.path.exists("/Applications/Docker.app/Contents/Resources/bin/docker")
        else None
    )
    docker_fastsurfer_image = False
    if not fs_install and docker_bin:
        try:
            proc = subprocess.run(
                [docker_bin, "images", "-q", "deepmi/fastsurfer"],
                capture_output=True,
                text=True,
                timeout=2.5,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                docker_fastsurfer_image = True
        except Exception:
            pass

    fallback_ok = os.path.exists(MNI_DKT_ATLAS_PATH) and os.path.exists(MNI_T1_BRAIN_PATH)
    fs_lic_path = find_freesurfer_license(fs_install)
    fs_lic_label = os.path.basename(fs_lic_path) if fs_lic_path else None
    if fs_lic_path == CACHED_FS_LICENSE_PATH and os.path.isfile(CACHED_FS_LICENSE_META_PATH):
        try:
            with open(CACHED_FS_LICENSE_META_PATH, "r", encoding="utf-8") as mf:
                fs_lic_label = json.load(mf).get("fileName") or fs_lic_label
        except Exception:
            pass
    fs_lic_info = {
        "available": bool(fs_lic_path),
        "path": fs_lic_path,
        "label": fs_lic_label,
    }

    if fs_install:
        os.environ["FASTSURFER_HOME"] = fs_install["fastsurferHome"]
        native_mode = "macos_native" if (IS_MAC or fs_install.get("setupScript")) else "linux_native"
        fs_status = {
            "available": True,
            "mode": native_mode,
            "fastsurferHome": fs_install["fastsurferHome"],
            "setupScript": fs_install["setupScript"],
            "cliPath": fs_install["cliPath"],
            "pythonBin": fs_install["pythonBin"],
            "freesurferHome": fs_install["freesurferHome"],
            "version": fs_install["version"],
            "dockerImage": docker_fastsurfer_image,
            "fallbackAvailable": fallback_ok,
            "fsLicense": fs_lic_info,
            "statusLabel": f"FastSurfer Ready ({os.path.basename(fs_install['fastsurferHome'])})",
        }
    elif docker_fastsurfer_image:
        fs_status = {
            "available": True,
            "mode": "docker",
            "fastsurferHome": None,
            "setupScript": None,
            "cliPath": docker_bin,
            "dockerImage": True,
            "fallbackAvailable": fallback_ok,
            "fsLicense": fs_lic_info,
            "statusLabel": "FastSurfer Docker Image Ready (deepmi/fastsurfer)",
        }
    else:
        fs_status = {
            "available": False,
            "mode": "fsl_freesurfer_dkt_fallback",
            "fastsurferHome": None,
            "setupScript": None,
            "cliPath": None,
            "dockerImage": False,
            "fallbackAvailable": fallback_ok,
            "fsLicense": fs_lic_info,
            "statusLabel": "FastSurfer Not Found — Template DKT Atlas Ready",
        }

    return {
        "fslBet": {
            "available": True,
            "nativeCli": bet_cli_ok,
            "path": BET_BIN if bet_cli_ok else "SciPy 3D Morphological BET Fallback",
        },
        "fastsurfer": fs_status,
        "fsLicense": fs_lic_info,
    }


def fsl_sampling_mat(img: nib.Nifti1Image) -> np.ndarray:
    """
    Constructs the FSL FLIRT scaled-voxel sampling matrix S for a NIfTI image.
    If det(affine) > 0 (RAS / neurological), FSL flips the X axis so internal coordinates are radiological.
    """
    aff = get_best_affine(img)
    zooms = np.linalg.norm(aff[:3, :3], axis=0)
    zooms = np.where(zooms > 1e-6, zooms, 1.0)
    nx = max(1, int(img.shape[0]))
    s = np.diag([zooms[0], zooms[1], zooms[2], 1.0])
    if np.linalg.det(aff[:3, :3]) > 0:
        s[0, 0] = -zooms[0]
        s[0, 3] = (nx - 1) * zooms[0]
    return s


def flirt_mat_to_world_ras(fsl_mat: np.ndarray, ref_img: nib.Nifti1Image, mov_img: nib.Nifti1Image) -> np.ndarray:
    """
    Converts a 4x4 FSL FLIRT .mat (scaled-voxel mov -> scaled-voxel ref) into a 4x4 World RAS
    transformation matrix W_ras such that A_new = W_ras @ A_orig.
    """
    s_ref = fsl_sampling_mat(ref_img)
    s_mov = fsl_sampling_mat(mov_img)
    a_ref = get_best_affine(ref_img)
    a_mov = get_best_affine(mov_img)
    return a_ref @ safe_inv4x4(s_ref) @ fsl_mat @ s_mov @ safe_inv4x4(a_mov)


def world_ras_to_flirt_mat(w_ras: np.ndarray, ref_img: nib.Nifti1Image, mov_img: nib.Nifti1Image) -> np.ndarray:
    """
    Converts a 4x4 World RAS transformation matrix W_ras into a 4x4 FSL FLIRT .mat matrix.
    """
    s_ref = fsl_sampling_mat(ref_img)
    s_mov = fsl_sampling_mat(mov_img)
    a_ref = get_best_affine(ref_img)
    a_mov = get_best_affine(mov_img)
    return s_ref @ safe_inv4x4(a_ref) @ w_ras @ a_mov @ safe_inv4x4(s_mov)


def compute_alignment_metrics(ref_img: nib.Nifti1Image, mov_img: nib.Nifti1Image, extra_world_mat=None):
    """
    Computes quantitative multimodal alignment metrics (Normalized Mutual Information,
    Normalized Cross-Correlation, and 3D Sobel Edge Overlap) between ref_img and mov_img
    in Reference world (RAS) space, subsampled for speed. Safe on 3D/4D and NaN volumes.
    """
    ref_data = get_clean_3d_data(ref_img, dtype=np.float32)
    mov_data = get_clean_3d_data(mov_img, dtype=np.float32)

    ref_aff = get_best_affine(ref_img)
    mov_aff = get_best_affine(mov_img)
    if extra_world_mat is not None:
        mov_aff = np.asarray(extra_world_mat, dtype=np.float64) @ mov_aff

    vox_to_vox = safe_inv4x4(mov_aff) @ ref_aff

    mov_in_ref = affine_transform(
        mov_data,
        vox_to_vox[:3, :3],
        offset=vox_to_vox[:3, 3],
        output_shape=ref_data.shape[:3],
        order=1,
        mode="constant",
        cval=0.0,
    )

    step = 2 if min(ref_data.shape[:3]) >= 16 else 1
    r_flat = ref_data[::step, ::step, ::step].ravel()
    m_flat = mov_in_ref[::step, ::step, ::step].ravel()
    mask = (r_flat > np.percentile(r_flat, 15)) | (m_flat > np.percentile(m_flat, 15))
    if np.sum(mask) < 50:
        mask = np.ones_like(r_flat, dtype=bool)

    r_fg = r_flat[mask]
    m_fg = m_flat[mask]

    # 1. Normalized Mutual Information (NMI = (H(R) + H(M)) / H(R, M))
    hist_2d, _, _ = np.histogram2d(r_fg, m_fg, bins=32)
    pxy = hist_2d / (np.sum(hist_2d) + 1e-12)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    nz_x = px > 0
    nz_y = py > 0
    nz_xy = pxy > 0
    hx = -np.sum(px[nz_x] * np.log(px[nz_x]))
    hy = -np.sum(py[nz_y] * np.log(py[nz_y]))
    hxy = -np.sum(pxy[nz_xy] * np.log(pxy[nz_xy]))
    nmi = float((hx + hy) / (hxy + 1e-12))

    # 2. Absolute Normalized Cross-Correlation (|NCC|)
    r_norm = r_fg - np.mean(r_fg)
    m_norm = m_fg - np.mean(m_fg)
    denom = float(np.linalg.norm(r_norm) * np.linalg.norm(m_norm))
    ncc = float(np.abs(np.dot(r_norm, m_norm) / denom)) if denom > 1e-9 else 0.0

    # 3. 3D Gradient Edge Overlap (Cosine similarity of 3D Sobel gradient magnitudes)
    r_sub = ref_data[::step, ::step, ::step]
    m_sub = mov_in_ref[::step, ::step, ::step]
    r_grad = np.hypot(np.hypot(sobel(r_sub, 0), sobel(r_sub, 1)), sobel(r_sub, 2)).ravel()
    m_grad = np.hypot(np.hypot(sobel(m_sub, 0), sobel(m_sub, 1)), sobel(m_sub, 2)).ravel()
    g_denom = float(np.linalg.norm(r_grad) * np.linalg.norm(m_grad))
    edge_overlap = float(np.dot(r_grad, m_grad) / g_denom) if g_denom > 1e-9 else 0.0

    return {
        "nmi": round(nmi if np.isfinite(nmi) else 1.0, 4),
        "ncc": round(ncc if np.isfinite(ncc) else 0.0, 4),
        "edgeOverlap": round(edge_overlap if np.isfinite(edge_overlap) else 0.0, 4),
    }, mov_in_ref


def params_to_world_matrix(params, center_ras):
    """
    Converts [tx, ty, tz (mm), rx, ry, rz (deg), sx, sy, sz, shx, shy, shz] around center_ras
    into a 4x4 world RAS transformation matrix (supports 6-DOF, 9-DOF, and 12-DOF).
    """
    tx, ty, tz = params[0], params[1], params[2]
    rx, ry, rz = np.deg2rad(params[3]), np.deg2rad(params[4]), np.deg2rad(params[5])
    sx = params[6] if len(params) > 6 else 1.0
    sy = params[7] if len(params) > 7 else 1.0
    sz = params[8] if len(params) > 8 else 1.0
    shx = params[9] if len(params) > 9 else 0.0
    shy = params[10] if len(params) > 10 else 0.0
    shz = params[11] if len(params) > 11 else 0.0

    cx, sx_r = np.cos(rx), np.sin(rx)
    cy, sy_r = np.cos(ry), np.sin(ry)
    cz, sz_r = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx_r], [0, sx_r, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy_r], [0, 1, 0], [-sy_r, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz_r, 0], [sz_r, cz, 0], [0, 0, 1]], dtype=np.float64)
    S = np.diag([sx, sy, sz])
    Sh = np.array([[1.0, shx, shy], [0.0, 1.0, shz], [0.0, 0.0, 1.0]], dtype=np.float64)

    R = Rz @ Ry @ Rx @ S @ Sh

    T_to_origin = np.eye(4, dtype=np.float64)
    T_to_origin[:3, 3] = -center_ras

    M_rot = np.eye(4, dtype=np.float64)
    M_rot[:3, :3] = R

    T_back = np.eye(4, dtype=np.float64)
    T_back[:3, 3] = center_ras + np.array([tx, ty, tz], dtype=np.float64)

    return T_back @ M_rot @ T_to_origin


def optimize_multimodal_registration(ref_img: nib.Nifti1Image, mov_img: nib.Nifti1Image, dof=6):
    """
    Fast multi-resolution 3D Normalized Mutual Information registration (supports 6, 9, and 12 DOF).
    Returns the optimal 9-parameter UI vector [tx, ty, tz, rx, ry, rz, sx, sy, sz],
    the 4x4 World RAS transform W_best, and the updated 4x4 NIfTI affine matrix for mov_img.
    """
    ref_data = get_clean_3d_data(ref_img, dtype=np.float32)
    mov_data = get_clean_3d_data(mov_img, dtype=np.float32)
    ref_aff = get_best_affine(ref_img)
    mov_aff = get_best_affine(mov_img)

    # Pivot around the Moving volume geometric center in RAS world space (matching frontend)
    nx, ny, nz = mov_data.shape[:3]
    center_ras = (mov_aff @ np.array([(nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0, 1.0]))[:3]

    r_thresh = ref_data > np.percentile(ref_data, 50)
    m_thresh = mov_data > np.percentile(mov_data, 50)
    r_ijk = np.argwhere(r_thresh).mean(axis=0) if np.any(r_thresh) else np.array(ref_data.shape[:3]) / 2.0
    m_ijk = np.argwhere(m_thresh).mean(axis=0) if np.any(m_thresh) else np.array(mov_data.shape[:3]) / 2.0

    r_com_ras = (ref_aff @ np.append(r_ijk[:3], 1.0))[:3]
    m_com_ras = (mov_aff @ np.append(m_ijk[:3], 1.0))[:3]
    init_shift = r_com_ras - m_com_ras

    step = 3 if min(ref_data.shape[:3]) >= 24 else 1
    r_sub = ref_data[::step, ::step, ::step]
    r_sub_aff = ref_aff.copy()
    r_sub_aff[:3, :3] *= step

    inv_mov_aff = safe_inv4x4(mov_aff)
    n_params = 6 if dof <= 6 else (9 if dof == 9 else 12)

    def cost_fn(p):
        full_p = np.zeros(12, dtype=np.float64)
        full_p[:6] = p[:6]
        full_p[6:9] = np.clip(p[6:9], 0.7, 1.4) if len(p) >= 9 else [1.0, 1.0, 1.0]
        full_p[9:12] = np.clip(p[9:12], -0.25, 0.25) if len(p) >= 12 else [0.0, 0.0, 0.0]
        W = params_to_world_matrix(full_p, center_ras)
        W_inv = safe_inv4x4(W)
        v2v = inv_mov_aff @ W_inv @ r_sub_aff
        m_warped = affine_transform(
            mov_data,
            v2v[:3, :3],
            offset=v2v[:3, 3],
            output_shape=r_sub.shape[:3],
            order=1,
            mode="constant",
            cval=0.0,
        )
        rf = r_sub.ravel()
        mf = m_warped.ravel()
        hist_2d, _, _ = np.histogram2d(rf, mf, bins=24)
        pxy = hist_2d / (np.sum(hist_2d) + 1e-12)
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        nz_x, nz_y, nz_xy = px > 0, py > 0, pxy > 0
        hx = -np.sum(px[nz_x] * np.log(px[nz_x]))
        hy = -np.sum(py[nz_y] * np.log(py[nz_y]))
        hxy = -np.sum(pxy[nz_xy] * np.log(pxy[nz_xy]))
        nmi = (hx + hy) / (hxy + 1e-12)
        return -nmi

    p0_ident = np.zeros(n_params, dtype=np.float64)
    p0_com = np.zeros(n_params, dtype=np.float64)
    p0_com[:3] = init_shift
    if n_params >= 9:
        p0_ident[6:9] = [1.0, 1.0, 1.0]
        p0_com[6:9] = [1.0, 1.0, 1.0]

    p_start = p0_com if cost_fn(p0_com) < cost_fn(p0_ident) else p0_ident

    res = minimize(
        cost_fn,
        p_start,
        method="Powell",
        options={"maxiter": 45, "ftol": 1e-3, "xtol": 1e-2},
    )
    best_p = res.x
    full_best_12 = np.zeros(12, dtype=np.float64)
    full_best_12[:6] = best_p[:6]
    full_best_12[6:9] = np.clip(best_p[6:9], 0.7, 1.4) if len(best_p) >= 9 else [1.0, 1.0, 1.0]
    full_best_12[9:12] = np.clip(best_p[9:12], -0.25, 0.25) if len(best_p) >= 12 else [0.0, 0.0, 0.0]

    W_best = params_to_world_matrix(full_best_12, center_ras)
    new_mov_affine = W_best @ mov_aff
    return full_best_12[:9], W_best, new_mov_affine


class MultimodalHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/sample-manifest":
            dicom_dir = os.path.join(ROOT_DIR, "sample_data", "dicom_t2_flair")
            dcm_files = sorted([f for f in os.listdir(dicom_dir) if f.endswith(".dcm")]) if os.path.exists(dicom_dir) else []
            payload = {
                "subject": "sub-01 (Single-Subject Multimodal Brain Dataset)",
                "niftiVolumes": [
                    {
                        "id": "t1w",
                        "name": "sub-01_T1w.nii.gz",
                        "label": "Pre-Op T1w Structural MRI (Reference)",
                        "modality": "T1w MRI",
                        "url": "/sample_data/nifti/sub-01_T1w.nii.gz",
                        "colormap": "gray",
                        "opacity": 1.0,
                        "role": "reference",
                    },
                    {
                        "id": "ct_misaligned",
                        "name": "sub-01_CT_postop_misaligned.nii.gz",
                        "label": "Post-Op CT + Skull (Moving)",
                        "modality": "CT (Misaligned)",
                        "url": "/sample_data/nifti/sub-01_CT_postop_misaligned.nii.gz",
                        "colormap": "hot",
                        "opacity": 0.65,
                        "cal_min": 400,
                        "cal_max": 2200,
                        "role": "moving",
                    },
                    {
                        "id": "seeg_contacts_child",
                        "name": "sub-01_CT_SEEG_contacts_mask.nii.gz",
                        "label": "SEEG Contacts Mask (Child of CT)",
                        "modality": "ROI / Contacts Mask",
                        "url": "/sample_data/nifti/sub-01_CT_SEEG_contacts_mask.nii.gz",
                        "colormap": "green",
                        "opacity": 0.9,
                        "cal_min": 0.5,
                        "cal_max": 2.0,
                        "role": "child",
                        "isMask": True,
                    },
                    {
                        "id": "t2_flair",
                        "name": "sub-01_T2_FLAIR.nii.gz",
                        "label": "3D T2-FLAIR MRI",
                        "modality": "T2-FLAIR",
                        "url": "/sample_data/nifti/sub-01_T2_FLAIR.nii.gz",
                        "colormap": "viridis",
                        "opacity": 0.0,
                        "visible": False,
                        "role": "overlay",
                    },
                    {
                        "id": "dti_fa",
                        "name": "sub-01_DTI_FA.nii.gz",
                        "label": "DTI Fractional Anisotropy (FA)",
                        "modality": "DTI FA",
                        "url": "/sample_data/nifti/sub-01_DTI_FA.nii.gz",
                        "colormap": "plasma",
                        "opacity": 0.0,
                        "visible": False,
                        "cal_min": 0.15,
                        "cal_max": 0.85,
                        "role": "overlay",
                    },
                ],
                "sampleDicomFolder": {
                    "folderName": "sample_data/dicom_t2_flair",
                    "seriesDescription": "Ax_T2_FLAIR_2mm (91 DICOM Slices)",
                    "sliceCount": len(dcm_files),
                    "files": [f"/sample_data/dicom_t2_flair/{f}" for f in dcm_files],
                },
            }
            self._send_json(200, payload)
            return
        if parsed.path == "/api/neuro-tools-status":
            status = detect_neuro_tools()
            status["dktLut"] = load_dkt_lut()
            self._send_json(200, status)
            return
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length) if length > 0 else b"{}"

        try:
            body = json.loads(raw_body.decode("utf-8"))
        except Exception as e:
            self._send_json(400, {"error": f"Invalid JSON payload: {e}"})
            return

        if parsed.path == "/api/convert-dicom":
            self._handle_convert_dicom(body)
        elif parsed.path == "/api/coregister":
            self._handle_coregister(body)
        elif parsed.path == "/api/export-nifti":
            self._handle_export_nifti(body)
        elif parsed.path == "/api/metrics":
            self._handle_metrics(body)
        elif parsed.path == "/api/save-project":
            self._handle_save_project(body)
        elif parsed.path == "/api/open-project":
            self._handle_open_project(body)
        elif parsed.path == "/api/skull-strip-bet":
            self._handle_skull_strip_bet(body)
        elif parsed.path == "/api/dkt-segment":
            self._handle_dkt_segment(body)
        elif parsed.path == "/api/set-fs-license":
            self._handle_set_fs_license(body)
        elif parsed.path == "/api/convert-volume":
            self._handle_convert_volume(body)
        elif parsed.path == "/api/drawing-save":
            self._handle_drawing_save(body)
        elif parsed.path == "/api/drawing-import":
            self._handle_drawing_import(body)
        elif parsed.path == "/api/drawing-morph":
            self._handle_drawing_morph(body)
        elif parsed.path == "/api/shutdown":
            self._handle_shutdown(body)
        else:
            self._send_json(404, {"error": "Unknown API endpoint"})

    def _handle_convert_dicom(self, body):
        """
        Converts uploaded DICOM slices (base64 list) or sample DICOM folder into NIfTI (.nii.gz)
        using Chris Rorden's dcm2niix, with automatic fallback to pydicom + nibabel if dcm2niix
        is absent or fails on the host OS.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            in_dir = os.path.join(tmpdir, "dcm_in")
            out_dir = os.path.join(tmpdir, "nii_out")
            os.makedirs(in_dir, exist_ok=True)
            os.makedirs(out_dir, exist_ok=True)

            if body.get("useSampleDicom"):
                src_dir = os.path.join(ROOT_DIR, "sample_data", "dicom_t2_flair")
                if os.path.exists(src_dir):
                    for fname in os.listdir(src_dir):
                        if fname.endswith(".dcm"):
                            shutil.copy2(os.path.join(src_dir, fname), os.path.join(in_dir, fname))
            else:
                files = body.get("files", [])
                if not files:
                    self._send_json(400, {"error": "No DICOM files provided"})
                    return
                for idx, fobj in enumerate(files):
                    safe_name = os.path.basename(fobj.get("name", f"slice_{idx:04d}.dcm"))
                    b64_data = fobj.get("dataBase64", "")
                    with open(os.path.join(in_dir, f"{idx:04d}_{safe_name}"), "wb") as f:
                        f.write(base64.b64decode(b64_data))

            stdout_log = ""
            stderr_log = ""
            if os.path.exists(DCM2NIIX_BIN) and os.access(DCM2NIIX_BIN, os.X_OK):
                try:
                    cmd = [DCM2NIIX_BIN, "-z", "y", "-f", "%d_%s", "-o", out_dir, in_dir]
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    stdout_log = proc.stdout or ""
                    stderr_log = proc.stderr or ""
                except Exception as e:
                    stderr_log = str(e)

            converted = []
            for fname in sorted(os.listdir(out_dir)):
                if fname.lower().endswith((".nii.gz", ".nii")):
                    fpath = os.path.join(out_dir, fname)
                    with open(fpath, "rb") as nf:
                        nii_bytes = nf.read()
                    base_no_ext = strip_neuro_ext(fname)
                    json_path = os.path.join(out_dir, base_no_ext + ".json")
                    meta = {}
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, "r", encoding="utf-8") as jf:
                                meta = json.load(jf)
                        except Exception:
                            pass
                    img = nib.load(fpath)
                    zooms = list(img.header.get_zooms())
                    while len(zooms) < 3:
                        zooms.append(1.0)
                    converted.append({
                        "name": fname,
                        "shape": list(img.shape),
                        "voxelSize": [round(float(x), 3) for x in zooms[:3]],
                        "metadata": meta,
                        "niftiBase64": base64.b64encode(nii_bytes).decode("ascii"),
                    })

            if not converted:
                # Fallback to pure-Python pydicom + nibabel converter
                try:
                    converted, py_log = convert_dicom_dir_pydicom(in_dir, out_dir)
                    stdout_log = (stdout_log + "\n" + py_log).strip()
                except Exception as py_err:
                    stderr_log = (stderr_log + f"\npydicom fallback error: {py_err}").strip()

            if not converted:
                self._send_json(500, {
                    "error": "DICOM conversion did not produce any NIfTI volumes",
                    "stdout": stdout_log,
                    "stderr": stderr_log,
                })
                return

            self._send_json(200, {
                "volumes": converted,
                "log": stdout_log,
            })

    def _handle_metrics(self, body):
        """Computes quantitative multimodal alignment metrics between Reference and Moving NIfTI buffers."""
        try:
            ref_bytes = base64.b64decode(body["refBase64"])
            mov_bytes = base64.b64decode(body["movBase64"])
            extra_mat = body.get("worldMatrix", None)

            with tempfile.TemporaryDirectory() as tmpdir:
                ref_ext = ".nii.gz" if body.get("refName", "").lower().endswith(".gz") or ref_bytes[:2] == b"\x1f\x8b" else ".nii"
                mov_ext = ".nii.gz" if body.get("movName", "").lower().endswith(".gz") or mov_bytes[:2] == b"\x1f\x8b" else ".nii"
                ref_path = os.path.join(tmpdir, f"ref{ref_ext}")
                mov_path = os.path.join(tmpdir, f"mov{mov_ext}")
                with open(ref_path, "wb") as f:
                    f.write(ref_bytes)
                with open(mov_path, "wb") as f:
                    f.write(mov_bytes)

                ref_img = nib.load(ref_path)
                mov_img = nib.load(mov_path)
                metrics, _ = compute_alignment_metrics(ref_img, mov_img, extra_world_mat=extra_mat)
                self._send_json(200, {"metrics": metrics})
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def _handle_coregister(self, body):
        """
        Runs automated multimodal coregistration between Reference and Moving volumes.
        Returns both the 4x4 World RAS matrix (W_ras) AND the 4x4 FSL FLIRT .mat matrix
        so transformations can be seamlessly applied to any child images!
        """
        try:
            ref_bytes = base64.b64decode(body["refBase64"])
            mov_bytes = base64.b64decode(body["movBase64"])
            mode = body.get("mode", "nmi_powell")
            dof = int(body.get("dof", 6))
            cost = body.get("cost", "normmi")

            with tempfile.TemporaryDirectory() as tmpdir:
                ref_ext = ".nii.gz" if body.get("refName", "").lower().endswith(".gz") or ref_bytes[:2] == b"\x1f\x8b" else ".nii"
                mov_ext = ".nii.gz" if body.get("movName", "").lower().endswith(".gz") or mov_bytes[:2] == b"\x1f\x8b" else ".nii"
                ref_path = os.path.join(tmpdir, f"ref{ref_ext}")
                mov_path = os.path.join(tmpdir, f"mov{mov_ext}")
                with open(ref_path, "wb") as f:
                    f.write(ref_bytes)
                with open(mov_path, "wb") as f:
                    f.write(mov_bytes)

                ref_img = nib.load(ref_path)
                mov_img = nib.load(mov_path)

                metrics_before, _ = compute_alignment_metrics(ref_img, mov_img)

                if mode == "fsl_flirt" and os.path.exists(FLIRT_BIN) and os.access(FLIRT_BIN, os.X_OK):
                    # Ensure 3D sanitized inputs for FSL FLIRT in case 4D or NaN-containing NIfTI was passed
                    ref_3d_path = os.path.join(tmpdir, "ref_3d.nii.gz")
                    mov_3d_path = os.path.join(tmpdir, "mov_3d.nii.gz")
                    ref_aff = get_best_affine(ref_img)
                    mov_aff = get_best_affine(mov_img)
                    nib.save(nib.Nifti1Image(get_clean_3d_data(ref_img), ref_aff), ref_3d_path)
                    nib.save(nib.Nifti1Image(get_clean_3d_data(mov_img), mov_aff), mov_3d_path)

                    out_path = os.path.join(tmpdir, "coreg_out.nii.gz")
                    mat_path = os.path.join(tmpdir, "coreg.mat")
                    env = os.environ.copy()
                    env["FSLDIR"] = FSLDIR
                    env["FSLOUTPUTTYPE"] = "NIFTI_GZ"
                    cmd = [
                        FLIRT_BIN,
                        "-in", mov_3d_path,
                        "-ref", ref_3d_path,
                        "-out", out_path,
                        "-omat", mat_path,
                        "-bins", "256",
                        "-cost", cost,
                        "-searchrx", "-30", "30",
                        "-searchry", "-30", "30",
                        "-searchrz", "-30", "30",
                        "-dof", str(dof),
                        "-interp", "trilinear",
                    ]
                    subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60)
                    if os.path.exists(mat_path):
                        fsl_mat = np.loadtxt(mat_path)
                        W_ras = flirt_mat_to_world_ras(fsl_mat, ref_img, mov_img)
                        new_mov_affine = W_ras @ mov_aff
                        metrics_after, _ = compute_alignment_metrics(ref_img, mov_img, extra_world_mat=W_ras)
                        self._send_json(200, {
                            "engine": f"FSL FLIRT ({cost}, {dof}-DOF)",
                            "metricsBefore": metrics_before,
                            "metricsAfter": metrics_after,
                            "worldMatrix": W_ras.tolist(),
                            "fslMat": fsl_mat.tolist(),
                            "newAffine": new_mov_affine.tolist(),
                        })
                        return

                # Default / NMI Powell Multi-Resolution Optimizer (supports 6, 9, and 12 DOF on all OSes)
                best_params, W_best, new_mov_affine = optimize_multimodal_registration(ref_img, mov_img, dof=dof)
                fsl_mat = world_ras_to_flirt_mat(W_best, ref_img, mov_img)
                metrics_after, _ = compute_alignment_metrics(ref_img, mov_img, extra_world_mat=W_best)

                self._send_json(200, {
                    "engine": f"3D Normalized Mutual Information Powell ({dof}-DOF)",
                    "metricsBefore": metrics_before,
                    "metricsAfter": metrics_after,
                    "worldMatrix": W_best.tolist(),
                    "fslMat": fsl_mat.tolist(),
                    "newAffine": new_mov_affine.tolist(),
                    "params": [round(float(x), 3) for x in best_params],
                })
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def _handle_export_nifti(self, body):
        """
        Exports a Moving or Child NIfTI volume (.nii.gz) with either:
          - exportMode='header_only': Updates sform/qform affine matrix losslessly without resampling
          - exportMode='resample_to_ref': Resamples volume onto Reference image's voxel grid and affine
            using interp='trilinear' (order=1) or 'nearest' (order=0).
        """
        try:
            src_bytes = base64.b64decode(body["sourceBase64"])
            new_affine = np.asarray(body["newAffine"], dtype=np.float64)
            export_mode = body.get("exportMode", "header_only")
            interp = body.get("interp", "trilinear")
            compress = bool(body.get("compress", True))

            with tempfile.TemporaryDirectory() as tmpdir:
                src_ext = ".nii.gz" if src_bytes[:2] == b"\x1f\x8b" else ".nii"
                src_path = os.path.join(tmpdir, f"src{src_ext}")
                with open(src_path, "wb") as f:
                    f.write(src_bytes)

                src_img = nib.load(src_path)
                src_data = np.asarray(src_img.dataobj)
                if np.issubdtype(src_data.dtype, np.floating):
                    src_data = np.nan_to_num(src_data, nan=0.0, posinf=0.0, neginf=0.0)

                out_ext = ".nii.gz" if compress else ".nii"
                out_path = os.path.join(tmpdir, f"exported{out_ext}")

                if export_mode == "resample_to_ref" and body.get("refBase64"):
                    ref_bytes = base64.b64decode(body["refBase64"])
                    ref_ext = ".nii.gz" if ref_bytes[:2] == b"\x1f\x8b" else ".nii"
                    ref_path = os.path.join(tmpdir, f"ref{ref_ext}")
                    with open(ref_path, "wb") as f:
                        f.write(ref_bytes)
                    ref_img = nib.load(ref_path)
                    ref_aff = get_best_affine(ref_img)
                    ref_3d_shape = get_clean_3d_data(ref_img).shape

                    # Map Reference voxel indices -> World RAS -> Source voxel indices (using new_affine)
                    vox_to_vox = safe_inv4x4(new_affine) @ ref_aff
                    order = 0 if interp == "nearest" else 1
                    src_3d = get_clean_3d_data(src_img, dtype=np.float32)
                    resampled = affine_transform(
                        src_3d,
                        vox_to_vox[:3, :3],
                        offset=vox_to_vox[:3, 3],
                        output_shape=ref_3d_shape,
                        order=order,
                        mode="constant",
                        cval=0.0,
                    )
                    if order == 0 and np.issubdtype(src_data.dtype, np.integer):
                        resampled = np.rint(resampled).astype(src_data.dtype)
                    out_nii = nib.Nifti1Image(resampled, ref_aff)
                    out_nii.set_sform(ref_aff, code=1)
                    out_nii.set_qform(ref_aff, code=1)
                else:
                    # Header-only update (lossless voxel data, updated sform & qform)
                    out_nii = nib.Nifti1Image(src_data, new_affine, header=src_img.header)
                    out_nii.set_sform(new_affine, code=1)
                    out_nii.set_qform(new_affine, code=1)

                nib.save(out_nii, out_path)
                with open(out_path, "rb") as f:
                    out_b64 = base64.b64encode(f.read()).decode("ascii")

                self._send_json(200, {
                    "niftiBase64": out_b64,
                    "shape": list(out_nii.shape),
                    "affine": out_nii.affine.tolist(),
                })
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def _pick_native_folder(self, prompt_text):
        """
        Attempts to open a native OS folder selection dialog across macOS, Linux, and Windows:
          - macOS: osascript (AppleScript Finder choose folder)
          - Linux: zenity or kdialog
          - Windows: PowerShell System.Windows.Forms.FolderBrowserDialog
          - Fallback: Python tkinter.filedialog (if available with a display)
        Returns (chosen_path, user_cancelled).
        """
        sys_plat = platform.system()
        safe_prompt = prompt_text.replace('"', '\\"').replace("'", "")

        if sys_plat == "Darwin" and os.path.exists("/usr/bin/osascript"):
            try:
                script = (
                    'tell current application\n'
                    'activate\n'
                    f'POSIX path of (choose folder with prompt "{safe_prompt}")\n'
                    'end tell'
                )
                proc = subprocess.run(
                    ["/usr/bin/osascript", "-e", script],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    return proc.stdout.strip(), False
                stderr_low = (proc.stderr or "").lower()
                if "user canceled" in stderr_low or "(-128)" in stderr_low:
                    return None, True
                if proc.returncode != 0:
                    print(
                        f"[NeuroCoreg] osascript folder dialog unavailable (rc={proc.returncode}): {(proc.stderr or '').strip()}",
                        flush=True,
                    )
            except Exception:
                pass

        elif sys_plat == "Linux" and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            zenity_bin = shutil.which("zenity")
            if zenity_bin:
                try:
                    proc = subprocess.run(
                        [zenity_bin, "--file-selection", "--directory", f"--title={safe_prompt}"],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    if proc.returncode == 0 and proc.stdout.strip():
                        return proc.stdout.strip(), False
                    if proc.returncode == 1:
                        return None, True
                except Exception:
                    pass
            kdialog_bin = shutil.which("kdialog")
            if kdialog_bin:
                try:
                    proc = subprocess.run(
                        [kdialog_bin, "--getexistingdirectory", os.path.expanduser("~"), "--title", safe_prompt],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    if proc.returncode == 0 and proc.stdout.strip():
                        return proc.stdout.strip(), False
                    if proc.returncode == 1:
                        return None, True
                except Exception:
                    pass

        elif sys_plat == "Windows":
            ps_bin = shutil.which("powershell") or shutil.which("pwsh")
            if ps_bin:
                try:
                    ps_cmd = (
                        "Add-Type -AssemblyName System.Windows.Forms; "
                        "$f = New-Object System.Windows.Forms.FolderBrowserDialog; "
                        f"$f.Description = '{safe_prompt}'; "
                        "$f.ShowNewFolderButton = $true; "
                        "if ($f.ShowDialog() -eq 'OK') { Write-Output $f.SelectedPath } else { exit 2 }"
                    )
                    proc = subprocess.run(
                        [ps_bin, "-NoProfile", "-STA", "-Command", ps_cmd],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    if proc.returncode == 0 and proc.stdout.strip():
                        return proc.stdout.strip(), False
                    if proc.returncode == 2:
                        return None, True
                except Exception:
                    pass

        # Cross-platform tkinter fallback (only when a display environment exists)
        if sys_plat in ("Darwin", "Windows") or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            try:
                tk_code = (
                    "import sys, tkinter as tk; from tkinter import filedialog; "
                    "root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True); "
                    f"p = filedialog.askdirectory(title={repr(safe_prompt)}); "
                    "root.destroy(); "
                    "print(p) if p else sys.exit(2)"
                )
                proc = subprocess.run(
                    [sys.executable, "-c", tk_code],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    return proc.stdout.strip(), False
                if proc.returncode == 2:
                    return None, True
            except Exception:
                pass

        return None, False

    def _pick_native_save_file(self, prompt_text: str, default_filename: str, default_ext: str = ".voi"):
        """
        Cross-platform native Save File dialog across macOS, Linux, and Windows.
        Supports mounted network drives (/Volumes, /mnt, Z:\\, \\\\server\\share).
        Returns (chosen_path, user_cancelled).
        """
        sys_plat = platform.system()
        safe_prompt = (prompt_text or "Save file").replace('"', '\\"').replace("'", "")
        safe_name = os.path.basename(default_filename or f"drawing{default_ext}").replace('"', "").replace("'", "")
        norm_ext = default_ext if default_ext.startswith(".") else f".{default_ext}"

        chosen = None
        cancelled = False

        if sys_plat == "Darwin" and os.path.exists("/usr/bin/osascript"):
            try:
                script = (
                    'tell current application\n'
                    'activate\n'
                    f'POSIX path of (choose file name with prompt "{safe_prompt}" default name "{safe_name}")\n'
                    'end tell'
                )
                proc = subprocess.run(
                    ["/usr/bin/osascript", "-e", script],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    chosen = proc.stdout.strip()
                else:
                    stderr_low = (proc.stderr or "").lower()
                    if "user canceled" in stderr_low or "(-128)" in stderr_low:
                        return None, True
                    if proc.returncode != 0:
                        print(
                            f"[NeuroCoreg] osascript save dialog unavailable (rc={proc.returncode}): {(proc.stderr or '').strip()}",
                            flush=True,
                        )
            except Exception:
                pass

        elif sys_plat == "Linux" and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            zenity_bin = shutil.which("zenity")
            if zenity_bin:
                try:
                    proc = subprocess.run(
                        [
                            zenity_bin,
                            "--file-selection",
                            "--save",
                            "--confirm-overwrite",
                            f"--filename={safe_name}",
                            f"--title={safe_prompt}",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    if proc.returncode == 0 and proc.stdout.strip():
                        chosen = proc.stdout.strip()
                    elif proc.returncode == 1:
                        return None, True
                except Exception:
                    pass
            if not chosen:
                kdialog_bin = shutil.which("kdialog")
                if kdialog_bin:
                    try:
                        proc = subprocess.run(
                            [kdialog_bin, "--getsavefilename", os.path.join(os.path.expanduser("~"), safe_name), f"*{norm_ext}", "--title", safe_prompt],
                            capture_output=True,
                            text=True,
                            timeout=60,
                        )
                        if proc.returncode == 0 and proc.stdout.strip():
                            chosen = proc.stdout.strip()
                        elif proc.returncode == 1:
                            return None, True
                    except Exception:
                        pass

        elif sys_plat == "Windows":
            ps_bin = shutil.which("powershell") or shutil.which("pwsh")
            if ps_bin:
                try:
                    ps_cmd = (
                        "Add-Type -AssemblyName System.Windows.Forms; "
                        "$s = New-Object System.Windows.Forms.SaveFileDialog; "
                        f"$s.Title = '{safe_prompt}'; "
                        f"$s.FileName = '{safe_name}'; "
                        "$s.OverwritePrompt = $true; "
                        "$s.Filter = 'MRIcroGL VOI (*.voi)|*.voi|Gzipped NIfTI (*.nii.gz)|*.nii.gz|NIfTI (*.nii)|*.nii|All files (*.*)|*.*'; "
                        "if ($s.ShowDialog() -eq 'OK') { Write-Output $s.FileName } else { exit 2 }"
                    )
                    proc = subprocess.run(
                        [ps_bin, "-NoProfile", "-STA", "-Command", ps_cmd],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    if proc.returncode == 0 and proc.stdout.strip():
                        chosen = proc.stdout.strip()
                    elif proc.returncode == 2:
                        return None, True
                except Exception:
                    pass

        if not chosen and not cancelled and (
            sys_plat in ("Darwin", "Windows") or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            try:
                tk_code = (
                    "import sys, tkinter as tk; from tkinter import filedialog; "
                    "root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True); "
                    f"p = filedialog.asksaveasfilename(title={repr(safe_prompt)}, initialfile={repr(safe_name)}, defaultextension={repr(norm_ext)}); "
                    "root.destroy(); "
                    "print(p) if p else sys.exit(2)"
                )
                proc = subprocess.run(
                    [sys.executable, "-c", tk_code],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    chosen = proc.stdout.strip()
                elif proc.returncode == 2:
                    return None, True
            except Exception:
                pass

        if not chosen:
            return None, False

        chosen = normalize_user_path(chosen)
        lower = chosen.lower()
        if not (lower.endswith(".voi") or lower.endswith(".nii.gz") or lower.endswith(".nii")):
            chosen = f"{chosen}{norm_ext}"
        return chosen, False

    def _handle_save_project(self, body):
        """
        Saves the full NeuroCoreg workspace bundle into a user-chosen local or mounted network folder:
          - project.json (manifest with roles, colormaps, opacity, windowing, affines, metrics)
          - nifti/*.nii.gz (volumes with updated coregistered sform/qform headers)
          - transforms/*.mat (World RAS and FSL FLIRT 4x4 matrices)
        Writes NIfTI files in a local temporary directory first and streams sequential bytes via
        shutil.copyfile so mounted network shares (SMB/CIFS, NFS, WebDAV, SSHFS, UNC paths) never
        fail on gzip header seeks, cross-device links, or chmod/utime metadata restrictions.
        """
        try:
            target_dir = body.get("targetDir")
            patient_id = (body.get("patientId") or "sub-01").strip()
            safe_patient = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in patient_id) or "sub-01"

            if not target_dir:
                chosen_dir, cancelled = self._pick_native_folder(
                    f"Select destination folder to save NeuroCoreg project ({safe_patient}):"
                )
                if cancelled:
                    self._send_json(200, {"cancelled": True})
                    return
                if not chosen_dir:
                    default_dir = get_default_projects_dir(f"{safe_patient}_project")
                    self._send_json(200, {"needsPath": True, "defaultDir": default_dir})
                    return
                chosen_dir = normalize_user_path(chosen_dir)
                # Create a clean project subfolder inside the chosen parent directory unless already named for it
                base_chosen = os.path.basename(os.path.normpath(chosen_dir))
                if base_chosen == f"{safe_patient}_project" or base_chosen == safe_patient:
                    target_dir = chosen_dir
                else:
                    target_dir = os.path.join(chosen_dir, f"{safe_patient}_project")

            target_dir = normalize_user_path(target_dir)
            nifti_dir = os.path.join(target_dir, "nifti")
            transforms_dir = os.path.join(target_dir, "transforms")
            os.makedirs(nifti_dir, exist_ok=True)
            os.makedirs(transforms_dir, exist_ok=True)

            manifest = body.get("manifest", {})
            volumes_payload = body.get("volumes", [])
            transforms_payload = body.get("transformFiles", [])
            drawings_payload = body.get("drawingFiles", [])

            saved_nifti_files = []
            saved_drawing_files = []
            with tempfile.TemporaryDirectory() as tmpdir:
                for idx, v in enumerate(volumes_payload):
                    raw_fname = os.path.basename(v.get("fileName") or f"volume_{idx}.nii.gz")
                    fname = f"{strip_neuro_ext(raw_fname)}.nii.gz"

                    raw_b64 = v.get("rawBase64")
                    curr_aff = np.asarray(v.get("currentAffine", np.eye(4)), dtype=np.float64)
                    dest_path = os.path.join(nifti_dir, fname)

                    if raw_b64:
                        src_bytes = base64.b64decode(raw_b64)
                        src_ext = ".nii.gz" if src_bytes[:2] == b"\x1f\x8b" else ".nii"
                        tmp_src = os.path.join(tmpdir, f"src_{idx}{src_ext}")
                        with open(tmp_src, "wb") as f:
                            f.write(src_bytes)
                        src_img = nib.load(tmp_src)
                        out_nii = nib.Nifti1Image(np.asarray(src_img.dataobj), curr_aff, header=src_img.header)
                        out_nii.set_sform(curr_aff, code=1)
                        out_nii.set_qform(curr_aff, code=1)
                        tmp_out = os.path.join(tmpdir, f"out_{idx}.nii.gz")
                        nib.save(out_nii, tmp_out)
                        shutil.copyfile(tmp_out, dest_path)
                        saved_nifti_files.append(fname)

                if drawings_payload:
                    drawings_dir = os.path.join(target_dir, "drawings")
                    os.makedirs(drawings_dir, exist_ok=True)
                    for idx, d in enumerate(drawings_payload):
                        raw_dname = os.path.basename(d.get("fileName") or f"drawing_{idx}.voi")
                        dname = f"{strip_neuro_ext(raw_dname)}.voi"
                        dest_draw = os.path.join(drawings_dir, dname)
                        if d.get("voxelsBase64") and d.get("dims"):
                            nx, ny, nz = [int(x) for x in d["dims"][:3]]
                            vox_raw = base64.b64decode(d["voxelsBase64"])
                            arr_3d = np.frombuffer(vox_raw, dtype=np.uint8).reshape((nx, ny, nz), order="F")
                            d_aff = np.asarray(d.get("affine", np.eye(4)), dtype=np.float64)
                            out_nii = build_canonical_uint8_nifti(arr_3d, d_aff)
                            tmp_draw = os.path.join(tmpdir, f"draw_{idx}.nii.gz")
                            nib.save(out_nii, tmp_draw)
                            shutil.copyfile(tmp_draw, dest_draw)
                            saved_drawing_files.append(dname)
                        elif d.get("voiBase64"):
                            voi_bytes = base64.b64decode(d["voiBase64"])
                            tmp_draw = os.path.join(tmpdir, f"draw_{idx}.voi")
                            with open(tmp_draw, "wb") as df:
                                df.write(voi_bytes)
                            shutil.copyfile(tmp_draw, dest_draw)
                            saved_drawing_files.append(dname)

            saved_mat_files = []
            for tfm in transforms_payload:
                tname = os.path.basename(tfm.get("fileName") or "transform.mat")
                tcontent = tfm.get("content") or ""
                tpath = os.path.join(transforms_dir, tname)
                with open(tpath, "w", encoding="utf-8") as f:
                    f.write(tcontent)
                saved_mat_files.append(tname)

            project_json_path = os.path.join(target_dir, "project.json")
            with open(project_json_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            self._send_json(200, {
                "saved": True,
                "projectDir": target_dir,
                "niftiFiles": saved_nifti_files,
                "transformFiles": saved_mat_files,
                "drawingFiles": saved_drawing_files,
            })
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def _handle_open_project(self, body):
        """Loads a saved NeuroCoreg project directory from local disk or a mounted network share."""
        try:
            target_dir = body.get("targetDir")
            if not target_dir:
                chosen_dir, cancelled = self._pick_native_folder("Select a saved NeuroCoreg project folder to open:")
                if cancelled:
                    self._send_json(200, {"cancelled": True})
                    return
                if not chosen_dir:
                    default_dir = get_default_projects_dir()
                    self._send_json(200, {"needsPath": True, "defaultDir": default_dir})
                    return
                target_dir = chosen_dir

            target_dir = normalize_user_path(target_dir)
            # If user selected the "nifti" or "drawings" subfolder inside a project, step up one level
            if os.path.basename(os.path.normpath(target_dir)).lower() in ("nifti", "drawings"):
                parent_cand = os.path.dirname(os.path.normpath(target_dir))
                if os.path.isfile(os.path.join(parent_cand, "project.json")):
                    target_dir = parent_cand

            project_json_path = os.path.join(target_dir, "project.json")
            if not os.path.exists(project_json_path):
                # Also check if target_dir contains a single *_project child directory with project.json
                try:
                    sub_projs = [
                        os.path.join(target_dir, d)
                        for d in os.listdir(target_dir)
                        if os.path.isdir(os.path.join(target_dir, d))
                        and os.path.isfile(os.path.join(target_dir, d, "project.json"))
                    ]
                    if len(sub_projs) == 1:
                        target_dir = sub_projs[0]
                        project_json_path = os.path.join(target_dir, "project.json")
                except Exception:
                    pass

            if not os.path.exists(project_json_path):
                self._send_json(400, {"error": f"No project.json found in {target_dir}"})
                return

            with open(project_json_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            nifti_dir = os.path.join(target_dir, "nifti")
            volumes_out = []
            for v_entry in manifest.get("volumes", []):
                fname = os.path.basename(v_entry.get("fileName") or v_entry.get("name") or "")
                cand_names = [fname, f"{strip_neuro_ext(fname)}.nii.gz", f"{strip_neuro_ext(fname)}.nii"]
                found_path = None
                for cname in cand_names:
                    for folder in (nifti_dir, target_dir):
                        p = os.path.join(folder, cname)
                        if os.path.isfile(p):
                            found_path = p
                            fname = cname
                            break
                    if found_path:
                        break
                if found_path:
                    with open(found_path, "rb") as nf:
                        b64 = base64.b64encode(nf.read()).decode("ascii")
                    volumes_out.append({
                        "fileName": fname,
                        "niftiBase64": b64,
                        "meta": v_entry,
                    })

            drawings_dir = os.path.join(target_dir, "drawings")
            drawings_out = []
            for d_entry in manifest.get("drawingLayers", []):
                dname = os.path.basename(d_entry.get("fileName") or "")
                if not dname:
                    continue
                cand_dnames = [dname, f"{strip_neuro_ext(dname)}.voi", f"{strip_neuro_ext(dname)}.nii.gz", f"{strip_neuro_ext(dname)}.nii"]
                found_dpath = None
                for cname in cand_dnames:
                    for folder in (drawings_dir, target_dir):
                        p = os.path.join(folder, cname)
                        if os.path.isfile(p):
                            found_dpath = p
                            dname = cname
                            break
                    if found_dpath:
                        break
                if found_dpath:
                    with open(found_dpath, "rb") as df:
                        b64 = base64.b64encode(df.read()).decode("ascii")
                    drawings_out.append({
                        "fileName": dname,
                        "voiBase64": b64,
                        "meta": d_entry,
                    })

            self._send_json(200, {
                "projectDir": target_dir,
                "manifest": manifest,
                "volumes": volumes_out,
                "drawings": drawings_out,
            })
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def _handle_skull_strip_bet(self, body):
        """
        Runs FSL BET on the selected T1w MRI volume if FSL is installed on the host OS,
        or automatically falls back to a pure-Python 3D morphological skull-stripping pipeline
        (scipy_morphological_skull_strip) so skull stripping works on any OS.
        """
        try:
            src_bytes = base64.b64decode(body["sourceBase64"])
            src_name = body.get("sourceName") or "T1w.nii.gz"
            frac = float(body.get("frac", 0.5))
            frac = max(0.05, min(0.95, frac))
            curr_affine = body.get("currentAffine")

            base_clean = strip_neuro_ext(os.path.basename(src_name))
            out_filename = f"{base_clean}_brain.nii.gz"

            with tempfile.TemporaryDirectory() as tmpdir:
                src_ext = ".nii.gz" if src_bytes[:2] == b"\x1f\x8b" else ".nii"
                raw_in_path = os.path.join(tmpdir, f"raw_in{src_ext}")
                with open(raw_in_path, "wb") as f:
                    f.write(src_bytes)

                src_img = nib.load(raw_in_path)
                aff = np.asarray(curr_affine, dtype=np.float64) if curr_affine is not None else get_best_affine(src_img)
                clean_3d = get_clean_3d_data(src_img, dtype=np.float32)

                in_path = os.path.join(tmpdir, "input_t1.nii.gz")
                updated_img = nib.Nifti1Image(clean_3d, aff)
                updated_img.set_sform(aff, code=1)
                updated_img.set_qform(aff, code=1)
                nib.save(updated_img, in_path)

                out_prefix = os.path.join(tmpdir, "brain_out")
                out_nii_path = f"{out_prefix}.nii.gz"
                engine_used = "FSL BET"

                fsl_bet_ok = False
                if os.path.exists(BET_BIN) and os.access(BET_BIN, os.X_OK):
                    env = os.environ.copy()
                    env["FSLDIR"] = FSLDIR
                    env["FSLOUTPUTTYPE"] = "NIFTI_GZ"
                    cmd = [BET_BIN, in_path, out_prefix, "-f", f"{frac:.2f}"]
                    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=90)
                    if proc.returncode == 0 and os.path.exists(out_nii_path):
                        fsl_bet_ok = True

                if not fsl_bet_ok:
                    # Pure-Python 3D Morphological Brain Extraction fallback
                    engine_used = "SciPy 3D Morphological Brain Extraction (Fallback)"
                    zooms = list(src_img.header.get_zooms())
                    while len(zooms) < 3:
                        zooms.append(1.0)
                    brain_arr, _ = scipy_morphological_skull_strip(clean_3d, zooms[:3], frac=frac)
                    fallback_nii = nib.Nifti1Image(brain_arr, aff)
                    fallback_nii.set_sform(aff, code=1)
                    fallback_nii.set_qform(aff, code=1)
                    nib.save(fallback_nii, out_nii_path)

                brain_img = nib.load(out_nii_path)
                brain_data = get_clean_3d_data(brain_img, dtype=np.float32)
                brain_voxels = int(np.count_nonzero(brain_data > 0))
                zooms = list(brain_img.header.get_zooms())
                while len(zooms) < 3:
                    zooms.append(1.0)
                vox_vol_mm3 = float(zooms[0] * zooms[1] * zooms[2])
                brain_vol_cm3 = round((brain_voxels * vox_vol_mm3) / 1000.0, 2)

                with open(out_nii_path, "rb") as bf:
                    out_b64 = base64.b64encode(bf.read()).decode("ascii")

                self._send_json(200, {
                    "engine": engine_used,
                    "fileName": out_filename,
                    "niftiBase64": out_b64,
                    "brainVoxels": brain_voxels,
                    "brainVolumeCm3": brain_vol_cm3,
                    "frac": frac,
                    "shape": list(brain_img.shape[:3]),
                })
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def _handle_set_fs_license(self, body):
        """
        Validates and saves a user-selected FreeSurfer license file (uploaded from the browser file picker
        or specified via a local path) into the user's writable cache so FastSurfer --fsaparc can use it.
        """
        try:
            file_name = os.path.basename(body.get("fileName") or "license.txt")
            file_path = body.get("filePath")
            content_text = body.get("contentText")
            content_b64 = body.get("contentBase64")

            if file_path:
                norm_p = normalize_user_path(file_path)
                is_valid, msg = validate_freesurfer_license_file(norm_p)
                if not is_valid:
                    self._send_json(400, {"valid": False, "error": msg})
                    return
                with open(norm_p, "r", encoding="utf-8", errors="replace") as rf:
                    content_text = rf.read()
                file_name = os.path.basename(norm_p)
            elif content_b64:
                content_text = base64.b64decode(content_b64).decode("utf-8", errors="replace")

            is_valid, msg = validate_freesurfer_license_content(content_text)
            if not is_valid:
                self._send_json(400, {"valid": False, "error": msg})
                return

            os.makedirs(FASTSURFER_CACHE_DIR, exist_ok=True)
            with open(CACHED_FS_LICENSE_PATH, "w", encoding="utf-8") as wf:
                wf.write(content_text)
            with open(CACHED_FS_LICENSE_META_PATH, "w", encoding="utf-8") as mf:
                json.dump(
                    {
                        "fileName": file_name,
                        "originalPath": normalize_user_path(file_path) if file_path else None,
                    },
                    mf,
                )
            os.environ["FS_LICENSE"] = CACHED_FS_LICENSE_PATH

            self._send_json(200, {
                "valid": True,
                "message": msg,
                "fsLicense": {
                    "available": True,
                    "path": CACHED_FS_LICENSE_PATH,
                    "label": file_name,
                },
            })
        except Exception as e:
            self._send_json(500, {"valid": False, "error": str(e)})

    def _handle_dkt_segment(self, body):
        """
        Runs whole-brain Desikan-Killiany-Tourville (DKT, aparc.DKTatlas+aseg) or Destrieux (aparc.a2009s+aseg)
        cortical + subcortical segmentation:
          - Supports atlasType:
              * 'dkt' (default): Desikan-Killiany-Tourville (aparc.DKTatlas+aseg, 104 regions, max ID 2035)
              * 'destrieux': Destrieux sulco-gyral atlas (aparc.a2009s+aseg, 148 cortical + subcortical, max ID 12175)
                Runs FastSurfer with `--fsaparc` and WITHOUT `--seg_only` when FastSurfer is available,
                and reverts to MNI152 template registration only when FastSurfer is unavailable (or when
                template_flirt mode is explicitly selected).
          - Supports dktMode:
              * 'fastsurfer_vinn' (default): FastSurferVINN (--seg_only for DKT; --fsaparc without --seg_only for Destrieux)
              * 'fastsurfer_full': Full FastSurfer (+ CerebNet, HypVINN, CC)
              * 'template_flirt': Quick MNI152 -> Subject template registration (via FSL FLIRT or SciPy 9-DOF affine fallback)
        """
        try:
            src_bytes = base64.b64decode(body["sourceBase64"])
            src_name = body.get("sourceName") or "T1w.nii.gz"
            curr_affine = body.get("currentAffine")
            dkt_mode = body.get("dktMode") or "fastsurfer_vinn"
            raw_atlas_type = str(body.get("atlasType") or "dkt").strip().lower()
            is_destrieux = raw_atlas_type in ("destrieux", "a2009s", "aparc.a2009s")
            atlas_type = "destrieux" if is_destrieux else "dkt"
            atlas_short = "Destrieux" if is_destrieux else "DKT"

            base_clean = strip_neuro_ext(os.path.basename(src_name)).replace("_brain", "")
            out_filename = f"{base_clean}_{atlas_short}_atlas.nii.gz"

            tools_info = detect_neuro_tools()
            fs_info = tools_info["fastsurfer"]
            dkt_lut = load_dkt_lut()

            with tempfile.TemporaryDirectory() as tmpdir:
                src_ext = ".nii.gz" if src_bytes[:2] == b"\x1f\x8b" else ".nii"
                raw_in_path = os.path.join(tmpdir, f"raw_in{src_ext}")
                with open(raw_in_path, "wb") as f:
                    f.write(src_bytes)

                src_img = nib.load(raw_in_path)
                target_aff = (
                    np.asarray(curr_affine, dtype=np.float64)
                    if curr_affine is not None
                    else get_best_affine(src_img)
                )
                src_3d = get_clean_3d_data(src_img, dtype=np.float32)
                in_t1_path = os.path.join(tmpdir, "subject_t1.nii.gz")
                t1_nii = nib.Nifti1Image(src_3d, target_aff)
                t1_nii.set_sform(target_aff, code=1)
                t1_nii.set_qform(target_aff, code=1)
                nib.save(t1_nii, in_t1_path)

                out_dkt_path = os.path.join(tmpdir, out_filename)
                engine_used = None

                # 1. Run FastSurfer (macOS native, Linux native, or Docker) or load verified cache
                if dkt_mode in ("fastsurfer_vinn", "fastsurfer_full") and fs_info.get("available"):
                    try:
                        import hashlib
                        os.makedirs(FASTSURFER_CACHE_DIR, exist_ok=True)
                        vox_hash = hashlib.sha256(src_3d.tobytes()).hexdigest()[:16]
                        cached_seg_path = os.path.join(
                            FASTSURFER_CACHE_DIR,
                            f"{atlas_type}_{dkt_mode}_{vox_hash}.nii.gz" if is_destrieux else f"{dkt_mode}_{vox_hash}.nii.gz",
                        )

                        # For DKT only: check if input matches sample sub-01_T1w with pre-generated FastSurferVINN DKT output
                        if (
                            not is_destrieux
                            and dkt_mode == "fastsurfer_vinn"
                            and not os.path.exists(cached_seg_path)
                            and os.path.exists(SAMPLE_FASTSURFER_DKT_PATH)
                            and os.path.exists(SAMPLE_T1_PATH)
                        ):
                            sample_t1_img = nib.load(SAMPLE_T1_PATH)
                            if sample_t1_img.shape[:3] == src_3d.shape[:3]:
                                sample_vox = get_clean_3d_data(sample_t1_img, dtype=np.float32)
                                mask_nz = src_3d > 0
                                if np.count_nonzero(mask_nz) > 1000 and np.allclose(
                                    src_3d[mask_nz], sample_vox[mask_nz], rtol=1e-2, atol=1.0
                                ):
                                    cached_seg_path = SAMPLE_FASTSURFER_DKT_PATH

                        if os.path.exists(cached_seg_path):
                            cached_img = nib.load(cached_seg_path)
                            seg_data = np.rint(get_clean_3d_data(cached_img)).astype(np.int16)
                            cached_aff = get_best_affine(cached_img)
                            if seg_data.shape[:3] != src_3d.shape[:3] or not np.allclose(cached_aff, target_aff, atol=1e-2):
                                vox2vox = safe_inv4x4(cached_aff) @ target_aff
                                seg_data = affine_transform(
                                    seg_data,
                                    vox2vox[:3, :3],
                                    offset=vox2vox[:3, 3],
                                    output_shape=src_3d.shape[:3],
                                    order=0,
                                    mode="constant",
                                    cval=0,
                                ).astype(np.int16)
                            dkt_nii = nib.Nifti1Image(seg_data, target_aff)
                            dkt_nii.set_sform(target_aff, code=1)
                            dkt_nii.set_qform(target_aff, code=1)
                            nib.save(dkt_nii, out_dkt_path)
                            if is_destrieux:
                                engine_used = f"FastSurfer ({fs_info.get('version', '2.7.0')}) --fsaparc Destrieux (aparc.a2009s+aseg)"
                            else:
                                engine_used = (
                                    f"FastSurferVINN ({fs_info.get('version', '2.7.0')}) DKT Segmentation"
                                    if dkt_mode == "fastsurfer_vinn"
                                    else f"FastSurfer Full ({fs_info.get('version', '2.7.0')}) Segmentation"
                                )
                        else:
                            fs_sd = os.path.join(tmpdir, "fs_out")
                            os.makedirs(fs_sd, exist_ok=True)
                            if is_destrieux:
                                # Per FastSurfer GitHub documentation: run with --fsaparc and WITHOUT --seg_only
                                # (note: --keepgeom is incompatible with the surface pipeline)
                                extra_flags = (
                                    ["--fsaparc", "--no_cereb", "--no_hypothal", "--no_cc", "--parallel", "--batch_size", "8", "--threads", "8"]
                                    if dkt_mode == "fastsurfer_vinn"
                                    else ["--fsaparc", "--parallel", "--batch_size", "8", "--threads", "8"]
                                )
                                run_timeout = 3600
                            else:
                                extra_flags = (
                                    ["--seg_only", "--no_cereb", "--no_hypothal", "--no_cc", "--no_biasfield", "--keepgeom", "--batch_size", "8", "--threads", "8"]
                                    if dkt_mode == "fastsurfer_vinn"
                                    else ["--seg_only", "--batch_size", "8", "--threads", "8"]
                                )
                                run_timeout = 600

                            if fs_info.get("mode") in ("macos_native", "linux_native") and fs_info.get("cliPath"):
                                fs_env = build_fastsurfer_macos_env(fs_info)
                                fs_lic = fs_env.get("FS_LICENSE")
                                if is_destrieux and fs_lic and os.path.isfile(fs_lic):
                                    extra_flags = [*extra_flags, "--fs_license", fs_lic]
                                setup_sh = fs_info.get("setupScript")
                                cli_path = fs_info["cliPath"]
                                fs_args = [
                                    cli_path,
                                    "--t1", in_t1_path,
                                    "--sid", "sub_dkt",
                                    "--sd", fs_sd,
                                    *extra_flags,
                                ]
                                quoted_cmd = " ".join(shlex.quote(a) for a in fs_args)
                                if setup_sh and os.path.exists(setup_sh):
                                    bash_cmd = f"source {shlex.quote(setup_sh)} >/dev/null 2>&1 && {quoted_cmd}"
                                else:
                                    bash_cmd = quoted_cmd
                                bash_bin = shutil.which("bash") or "/bin/bash"
                                subprocess.run(
                                    [bash_bin, "-c", bash_cmd],
                                    env=fs_env,
                                    capture_output=True,
                                    text=True,
                                    timeout=run_timeout,
                                )
                            elif fs_info.get("mode") == "docker":
                                docker_bin = shutil.which("docker") or "docker"
                                docker_cmd = [
                                    docker_bin, "run", "--rm",
                                    "-v", f"{tmpdir}:/data",
                                    "deepmi/fastsurfer:latest",
                                    "--t1", "/data/subject_t1.nii.gz",
                                    "--sid", "sub_dkt",
                                    "--sd", "/data/fs_out",
                                    *extra_flags,
                                ]
                                subprocess.run(docker_cmd, capture_output=True, text=True, timeout=run_timeout)

                            if is_destrieux:
                                cand_mgz = [
                                    os.path.join(fs_sd, "sub_dkt", "mri", "aparc.a2009s+aseg.mgz"),
                                ]
                                # If surface annotations (lh/rh.aparc.a2009s.annot) exist but aparc.a2009s+aseg.mgz wasn't mapped yet:
                                sub_mri = os.path.join(fs_sd, "sub_dkt", "mri")
                                sub_lbl = os.path.join(fs_sd, "sub_dkt", "label")
                                sub_surf = os.path.join(fs_sd, "sub_dkt", "surf")
                                if (
                                    not os.path.exists(cand_mgz[0])
                                    and os.path.exists(os.path.join(sub_lbl, "lh.aparc.a2009s.annot"))
                                    and os.path.exists(os.path.join(sub_lbl, "rh.aparc.a2009s.annot"))
                                    and os.path.exists(os.path.join(sub_mri, "aseg.mgz"))
                                ):
                                    fs_env = build_fastsurfer_macos_env(fs_info)
                                    surf2vol_bin = shutil.which("mri_surf2volseg", path=fs_env.get("PATH"))
                                    if surf2vol_bin:
                                        subprocess.run(
                                            [
                                                surf2vol_bin,
                                                "--o", cand_mgz[0],
                                                "--label-cortex",
                                                "--i", os.path.join(sub_mri, "aseg.mgz"),
                                                "--lh-annot", os.path.join(sub_lbl, "lh.aparc.a2009s.annot"), "11100",
                                                "--lh-cortex-mask", os.path.join(sub_lbl, "lh.cortex.label"),
                                                "--lh-white", os.path.join(sub_surf, "lh.white"),
                                                "--lh-pial", os.path.join(sub_surf, "lh.pial"),
                                                "--rh-annot", os.path.join(sub_lbl, "rh.aparc.a2009s.annot"), "12100",
                                                "--rh-cortex-mask", os.path.join(sub_lbl, "rh.cortex.label"),
                                                "--rh-white", os.path.join(sub_surf, "rh.white"),
                                                "--rh-pial", os.path.join(sub_surf, "rh.pial"),
                                            ],
                                            env=fs_env,
                                            capture_output=True,
                                            text=True,
                                            timeout=300,
                                        )
                            else:
                                cand_mgz = [
                                    os.path.join(fs_sd, "sub_dkt", "mri", "aparc.DKTatlas+aseg.deep.mgz"),
                                    os.path.join(fs_sd, "sub_dkt", "mri", "aparc.DKTatlas+aseg.orig.mgz"),
                                    os.path.join(fs_sd, "sub_dkt", "mri", "aparc.DKTatlas+aseg.mgz"),
                                ]

                            for mgz_p in cand_mgz:
                                if os.path.exists(mgz_p):
                                    fs_seg_img = nib.load(mgz_p)
                                    seg_data = np.rint(get_clean_3d_data(fs_seg_img)).astype(np.int16)
                                    fs_aff = get_best_affine(fs_seg_img)
                                    if seg_data.shape[:3] == src_3d.shape[:3] and np.allclose(fs_aff, target_aff, atol=1e-2):
                                        resampled_seg = seg_data
                                    else:
                                        vox2vox = safe_inv4x4(fs_aff) @ target_aff
                                        resampled_seg = affine_transform(
                                            seg_data,
                                            vox2vox[:3, :3],
                                            offset=vox2vox[:3, 3],
                                            output_shape=src_3d.shape[:3],
                                            order=0,
                                            mode="constant",
                                            cval=0,
                                        ).astype(np.int16)
                                    dkt_nii = nib.Nifti1Image(resampled_seg, target_aff)
                                    dkt_nii.set_sform(target_aff, code=1)
                                    dkt_nii.set_qform(target_aff, code=1)
                                    nib.save(dkt_nii, out_dkt_path)
                                    try:
                                        nib.save(dkt_nii, cached_seg_path)
                                    except Exception:
                                        pass
                                    if is_destrieux:
                                        engine_used = f"FastSurfer ({fs_info.get('version', '2.7.0')}) --fsaparc Destrieux (aparc.a2009s+aseg)"
                                    else:
                                        engine_used = (
                                            f"FastSurferVINN ({fs_info.get('version', '2.7.0')}) DKT Segmentation"
                                            if dkt_mode == "fastsurfer_vinn"
                                            else f"FastSurfer Full ({fs_info.get('version', '2.7.0')}) Segmentation"
                                        )
                                    break
                    except Exception as fs_err:
                        print(f"FastSurfer execution warning (falling back to template {atlas_short}): {fs_err}", flush=True)

                # 2. Fast Template Atlas Registration (FSL FLIRT or Pure-Python SciPy 9-DOF fallback)
                if not os.path.exists(out_dkt_path):
                    template_atlas_path = MNI_DESTRIEUX_ATLAS_PATH if is_destrieux else MNI_DKT_ATLAS_PATH
                    if not (os.path.exists(template_atlas_path) and os.path.exists(MNI_T1_BRAIN_PATH)):
                        self._send_json(500, {
                            "error": f"Neither FastSurfer nor the MNI152 {atlas_short} Atlas template was found on disk."
                        })
                        return

                    # Skull-strip subject T1w first if it isn't already skull-stripped for sharper template registration
                    sub_brain_path = os.path.join(tmpdir, "sub_brain.nii.gz")
                    if "_brain" in src_name.lower():
                        sub_brain_path = in_t1_path
                    elif os.path.exists(BET_BIN) and os.access(BET_BIN, os.X_OK):
                        env = os.environ.copy()
                        env["FSLDIR"] = FSLDIR
                        env["FSLOUTPUTTYPE"] = "NIFTI_GZ"
                        bet_prefix = os.path.join(tmpdir, "sub_brain")
                        subprocess.run(
                            [BET_BIN, in_t1_path, bet_prefix, "-f", "0.5"],
                            env=env,
                            capture_output=True,
                            text=True,
                            timeout=60,
                        )
                        if not os.path.exists(sub_brain_path):
                            sub_brain_path = in_t1_path
                    else:
                        zooms = list(src_img.header.get_zooms())
                        while len(zooms) < 3:
                            zooms.append(1.0)
                        brain_arr, _ = scipy_morphological_skull_strip(src_3d, zooms[:3], frac=0.5)
                        nib.save(nib.Nifti1Image(brain_arr, target_aff), sub_brain_path)

                    flirt_ok = False
                    if os.path.exists(FLIRT_BIN) and os.access(FLIRT_BIN, os.X_OK):
                        env = os.environ.copy()
                        env["FSLDIR"] = FSLDIR
                        env["FSLOUTPUTTYPE"] = "NIFTI_GZ"
                        mni2sub_mat = os.path.join(tmpdir, "mni2sub.mat")
                        flirt_reg_cmd = [
                            FLIRT_BIN,
                            "-in", MNI_T1_BRAIN_PATH,
                            "-ref", sub_brain_path,
                            "-omat", mni2sub_mat,
                            "-bins", "256",
                            "-cost", "corratio",
                            "-searchrx", "-30", "30",
                            "-searchry", "-30", "30",
                            "-searchrz", "-30", "30",
                            "-dof", "12",
                        ]
                        subprocess.run(flirt_reg_cmd, env=env, capture_output=True, text=True, timeout=90)
                        if os.path.exists(mni2sub_mat):
                            flirt_apply_cmd = [
                                FLIRT_BIN,
                                "-in", template_atlas_path,
                                "-ref", in_t1_path,
                                "-applyxfm",
                                "-init", mni2sub_mat,
                                "-out", out_dkt_path,
                                "-interp", "nearestneighbour",
                            ]
                            subprocess.run(flirt_apply_cmd, env=env, capture_output=True, text=True, timeout=60)
                            if os.path.exists(out_dkt_path):
                                flirt_ok = True
                                engine_used = (
                                    "FSL FLIRT 12-DOF + FreeSurfer Destrieux Atlas (aparc.a2009s+aseg)"
                                    if is_destrieux
                                    else "FSL FLIRT 12-DOF + FreeSurfer DKT Atlas (aparc.DKTatlas+aseg)"
                                )

                    if not flirt_ok:
                        # Pure-Python SciPy 9-DOF MNI152 -> Subject Registration Fallback
                        sub_brain_img = nib.load(sub_brain_path)
                        mni_brain_img = nib.load(MNI_T1_BRAIN_PATH)
                        mni_dkt_img = nib.load(template_atlas_path)
                        _, _, new_mni_aff = optimize_multimodal_registration(sub_brain_img, mni_brain_img, dof=9)
                        vox_to_mni = safe_inv4x4(new_mni_aff) @ target_aff
                        dkt_mni_data = np.rint(get_clean_3d_data(mni_dkt_img)).astype(np.int16)
                        warped_dkt = affine_transform(
                            dkt_mni_data,
                            vox_to_mni[:3, :3],
                            offset=vox_to_mni[:3, 3],
                            output_shape=src_3d.shape[:3],
                            order=0,
                            mode="constant",
                            cval=0,
                        ).astype(np.int16)
                        clean_dkt_nii = nib.Nifti1Image(warped_dkt, target_aff)
                        clean_dkt_nii.set_sform(target_aff, code=1)
                        clean_dkt_nii.set_qform(target_aff, code=1)
                        nib.save(clean_dkt_nii, out_dkt_path)
                        engine_used = (
                            "SciPy 9-DOF Template Registration + FreeSurfer Destrieux Atlas (aparc.a2009s+aseg)"
                            if is_destrieux
                            else "SciPy 9-DOF Template Registration + FreeSurfer DKT Atlas"
                        )
                    else:
                        # Ensure clean integer int16 labels & exact subject sform/qform affine
                        dkt_raw_img = nib.load(out_dkt_path)
                        dkt_int_data = np.rint(get_clean_3d_data(dkt_raw_img)).astype(np.int16)
                        clean_dkt_nii = nib.Nifti1Image(dkt_int_data, target_aff)
                        clean_dkt_nii.set_sform(target_aff, code=1)
                        clean_dkt_nii.set_qform(target_aff, code=1)
                        nib.save(clean_dkt_nii, out_dkt_path)

                # 3. Mask the Atlas with the Subject's Brain Mask so all voxels outside the brain are 0
                final_img = nib.load(out_dkt_path)
                unmasked_data = np.rint(get_clean_3d_data(final_img)).astype(np.int16)

                # Save an unmasked copy so the frontend can dynamically re-mask when BET is re-run with a new -f
                unmasked_path = os.path.join(tmpdir, "unmasked_dkt.nii.gz")
                unmasked_nii = nib.Nifti1Image(unmasked_data, target_aff)
                unmasked_nii.set_sform(target_aff, code=1)
                unmasked_nii.set_qform(target_aff, code=1)
                nib.save(unmasked_nii, unmasked_path)
                with open(unmasked_path, "rb") as uf:
                    unmasked_b64 = base64.b64encode(uf.read()).decode("ascii")

                brain_mask_bool = None
                if body.get("brainMaskBase64"):
                    try:
                        bm_bytes = base64.b64decode(body["brainMaskBase64"])
                        bm_ext = ".nii.gz" if bm_bytes[:2] == b"\x1f\x8b" else ".nii"
                        bm_path = os.path.join(tmpdir, f"user_brain_mask{bm_ext}")
                        with open(bm_path, "wb") as bmf:
                            bmf.write(bm_bytes)
                        bm_img = nib.load(bm_path)
                        bm_3d = get_clean_3d_data(bm_img, dtype=np.float32)
                        bm_aff = (
                            np.asarray(body["brainMaskAffine"], dtype=np.float64)
                            if body.get("brainMaskAffine") is not None
                            else get_best_affine(bm_img)
                        )
                        if bm_3d.shape[:3] == unmasked_data.shape[:3] and np.allclose(bm_aff, target_aff, atol=1e-2):
                            brain_mask_bool = bm_3d > 0
                        else:
                            vox2bm = safe_inv4x4(bm_aff) @ target_aff
                            warped_bm = affine_transform(
                                (bm_3d > 0).astype(np.float32),
                                vox2bm[:3, :3],
                                offset=vox2bm[:3, 3],
                                output_shape=unmasked_data.shape[:3],
                                order=0,
                                mode="constant",
                                cval=0.0,
                            )
                            brain_mask_bool = warped_bm > 0.5
                    except Exception as bm_err:
                        print(f"Warning reading brainMaskBase64: {bm_err}", flush=True)

                if brain_mask_bool is None:
                    sub_brain_cand = os.path.join(tmpdir, "sub_brain.nii.gz")
                    if "_brain" in src_name.lower():
                        brain_mask_bool = src_3d > 0
                    elif os.path.exists(sub_brain_cand):
                        sb_img = nib.load(sub_brain_cand)
                        brain_mask_bool = get_clean_3d_data(sb_img, dtype=np.float32) > 0
                    else:
                        bet_frac = max(0.05, min(0.95, float(body.get("betFrac", 0.5))))
                        if os.path.exists(BET_BIN) and os.access(BET_BIN, os.X_OK):
                            env = os.environ.copy()
                            env["FSLDIR"] = FSLDIR
                            env["FSLOUTPUTTYPE"] = "NIFTI_GZ"
                            bet_prefix = os.path.join(tmpdir, "auto_bet_mask")
                            subprocess.run(
                                [BET_BIN, in_t1_path, bet_prefix, "-f", f"{bet_frac:.2f}"],
                                env=env,
                                capture_output=True,
                                text=True,
                                timeout=60,
                            )
                            bet_out_nii = f"{bet_prefix}.nii.gz"
                            if os.path.exists(bet_out_nii):
                                sb_img = nib.load(bet_out_nii)
                                brain_mask_bool = get_clean_3d_data(sb_img, dtype=np.float32) > 0
                        if brain_mask_bool is None:
                            zooms = list(src_img.header.get_zooms())
                            while len(zooms) < 3:
                                zooms.append(1.0)
                            _, mask_arr = scipy_morphological_skull_strip(src_3d, zooms[:3], frac=bet_frac)
                            brain_mask_bool = mask_arr > 0

                # Zero out non-brain / exterior / skull / extraventricular CSF / unknown FreeSurfer LUT IDs
                non_brain_fs_ids = np.array(
                    [
                        0, 1, 6, 24, 29, 30, 40, 45, 61, 62, 80, 81, 82, 85, 98,
                        *range(118, 155), 165, 166, 167, 168, 224,
                        *range(256, 267), 701, *range(901, 932),
                        1000, 1001, 1100, 2000, 2001, 2100, 11100, 12100,
                    ],
                    dtype=np.int16,
                )
                valid_brain_label_mask = (unmasked_data > 0) & (~np.isin(unmasked_data, non_brain_fs_ids))
                masked_data = np.where(brain_mask_bool & valid_brain_label_mask, unmasked_data, 0).astype(np.int16)
                masked_nii = nib.Nifti1Image(masked_data, target_aff)
                masked_nii.set_sform(target_aff, code=1)
                masked_nii.set_qform(target_aff, code=1)
                nib.save(masked_nii, out_dkt_path)

                unique_labels = [int(x) for x in np.unique(masked_data) if int(x) != 0]

                with open(out_dkt_path, "rb") as df:
                    out_b64 = base64.b64encode(df.read()).decode("ascii")

                self._send_json(200, {
                    "engine": engine_used,
                    "atlasType": atlas_type,
                    "atlasLabel": f"{atlas_short} Atlas",
                    "maxLutId": 12175 if is_destrieux else 2035,
                    "fastsurferDetected": bool(fs_info.get("available")),
                    "fastsurferStatus": fs_info,
                    "fileName": out_filename,
                    "niftiBase64": out_b64,
                    "unmaskedNiftiBase64": unmasked_b64,
                    "brainMasked": True,
                    "regionCount": len(unique_labels),
                    "shape": list(masked_data.shape[:3]),
                    "dktLut": dkt_lut,
                })
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def _handle_convert_volume(self, body):
        """
        Converts FreeSurfer .mgz / .mgh (or any nibabel-supported volume) into compressed NIfTI (.nii.gz),
        properly preserving .mgz and .mgh extensions when loading via nibabel.
        """
        try:
            src_bytes = base64.b64decode(body["sourceBase64"])
            src_name = body.get("sourceName") or "volume.mgz"
            is_dkt = bool(body.get("isDktAtlas", False))

            lower_name = src_name.lower()
            if lower_name.endswith(".mgz"):
                ext = ".mgz"
            elif lower_name.endswith(".mgh"):
                ext = ".mgh"
            elif lower_name.endswith(".nii.gz") or src_bytes[:2] == b"\x1f\x8b":
                ext = ".nii.gz"
            else:
                ext = ".nii"

            base_clean = strip_neuro_ext(os.path.basename(src_name))
            out_filename = f"{base_clean}.nii.gz"

            with tempfile.TemporaryDirectory() as tmpdir:
                in_path = os.path.join(tmpdir, f"input{ext}")
                with open(in_path, "wb") as f:
                    f.write(src_bytes)

                img = nib.load(in_path)
                clean_arr = get_clean_3d_data(img, dtype=np.float32)
                detected_atlas_type = None
                if is_dkt:
                    data = np.rint(clean_arr).astype(np.int16)
                    non_brain_fs_ids = np.array(
                        [
                            0, 1, 6, 24, 29, 30, 40, 45, 61, 62, 80, 81, 82, 85, 98,
                            *range(118, 155), 165, 166, 167, 168, 224,
                            *range(256, 267), 701, *range(901, 932),
                            1000, 1001, 1100, 2000, 2001, 2100, 11100, 12100,
                        ],
                        dtype=np.int16,
                    )
                    data[np.isin(data, non_brain_fs_ids) | (data < 0)] = 0
                    max_lbl = int(np.max(data)) if data.size > 0 else 0
                    if max_lbl >= 11100 or "a2009s" in lower_name or "destrieux" in lower_name:
                        detected_atlas_type = "destrieux"
                    else:
                        detected_atlas_type = "dkt"
                else:
                    data = clean_arr
                aff = get_best_affine(img)

                out_path = os.path.join(tmpdir, out_filename)
                out_nii = nib.Nifti1Image(data, aff)
                out_nii.set_sform(aff, code=1)
                out_nii.set_qform(aff, code=1)
                nib.save(out_nii, out_path)

                with open(out_path, "rb") as f:
                    out_b64 = base64.b64encode(f.read()).decode("ascii")

                self._send_json(200, {
                    "fileName": out_filename,
                    "niftiBase64": out_b64,
                    "shape": list(out_nii.shape[:3]),
                    "atlasType": detected_atlas_type,
                    "dktLut": load_dkt_lut() if is_dkt else None,
                })
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def _handle_drawing_save(self, body):
        """
        Exports a uint8 drawing layer as MRIcroGL .voi (gzipped uint8 NIfTI), .nii.gz, or .nii,
        and either returns base64 bytes for browser download or writes directly via native OS Save File dialog.
        """
        try:
            dims = body.get("dims") or []
            if len(dims) < 3:
                self._send_json(400, {"error": "Missing drawing dims [nx, ny, nz]."})
                return
            nx, ny, nz = (int(dims[0]), int(dims[1]), int(dims[2]))
            vox_bytes = base64.b64decode(body["voxelsBase64"])
            expected_len = nx * ny * nz
            if len(vox_bytes) != expected_len:
                self._send_json(400, {"error": f"Voxel buffer length {len(vox_bytes)} != {nx}*{ny}*{nz}."})
                return

            arr = np.frombuffer(vox_bytes, dtype=np.uint8).reshape((nx, ny, nz), order="F")
            aff = np.asarray(body.get("affine", np.eye(4)), dtype=np.float64)
            fmt = (body.get("format") or "voi").lower().lstrip(".")
            if fmt not in ("voi", "nii.gz", "nii"):
                fmt = "voi"
            ext = f".{fmt}"
            base_stem = strip_neuro_ext(os.path.basename(body.get("fileName") or "drawing_voi"))
            base_stem = re.sub(r"[^a-zA-Z0-9._-]", "_", base_stem) or "drawing_voi"
            suggested_name = f"{base_stem}{ext}"

            out_nii = build_canonical_uint8_nifti(arr, aff)
            out_bytes = save_nifti_bytes_for_format(out_nii, fmt)
            out_b64 = base64.b64encode(out_bytes).decode("ascii")

            mode = (body.get("mode") or "download").lower()
            if mode != "save_as":
                self._send_json(200, {
                    "fileName": suggested_name,
                    "format": fmt,
                    "fileBase64": out_b64,
                    "byteLength": len(out_bytes),
                })
                return

            target_path = body.get("targetPath")
            if target_path:
                dest_path = normalize_user_path(target_path)
            else:
                chosen_path, cancelled = self._pick_native_save_file(
                    f"Save Drawing Layer ({suggested_name}):",
                    suggested_name,
                    ext,
                )
                if cancelled:
                    self._send_json(200, {"cancelled": True})
                    return
                if not chosen_path:
                    self._send_json(200, {
                        "needsFallback": True,
                        "fileName": suggested_name,
                        "format": fmt,
                        "fileBase64": out_b64,
                        "byteLength": len(out_bytes),
                    })
                    return
                dest_path = chosen_path

            dest_dir = os.path.dirname(os.path.abspath(dest_path))
            if dest_dir:
                os.makedirs(dest_dir, exist_ok=True)

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_out = os.path.join(tmpdir, "export_draw.bin")
                with open(tmp_out, "wb") as tf:
                    tf.write(out_bytes)
                shutil.copyfile(tmp_out, dest_path)

                labels_path = None
                labels_meta = body.get("labelsMetadata")
                if labels_meta:
                    labels_fname = f"{strip_neuro_ext(os.path.basename(dest_path))}_labels.json"
                    labels_path = os.path.join(dest_dir, labels_fname)
                    tmp_lbl = os.path.join(tmpdir, "labels.json")
                    with open(tmp_lbl, "w", encoding="utf-8") as lf:
                        json.dump(labels_meta, lf, indent=2)
                    shutil.copyfile(tmp_lbl, labels_path)

            self._send_json(200, {
                "saved": True,
                "savedPath": dest_path,
                "labelsPath": labels_path,
                "fileName": os.path.basename(dest_path),
            })
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def _handle_drawing_import(self, body):
        """
        Loads a .voi, .nii.gz, .nii, or .mgz file and optionally resamples it via nearest-neighbor
        (order=0) onto the workspace Reference grid as a clean uint8 voxel buffer.
        """
        try:
            src_bytes = base64.b64decode(body["sourceBase64"])
            src_name = body.get("sourceName") or "imported.voi"
            with tempfile.TemporaryDirectory() as tmpdir:
                src_img = load_neuro_image_from_bytes(src_bytes, src_name, tmpdir)
                src_3d = get_clean_3d_data(src_img, dtype=np.float32)
                src_aff = (
                    np.asarray(body["sourceAffine"], dtype=np.float64)
                    if body.get("sourceAffine")
                    else get_best_affine(src_img)
                )

            thresh_min = body.get("thresholdMin")
            if thresh_min is not None:
                try:
                    tval = float(thresh_min)
                    src_3d = (src_3d >= tval).astype(np.float32)
                except Exception:
                    pass

            ref_dims = body.get("refDims")
            ref_aff_raw = body.get("refAffine")
            resampled = False
            if ref_dims and len(ref_dims) >= 3 and ref_aff_raw:
                target_shape = (int(ref_dims[0]), int(ref_dims[1]), int(ref_dims[2]))
                ref_aff = np.asarray(ref_aff_raw, dtype=np.float64)
                if src_3d.shape == target_shape and np.allclose(src_aff, ref_aff, atol=1e-2):
                    out_arr = np.rint(np.clip(src_3d, 0, 255)).astype(np.uint8)
                    out_aff = ref_aff
                else:
                    vox_to_vox = safe_inv4x4(src_aff) @ ref_aff
                    res = affine_transform(
                        src_3d,
                        vox_to_vox[:3, :3],
                        offset=vox_to_vox[:3, 3],
                        output_shape=target_shape,
                        order=0,
                        mode="constant",
                        cval=0.0,
                    )
                    out_arr = np.rint(np.clip(res, 0, 255)).astype(np.uint8)
                    out_aff = ref_aff
                    resampled = True
            else:
                out_arr = np.rint(np.clip(src_3d, 0, 255)).astype(np.uint8)
                out_aff = src_aff

            uniq = [int(x) for x in np.unique(out_arr) if int(x) > 0]
            if len(uniq) == 1 and uniq[0] > 6:
                out_arr[out_arr > 0] = 1
                uniq = [1]

            suggested_mode = "multilabel" if len(uniq) > 1 else "binary"
            vox_b64 = base64.b64encode(out_arr.tobytes(order="F")).decode("ascii")
            self._send_json(200, {
                "voxelsBase64": vox_b64,
                "dims": list(out_arr.shape),
                "affine": out_aff.tolist(),
                "resampled": resampled,
                "uniqueLabels": uniq,
                "suggestedMode": suggested_mode,
                "voxelCount": int(np.count_nonzero(out_arr)),
            })
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def _handle_drawing_morph(self, body):
        """
        Applies server-side 3D morphological cleanup (fill_holes, largest_island, dilate, erode),
        signed-distance slice interpolation (interpolate_slices), or intensity-guided region growing.
        """
        try:
            dims = body.get("dims") or []
            if len(dims) < 3:
                self._send_json(400, {"error": "Missing drawing dims [nx, ny, nz]."})
                return
            nx, ny, nz = (int(dims[0]), int(dims[1]), int(dims[2]))
            vox_bytes = base64.b64decode(body["voxelsBase64"])
            if len(vox_bytes) != nx * ny * nz:
                self._send_json(400, {"error": "Drawing buffer size mismatch."})
                return

            arr = np.frombuffer(vox_bytes, dtype=np.uint8).reshape((nx, ny, nz), order="F")
            aff = np.asarray(body.get("affine", np.eye(4)), dtype=np.float64)
            op = body.get("op") or ""
            target_label = int(body.get("targetLabel", 1))
            overwrite = bool(body.get("overwrite", True))

            out_arr, changed, msg = apply_drawing_morph_op(
                arr,
                aff,
                op,
                target_label=target_label,
                overwrite=overwrite,
                params=body,
            )

            self._send_json(200, {
                "voxelsBase64": base64.b64encode(out_arr.tobytes(order="F")).decode("ascii"),
                "changedVoxels": int(changed),
                "voxelCount": int(np.count_nonzero(out_arr)),
                "message": msg,
            })
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def _handle_shutdown(self, body):
        """Cleanly shuts down the ThreadingHTTPServer after sending the HTTP 200 response."""
        self._send_json(200, {"shutdown": True, "message": "NeuroCoreg Studio server shutting down."})

        def _stop_server():
            import time
            time.sleep(0.15)
            try:
                self.server.shutdown()
                self.server.server_close()
            except Exception:
                pass
            os._exit(0)

        import threading
        threading.Thread(target=_stop_server, daemon=True).start()

    def _send_json(self, status, payload):
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class ReusableHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    explicit_port = os.environ.get("PORT")
    base_port = int(explicit_port) if explicit_port else 8765
    server = None
    chosen_port = base_port
    for cand_port in ([base_port] if explicit_port else range(base_port, base_port + 35)):
        try:
            server = ReusableHTTPServer((host, cand_port), MultimodalHandler)
            chosen_port = cand_port
            break
        except OSError:
            if explicit_port:
                raise
            continue
    if server is None:
        server = ReusableHTTPServer((host, 0), MultimodalHandler)
        chosen_port = server.server_address[1]
    print(f"NiiVue Multimodal Coregistration Server listening on http://{host}:{chosen_port}", flush=True)
    server.serve_forever()
