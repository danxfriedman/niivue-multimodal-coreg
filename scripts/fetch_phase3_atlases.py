#!/usr/bin/env python3
"""
Download, inspect, register (THOMAS -> MNI152NLin6Asym 1mm), normalize
(int16 + intent_code=1002 + sform/qform=4), and package:
  - THOMAS Thalamic Nuclei Atlas (thalamicseg/thomas_new + thalamicseg/thomas-v0)
  - Tian Melbourne Subcortex S1-S4 (yetianmed/subcortex)
into sample_data/atlases/.
"""

import colorsys
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
ATLASES_DIR = ROOT_DIR / "sample_data" / "atlases"
RAW_CACHE_DIR = Path("/tmp/neurocoreg_phase3_atlas_downloads")

FSLDIR = os.environ.get("FSLDIR", "/usr/local/fsl")
FLIRT_BIN = os.path.join(FSLDIR, "bin", "flirt")
MNI_T1_1MM = os.path.join(FSLDIR, "data", "standard", "MNI152_T1_1mm.nii.gz")
MNI_T1_2MM = os.path.join(FSLDIR, "data", "standard", "MNI152_T1_2mm.nii.gz")
FSL_HO_SUB_1MM = os.path.join(
    FSLDIR, "data", "atlases", "HarvardOxford", "HarvardOxford-sub-maxprob-thr25-1mm.nii.gz"
)

GITHUB_TIAN_API = (
    "https://api.github.com/repos/yetianmed/subcortex/contents/"
    "Group-Parcellation/3T/Subcortex-Only"
)

THOMAS_TEMPLATE_URL = "https://media.githubusercontent.com/media/thalamicseg/thomas_new/master/origtemplate.nii.gz"
THOMAS_ATLAS_URL = "https://media.githubusercontent.com/media/thalamicseg/thomas-v0/master/atlas_with_CL_VPM.nii.gz"
THOMAS_CTBL_URL = "https://raw.githubusercontent.com/thalamicseg/thomas_new/master/CustomAtlas.ctbl"

# Canonical THOMAS anatomical definitions
THOMAS_NUCLEUS_INFO = {
    1: ("Thalamus-Whole", "Whole Thalamus"),
    2: ("AV", "Anteroventral Nucleus (AV)"),
    4: ("VA", "Ventral Anterior Nucleus (VA)"),
    5: ("VLa", "Ventral Lateral Anterior Nucleus (VLa)"),
    6: ("VLp", "Ventral Lateral Posterior / VIM Nucleus (VLp)"),
    7: ("VPL", "Ventral Posterolateral Nucleus (VPL)"),
    8: ("Pul", "Pulvinar Nucleus (Pul)"),
    9: ("LGN", "Lateral Geniculate Nucleus (LGN)"),
    10: ("MGN", "Medial Geniculate Nucleus (MGN)"),
    11: ("CM", "Centromedian Nucleus (CM)"),
    12: ("MD-Pf", "Mediodorsal-Parafascicular Nucleus (MD-Pf)"),
    13: ("Hb", "Habenula (Hb)"),
    14: ("MTT", "Mammillothalamic Tract (MTT)"),
    17: ("CL", "Central Lateral Nucleus (CL)"),
    18: ("VPM", "Ventral Posteromedial Nucleus (VPM)"),
}


def http_get_json(url: str) -> dict | list:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "niivue-multimodal-coreg-atlas-fetcher/1.0",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 200:
        # Guard against cached git-lfs pointer stub
        head = dest.read_bytes()[:40]
        if not head.startswith(b"version https://git-lfs"):
            print(f"  [cache] {dest.name} ({dest.stat().st_size:,} bytes)")
            return dest
    print(f"  [fetch] {url} -> {dest.name}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "niivue-multimodal-coreg-atlas-fetcher/1.0"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = resp.read()
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tf:
        tf.write(data)
        tmp_path = Path(tf.name)
    tmp_path.replace(dest)
    print(f"  [saved] {dest.name} ({dest.stat().st_size:,} bytes)")
    return dest


def generate_distinct_rgb(idx: int, total: int, hemi: str = "C") -> tuple[int, int, int]:
    """Golden-angle HSV palette with clean saturation and mild L/R brightness distinction."""
    hue = ((idx * 0.618033988749895) + (0.04 if hemi == "R" else 0.0)) % 1.0
    sat = 0.72 + 0.14 * ((idx % 3) / 2.0)
    val = 0.88 if hemi != "R" else 0.94
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def save_label_nifti_int16(
    data: np.ndarray,
    affine: np.ndarray,
    out_path: Path,
    descrip: str,
    sform_code: int = 4,
) -> dict:
    """Save int16 label NIfTI with NII_INTENT_LABEL (1002) and exact slope=1/inter=0."""
    arr = np.rint(data).astype(np.int16)
    max_val = int(arr.max()) if arr.size else 0
    img = nib.Nifti1Image(arr, affine)
    hdr = img.header
    hdr.set_data_dtype(np.int16)
    hdr.set_intent(1002, (), "Label")
    hdr.set_slope_inter(1.0, 0.0)
    hdr["cal_min"] = 0.0
    hdr["cal_max"] = float(max(max_val, 1))
    hdr["descrip"] = descrip.encode("ascii", "ignore")[:79]
    img.set_sform(affine, code=sform_code)
    img.set_qform(affine, code=sform_code)
    nib.save(img, str(out_path))
    nz = arr[arr > 0]
    uniq = [int(v) for v in np.unique(nz).tolist()]
    return {
        "path": str(out_path),
        "sizeBytes": out_path.stat().st_size,
        "shape": list(arr.shape),
        "uniquePositiveIds": uniq,
        "nonzeroVoxels": int(nz.size),
    }


def compute_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return 0.0
    return (2.0 * int(np.logical_and(a, b).sum())) / float(denom)


def fetch_and_build_tian() -> list[dict]:
    print("\n=== Fetching Tian Melbourne Subcortex S1-S4 ===")
    entries = http_get_json(GITHUB_TIAN_API)
    by_name = {e["name"]: e for e in entries if isinstance(e, dict) and e.get("download_url")}

    results = []
    for scale in (1, 2, 3, 4):
        # Prefer MNI152NLin6Asym 1mm (`Tian_Subcortex_S{scale}_3T_1mm.nii.gz`), else 2mm `.nii`
        preferred_1mm = f"Tian_Subcortex_S{scale}_3T_1mm.nii.gz"
        fallback_2mm = f"Tian_Subcortex_S{scale}_3T.nii"
        if preferred_1mm in by_name:
            nii_name = preferred_1mm
            res_tag = "1mm"
        elif f"Tian_Subcortex_S{scale}_3T.nii.gz" in by_name:
            nii_name = f"Tian_Subcortex_S{scale}_3T.nii.gz"
            res_tag = "2mm"
        elif fallback_2mm in by_name:
            nii_name = fallback_2mm
            res_tag = "2mm"
        else:
            raise RuntimeError(f"Could not find Tian S{scale} MNI152NLin6Asym file in {sorted(by_name)}")

        lbl_name = f"Tian_Subcortex_S{scale}_3T_label.txt"
        if lbl_name not in by_name:
            raise RuntimeError(f"Could not find Tian label file {lbl_name}")

        raw_nii = download_file(by_name[nii_name]["download_url"], RAW_CACHE_DIR / nii_name)
        raw_lbl = download_file(by_name[lbl_name]["download_url"], RAW_CACHE_DIR / lbl_name)

        img = nib.load(str(raw_nii))
        data = np.asarray(img.dataobj)
        if data.ndim == 4 and data.shape[3] == 1:
            data = data[..., 0]
        uniq_ids = [int(v) for v in np.unique(np.rint(data)) if int(v) > 0]

        lines = [ln.strip() for ln in raw_lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
        parsed_pairs = []
        for idx_line, line in enumerate(lines, start=1):
            parts = re.split(r"[\t, ]+", line, maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                parsed_pairs.append((int(parts[0]), parts[1].strip()))
            else:
                parsed_pairs.append((idx_line, line.strip()))

        lut_labels = []
        for idx_pos, (rid, rname) in enumerate(parsed_pairs):
            hemi = (
                "R"
                if (rname.endswith("-rh") or rname.endswith("_R") or "-R-" in rname)
                else ("L" if (rname.endswith("-lh") or rname.endswith("_L") or "-L-" in rname) else "C")
            )
            r, g, b = generate_distinct_rgb(idx_pos, max(len(parsed_pairs), 1), hemi=hemi)
            lut_labels.append(
                {
                    "id": rid,
                    "name": rname,
                    "shortName": rname,
                    "hemisphere": hemi,
                    "r": r,
                    "g": g,
                    "b": b,
                    "a": 255,
                }
            )

        out_nii = ATLASES_DIR / f"Tian_Subcortex_S{scale}_3T.nii.gz"
        out_lut = ATLASES_DIR / f"tian_s{scale}_lut.json"
        stats = save_label_nifti_int16(
            data,
            img.affine,
            out_nii,
            f"Tian Melbourne Subcortex Scale {scale} (3T MNI152NLin6Asym {res_tag})",
            sform_code=4,
        )
        lut_payload = {
            "atlasId": f"tian_s{scale}",
            "name": f"Tian Melbourne Subcortex Scale {scale} ({len(lut_labels)} ROIs)",
            "space": "MNI152NLin6Asym",
            "resolution": res_tag,
            "source": "https://github.com/yetianmed/subcortex",
            "labels": lut_labels,
        }
        out_lut.write_text(json.dumps(lut_payload, indent=2), encoding="utf-8")

        missing_in_lut = set(uniq_ids) - {e["id"] for e in lut_labels}
        if missing_in_lut:
            raise RuntimeError(f"Tian S{scale} voxel IDs missing from LUT: {missing_in_lut}")

        print(
            f"  [Tian S{scale}] res={res_tag} shape={stats['shape']} ids=1..{max(uniq_ids)} "
            f"(count={len(uniq_ids)}, LUT={len(lut_labels)}) size={stats['sizeBytes']:,} B"
        )
        results.append(stats)

    return results


def parse_ctbl_colors(ctbl_text: str) -> dict[int, tuple[str, tuple[int, int, int]]]:
    parsed = {}
    for ln in ctbl_text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) >= 5 and parts[0].isdigit():
            idx = int(parts[0])
            name = parts[1]
            r, g, b = int(parts[2]), int(parts[3]), int(parts[4])
            parsed[idx] = (name, (r, g, b))
    return parsed


def fetch_and_build_thomas() -> dict:
    print("\n=== Fetching & Registering THOMAS Thalamic Atlas -> MNI152NLin6Asym 1mm ===")
    if (
        not os.path.isfile(FLIRT_BIN)
        or not os.path.isfile(MNI_T1_1MM)
        or not os.path.isfile(MNI_T1_2MM)
        or not os.path.isfile(FSL_HO_SUB_1MM)
    ):
        raise RuntimeError(
            f"Required FSL tools/templates missing: {FLIRT_BIN}, {MNI_T1_1MM}, {MNI_T1_2MM}, {FSL_HO_SUB_1MM}"
        )

    raw_tmpl = download_file(THOMAS_TEMPLATE_URL, RAW_CACHE_DIR / "thomas_origtemplate.nii.gz")
    raw_atlas = download_file(THOMAS_ATLAS_URL, RAW_CACHE_DIR / "thomas_atlas_with_CL_VPM.nii.gz")
    raw_ctbl = download_file(THOMAS_CTBL_URL, RAW_CACHE_DIR / "thomas_CustomAtlas.ctbl")

    ctbl_colors = parse_ctbl_colors(raw_ctbl.read_text(encoding="utf-8", errors="replace"))
    ho_img = nib.load(FSL_HO_SUB_1MM)
    ho_data = np.rint(np.asarray(ho_img.dataobj)).astype(np.int16)
    ho_left_thal = ho_data == 4
    ho_right_thal = ho_data == 15

    mni_t1_img = nib.load(MNI_T1_1MM)
    mni_shape = mni_t1_img.shape[:3]
    mni_aff = mni_t1_img.affine.copy()

    fsl_env = os.environ.copy()
    fsl_env["FSLDIR"] = FSLDIR
    fsl_env["FSLOUTPUTTYPE"] = "NIFTI_GZ"

    with tempfile.TemporaryDirectory() as tmpdir:
        ds_tmpl_path = os.path.join(tmpdir, "thomas_orig_ds2.nii.gz")
        t2m_mat = os.path.join(tmpdir, "thomas_to_mni152.mat")
        warped_left_path = os.path.join(tmpdir, "warped_thomas_left.nii.gz")

        # Downsample high-res origtemplate by 2x for fast, accurate 12-DOF normmi FLIRT -> MNI152_T1_2mm
        tmpl_img = nib.load(str(raw_tmpl))
        tmpl_ds_arr = np.asarray(tmpl_img.dataobj, dtype=np.float32)[::2, ::2, ::2]
        tmpl_ds_aff = tmpl_img.affine.copy()
        tmpl_ds_aff[:3, :3] *= 2.0
        nib.save(nib.Nifti1Image(tmpl_ds_arr, tmpl_ds_aff), ds_tmpl_path)

        print("  [FLIRT] 12-DOF normmi downsampled origtemplate (2x) -> MNI152_T1_2mm...")
        subprocess.run(
            [
                FLIRT_BIN,
                "-in", ds_tmpl_path,
                "-ref", MNI_T1_2MM,
                "-omat", t2m_mat,
                "-dof", "12",
                "-cost", "normmi",
                "-searchrx", "-30", "30",
                "-searchry", "-30", "30",
                "-searchrz", "-30", "30",
            ],
            env=fsl_env,
            check=True,
            capture_output=True,
            text=True,
        )

        print("  [FLIRT] Applying nearest-neighbour warp to full-res THOMAS labels onto MNI152_T1_1mm...")
        subprocess.run(
            [
                FLIRT_BIN,
                "-in", str(raw_atlas),
                "-ref", MNI_T1_1MM,
                "-applyxfm",
                "-init", t2m_mat,
                "-interp", "nearestneighbour",
                "-out", warped_left_path,
            ],
            env=fsl_env,
            check=True,
            capture_output=True,
            text=True,
        )

        left_arr = np.rint(np.asarray(nib.load(warped_left_path).dataobj)).astype(np.int16)

    # Build Right hemisphere by exact RAS x-mirror (-x, y, z) on MNI152_T1_1mm grid
    inv_mni = np.linalg.inv(mni_aff)
    merged = np.zeros(mni_shape, dtype=np.int16)

    nz_l = np.argwhere(left_arr > 0)
    xyz_l = nib.affines.apply_affine(mni_aff, nz_l)
    for idx_pt, (ii, jj, kk) in enumerate(nz_l):
        lbl = int(left_arr[ii, jj, kk])
        if lbl <= 0:
            continue
        x_ras = float(xyz_l[idx_pt, 0])
        if x_ras <= 0.5:
            merged[ii, jj, kk] = lbl

    # Mirror Left voxels across x=0 into Right hemisphere (id + 100)
    mirrored_xyz = xyz_l.copy()
    mirrored_xyz[:, 0] = -mirrored_xyz[:, 0]
    mirrored_ijk = np.rint(nib.affines.apply_affine(inv_mni, mirrored_xyz)).astype(int)
    for idx_pt, (ii, jj, kk) in enumerate(nz_l):
        lbl = int(left_arr[ii, jj, kk])
        if lbl <= 0:
            continue
        ri, rj, rk = int(mirrored_ijk[idx_pt, 0]), int(mirrored_ijk[idx_pt, 1]), int(mirrored_ijk[idx_pt, 2])
        if 0 <= ri < mni_shape[0] and 0 <= rj < mni_shape[1] and 0 <= rk < mni_shape[2]:
            rx_ras = float(mirrored_xyz[idx_pt, 0])
            if rx_ras > 0.0:
                merged[ri, rj, rk] = lbl + 100

    l_mask = (merged > 0) & (merged < 100)
    r_mask = merged >= 100
    cont_l = float(np.logical_and(l_mask, ho_left_thal).sum()) / max(1, int(l_mask.sum()))
    cont_r = float(np.logical_and(r_mask, ho_right_thal).sum()) / max(1, int(r_mask.sum()))
    dice_l_final = compute_dice(l_mask, ho_left_thal)
    dice_r_final = compute_dice(r_mask, ho_right_thal)
    print(
        f"  [Alignment vs HO thr25] Containment: L={cont_l:.4f}, R={cont_r:.4f} | "
        f"Dice: L={dice_l_final:.4f}, R={dice_r_final:.4f}"
    )
    if min(cont_l, cont_r) < 0.85 or min(dice_l_final, dice_r_final) < 0.58:
        raise RuntimeError(
            f"THOMAS -> MNI152 validation failed "
            f"(cont L={cont_l:.4f}, R={cont_r:.4f}; dice L={dice_l_final:.4f}, R={dice_r_final:.4f})"
        )

    # Validate stereotactic MNI152 RAS centroid bounds across all nuclei
    for lid in np.unique(merged):
        if lid <= 0:
            continue
        pts = np.argwhere(merged == lid)
        c = nib.affines.apply_affine(mni_aff, pts).mean(axis=0)
        cx, cy, cz = float(c[0]), float(c[1]), float(c[2])
        expected_hemi_ok = (cx < 0.0) if lid < 100 else (cx > 0.0)
        if not expected_hemi_ok or not (-35.0 <= cy <= -5.0) or not (-10.0 <= cz <= 15.0):
            raise RuntimeError(f"THOMAS centroid out of stereotactic bounds for id={lid}: RAS={c.tolist()}")

    out_nii = ATLASES_DIR / "MNI152_THOMAS_atlas.nii.gz"
    stats = save_label_nifti_int16(
        merged,
        mni_aff,
        out_nii,
        "THOMAS Thalamic Nuclei Atlas (MNI152NLin6Asym 1mm, L=2..18, R=102..118)",
        sform_code=4,
    )

    present_ids = set(stats["uniquePositiveIds"])
    base_ids = sorted({bid if bid < 100 else bid - 100 for bid in present_ids})
    lut_labels = []
    print("  [THOMAS MNI152 nuclei centroids RAS]:")
    for base_id in base_ids:
        ctbl_name, ctbl_rgb = ctbl_colors.get(base_id, (f"Nuc-{base_id}", generate_distinct_rgb(base_id, 18)))
        abbr, full_name = THOMAS_NUCLEUS_INFO.get(base_id, (ctbl_name, f"Thalamic {ctbl_name}"))
        for hemi, lid in (("L", base_id), ("R", base_id + 100)):
            if lid not in present_ids:
                continue
            pts = np.argwhere(merged == lid)
            centroid = nib.affines.apply_affine(mni_aff, pts).mean(axis=0)
            rgb = (
                ctbl_rgb
                if hemi == "L"
                else (
                    min(255, int(round(ctbl_rgb[0] * 0.92 + 16))),
                    min(255, int(round(ctbl_rgb[1] * 0.92 + 12))),
                    min(255, int(round(ctbl_rgb[2] * 0.95 + 10))),
                )
            )
            lut_labels.append(
                {
                    "id": lid,
                    "name": f"{'Left' if hemi == 'L' else 'Right'} {full_name}",
                    "shortName": f"{hemi}-{abbr}",
                    "hemisphere": hemi,
                    "r": rgb[0],
                    "g": rgb[1],
                    "b": rgb[2],
                    "a": 255,
                }
            )
            print(
                f"    id={lid:3d} {hemi}-{abbr:6s} vox={len(pts):5d} "
                f"RAS=[{centroid[0]:5.1f}, {centroid[1]:5.1f}, {centroid[2]:5.1f}]"
            )

    lut_labels.sort(key=lambda x: x["id"])
    out_lut = ATLASES_DIR / "thomas_lut.json"
    lut_payload = {
        "atlasId": "thomas",
        "name": "THOMAS Thalamic Nuclei Atlas (MNI152)",
        "space": "MNI152NLin6Asym",
        "resolution": "1mm",
        "source": "https://github.com/thalamicseg/thomas_new & https://github.com/thalamicseg/thomas-v0",
        "labels": lut_labels,
    }
    out_lut.write_text(json.dumps(lut_payload, indent=2), encoding="utf-8")

    missing = present_ids - {e["id"] for e in lut_labels}
    if missing:
        raise RuntimeError(f"THOMAS merged IDs missing from LUT: {missing}")

    stats["diceLeftHO"] = round(dice_l_final, 4)
    stats["diceRightHO"] = round(dice_r_final, 4)
    return stats


def main() -> int:
    ATLASES_DIR.mkdir(parents=True, exist_ok=True)
    tian_stats = fetch_and_build_tian()
    thomas_stats = fetch_and_build_thomas()

    total_bytes = thomas_stats["sizeBytes"] + sum(s["sizeBytes"] for s in tian_stats)
    print("\n=== Summary ===")
    print(f"Output directory: {ATLASES_DIR}")
    print(f"THOMAS: shape={thomas_stats['shape']} ids={len(thomas_stats['uniquePositiveIds'])} size={thomas_stats['sizeBytes']:,} B")
    for idx, ts in enumerate(tian_stats, start=1):
        print(f"Tian S{idx}: shape={ts['shape']} ids={len(ts['uniquePositiveIds'])} size={ts['sizeBytes']:,} B")
    print(f"Total NIfTI payload: {total_bytes / 1024.0:.1f} KB ({total_bytes / (1024.0 * 1024.0):.3f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
