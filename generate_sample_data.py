#!/usr/bin/env python3
"""
Generates single-subject multimodal neuroimaging sample datasets (NIfTI + DICOM series folder)
for testing the NiiVue (niivue/mono) Multimodal Brain Viewer & Coregistration Tool.
"""
import os
import datetime
import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform, gaussian_filter
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NIFTI_DIR = os.path.join(BASE_DIR, "sample_data", "nifti")
DICOM_DIR = os.path.join(BASE_DIR, "sample_data", "dicom_t2_flair")

os.makedirs(NIFTI_DIR, exist_ok=True)
os.makedirs(DICOM_DIR, exist_ok=True)


def _synthesize_fallback_mni_head_and_brain():
    """Creates a synthetic 2mm MNI-like 3D head and brain if FSL standard data is not installed."""
    shape = (91, 109, 91)
    affine = np.array([
        [-2.0,  0.0,  0.0,   90.0],
        [ 0.0,  2.0,  0.0, -126.0],
        [ 0.0,  0.0,  2.0,  -72.0],
        [ 0.0,  0.0,  0.0,    1.0],
    ], dtype=np.float64)
    grid_x, grid_y, grid_z = np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        np.arange(shape[2], dtype=np.float32),
        indexing="ij",
    )
    cx, cy, cz = 45.0, 54.0, 45.0
    r_brain = ((grid_x - cx) / 31.0) ** 2 + ((grid_y - cy) / 39.0) ** 2 + ((grid_z - cz) / 30.0) ** 2
    r_head = ((grid_x - cx) / 36.0) ** 2 + ((grid_y - cy) / 44.0) ** 2 + ((grid_z - cz) / 35.0) ** 2
    r_vent = ((grid_x - cx) / 8.0) ** 2 + ((grid_y - (cy + 2.0)) / 15.0) ** 2 + ((grid_z - (cz + 2.0)) / 7.0) ** 2

    t1_brain = np.zeros(shape, dtype=np.float32)
    brain_mask = r_brain <= 1.0
    t1_brain[brain_mask] = 5500.0 - 1800.0 * r_brain[brain_mask]
    t1_brain[r_vent <= 1.0] = 1200.0
    t1_brain = gaussian_filter(t1_brain, sigma=0.8)

    t1_head = t1_brain.copy()
    skull_ring = (r_head <= 1.0) & (r_brain > 1.04)
    t1_head[skull_ring] = 2600.0
    t1_head = gaussian_filter(t1_head, sigma=0.7)
    return t1_head, t1_brain, affine


def create_multimodal_nifti():
    # Use FSL 2mm MNI152 brain + skull if installed, or local sample template / synthetic fallback
    fsl_root = os.environ.get("FSLDIR") or "/usr/local/fsl"
    fsl_std = os.path.join(fsl_root, "data", "standard")
    omm_dir = os.path.join(fsl_root, "data", "omm", "Oxford-MM-1")

    t1_path = os.path.join(fsl_std, "MNI152_T1_2mm.nii.gz")
    t1_brain_path = os.path.join(fsl_std, "MNI152_T1_2mm_brain.nii.gz")
    local_brain_path = os.path.join(NIFTI_DIR, "MNI152_T1_brain_2mm.nii.gz")
    t2_omm_path = os.path.join(omm_dir, "OMM-1_T2_FLAIR_brain.nii.gz")
    fa_path = os.path.join(fsl_std, "FMRIB58_FA_1mm.nii.gz")

    if os.path.exists(t1_path) and os.path.exists(t1_brain_path):
        t1_img = nib.load(t1_path)
        t1_data = t1_img.get_fdata(dtype=np.float32)
        affine = t1_img.affine.copy()
        t1_brain_img = nib.load(t1_brain_path)
        t1_brain = t1_brain_img.get_fdata(dtype=np.float32)
    elif os.path.exists(local_brain_path):
        t1_brain_img = nib.load(local_brain_path)
        t1_brain = t1_brain_img.get_fdata(dtype=np.float32)
        affine = t1_brain_img.affine.copy()
        t1_data, _, _ = _synthesize_fallback_mni_head_and_brain()
        t1_data = np.maximum(t1_data, t1_brain)
    else:
        t1_data, t1_brain, affine = _synthesize_fallback_mni_head_and_brain()

    # 1. Save Reference T1w MRI (sub-01_T1w.nii.gz)
    t1_out = os.path.join(NIFTI_DIR, "sub-01_T1w.nii.gz")
    nib.save(nib.Nifti1Image(t1_data.astype(np.int16), affine), t1_out)
    print(f"Created {t1_out} shape={t1_data.shape}")

    # 2. Create T2-FLAIR volume in the same 2mm space
    if os.path.exists(t2_omm_path):
        t2_omm = nib.load(t2_omm_path)
        # Resample 1mm OMM T2 FLAIR to 2mm MNI grid via world coordinates
        inv_t2_aff = np.linalg.inv(t2_omm.affine)
        rel_mat = inv_t2_aff @ affine
        t2_data = affine_transform(
            t2_omm.get_fdata(dtype=np.float32),
            rel_mat[:3, :3],
            offset=rel_mat[:3, 3],
            output_shape=t1_data.shape,
            order=1
        )
    else:
        # Invert brain contrast to mimic T2w
        mask = t1_brain > 500
        t2_data = np.zeros_like(t1_brain)
        t2_data[mask] = np.clip(9000 - t1_brain[mask], 500, 8500)

    t2_out = os.path.join(NIFTI_DIR, "sub-01_T2_FLAIR.nii.gz")
    nib.save(nib.Nifti1Image(t2_data.astype(np.int16), affine), t2_out)
    print(f"Created {t2_out} shape={t2_data.shape}")

    # 3. Create Simulated Post-Op CT (Skull + Ventricles + Soft Tissue + SEEG electrode contacts)
    # Derive skull from difference between head T1 and brain T1, plus bright cranial vault
    skull_raw = np.clip(t1_data - t1_brain * 0.85, 0, None)
    ct_data = np.zeros_like(t1_data, dtype=np.float32)
    # Soft tissue ~ 30-60 HU (shifted to positive 0..2500 scale for easy viewing)
    mask_head = t1_data > 800
    ct_data[mask_head] = 200.0 + (t1_brain[mask_head] / (np.max(t1_brain) + 1e-5)) * 180.0
    # Dense bone / skull table ~ 1200-2000
    skull_shell = gaussian_filter((t1_data > 1600).astype(np.float32) - (t1_brain > 500).astype(np.float32), sigma=0.8)
    ct_data += np.clip(skull_shell, 0, 1) * 1400.0 + (skull_raw / (np.max(skull_raw) + 1e-5)) * 800.0

    # Add synthetic hyperdense SEEG depth electrode contacts along 2 trajectories (like seegcoreg.sh)
    seeg_mask_data = np.zeros_like(t1_data, dtype=np.float32)
    for step in range(8):
        x_idx = int(22 + step * 2)
        y_idx = int(54)
        z_idx = int(38 + step * 1)
        ct_data[x_idx-1:x_idx+1, y_idx-1:y_idx+1, z_idx-1:z_idx+1] = 2400.0
        seeg_mask_data[x_idx-1:x_idx+1, y_idx-1:y_idx+1, z_idx-1:z_idx+1] = 1.0
        # Second trajectory (posterior temporal)
        x2 = int(24 + step * 2)
        y2 = int(38)
        z2 = int(34 + step * 1)
        ct_data[x2-1:x2+1, y2-1:y2+1, z2-1:z2+1] = 2400.0
        seeg_mask_data[x2-1:x2+1, y2-1:y2+1, z2-1:z2+1] = 2.0

    # Save aligned CT ground truth AND a misaligned CT (so user can coregister it!)
    ct_aligned_out = os.path.join(NIFTI_DIR, "sub-01_CT_aligned_gt.nii.gz")
    nib.save(nib.Nifti1Image(ct_data.astype(np.int16), affine), ct_aligned_out)

    # Apply an intentional rigid misalignment to the CT sform/qform header:
    # Translation: Tx = +12mm, Ty = -10mm, Tz = +6mm; Rotation: Rz = +12 deg, Rx = -6 deg
    theta_z = np.deg2rad(12.0)
    theta_x = np.deg2rad(-6.0)
    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0, 0],
        [np.sin(theta_z),  np.cos(theta_z), 0, 0],
        [0,                0,               1, 0],
        [0,                0,               0, 1]
    ], dtype=np.float64)
    Rx = np.array([
        [1, 0,                0,               0],
        [0, np.cos(theta_x), -np.sin(theta_x), 0],
        [0, np.sin(theta_x),  np.cos(theta_x), 0],
        [0, 0,                0,               1]
    ], dtype=np.float64)
    T_shift = np.array([
        [1, 0, 0, 12.0],
        [0, 1, 0, -10.0],
        [0, 0, 1,  6.0],
        [0, 0, 0,  1.0]
    ], dtype=np.float64)
    misaligned_affine = T_shift @ Rz @ Rx @ affine
    ct_misaligned_out = os.path.join(NIFTI_DIR, "sub-01_CT_postop_misaligned.nii.gz")
    ct_nii = nib.Nifti1Image(ct_data.astype(np.int16), misaligned_affine)
    ct_nii.set_sform(misaligned_affine, code=1)
    ct_nii.set_qform(misaligned_affine, code=1)
    nib.save(ct_nii, ct_misaligned_out)
    print(f"Created {ct_misaligned_out} (intentionally misaligned for coregistration testing)")

    # Save Child Volume in the same misaligned Post-Op CT space: SEEG Electrode Contacts Mask
    seeg_mask_out = os.path.join(NIFTI_DIR, "sub-01_CT_SEEG_contacts_mask.nii.gz")
    seeg_nii = nib.Nifti1Image(seeg_mask_data.astype(np.uint8), misaligned_affine)
    seeg_nii.set_sform(misaligned_affine, code=1)
    seeg_nii.set_qform(misaligned_affine, code=1)
    nib.save(seeg_nii, seeg_mask_out)
    print(f"Created {seeg_mask_out} (child volume in misaligned CT space)")

    # 4. Create DTI Fractional Anisotropy (FA) volume in 2mm space
    fa_out = os.path.join(NIFTI_DIR, "sub-01_DTI_FA.nii.gz")
    if os.path.exists(fa_path):
        fa_1mm = nib.load(fa_path)
        inv_fa_aff = np.linalg.inv(fa_1mm.affine)
        rel_mat = inv_fa_aff @ affine
        fa_data = affine_transform(
            fa_1mm.get_fdata(dtype=np.float32),
            rel_mat[:3, :3],
            offset=rel_mat[:3, 3],
            output_shape=t1_data.shape,
            order=1
        )
    else:
        wm_mask = (t1_brain > np.percentile(t1_brain[t1_brain > 0], 55)) if np.any(t1_brain > 0) else (t1_brain > 0)
        fa_data = gaussian_filter(wm_mask.astype(np.float32) * 0.72, sigma=1.1)
    nib.save(nib.Nifti1Image(fa_data.astype(np.float32), affine), fa_out)
    print(f"Created {fa_out} shape={fa_data.shape}")

    return t2_data.astype(np.uint16), affine

def create_dicom_series(vol_data, affine):
    """Writes a 3D volume as a folder of 2D axial DICOM (.dcm) slices compatible with dcm2niix."""
    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_uid = generate_uid()
    nx, ny, nz = vol_data.shape

    # Clear existing dcm files
    for f in os.listdir(DICOM_DIR):
        if f.endswith(".dcm"):
            os.remove(os.path.join(DICOM_DIR, f))

    # Scale intensities to 0..4095 uint16
    vmax = float(np.max(vol_data)) if np.max(vol_data) > 0 else 1.0
    scaled = np.clip((vol_data.astype(np.float32) / vmax) * 4000.0, 0, 4095).astype(np.uint16)

    # Extract voxel spacing from affine
    dx = float(np.linalg.norm(affine[:3, 0]))
    dy = float(np.linalg.norm(affine[:3, 1]))
    dz = float(np.linalg.norm(affine[:3, 2]))
    origin = affine[:3, 3]

    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")

    for k in range(nz):
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = MRImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()

        filename = os.path.join(DICOM_DIR, f"IM_{k+1:04d}.dcm")
        ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        ds.PatientName = "SUB01^MULTIMODAL"
        ds.PatientID = "SUB-01"
        ds.PatientBirthDate = "19850101"
        ds.PatientSex = "O"
        ds.StudyDate = date_str
        ds.SeriesDate = date_str
        ds.AcquisitionDate = date_str
        ds.ContentDate = date_str
        ds.StudyTime = time_str
        ds.SeriesTime = time_str
        ds.AcquisitionTime = time_str
        ds.ContentTime = time_str
        ds.AccessionNumber = "ACC001"
        ds.Modality = "MR"
        ds.Manufacturer = "NiiVueMonoSynthetic"
        ds.InstitutionName = "Neuroimaging Core"
        ds.StudyDescription = "Multimodal Brain Coregistration Study"
        ds.SeriesDescription = "Ax_T2_FLAIR_2mm"
        ds.ScanningSequence = "SE"
        ds.SequenceVariant = "SK"
        ds.ScanOptions = "FS"
        ds.MRAcquisitionType = "3D"
        ds.RepetitionTime = "9000"
        ds.EchoTime = "120"
        ds.InversionTime = "2500"
        ds.FlipAngle = "90"

        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.SOPClassUID = MRImageStorage
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SeriesNumber = 2
        ds.AcquisitionNumber = 1
        ds.InstanceNumber = k + 1

        # Standard axial LPS orientation
        # NIfTI is RAS (+X Right, +Y Anterior, +Z Superior)
        # DICOM is LPS (+X Left, +Y Posterior, +Z Superior)
        # Flip X and Y when writing pixel array so LPS coordinates match RAS world space!
        slice_2d = np.flip(np.flip(scaled[:, :, k], axis=0), axis=1).T  # shape (ny, nx)
        ds.Rows, ds.Columns = slice_2d.shape
        ds.PixelSpacing = [f"{dy:.4f}", f"{dx:.4f}"]
        ds.SliceThickness = f"{dz:.4f}"
        ds.SpacingBetweenSlices = f"{dz:.4f}"
        z_pos = float(origin[2] + k * dz)
        # Origin of top-left pixel in LPS
        x_lps = float(-(origin[0] + (nx - 1) * (-dx)))
        y_lps = float(-(origin[1] + (ny - 1) * dy))
        ds.ImagePositionPatient = [f"{x_lps:.4f}", f"{y_lps:.4f}", f"{z_pos:.4f}"]
        ds.ImageOrientationPatient = ["1", "0", "0", "0", "1", "0"]
        ds.SliceLocation = f"{z_pos:.4f}"
        ds.PatientPosition = "HFS"

        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.WindowCenter = "1500"
        ds.WindowWidth = "3000"
        ds.RescaleIntercept = "0"
        ds.RescaleSlope = "1"
        ds.PixelData = slice_2d.tobytes()

        ds.save_as(filename, write_like_original=False)

    print(f"Created {nz} DICOM slices in {DICOM_DIR}")

if __name__ == "__main__":
    t2_vol, aff = create_multimodal_nifti()
    create_dicom_series(t2_vol, aff)
