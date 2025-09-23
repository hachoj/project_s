# reconstruction.py
import os
import math
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def extract_slices(dicom_patient_dir):
    """
    Load DICOM slices from a folder, extract the rotation angle (from origin[2]),
    and return a dict keyed by filename with per-slice metadata and a 2D array.
    """
    # Ensure deterministic sweep ordering (filenames with numeric order benefit from sorting)
    patient_slices_paths = sorted(os.listdir(dicom_patient_dir))

    reader = sitk.ImageFileReader()
    slices = {}

    for slice_path in patient_slices_paths:
        filepath = os.path.join(dicom_patient_dir, slice_path)
        reader.SetFileName(filepath)
        reader.ReadImageInformation()

        # sitk image (do not keep a dangling callable)
        img = reader.Execute()

        # Convert to numpy; assume first component channel if multi-component
        vol = sitk.GetArrayFromImage(img)  # (1, H, W, C?) or (H, W) depending on input
        if vol.ndim == 4:
            # intensity stored as R=G=B → use R
            vol_2d = vol[0, :, :, 0]
        elif vol.ndim == 3:
            # (1, H, W) → squeeze
            vol_2d = vol[0, :, :]
        else:
            # already (H, W)
            vol_2d = vol

        origin = img.GetOrigin()
        spacing = img.GetSpacing()  # assume isotropic in-plane spacing in [0] and [1]
        size = img.GetSize()  # (W, H, [C])

        angle = origin[2]  # degrees, as per dataset/paper

        slices[slice_path] = {
            "angle": float(angle),
            "slice_volume": vol_2d.astype(np.uint8, copy=False),  # keep uint8 like ref
            "shape": size,  # (W, H, [C])
            "spacing": spacing,  # (sx, sy, [sz])
        }

    return slices


def reconstruct_volume(extracted_slices):
    """
    Reconstruct a 3D volume consistent with the reference implementation:
      - Probe radius = 12.5 mm
      - Spacing downsample: factor=6 in LR/AP, and factor*4 in SI
      - Angle convention: theta = -degrees(atan(distance_to_midline / distance_to_posterior))
      - Angular gating to [min(angles), max(angles)]
      - Nearest-angle (NN) slice selection
      - j (row) from radial distance; i (col) from SI index & spacing ratio
    Output array layout: (LR, AP, SI) in uint8
    """
    # --- Gather slices and angles in deterministic order ---
    items = list(extracted_slices.items())
    # Preserve the same (sorted) file-order that extract_slices used
    items.sort(key=lambda kv: kv[0])

    angles_list = []
    slices_list = []
    for _, v in items:
        angles_list.append(v["angle"])
        slices_list.append(v["slice_volume"])

    angles_arr = np.asarray(angles_list, dtype=float)  # (S,)
    angle_min = float(np.min(angles_arr))
    angle_max = float(np.max(angles_arr))

    # Stack slices as (S, H, W)
    H = int(slices_list[0].shape[0])  # height (rows)
    W = int(slices_list[0].shape[1])  # width  (cols)
    slices_stack = np.stack(slices_list, axis=0)  # (S, H, W)

    # --- Geometry from a representative slice ---
    arbitrary = items[0][1]
    spacing_xy = float(arbitrary["spacing"][0])  # assume isotropic in-plane spacing

    height_num_pixels = H
    width_num_pixels = W
    height_mm = spacing_xy * height_num_pixels
    width_mm = spacing_xy * width_num_pixels

    # Reference probe radius
    probe_radius = 12.5  # mm

    # Physical extents (reference formulas)
    AP_mm = height_mm + probe_radius
    LR_mm = 2.0 * AP_mm
    SI_mm = width_mm

    # Downsample spacings (reference)
    factor = 3
    spacing_LR = spacing_xy * factor
    spacing_AP = spacing_xy * factor
    spacing_SI = spacing_xy * factor * 4

    # Discrete sizes
    LR_num = int(LR_mm / spacing_LR)
    AP_num = int(AP_mm / spacing_AP)
    SI_num = int(SI_mm / spacing_SI)

    # Allocate output volume (LR, AP, SI) as uint8 (reference uses sitkUInt8)
    vol = np.zeros((LR_num, AP_num, SI_num), dtype=np.uint8)

    # --- Reconstruction (triple loop like the reference for 1:1 behavior) ---
    # Precompute for speed
    height_plus_r = height_mm + probe_radius
    inv_spacing_xy = 1.0 / spacing_xy

    for z in tqdm(range(SI_num), desc="Reconstructing (z)", unit="slice"):
        # SI → width column index in source slices
        i = int(z * spacing_SI * inv_spacing_xy)
        if i >= width_num_pixels:
            continue

        for x in range(LR_num):
            # distance to midline along LR
            distance_to_midline = x * spacing_LR - (height_mm + probe_radius)

            for y in range(AP_num):
                # distance to posterior along AP (posterior → anterior direction)
                distance_to_posterior = height_plus_r - y * spacing_AP

                # Angle as in reference (note the negative sign)
                # Use atan (not atan2) to match the ref exactly
                theta = -math.degrees(
                    math.atan(distance_to_midline / distance_to_posterior)
                )

                # Angular coverage gate
                if theta < angle_min or theta > angle_max:
                    continue

                # Nearest angle slice
                idx = int(np.abs(angles_arr - theta).argmin())

                # Radial distance and row index j in the 2D slice
                d = math.sqrt(distance_to_midline**2 + distance_to_posterior**2)
                j = int((height_plus_r - d) * inv_spacing_xy)
                if 0 <= j < height_num_pixels:
                    vol[x, y, z] = slices_stack[idx, j, i]

    return {
        "volume": vol,  # (LR, AP, SI) uint8
        "spacing_LR": spacing_LR,
        "spacing_AP": spacing_AP,
        "spacing_SI": spacing_SI,
    }


def save_volume(extracted_slices, reconstructed, patient_dir):
    """
    Save the reconstructed volume as NIfTI with spacing matching the reference.
    The in-memory array is (LR, AP, SI); SimpleITK expects (z, y, x) when using GetImageFromArray.
    """
    vol = reconstructed["volume"]  # (LR, AP, SI)
    spacing_LR = reconstructed["spacing_LR"]
    spacing_AP = reconstructed["spacing_AP"]
    spacing_SI = reconstructed["spacing_SI"]

    # Convert to (SI, AP, LR) = (z, y, x) for SimpleITK
    itk_img = sitk.GetImageFromArray(vol.transpose(2, 1, 0))  # uint8 preserved

    # Set metadata like the reference
    itk_img.SetOrigin((0.0, 0.0, 0.0))
    # Spacing order in SimpleITK is (x, y, z) == (LR, AP, SI)
    itk_img.SetSpacing((float(spacing_LR), float(spacing_AP), float(spacing_SI)))
    itk_img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    out_path = f"{patient_dir}.nii.gz"
    sitk.WriteImage(itk_img, out_path)
    print(f"Saved: {out_path}")
