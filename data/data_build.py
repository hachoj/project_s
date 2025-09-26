import os
import numpy as np
from tqdm import tqdm
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.reconstruction import extract_slices

# Directories
VOL_DIR_POS = "../project_n/data/Original_Data_in_DICOM_Format/positive"
VOL_DIR_NEG = "../project_n/data/Original_Data_in_DICOM_Format/negative"

# Maximum total angle span between the previous and next slice (degrees).
# Note: this is the full window span (prev → next), not per-slice.
ACCEPTABLE_ANGLE_DIFFERENCE = 1.8

OUT_DIR = (
    "data/datasets/data_"
    + str(ACCEPTABLE_ANGLE_DIFFERENCE)
)

os.makedirs(OUT_DIR, exist_ok=True)


def process_patient_slices(patient_dict: dict) -> dict:
    """Return a dict with entries sorted by angle (ascending)."""
    sorted_items = sorted(patient_dict.items(), key=lambda kv: kv[1]["angle"])
    return dict(sorted_items)


def process_patient(
    patient_dict: dict, out_path: str, patient_name: str, label: str
) -> int:
    """Build 3-slice windows and save triplets (prev, curr, next) with angles.

    - Saves npz per middle slice containing:
        - slices: [3, H, W] uint8, (prev, curr, next)
        - angles: [3] float32, absolute angles in degrees (prev, curr, next)
    - Skips windows whose prev→next total angle exceeds ACCEPTABLE_ANGLE_DIFFERENCE
    - Skips degenerate windows where prev and next angles are identical
    """
    angles = sorted([value["angle"] for value in patient_dict.values()])
    angles = np.array(angles, dtype=np.float32)
    angle_patient_dict = {
        np.float32(value["angle"]): value for value in patient_dict.values()
    }

    # Prefix label to avoid collisions if patient IDs overlap across classes
    patient_dir = os.path.join(out_path, f"{label}_{patient_name}")
    os.makedirs(patient_dir, exist_ok=True)

    previous_angle = None
    next_angle = None

    slice_number = 0
    total_attempts = len(angles) - 2

    i = 0

    # Require at least 3 distinct angles to form a window
    if len(angles) < 3:
        print(f"Patient {patient_name} ({label}): 0/0 windows saved (insufficient slices).")
        return 0

    while i < len(angles) - 2:
        if previous_angle is None:
            previous_angle = angles[i]
            next_angle = angles[i+2]
            i += 1
        else:
            gap = float(np.abs(previous_angle - next_angle))
            # Skip degenerate/duplicate angle windows to avoid divide-by-zero
            if gap <= 1e-8:
                previous_angle = angles[i]
                next_angle = angles[i+2]
                i += 1
                continue

            if gap <= ACCEPTABLE_ANGLE_DIFFERENCE:
                slice_number += 1

                file_name = f"patch_{slice_number:03d}.npz"
                path = os.path.join(patient_dir, file_name)

                previous_slice = angle_patient_dict[previous_angle]["slice_volume"]  # [H,W]
                current_slice = angle_patient_dict[angles[i]]["slice_volume"]  # [H,W]
                next_slice = angle_patient_dict[next_angle]["slice_volume"]  # [H,W]

                slices = np.stack([previous_slice, current_slice, next_slice], axis=0)  # [3,H,W]

                np.savez(
                    path,
                    slices=slices,
                    angles=np.array([previous_angle, angles[i], next_angle], dtype=np.float32),
                )

                # np.savez(
                #     path.replace(".npz", "_rev.npz"),
                #     slices=slices[::-1],
                #     angles=np.array([-1*next_angle, -1*angles[i], -1*previous_angle], dtype=np.float32),
                # )

                previous_angle = angles[i]
                next_angle = angles[i+2]
                i += 1
            else:
                previous_angle = angles[i]
                next_angle = angles[i+2]
                i += 1
                

    print(
        f"Patient {patient_name} ({label}): {slice_number}/{total_attempts} windows saved."
    )

    return slice_number


if __name__ == "__main__":
    positive_patient_dirs = [
        d for d in os.listdir(VOL_DIR_POS) if os.path.isdir(os.path.join(VOL_DIR_POS, d))
    ]
    negative_patient_dirs = [
        d for d in os.listdir(VOL_DIR_NEG) if os.path.isdir(os.path.join(VOL_DIR_NEG, d))
    ]

    total_slices = 0

    # positive patients
    for patient in tqdm(positive_patient_dirs, desc="positive patients"):
        patient_path = os.path.join(VOL_DIR_POS, patient)
        patient_dict = extract_slices(patient_path)
        sorted_normalized_patient_dict = process_patient_slices(patient_dict)
        num_patches = process_patient(
            sorted_normalized_patient_dict, OUT_DIR, patient, "positive"
        )
        total_slices += num_patches

    # negative patients
    for patient in tqdm(negative_patient_dirs, desc="negative patients"):
        patient_path = os.path.join(VOL_DIR_NEG, patient)
        patient_dict = extract_slices(patient_path)
        sorted_normalized_patient_dict = process_patient_slices(patient_dict)
        num_patches = process_patient(
            sorted_normalized_patient_dict, OUT_DIR, patient, "negative"
        )
        total_slices += num_patches

    print(f"Total slices saved: {total_slices}")
