import os
import numpy as np
from tqdm import tqdm
import math
from bisect import bisect_left
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.reconstruction import extract_slices

# Directories
VOL_DIR_POS = "../project_n/data/Original_Data_in_DICOM_Format/positive"
VOL_DIR_NEG = "../project_n/data/Original_Data_in_DICOM_Format/negative"

ARC_SIZE = 20  # degrees
STEP_SIZE = 10.0  # degrees - how much to skip after a successful save
ACCEPTABLE_AVERAGE_ANGLE_DIFFERENCE = 1.0  # deg/slice
ACCEPTABLE_MAX_ANGLE_DIFFERENCE = 1.5  # deg/slice

OUT_DIR = (
    "data/datasets/data_"
    + str(ARC_SIZE)
    + "_"
    + str(STEP_SIZE)
    + "_"
    + str(ACCEPTABLE_AVERAGE_ANGLE_DIFFERENCE)
)

os.makedirs(OUT_DIR, exist_ok=True)


def process_patient_slices(patient_dict: dict) -> dict:
    # # since the datatype of these slices is np.uint8 you can just normalize 0-255 to [0,1]
    # for key, value in patient_dict.items():
    #     slice_volume = patient_dict[key]["slice_volume"]
    #     patient_dict[key]["slice_volume"] = np.float32(slice_volume) / 255.0

    # sort entries by angle
    sorted_items = sorted(patient_dict.items(), key=lambda kv: kv[1]["angle"])
    sorted_data = dict(sorted_items)

    return sorted_data


def _gap_stats(sorted_angles_window: np.ndarray):
    """
    Compute empirical consecutive-gap stats for a sorted array of angles within a window.
    Returns (mean_gap, max_gap, min_gap, std_gap). If fewer than 2 angles, returns Nones.
    """
    if sorted_angles_window.size < 2:
        return None, None, None, None
    diffs = np.diff(sorted_angles_window)
    return (
        float(np.mean(diffs)),
        float(np.max(diffs)),
        float(np.min(diffs)),
        float(np.std(diffs)) if diffs.size > 1 else 0.0,
    )


def process_patient(
    patient_dict: dict, out_path: str, patient_name: str, label: str
) -> int:
    angles = sorted([value["angle"] for value in patient_dict.values()])
    angles = np.array(angles, dtype=np.float32)
    angle_patient_dict = {
        np.float32(value["angle"]): value for value in patient_dict.values()
    }

    min_angle = math.floor(angles[0])
    max_angle = math.ceil(angles[-1])

    patient_dir = os.path.join(out_path, patient_name)
    os.makedirs(patient_dir, exist_ok=True)

    # Adaptive sliding window
    slice_number = 0
    total_attempts = 0
    start_angle = min_angle

    while start_angle <= max_angle - ARC_SIZE:
        total_attempts += 1
        end_angle = start_angle + ARC_SIZE
        left = bisect_left(angles, start_angle)
        right = bisect_left(angles, end_angle)  # end exclusive
        count = right - left

        if count <= 1:
            start_angle += 1  # Move by 1 degree if not enough slices
            continue

        bounded_angles = np.array(angles[left:right], dtype=np.float32)

        mean_gap, max_gap, min_gap, std_gap = _gap_stats(bounded_angles)

        mean_ok = mean_gap <= ACCEPTABLE_AVERAGE_ANGLE_DIFFERENCE
        max_ok = max_gap <= ACCEPTABLE_MAX_ANGLE_DIFFERENCE

        if mean_ok and max_ok:
            volume_list = []
            for angle in bounded_angles:
                volume_list.append([angle_patient_dict[angle]["slice_volume"], angle])

            window_slices = np.stack(
                [entry[0] for entry in volume_list], axis=0
            ).astype(np.float32)
            window_angles = np.asarray(
                [entry[1] for entry in volume_list], dtype=np.float32
            )

            # Optional sanity checks:
            assert (
                window_slices.ndim == 3
                and window_slices.shape[0] == window_angles.shape[0]
            )

            file_name = f"patch_{slice_number:03d}.npz"
            path = os.path.join(patient_dir, file_name)

            np.savez(
                path,
                slices=window_slices,
                angles=window_angles,
                average_angle_difference=np.float32(mean_gap),
            )

            reversed_window_slices = np.flip(window_slices, axis=0)
            reversed_window_angles = np.flip(window_angles, axis=0) * -1.0
            reversed_file_name = f"patch_{slice_number:03d}_rev.npz"
            reversed_path = os.path.join(patient_dir, reversed_file_name)
            np.savez(
                reversed_path,
                slices=reversed_window_slices,
                angles=reversed_window_angles,
                average_angle_difference=np.float32(mean_gap),
            )

            slice_number += 1

            # Jump by STEP_SIZE after successful save
            start_angle += STEP_SIZE
        else:
            # Move by 1 degree if density criterion not met
            start_angle += 1

    print(
        f"Patient {patient_name} ({label}): {slice_number}/{total_attempts} windows saved."
    )

    return slice_number


if __name__ == "__main__":
    positive_patient_dirs = os.listdir(VOL_DIR_POS)
    negative_patient_dirs = os.listdir(VOL_DIR_NEG)

    patient_stats = {}

    # positive patients
    for patient in tqdm(positive_patient_dirs, desc="positive patients"):
        patient_path = os.path.join(VOL_DIR_POS, patient)
        patient_dict = extract_slices(patient_path)

        # no longer normalize data since there's no point and adds more flexibility
        # to the downstream processing
        sorted_normalized_patient_dict = process_patient_slices(patient_dict)
        num_patches = process_patient(
            sorted_normalized_patient_dict, OUT_DIR, patient, "positive"
        )
        patient_stats[patient] = {"patches": num_patches, "label": "positive"}

    # negative patients
    for patient in tqdm(negative_patient_dirs, desc="negative patients"):
        patient_path = os.path.join(VOL_DIR_NEG, patient)
        patient_dict = extract_slices(patient_path)

        # no longer normalize data since there's no point and adds more flexibility
        # to the downstream processing
        sorted_normalized_patient_dict = process_patient_slices(patient_dict)
        num_patches = process_patient(
            sorted_normalized_patient_dict, OUT_DIR, patient, "negative"
        )
        patient_stats[patient] = {"patches": num_patches, "label": "negative"}

    # Print summary statistics
    print("\n" + "=" * 50)
    print("PATIENT SUMMARY")
    print("=" * 50)

    total_patches = sum(stats["patches"] for stats in patient_stats.values())
    patients_with_data = [
        p for p, stats in patient_stats.items() if stats["patches"] > 0
    ]

    print(f"Total patients: {len(patient_stats)}")
    print(f"Patients with valid patches: {len(patients_with_data)}")
    print(f"Total patches: {total_patches}")
    print(f"Average patches per patient: {total_patches/len(patients_with_data):.1f}")

    print("\nPatient details (sorted by patch count):")
    sorted_patients = sorted(
        patient_stats.items(), key=lambda x: x[1]["patches"], reverse=True
    )
    for patient, stats in sorted_patients:
        if stats["patches"] > 0:
            print(f"  {patient}: {stats['patches']} patches ({stats['label']})")
