import os
import shutil
from pathlib import Path
import json


def manual_patient_split(data_dir, train_dir, val_dir):
    """
    Manually split patients between train and val to achieve roughly 95/5 split
    while keeping all patches from each patient together.
    """

    # Get all patient directories
    patient_dirs = [d for d in Path(data_dir).iterdir() if d.is_dir()]

    # Count patches per patient
    patient_patches = {}
    for patient_dir in patient_dirs:
        num_patches = len(list(patient_dir.glob("*.npz")))
        if num_patches > 0:
            patient_patches[patient_dir.name] = num_patches

    # Sort patients by patch count (descending)
    sorted_patients = sorted(patient_patches.items(), key=lambda x: x[1], reverse=True)

    total_patches = sum(patient_patches.values())
    target_val_patches = int(total_patches * 0.05)

    print(f"Total patches: {total_patches}")
    print(f"Target val patches (~5%): {target_val_patches}")
    print(f"Target train patches (~95%): {total_patches - target_val_patches}")

    # Greedy assignment to validation set
    val_patients = []
    val_patch_count = 0

    for patient, patch_count in sorted_patients:
        if (
            val_patch_count + patch_count <= target_val_patches * 1.2
        ):  # Allow 20% overage
            val_patients.append(patient)
            val_patch_count += patch_count

    train_patients = [p for p, _ in sorted_patients if p not in val_patients]
    train_patch_count = total_patches - val_patch_count

    print(f"\nSplit results:")
    print(
        f"Train: {len(train_patients)} patients, {train_patch_count} patches ({train_patch_count/total_patches*100:.1f}%)"
    )
    print(
        f"Val: {len(val_patients)} patients, {val_patch_count} patches ({val_patch_count/total_patches*100:.1f}%)"
    )

    print(f"\nValidation patients: {val_patients}")

    # Create train and val directories
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(val_dir).mkdir(parents=True, exist_ok=True)

    # Copy patches to appropriate directories (flattened structure)
    print("\nCopying files...")

    # Train patients
    for patient in train_patients:
        patient_dir = Path(data_dir) / patient
        for patch_file in patient_dir.glob("*.npz"):
            # Create new filename with patient prefix to avoid collisions
            new_name = f"{patient}_{patch_file.name}"
            shutil.copy(patch_file, Path(train_dir) / new_name)

    # Val patients
    for patient in val_patients:
        patient_dir = Path(data_dir) / patient
        for patch_file in patient_dir.glob("*.npz"):
            # Create new filename with patient prefix to avoid collisions
            new_name = f"{patient}_{patch_file.name}"
            shutil.copy(patch_file, Path(val_dir) / new_name)

    # Save split info for reproducibility
    split_info = {
        "train_patients": train_patients,
        "val_patients": val_patients,
        "train_patches": train_patch_count,
        "val_patches": val_patch_count,
        "total_patches": total_patches,
    }

    with open(Path(data_dir) / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\nSplit info saved to {Path(data_dir) / 'split_info.json'}")
    print("Done!")


if __name__ == "__main__":
    # Adjust these paths as needed
    manual_patient_split(
        data_dir="/home/chojnowski.h/weishao/chojnowski.h/project_i/data/datasets/data_20_10.0_1.0",
        train_dir="/home/chojnowski.h/weishao/chojnowski.h/project_i/data/datasets/train",
        val_dir="/home/chojnowski.h/weishao/chojnowski.h/project_i/data/datasets/val",
    )
