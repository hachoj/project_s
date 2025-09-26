import os
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


def _collect_patient_files(data_path: Path):
    """Return a map of patient id -> list of patch files (single scan per patient)."""

    patient_files = {}

    with os.scandir(data_path) as patient_entries:
        for patient_entry in patient_entries:
            if not patient_entry.is_dir():
                continue

            with os.scandir(patient_entry.path) as patch_entries:
                patches = [
                    Path(patch_entry.path)
                    for patch_entry in patch_entries
                    if patch_entry.is_file() and patch_entry.name.endswith(".npz")
                ]

            if patches:
                patient_files[patient_entry.name] = patches

    return patient_files


def _copy_patient_files(patients, destination, patient_files, desc, max_workers=None):
    """Copy all patch files for the provided patients."""

    destination.mkdir(parents=True, exist_ok=True)
    total_files = sum(len(patient_files[patient]) for patient in patients)

    if total_files == 0:
        return

    workers = max_workers or min(32, (os.cpu_count() or 1) * 2)

    def _copy_single(src: Path, dst: Path) -> None:
        # shutil.copy uses optimized OS copyfile implementations when available
        shutil.copy(src, dst)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []

        for patient in patients:
            prefix = f"{patient}_"
            for patch_file in patient_files[patient]:
                target = destination / f"{prefix}{patch_file.name}"
                futures.append(executor.submit(_copy_single, patch_file, target))

        for future in tqdm(
            as_completed(futures), total=total_files, desc=desc, unit="file"
        ):
            future.result()


def manual_patient_split(data_dir, train_dir, val_dir):
    """
    Manually split patients between train and val to achieve roughly 95/5 split
    while keeping all patches from each patient together.
    """

    data_path = Path(data_dir)
    patient_files = _collect_patient_files(data_path)

    if not patient_files:
        print("No patient data found.")
        return

    patient_patches = {patient: len(files) for patient, files in patient_files.items()}

    sorted_patients = sorted(
        patient_patches.items(), key=lambda item: item[1], reverse=True
    )

    total_patches = sum(patient_patches.values())
    target_val_patches = int(total_patches * 0.05)

    print(f"Total patches: {total_patches}")
    print(f"Target val patches (~5%): {target_val_patches}")
    print(f"Target train patches (~95%): {total_patches - target_val_patches}")

    val_patients = []
    val_patch_count = 0

    for patient, patch_count in sorted_patients:
        if val_patch_count + patch_count <= target_val_patches * 1.2:  # Allow 20% overage
            val_patients.append(patient)
            val_patch_count += patch_count

    train_patients = [patient for patient, _ in sorted_patients if patient not in val_patients]
    train_patch_count = total_patches - val_patch_count

    print("\nSplit results:")
    print(
        f"Train: {len(train_patients)} patients, {train_patch_count} patches ({train_patch_count/total_patches*100:.1f}%)"
    )
    print(
        f"Val: {len(val_patients)} patients, {val_patch_count} patches ({val_patch_count/total_patches*100:.1f}%)"
    )

    print(f"\nValidation patients: {val_patients}")

    print("\nCopying files...")
    _copy_patient_files(train_patients, Path(train_dir), patient_files, "Copying train files")
    _copy_patient_files(val_patients, Path(val_dir), patient_files, "Copying val files")

    split_info = {
        "train_patients": train_patients,
        "val_patients": val_patients,
        "train_patches": train_patch_count,
        "val_patches": val_patch_count,
        "total_patches": total_patches,
    }

    with open(data_path / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    print(f"\nSplit info saved to {data_path / 'split_info.json'}")
    print("Done!")


if __name__ == "__main__":
    # Adjust these paths as needed
    manual_patient_split(
        data_dir="/home/chojnowski.h/weishao/chojnowski.h/project_s/data/datasets/data_1.8_rev",
        train_dir="/home/chojnowski.h/weishao/chojnowski.h/project_s/data/datasets/train_rev",
        val_dir="/home/chojnowski.h/weishao/chojnowski.h/project_s/data/datasets/val_rev",
    )
