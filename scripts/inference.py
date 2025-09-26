import torch
import yaml
import numpy as np
import time

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from model.utils import (
    reconstruct_angle_linear,
    reconstruct_angle_sr,
)
from model.model import ProjectI
from model.reconstruction import extract_slices, reconstruct_volume, save_volume

EVERY_X_SLICES = 2

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--------------------------------")
    print(f"Loading config file...")

    with open("configs/model/model_config.yaml") as f:
        model_cfg = yaml.safe_load(f)

    with open("configs/model/encoder.yaml") as f:
        encoder_cfg = yaml.safe_load(f)

    with open("configs/model/decoder_config.yaml") as f:
        decoder_cfg = yaml.safe_load(f)

    with open("configs/inference/inference_config.yaml") as f:
        inference_cfg = yaml.safe_load(f)

    # paths
    model_path = inference_cfg["model_path"]
    image_path = inference_cfg["image_path"]
    save_path = inference_cfg["save_path"]
    patient_name = image_path.split("/")[-1]

    patch_size = model_cfg["patch_size"]
    stride = model_cfg["stride"]
    zero_one = bool(model_cfg.get("zero_one", False))  # model expects inputs in [-1,1] if zero_one=False

    # reconstruction params
    target_step_deg = inference_cfg["target_step_deg"]

    just_sr = inference_cfg.get("just_sr", False)

    print(f"Config file loaded successfully")
    print(f"--------------------------------")
    print(f"Building model...")

    model = ProjectI(
        embd_dim=model_cfg["embd_dim"],
        patch_size=model_cfg["patch_size"],
        num_harmonics=model_cfg["num_harmonics"],
        num_heads=model_cfg["num_heads"],
        encoder_config=encoder_cfg,
        decoder_config=decoder_cfg,
    ).to(device)

    model = torch.compile(model)

    model.load_state_dict(torch.load(model_path)["model_state"])

    print(f"Model built successfully")
    print(f"--------------------------------")
    print(f"Loading data...")

    extracted_slices = extract_slices(image_path)
    # Get the first key in extracted_slices instead of hardcoding
    first_key = next(iter(extracted_slices))
    shape = extracted_slices[first_key]["shape"]
    spacing = extracted_slices[first_key]["spacing"]

    print(f"Data loaded successfully")
    print(f"--------------------------------")

    if not just_sr:
        print(f"Saving original volume...")

        save_volume(
            extracted_slices,
            reconstruct_volume(extracted_slices),
            save_path + patient_name + "_original",
        )

        print(f"Original volume saved successfully")
        print(f"--------------------------------")

    print(f"Beginning inference...")

    reconstructed_slices = dict()

    slices = []
    angles = []

    for value in extracted_slices.values():
        angles.append(value["angle"])
        slices.append(value["slice_volume"])

    idx = torch.arange(0, len(extracted_slices), EVERY_X_SLICES)

    angles = torch.tensor(np.array(angles))
    slices = torch.tensor(np.array(slices))

    slices_lr = slices[idx]  # (T, H, W)
    angles_lr = angles[idx]  # (T,)


    if not just_sr:
        print(f"Saving lr volume with every {EVERY_X_SLICES} slices")
        reconstructed_slices_lr = dict()
        for linear_angle, linear_slice_HW in zip(angles_lr, slices_lr):
            reconstructed_slices_lr[linear_angle.item()] = {
                "angle": linear_angle.item(),
                "slice_volume": linear_slice_HW.numpy().astype(np.uint8),
                "shape": shape,
                "spacing": spacing,
            }
        reconstructed_volume_lr = reconstruct_volume(reconstructed_slices_lr)
        save_volume(
            reconstructed_slices_lr,
            reconstructed_volume_lr,
            save_path + patient_name + f"_lr_every_{EVERY_X_SLICES}_slices",
        )

        print(f"LR volume saved successfully")
        print(f"--------------------------------")

    with torch.no_grad():
        slices_lr = slices_lr.to(device)
        angles_lr = angles_lr.to(device)

        if not just_sr:
            print(f"Starting linear inference...")
            pred_uniform, grid_deg = reconstruct_angle_linear(
                slices_THW=slices_lr,
                angles_deg_T=angles_lr,
                target_step_deg=target_step_deg,
            )

            # pred_uniform: [R, 1, H, W], grid_deg: [R]
            for linear_angle, linear_slice_CHW in zip(grid_deg, pred_uniform):
                # take the single channel → [H, W]
                linear_slice_HW = linear_slice_CHW[0]
                reconstructed_slices[linear_angle.item()] = {
                    "angle": linear_angle.item(),
                    "slice_volume": linear_slice_HW
                    .cpu()
                    .numpy()
                    .astype(np.uint8),
                    "shape": shape,
                    "spacing": spacing,
                }

            reconstructed_volume = reconstruct_volume(reconstructed_slices)
            save_volume(
                reconstructed_slices,
                reconstructed_volume,
            save_path + patient_name + "_lin_" + str(target_step_deg) + "_deg_" + str(EVERY_X_SLICES) + "_slices",
            )
            print(f"Linear inference completed successfully")
            print(f"--------------------------------")

        print(f"Starting SR inference...")
        start_time = time.time()
        pred_uniform, grid_deg = reconstruct_angle_sr(
            model,
            slices_THW=slices_lr,  # (T,H,W)
            angles_deg_T=angles_lr,  # (T,)
            target_step_deg=target_step_deg,
            zero_one=zero_one,
            patch_size=patch_size,
            stride_hw=stride,
        )
        end_time = time.time()
        print(f"SR inference time: {end_time - start_time} seconds")

        reconstructed_slices = dict()

        for sr_angle, sr_slices in zip(grid_deg, pred_uniform):
            slice_np = (
                ((sr_slices * 255.0) if zero_one else (sr_slices + 1) * (127.5))
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            slice_np = np.clip(slice_np, 0, 255).astype(np.uint8)
            reconstructed_slices[sr_angle.item()] = {
                "angle": sr_angle.item(),
                "slice_volume": slice_np,
                "shape": shape,
                "spacing": spacing,
            }

        reconstructed_volume = reconstruct_volume(reconstructed_slices)
        save_volume(
            reconstructed_slices,
            reconstructed_volume,
            save_path
            + patient_name
            + "_inf_"
            + str(target_step_deg)
            + "_stp_"
            + str(EVERY_X_SLICES)
            + "slices",
        )
    print(f"SR inference completed successfully")
    print(f"--------------------------------")
