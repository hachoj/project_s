import torch
import yaml
import numpy as np
import cv2
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.utils import (
    reconstruct_removed_hw,
    reconstruct_angle_linear,
    reconstruct_angle_sr,
)
from model.model import ProjectI
from model.reconstruction import extract_slices, reconstruct_volume, save_volume

EVERY_X_SLICES = 1

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"--------------------------------")
    print(f"Loading config file...")

    with open("configs/model/model_config.yaml") as f:
        model_cfg = yaml.safe_load(f)

    with open(f"configs/model/{model_cfg['encoder_type']}_config.yaml") as f:
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

    # data params
    patch_size = inference_cfg["patch_size"]
    stride = inference_cfg["stride"]

    # reconstruction params
    target_step_deg = inference_cfg["target_step_deg"]
    arc_deg = inference_cfg["arc_deg"]
    stride_deg = inference_cfg["stride_deg"]

    # augmentation params
    gaussian_noise_mean = inference_cfg.get("gaussian_noise_mean", 0.0)
    gaussian_noise_std = inference_cfg.get("gaussian_noise_std", 0.0)
    gaussian_noise_scale = inference_cfg.get("gaussian_noise_scale", 1.0)

    zero_one = inference_cfg.get("zero_one", False)
    angle_norm = inference_cfg.get("angle_norm", 3)

    just_sr = inference_cfg.get("just_sr", False)

    print(f"Config file loaded successfully")
    print(f"--------------------------------")
    print(f"Building model...")

    model = ProjectI(
        embd_dim=model_cfg["embd_dim"],
        encoder_type=model_cfg["encoder_type"],
        is_attention_resample=model_cfg["is_attention_resample"],
        encoder_config=encoder_cfg,
        decoder_config=decoder_cfg,
    ).to(device)

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
            arc_deg=arc_deg,
            stride_deg=stride_deg,
            target_step_deg=target_step_deg,
            patch_size=patch_size,
            stride_hw=stride,
            zero_one=zero_one,
            angle_norm=angle_norm,
            reconstruct_removed_hw_fn=reconstruct_removed_hw,
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
            if gaussian_noise_std and gaussian_noise_std > 0.0:
                noise = np.random.normal(
                    loc=gaussian_noise_mean,
                    scale=gaussian_noise_std,
                    size=(
                        int(slice_np.shape[0] // gaussian_noise_scale),
                        int(slice_np.shape[1] // gaussian_noise_scale),
                    ),
                )
                noise = cv2.resize(
                    noise,
                    (slice_np.shape[1], slice_np.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                slice_np = slice_np + noise
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
            + "_inference_"
            + str(target_step_deg)
            + "stp"
            + str(stride_deg)
            + "strd"
            + str(patch_size[0])
            + "ptch"
            + str(EVERY_X_SLICES)
            + "slices",
        )
    print(f"SR inference completed successfully")
    print(f"--------------------------------")
