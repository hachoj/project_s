import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader


class MicroUSTrain(Dataset):
    def __init__(
        self,
        data_directory_path,
        patch_size,
        zero_one=False,
    ):
        self.items = []
        self.patch_size = patch_size  # (ph, pw)
        self.zero_one = zero_one
        for fn in sorted(os.listdir(data_directory_path)):
            if not fn.endswith(".npz"):
                continue
            path = os.path.join(data_directory_path, fn)
            # only read the small metadata once
            with np.load(path, allow_pickle=False) as f:
                angle = float(f["angle"])
            self.items.append(
                {
                    "path": path,
                    "angle": angle,
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        path = rec["path"]
        angle = rec["angle"]

        # lazy load the slices + angles
        with np.load(path, allow_pickle=False) as f:
            if self.zero_one:
                slices = torch.from_numpy(f["slices"]).float() / 255.0
            else:
                slices = torch.from_numpy(f["slices"]).float() / 255.0 * 2.0 - 1.0
            angle = torch.from_numpy(angle).float()

        # sanity checks
        assert angle < 1.0, "relative angle difference must be < 1"

        ph, pw = self.patch_size
        _, H, W = slices.shape
        assert ph <= H and pw <= W, "patch size > image size"
        h, w = _rand_crop2d_content_aware(slices, ph, pw)
        slices = slices[:, h : h + ph, w : w + pw]

        return slices, angle


def _worker_init_fn(_):
    # Keep dataloader workers lightweight to avoid CPU oversubscription
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def _rand_crop2d_content_aware(slices, ph, pw, is_train=True):
    H = int(slices.shape[-2])
    W = int(slices.shape[-1])
    if is_train:
        h = 0 if H == ph else np.random.randint(0, H - ph + 1)
        w = 0 if W == pw else np.random.randint(0, W - pw + 1)
        if slices[:, h, w : w + pw].sum() == 0:
            return _rand_crop2d_content_aware(slices, ph, pw)
        if slices[:, h + ph - 1, w : w + pw].sum() == 0:
            return _rand_crop2d_content_aware(slices, ph, pw)
        if slices[:, h : h + ph, w].sum() == 0:
            return _rand_crop2d_content_aware(slices, ph, pw)
        if slices[:, h : h + ph, w + pw - 1].sum() == 0:
            return _rand_crop2d_content_aware(slices, ph, pw)
    else:
        h = 0 if H == ph else int(H / 2) - int(ph / 2)
        w = 0 if W == pw else int(W / 2) - int(pw / 2)

    return int(h), int(w)


def build_train_dataloader(
    data_dir,
    patch_size,
    zero_one,
    num_workers,
    pin_memory,
    batch_size,
):
    dataset = MicroUSTrain(data_dir, patch_size, zero_one)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=_worker_init_fn,
    )


class MicroUSVal(Dataset):
    def __init__(
        self,
        data_directory_path,
        patch_size,
        zero_one=False,
    ):
        self.items = []
        self.patch_size = patch_size  # (ph, pw)
        self.zero_one = zero_one
        for fn in sorted(os.listdir(data_directory_path)):
            if not fn.endswith(".npz"):
                continue
            path = os.path.join(data_directory_path, fn)
            # only read the small metadata once
            with np.load(path, allow_pickle=False) as f:
                angle = float(f["angle"])
            self.items.append(
                {
                    "path": path,
                    "angle": angle,
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        path = rec["path"]
        angle = rec["angle"]

        # lazy load the slices + angles
        with np.load(path, allow_pickle=False) as f:
            if self.zero_one:
                slices = torch.from_numpy(f["slices"]).float() / 255.0
            else:
                slices = torch.from_numpy(f["slices"]).float() / 255.0 * 2.0 - 1.0
            angle = torch.from_numpy(angle).float()

        assert angle < 1.0, "relative angle difference must be < 1"

        ph, pw = self.patch_size
        _, H, W = non_removed_slices.shape
        assert ph <= H and pw <= W, "patch size > image size"
        h, w = _rand_crop2d_content_aware(non_removed_slices, ph, pw, is_train=False)
        slices = slices[:, h : h + ph, w : w + pw]

        return (
            slices,
            angle,
        )


def build_val_dataloader(
    data_dir,
    patch_size,
    desired_angle_difference,
    zero_one=False,
):
    dataset = MicroUSVal(data_dir, patch_size, zero_one)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=_worker_init_fn,
    )
