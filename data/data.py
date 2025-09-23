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
        angle_norm=3,
        desired_angle_difference=1.5,
        non_uniform_desired_angle_difference=(1.5, 2.0),
        arc_size=30,
    ):
        self.items = []
        self.desired_angle_difference = desired_angle_difference
        self.non_uniform_desired_angle_difference = non_uniform_desired_angle_difference
        self.patch_size = patch_size  # (ph, pw)
        self.zero_one = zero_one
        self.arc_size = arc_size
        self.angle_norm = angle_norm
        for fn in sorted(os.listdir(data_directory_path)):
            if not fn.endswith(".npz"):
                continue
            path = os.path.join(data_directory_path, fn)
            # only read the small metadata once
            with np.load(path, allow_pickle=False) as f:
                average_angle_difference = float(f["average_angle_difference"])
            self.items.append(
                {
                    "path": path,
                    "average_angle_difference": average_angle_difference,
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        path = rec["path"]
        average_angle_difference = rec["average_angle_difference"]

        # lazy load the slices + angles
        with np.load(path, allow_pickle=False) as f:
            if self.zero_one:
                slices = torch.from_numpy(f["slices"]).float() / 255.0
            else:
                slices = torch.from_numpy(f["slices"]).float() / 255.0 * 2.0 - 1.0
            angles = torch.from_numpy(f["angles"]).float()

        num_slices = angles.shape[0]

        if self.non_uniform_desired_angle_difference:
            desired_angle_difference = float(
                np.random.uniform(
                    self.non_uniform_desired_angle_difference[0],
                    self.non_uniform_desired_angle_difference[1],
                )
            )
        else:
            desired_angle_difference = float(self.desired_angle_difference)

        # sanity checks
        assert slices.shape[0] == angles.shape[0], "num slices mismatch"
        assert average_angle_difference <= 1.0, "average angle difference must be ≤ 1"
        assert (
            desired_angle_difference > average_angle_difference
        ), "desired angle difference must be greater than average angle difference"

        # choose which slices to remove
        remove_n_slices = int(
            num_slices * (1.0 - (average_angle_difference / desired_angle_difference))
        )
        remove_n_slices = max(0, min(remove_n_slices, num_slices - 1))
        if remove_n_slices == 0 and num_slices == 1:
            raise ValueError("Only one slice in the patch, cannot remove any")

        idx_remove = torch.randperm(num_slices)[:remove_n_slices]
        keep = torch.ones(num_slices, dtype=torch.bool)
        keep[idx_remove] = False

        non_removed_slices = slices[keep]
        non_removed_angles = angles[keep]
        removed_slices = slices[idx_remove]
        removed_angles = angles[idx_remove]

        ph, pw = self.patch_size
        H, W = non_removed_slices.shape[-2:]
        assert ph <= H and pw <= W, "patch size > image size"
        h, w = _rand_crop2d_content_aware(non_removed_slices, ph, pw)
        slices = slices[:, h : h + ph, w : w + pw]
        non_removed_slices = non_removed_slices[:, h : h + ph, w : w + pw]
        removed_slices = removed_slices[:, h : h + ph, w : w + pw]

        # normalize angles to [-pi/3, pi/3]
        min_angle = angles.min()
        max_angle = min_angle + self.arc_size
        extra_room = max_angle - angles.max()
        assert (
            max_angle >= angles.max()
        ), f"arc size is too small: min_angle: {min_angle}, max_angle: {max_angle}, angles.max(): {angles.max()}"

        non_removed_angles = (non_removed_angles - min_angle) / (
            max_angle - min_angle
        ) * np.pi / (self.angle_norm/2) - np.pi / self.angle_norm
        removed_angles = (removed_angles - min_angle) / (
            max_angle - min_angle
        ) * np.pi / (self.angle_norm/2) - np.pi / self.angle_norm
        angles = (angles - min_angle) / (max_angle - min_angle) * np.pi / (self.angle_norm/2) - np.pi / self.angle_norm

        # randomly shift angles within the extra room to avoid always starting at -pi/6
        # and having similar angles all the time
        extra_room = extra_room / (max_angle - min_angle) * np.pi / (self.angle_norm/2)
        add_angle = np.random.uniform(0, extra_room)
        non_removed_angles += add_angle
        removed_angles += add_angle
        angles += add_angle


        return (
            non_removed_slices,
            non_removed_angles,
            removed_slices,
            removed_angles,
            slices,
            angles,
        )


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
    desired_angle_difference,
    non_uniform_desired_angle_difference,
    zero_one,
    angle_norm,
    arc_size,
    num_workers,
    pin_memory,
):
    dataset = MicroUSTrain(
        data_dir,
        patch_size,
        zero_one,
        angle_norm,
        desired_angle_difference,
        non_uniform_desired_angle_difference,
        arc_size,
    )
    return DataLoader(
        dataset,
        batch_size=1,
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
        angle_norm=3,
        desired_angle_difference=1.5,
        arc_size=45,
    ):
        self.items = []
        self.desired_angle_difference = desired_angle_difference
        self.patch_size = patch_size  # (ph, pw)
        self.arc_size = arc_size
        self.zero_one = zero_one
        self.angle_norm = angle_norm
        for fn in sorted(os.listdir(data_directory_path)):
            if not fn.endswith(".npz"):
                continue
            path = os.path.join(data_directory_path, fn)
            # only read the small metadata once
            with np.load(path, allow_pickle=False) as f:
                average_angle_difference = float(f["average_angle_difference"])
            self.items.append(
                {
                    "path": path,
                    "average_angle_difference": average_angle_difference,
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        path = rec["path"]
        average_angle_difference = rec["average_angle_difference"]

        # lazy load the slices + angles
        with np.load(path, allow_pickle=False) as f:
            if self.zero_one:
                slices = torch.from_numpy(f["slices"]).float() / 255.0
            else:
                slices = torch.from_numpy(f["slices"]).float() / 255.0 * 2.0 - 1.0
            angles = torch.from_numpy(f["angles"]).float()

        num_slices = angles.shape[0]

        desired_angle_difference = float(self.desired_angle_difference)

        # sanity checks
        assert slices.shape[0] == angles.shape[0], "num slices mismatch"
        assert average_angle_difference <= 1.0, "average angle difference must be ≤ 1"
        assert (
            desired_angle_difference > average_angle_difference
        ), "desired angle difference must be greater than average angle difference"

        # choose which slices to remove
        remove_n_slices = int(
            num_slices * (1.0 - (average_angle_difference / desired_angle_difference))
        )
        remove_n_slices = max(0, min(remove_n_slices, num_slices - 1))
        if remove_n_slices == 0 and num_slices == 1:
            raise ValueError("Only one slice in the patch, cannot remove any")

        idx_remove = torch.randperm(num_slices)[:remove_n_slices]
        keep = torch.ones(num_slices, dtype=torch.bool)
        keep[idx_remove] = False

        non_removed_slices = slices[keep]
        non_removed_angles = angles[keep]
        removed_slices = slices[idx_remove]
        removed_angles = angles[idx_remove]

        ph, pw = self.patch_size
        H, W = non_removed_slices.shape[-2:]
        assert ph <= H and pw <= W, "patch size > image size"
        h, w = _rand_crop2d_content_aware(non_removed_slices, ph, pw, is_train=False)
        slices = slices[:, h : h + ph, w : w + pw]
        non_removed_slices = non_removed_slices[:, h : h + ph, w : w + pw]
        removed_slices = removed_slices[:, h : h + ph, w : w + pw]

        # normalize angles to [-pi/8, pi/8]
        min_angle = angles.min()
        max_angle = min_angle + self.arc_size
        assert max_angle >= angles.max(), "arc size is too small"

        non_removed_angles = (non_removed_angles - min_angle) / (
            max_angle - min_angle
        ) * np.pi / (self.angle_norm/2) - np.pi / self.angle_norm
        removed_angles = (removed_angles - min_angle) / (
            max_angle - min_angle
        ) * np.pi / (self.angle_norm/2) - np.pi / self.angle_norm
        angles = (angles - min_angle) / (max_angle - min_angle) * np.pi / (self.angle_norm/2) - np.pi / self.angle_norm

        return (
            non_removed_slices,
            non_removed_angles,
            removed_slices,
            removed_angles,
            slices,
            angles,
        )


def build_val_dataloader(
    data_dir,
    patch_size,
    desired_angle_difference,
    arc_size,
    zero_one=False,
    angle_norm=3,
    num_workers=0,
    pin_memory=True,
):
    ds = MicroUSVal(data_dir, patch_size, zero_one, angle_norm, desired_angle_difference, arc_size)
    return DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=_worker_init_fn,
    )
