"""PyTorch Dataset for Stage 1: Crater Detection on DeepMoon HDF5 tiles.

DeepMoon (Silburt 2019, Zenodo 1133969) ships pre-tiled 256x256 DEM images
paired with float crater masks, packed into HDF5:
  - `input_images`   shape (N, 256, 256), uint8
  - `target_masks`   shape (N, 256, 256), float32 in [0, 1]

We treat this as binary segmentation: predict a crater-ring mask from the DEM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class DeepMoonCraterDataset(Dataset):
    """DeepMoon HDF5 tiles -> (1-ch DEM tensor, binary mask tensor).

    Opens the HDF5 file lazily per worker so it survives DataLoader forking.
    """

    def __init__(
        self,
        hdf5_path: Path | str,
        transform: Optional[Callable] = None,
        binary_threshold: float = 0.0,
    ) -> None:
        self.hdf5_path = str(hdf5_path)
        self.transform = transform
        self.binary_threshold = binary_threshold
        self._file: Optional[h5py.File] = None
        with h5py.File(self.hdf5_path, "r") as f:
            self.length = f["input_images"].shape[0]

    def _open(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r", swmr=True)
        return self._file

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        f = self._open()
        img = f["input_images"][idx][...]  # (256, 256) uint8
        mask = f["target_masks"][idx][...]  # (256, 256) float32 [0, 1]

        # Binary target: any nonzero ring pixel counts as crater.
        mask_bin = (mask > self.binary_threshold).astype(np.uint8)

        if self.transform is not None:
            t = self.transform(image=img, mask=mask_bin)
            img, mask_bin = t["image"], t["mask"]

        # Normalize DEM to [0, 1] float and add channel dim -> (1, H, W).
        img_t = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        mask_t = torch.from_numpy(mask_bin.astype(np.float32))  # (H, W) float for BCE
        return {"image": img_t, "mask": mask_t}


# Backwards-compat alias referenced in older planning docs.
CraterDataset = DeepMoonCraterDataset
