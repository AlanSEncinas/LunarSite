"""PyTorch Dataset for Stage 2: Terrain Hazard Segmentation.

Loads the Kaggle Artificial Lunar Rocky Landscape dataset.
Images are RGB renders, masks are color-coded PNGs with 4 classes:
    0 = background (black),  1 = small rocks (red),
    2 = large rocks (green), 3 = sky (blue).
"""

from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Color -> class index mapping for the clean masks
COLOR_TO_CLASS = {
    (0, 0, 0): 0,        # background
    (255, 0, 0): 1,       # small rocks
    (0, 255, 0): 2,       # large rocks
    (0, 0, 255): 3,       # sky
}


def color_mask_to_index(mask_rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB color-coded mask to a class index mask.

    Args:
        mask_rgb: (H, W, 3) uint8 array with color-coded classes.

    Returns:
        (H, W) int64 array with class indices 0-3.
    """
    h, w = mask_rgb.shape[:2]
    index_mask = np.zeros((h, w), dtype=np.int64)
    for color, idx in COLOR_TO_CLASS.items():
        match = np.all(mask_rgb == color, axis=-1)
        index_mask[match] = idx
    return index_mask


class LunarTerrainDataset(Dataset):
    """Dataset for lunar terrain segmentation.

    Expects paired files: render####.png in image_dir and clean####.png in mask_dir.

    Args:
        image_dir: Path to directory containing RGB render images.
        mask_dir: Path to directory containing color-coded clean masks.
        transform: Optional albumentations transform pipeline.
    """

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        transform: Optional[Callable] = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        self.image_paths = sorted(self.image_dir.glob("render*.png"))
        self.mask_paths = sorted(self.mask_dir.glob("clean*.png"))

        assert len(self.image_paths) == len(self.mask_paths), (
            f"Mismatch: {len(self.image_paths)} images vs {len(self.mask_paths)} masks"
        )
        assert len(self.image_paths) > 0, f"No images found in {self.image_dir}"

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        """Return dict with 'image' (C,H,W float32) and 'mask' (H,W int64)."""
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_rgb = cv2.imread(str(self.mask_paths[idx]))
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
        mask = color_mask_to_index(mask_rgb)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Convert image HWC uint8 -> CHW float32 [0, 1]
        image = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32) / 255.0)
        mask = torch.from_numpy(mask.astype(np.int64))

        return {"image": image, "mask": mask}
