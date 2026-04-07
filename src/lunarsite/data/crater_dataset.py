"""PyTorch Dataset for Stage 1: Crater Detection.

Loads DEM tiles (single-channel elevation) with binary crater masks.
"""

from pathlib import Path
from typing import Optional, Callable

from torch.utils.data import Dataset


class CraterDataset(Dataset):
    """Dataset for crater detection on DEM tiles.

    Args:
        image_dir: Path to DEM tile directory.
        mask_dir: Path to binary crater mask directory.
        transform: Optional albumentations transform pipeline.
    """

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        transform: Optional[Callable] = None,
    ) -> None:
        # TODO: Load DEM/mask file paths
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_paths: list[Path] = []
        self.mask_paths: list[Path] = []

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        """Return dict with keys 'image' (1,H,W float tensor) and 'mask' (H,W long tensor)."""
        # TODO: Load DEM tile and mask, normalize, apply transforms
        pass
