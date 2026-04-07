"""PyTorch Dataset for LuSNAR: Lunar Segmentation, Navigation, and Rendering.

Multi-task lunar benchmark with 5 segmentation classes from Unreal Engine
simulations across 9 lunar scenes. 108 GB total with stereo RGB, semantic
labels, depth maps, and LiDAR point clouds.

Classes: {0: regolith, 1: craters, 2: rocks, 3: mountains, 4: sky}

Reference: github.com/zqyu9/LuSNAR-dataset (July 2024)
HuggingFace: JeremyLuo/LuSNAR
"""

from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset


# LuSNAR class mapping
LUSNAR_CLASSES = {
    0: "regolith",
    1: "craters",
    2: "rocks",
    3: "mountains",
    4: "sky",
}

# Map LuSNAR 5-class to our 4-class scheme for joint training
# LuSNAR: regolith(0), craters(1), rocks(2), mountains(3), sky(4)
# Ours:   background(0), small_rocks(1), large_rocks(2), sky(3)
LUSNAR_TO_LUNAR = {
    0: 0,  # regolith -> background
    1: 0,  # craters -> background (crater rims are terrain)
    2: 2,  # rocks -> large_rocks (LuSNAR doesn't distinguish size)
    3: 0,  # mountains -> background
    4: 3,  # sky -> sky
}


class LuSNARDataset(Dataset):
    """Dataset for LuSNAR lunar segmentation benchmark.

    Supports both native 5-class labels and remapped 4-class labels
    for joint training with the Kaggle Lunar Landscape dataset.

    Args:
        data_dir: Path to extracted LuSNAR data.
        scene_ids: List of scene IDs to include (1-9). None = all.
        transform: Optional albumentations transform pipeline.
        remap_classes: If True, remap 5 classes to our 4-class scheme.
        include_depth: If True, also load depth maps.
    """

    def __init__(
        self,
        data_dir: Path,
        scene_ids: Optional[list[int]] = None,
        transform: Optional[Callable] = None,
        remap_classes: bool = True,
        include_depth: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.remap_classes = remap_classes
        self.include_depth = include_depth

        # Discover image/mask pairs across scenes
        self.image_paths: list[Path] = []
        self.mask_paths: list[Path] = []
        self.depth_paths: list[Path] = []

        scenes = scene_ids or list(range(1, 10))
        for scene_id in scenes:
            scene_dir = self.data_dir / f"scene_{scene_id}"
            if not scene_dir.exists():
                continue

            rgb_dir = scene_dir / "rgb"
            seg_dir = scene_dir / "semantic"
            depth_dir = scene_dir / "depth"

            if not rgb_dir.exists() or not seg_dir.exists():
                continue

            rgb_files = sorted(rgb_dir.glob("*.png"))
            for rgb_path in rgb_files:
                seg_path = seg_dir / rgb_path.name
                if seg_path.exists():
                    self.image_paths.append(rgb_path)
                    self.mask_paths.append(seg_path)
                    if include_depth and depth_dir.exists():
                        depth_path = depth_dir / rgb_path.name
                        self.depth_paths.append(depth_path if depth_path.exists() else None)

        assert len(self.image_paths) > 0, f"No images found in {self.data_dir}"

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        """Return dict with 'image', 'mask', and optionally 'depth'."""
        import cv2

        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(np.int64)

        if self.remap_classes:
            remapped = np.zeros_like(mask)
            for src, dst in LUSNAR_TO_LUNAR.items():
                remapped[mask == src] = dst
            mask = remapped

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32) / 255.0)
        mask = torch.from_numpy(mask.astype(np.int64))

        result = {"image": image, "mask": mask}

        if self.include_depth and self.depth_paths and self.depth_paths[idx] is not None:
            depth = cv2.imread(str(self.depth_paths[idx]), cv2.IMREAD_UNCHANGED)
            if depth is not None:
                result["depth"] = torch.from_numpy(depth.astype(np.float32))

        return result
