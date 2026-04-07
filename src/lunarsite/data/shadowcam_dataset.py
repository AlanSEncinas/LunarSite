"""PyTorch Dataset for ShadowCam imagery of permanently shadowed regions.

ShadowCam (aboard KPLO/Danuri) is 200x more sensitive than LROC NAC,
specifically designed to image lunar PSRs at ~2m resolution.

Data available at: data.ser.asu.edu (ASU ShadowCam portal)
PDS archive: pds.shadowcam.asu.edu

Note: ShadowCam images are grayscale, extremely faint, and require
preprocessing (HORUS-style denoising or CLAHE) before feature extraction.

References:
    - ShadowCam instrument: shadowcam.im-ldi.com
    - SfS-SI for PSR 3D (Acta Astronautica 2026)
    - SS-SFS for PSR terrain (Icarus 2025)
"""

from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset


class ShadowCamDataset(Dataset):
    """Dataset for ShadowCam PSR imagery.

    Loads calibrated ShadowCam CDR images (grayscale, 16-bit or 32-bit)
    and applies enhancement preprocessing for downstream tasks.

    Args:
        image_dir: Path to directory containing ShadowCam CDR images.
        transform: Optional albumentations transform pipeline.
        enhance: If True, apply CLAHE enhancement during loading.
        tile_size: If set, tile large images into this size.
    """

    def __init__(
        self,
        image_dir: Path,
        transform: Optional[Callable] = None,
        enhance: bool = True,
        tile_size: Optional[int] = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.enhance = enhance
        self.tile_size = tile_size

        # Support common formats: TIFF (PDS CDR), PNG, IMG
        self.image_paths = sorted(
            list(self.image_dir.glob("*.tif"))
            + list(self.image_dir.glob("*.tiff"))
            + list(self.image_dir.glob("*.png"))
            + list(self.image_dir.glob("*.img"))
        )

        if tile_size and len(self.image_paths) > 0:
            self._build_tile_index()
        else:
            self.tiles = None

    def _build_tile_index(self) -> None:
        """Build an index of tiles for large images."""
        import cv2

        self.tiles = []
        for img_path in self.image_paths:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            h, w = img.shape[:2]
            for y in range(0, h - self.tile_size + 1, self.tile_size):
                for x in range(0, w - self.tile_size + 1, self.tile_size):
                    self.tiles.append((img_path, y, x))

    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE enhancement to extremely faint PSR imagery."""
        import cv2

        # Normalize to 8-bit for CLAHE
        if image.dtype == np.uint16:
            # Use log scaling for high dynamic range
            img_float = image.astype(np.float32)
            img_float = np.log1p(img_float)
            img_float = (img_float / img_float.max() * 255).astype(np.uint8)
        elif image.dtype == np.float32:
            img_float = (image / image.max() * 255).astype(np.uint8) if image.max() > 0 else image.astype(np.uint8)
        else:
            img_float = image

        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        return clahe.apply(img_float)

    def __len__(self) -> int:
        if self.tiles is not None:
            return len(self.tiles)
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        """Return dict with 'image' (1, H, W) float tensor."""
        import cv2

        if self.tiles is not None:
            img_path, y, x = self.tiles[idx]
            image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            image = image[y : y + self.tile_size, x : x + self.tile_size]
        else:
            image = cv2.imread(str(self.image_paths[idx]), cv2.IMREAD_UNCHANGED)

        if image is None:
            raise RuntimeError(f"Failed to load image at index {idx}")

        # Ensure 2D grayscale
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.enhance:
            image = self._enhance_image(image)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        # Convert to tensor: (1, H, W) float32
        if image.dtype == np.uint8:
            tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
        else:
            tensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            if tensor.max() > 0:
                tensor = tensor / tensor.max()

        return {"image": tensor, "path": str(self.image_paths[idx] if self.tiles is None else self.tiles[idx][0])}
