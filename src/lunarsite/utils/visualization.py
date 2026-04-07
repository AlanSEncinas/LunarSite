"""Visualization utilities for segmentation masks, predictions, and SHAP plots."""

from pathlib import Path
from typing import Optional

import numpy as np


# Class colormap for terrain segmentation (RGB)
TERRAIN_COLORS = {
    0: (0, 0, 0),        # background - black
    1: (255, 165, 0),    # small rocks - orange
    2: (255, 0, 0),      # large rocks - red
    3: (135, 206, 235),  # sky - light blue
}


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay a colored segmentation mask on an image.

    Args:
        image: RGB image (H, W, 3), uint8.
        mask: Class index mask (H, W), int.
        alpha: Transparency for overlay.

    Returns:
        Blended image (H, W, 3), uint8.
    """
    # TODO: Implement color overlay
    pass


def plot_training_curves(log_path: Path, output_path: Optional[Path] = None) -> None:
    """Plot loss and metric curves from a training log.

    Args:
        log_path: Path to CSV or JSON training log.
        output_path: If provided, save figure here instead of showing.
    """
    # TODO: Implement training curve plotting
    pass
