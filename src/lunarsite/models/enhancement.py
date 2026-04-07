"""Dark image enhancement for lunar PSR and low-light imagery.

Implements enhancement preprocessing for extremely dark lunar images,
drawing from HORUS (Nature Comms 2021), DarkIR (CVPR 2025), and
Zero-DCE approaches.

These modules preprocess dark imagery before feeding into the segmentation
or depth estimation pipelines.

References:
    - HORUS: nature.com/articles/s41467-021-25882-z
    - DarkIR: github.com/cidautai/DarkIR
    - Zero-DCE: github.com/Li-Chongyi/Zero-DCE
    - SCI (CVPR 2022): github.com/vis-opt-group/SCI
"""

import torch
import torch.nn as nn
import numpy as np


class AdaptiveHistogramEnhancer:
    """CLAHE-based enhancement for low-light lunar imagery.

    Applies Contrast Limited Adaptive Histogram Equalization,
    which is the standard preprocessing for LROC NAC dark images.
    """

    def __init__(self, clip_limit: float = 3.0, grid_size: int = 8):
        self.clip_limit = clip_limit
        self.grid_size = grid_size

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Enhance a dark image using CLAHE.

        Args:
            image: Grayscale (H, W) or RGB (H, W, 3) uint8 image.

        Returns:
            Enhanced image, same shape and dtype.
        """
        import cv2

        if image.ndim == 3:
            # Convert to LAB, apply CLAHE to L channel
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=(self.grid_size, self.grid_size),
            )
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=(self.grid_size, self.grid_size),
            )
            return clahe.apply(image)


class ZeroDCELight(nn.Module):
    """Lightweight zero-reference deep curve estimation for dark images.

    Learns image-specific tone curves without paired training data.
    Adapted from Zero-DCE++ for efficiency on large lunar image tiles.

    The key idea: instead of learning an enhancement function directly,
    learn parameters of a curve that maps dark pixels to brighter ones,
    with losses that preserve naturalness.
    """

    def __init__(self, num_iterations: int = 8):
        super().__init__()
        self.num_iterations = num_iterations

        # Lightweight curve parameter estimator
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3 * num_iterations, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhance dark image via learned curve estimation.

        Args:
            x: Input image (N, 3, H, W) in [0, 1].

        Returns:
            Enhanced image (N, 3, H, W) in [0, 1].
        """
        curve_params = self.net(x)
        # Split into per-iteration curve parameters
        curves = torch.split(curve_params, 3, dim=1)

        enhanced = x
        for curve in curves:
            enhanced = enhanced + curve * (enhanced - enhanced * enhanced)

        return enhanced.clamp(0, 1)


class HORUSDenoiser(nn.Module):
    """HORUS-inspired denoising for extremely faint PSR imagery.

    Two-stage approach:
    1. DestripeNet: Remove systematic CCD stripe noise
    2. PhotonNet: Remove shot noise, read noise, and compression artifacts

    Simplified from the original HORUS architecture for practical use.
    The original used 70,000+ dark calibration frames for training.

    Reference: Bickel et al., Nature Communications (2021)
    """

    def __init__(self):
        super().__init__()

        # Stage 1: Destripe — 1D convolutions along detector columns
        self.destripe = nn.Sequential(
            nn.Conv2d(1, 16, (1, 7), padding=(0, 3)),  # horizontal
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, (1, 7), padding=(0, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, (1, 7), padding=(0, 3)),
        )

        # Stage 2: Denoise — U-Net style encoder-decoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
        )

        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Denoise an extremely faint grayscale image.

        Args:
            x: Input (N, 1, H, W) — raw or minimally processed PSR image.

        Returns:
            Denoised image (N, 1, H, W).
        """
        # Stage 1: Destripe (residual)
        stripe_noise = self.destripe(x)
        destriped = x - stripe_noise

        # Stage 2: Denoise (U-Net with skip connections)
        e1 = self.encoder1(destriped)
        e2 = self.encoder2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.decoder2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final(d1)
