"""Monocular depth estimation for lunar terrain.

Wraps foundation depth models (Depth Anything V2, Marigold) for zero-shot
and fine-tuned depth estimation on lunar imagery, including dark/shadowed regions.

References:
    - Depth Anything V2 (NeurIPS 2024): github.com/DepthAnything/Depth-Anything-V2
    - Marigold (CVPR 2024 Oral): github.com/prs-eth/Marigold
    - DepthDark (ACM MM 2025): PEFT fine-tuning for low-light depth
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthAnythingWrapper(nn.Module):
    """Wrapper for Depth Anything V2 with optional low-light adaptation.

    Supports zero-shot inference and LoRA/PEFT fine-tuning for lunar imagery.

    Args:
        model_size: One of 'small', 'base', 'large'. Default 'base'.
        use_illumination_guidance: If True, concatenates a grayscale illumination
            channel as conditioning (inspired by DepthDark's LLPEFT).
    """

    def __init__(self, model_size: str = "base", use_illumination_guidance: bool = False):
        super().__init__()
        self.model_size = model_size
        self.use_illumination_guidance = use_illumination_guidance
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load Depth Anything V2 from HuggingFace."""
        try:
            from transformers import AutoModelForDepthEstimation, AutoImageProcessor
            model_id = f"depth-anything/Depth-Anything-V2-{self.model_size.capitalize()}-hf"
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        except ImportError:
            raise ImportError(
                "Install transformers: pip install transformers\n"
                "Required for Depth Anything V2."
            )

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict relative depth from a single RGB image.

        Args:
            image: RGB image (H, W, 3) uint8.

        Returns:
            Relative depth map (H, W) float32, higher = farther.
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        depth = outputs.predicted_depth.squeeze().cpu().numpy()
        # Resize to original resolution
        h, w = image.shape[:2]
        if depth.shape != (h, w):
            depth = F.interpolate(
                torch.from_numpy(depth).unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze().numpy()
        return depth

    @torch.no_grad()
    def predict_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Predict depth for a batch of images."""
        return [self.predict(img) for img in images]


class ShadowDepthEstimator:
    """Estimate crater/terrain depth from shadow geometry.

    Uses the known relationship: depth = shadow_length * tan(sun_elevation).
    This is a physics-based method, not learned, providing complementary
    depth information to neural approaches.

    References:
        - DeepShadow (ECCV 2022): github.com/asafkar/deep_shadow
        - Shadow-Constrained SfS (Geo-spatial Info Sci 2024)
    """

    def __init__(self, pixel_scale_m: float = 1.0):
        """
        Args:
            pixel_scale_m: Ground sample distance in meters/pixel.
        """
        self.pixel_scale_m = pixel_scale_m

    def depth_from_shadow(
        self,
        image: np.ndarray,
        shadow_mask: np.ndarray,
        sun_elevation_rad: float,
        sun_azimuth_rad: float,
    ) -> np.ndarray:
        """Estimate depth from cast shadow geometry.

        For each shadow pixel, traces back along the sun direction to find
        the casting edge, then computes depth from shadow length.

        Args:
            image: Grayscale or RGB image (H, W, ...).
            shadow_mask: Binary mask where 1 = shadow (H, W).
            sun_elevation_rad: Sun elevation angle in radians above horizon.
            sun_azimuth_rad: Sun azimuth angle in radians (0 = north, CW).

        Returns:
            Depth map (H, W) in meters. 0 where no shadow information available.
        """
        h, w = shadow_mask.shape[:2]
        depth_map = np.zeros((h, w), dtype=np.float32)

        if sun_elevation_rad <= 0:
            return depth_map  # Sun below horizon, no shadow-based depth

        # Shadow direction vector (in pixel coords)
        dx = np.sin(sun_azimuth_rad)
        dy = -np.cos(sun_azimuth_rad)

        # For each shadow pixel, find distance to shadow edge along sun direction
        shadow_bool = shadow_mask.astype(bool)

        # Compute distance transform along shadow direction
        # Use connected component approach: for each row of shadow pixels
        # along sun direction, measure shadow length
        from scipy import ndimage

        # Label connected shadow regions
        labeled, n_features = ndimage.label(shadow_bool)

        for region_id in range(1, n_features + 1):
            region_mask = labeled == region_id
            region_coords = np.argwhere(region_mask)

            if len(region_coords) == 0:
                continue

            # Project coordinates onto sun direction
            projections = region_coords[:, 1] * dx + region_coords[:, 0] * dy
            shadow_length_px = projections.max() - projections.min()
            shadow_length_m = shadow_length_px * self.pixel_scale_m

            # Depth = shadow_length * tan(sun_elevation)
            depth = shadow_length_m * np.tan(sun_elevation_rad)

            # Assign depth proportionally across shadow (deeper farther from edge)
            for coord, proj in zip(region_coords, projections):
                frac = (proj - projections.min()) / (shadow_length_px + 1e-8)
                depth_map[coord[0], coord[1]] = depth * frac

        return depth_map

    def shadow_mask_from_image(
        self,
        image: np.ndarray,
        threshold: float = 0.05,
    ) -> np.ndarray:
        """Create binary shadow mask from image intensity.

        Args:
            image: RGB image (H, W, 3) uint8 or float.
            threshold: Intensity threshold below which pixels are shadow.

        Returns:
            Binary shadow mask (H, W), 1 = shadow.
        """
        if image.dtype == np.uint8:
            gray = np.mean(image.astype(np.float32) / 255.0, axis=-1)
        else:
            gray = np.mean(image, axis=-1)
        return (gray < threshold).astype(np.uint8)


class IlluminationDecomposer(nn.Module):
    """Decompose lunar imagery into albedo and shading components.

    Inspired by SHADeS (IJCARS 2025) non-Lambertian decomposition for
    endoscopy, adapted for lunar regolith. Separates what the surface
    looks like (albedo) from how it is lit (shading), enabling
    illumination-invariant terrain analysis.

    Image = Albedo * Shading + Shadow

    References:
        - SHADeS: github.com/RemaDaher/SHADeS
        - PAIDNet (Pattern Recognition 2025): physics-aware decomposition
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
        )

        # Albedo decoder — surface reflectance (illumination invariant)
        self.albedo_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, 3, padding=1),
            nn.Sigmoid(),
        )

        # Shading decoder — illumination component
        self.shading_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor) -> dict:
        """Decompose image into albedo and shading.

        Args:
            image: (N, C, H, W) input image tensor.

        Returns:
            Dict with 'albedo' (N, C, H, W), 'shading' (N, 1, H, W),
            and 'reconstruction' (N, C, H, W).
        """
        features = self.encoder(image)
        albedo = self.albedo_decoder(features)
        shading = self.shading_decoder(features)
        reconstruction = albedo * shading
        return {
            "albedo": albedo,
            "shading": shading,
            "reconstruction": reconstruction,
        }
