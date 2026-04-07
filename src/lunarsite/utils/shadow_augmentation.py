"""Lunar-specific augmentations targeting the sim-to-real domain gap.

Standard augmentations (flip, rotate, brightness) miss the biggest source of
domain shift in lunar imagery: extreme shadow variation. The Moon has no
atmosphere, creating razor-sharp shadows that change dramatically with
sun angle. These augmentations simulate that variation.

References:
    - Chrono lunar sim (2024): Hapke photometric functions for regolith
    - OmniLRS: physically accurate lunar rendering
    - HORUS: demonstrated PSR imaging needs specialized processing
"""

import numpy as np
import cv2
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform


class LunarShadowRotation(ImageOnlyTransform):
    """Simulate varying sun angles by rotating shadow direction.

    At the lunar south pole, sun elevation is always <1.5 degrees.
    Small changes in sun azimuth drastically change shadow patterns.
    This augmentation rotates the shadow-casting direction.
    """

    def __init__(self, angle_range: tuple[float, float] = (-30, 30), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.angle_range = angle_range

    def apply(self, img: np.ndarray, angle: float = 0, **params) -> np.ndarray:
        # Detect shadow regions (very dark pixels)
        gray = np.mean(img.astype(np.float32), axis=-1)
        shadow_mask = gray < 25  # near-black

        if shadow_mask.sum() < 100:
            return img  # No significant shadows to rotate

        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Rotate shadow mask and blend
        rotated_shadow = cv2.warpAffine(
            shadow_mask.astype(np.uint8), M, (w, h),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        ).astype(bool)

        result = img.copy()
        # Darken pixels where rotated shadow falls
        result[rotated_shadow] = np.clip(result[rotated_shadow] * 0.1, 0, 255).astype(np.uint8)
        # Lighten pixels where original shadow was but rotated shadow isn't
        revealed = shadow_mask & ~rotated_shadow
        result[revealed] = np.clip(result[revealed] * 3.0 + 30, 0, 200).astype(np.uint8)

        return result

    def get_params(self) -> dict:
        angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
        return {"angle": angle}

    def get_transform_init_args_names(self) -> tuple:
        return ("angle_range",)


class ExtremeContrastAugmentation(ImageOnlyTransform):
    """Simulate the extreme dynamic range of airless body imagery.

    Lunar images have no atmospheric scattering to fill shadows.
    Sunlit regions can be 10,000x brighter than shadowed regions.
    This augmentation pushes contrast far beyond standard ranges.
    """

    def __init__(
        self,
        shadow_darken_range: tuple[float, float] = (0.01, 0.15),
        sunlit_brighten_range: tuple[float, float] = (1.2, 2.5),
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.shadow_darken_range = shadow_darken_range
        self.sunlit_brighten_range = sunlit_brighten_range

    def apply(
        self,
        img: np.ndarray,
        shadow_factor: float = 0.05,
        sunlit_factor: float = 1.5,
        **params,
    ) -> np.ndarray:
        gray = np.mean(img.astype(np.float32), axis=-1)
        median = np.median(gray)

        result = img.astype(np.float32)

        # Darken shadows more
        shadow_mask = gray < median * 0.5
        result[shadow_mask] *= shadow_factor

        # Brighten sunlit areas
        sunlit_mask = gray > median * 1.5
        result[sunlit_mask] *= sunlit_factor

        return np.clip(result, 0, 255).astype(np.uint8)

    def get_params(self) -> dict:
        return {
            "shadow_factor": np.random.uniform(*self.shadow_darken_range),
            "sunlit_factor": np.random.uniform(*self.sunlit_brighten_range),
        }

    def get_transform_init_args_names(self) -> tuple:
        return ("shadow_darken_range", "sunlit_brighten_range")


class HapkeBRDFPerturbation(ImageOnlyTransform):
    """Simulate varying regolith reflectance properties.

    Lunar regolith follows the Hapke BRDF model with a strong opposition
    effect (retroreflection at zero phase angle). Different compositions
    (highland anorthosite vs mare basalt) have different albedos and
    phase function parameters.

    This augmentation varies the apparent albedo and phase function
    to simulate different regolith compositions and viewing geometries.
    """

    def __init__(
        self,
        albedo_range: tuple[float, float] = (0.5, 1.5),
        phase_darkening_range: tuple[float, float] = (0.7, 1.3),
        always_apply=False,
        p=0.4,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.albedo_range = albedo_range
        self.phase_darkening_range = phase_darkening_range

    def apply(
        self,
        img: np.ndarray,
        albedo_factor: float = 1.0,
        phase_factor: float = 1.0,
        **params,
    ) -> np.ndarray:
        result = img.astype(np.float32)

        # Albedo scaling — overall brightness variation
        result *= albedo_factor

        # Phase function — darken/brighten based on pixel brightness
        # Simulates different viewing angles relative to sun
        gray = np.mean(result, axis=-1, keepdims=True)
        normalized = gray / (gray.max() + 1e-8)
        # Higher phase angle = more darkening of moderate-brightness areas
        phase_mask = normalized * (1 - normalized) * 4  # peaks at 0.5
        result *= (1 - phase_mask * (1 - phase_factor))

        return np.clip(result, 0, 255).astype(np.uint8)

    def get_params(self) -> dict:
        return {
            "albedo_factor": np.random.uniform(*self.albedo_range),
            "phase_factor": np.random.uniform(*self.phase_darkening_range),
        }

    def get_transform_init_args_names(self) -> tuple:
        return ("albedo_range", "phase_darkening_range")


class SyntheticCraterOverlay(ImageOnlyTransform):
    """Add synthetic crater shadows to terrain images.

    Procedurally generates circular crater-like shadow patterns
    to increase crater detection robustness.
    """

    def __init__(
        self,
        num_craters_range: tuple[int, int] = (0, 3),
        radius_range: tuple[int, int] = (10, 60),
        always_apply=False,
        p=0.3,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.num_craters_range = num_craters_range
        self.radius_range = radius_range

    def apply(self, img: np.ndarray, craters: list = None, **params) -> np.ndarray:
        if not craters:
            return img

        result = img.copy()
        h, w = img.shape[:2]

        for cx, cy, r, depth in craters:
            # Create gradient shadow inside crater
            y, x = np.ogrid[-cy:h - cy, -cx:w - cx]
            dist = np.sqrt(x * x + y * y).astype(np.float32)
            crater_mask = dist < r

            # Rim is bright, interior darkens toward center
            brightness = np.clip(dist / r, 0, 1)
            brightness = brightness ** depth  # depth controls shadow intensity

            for c in range(3):
                channel = result[:, :, c].astype(np.float32)
                channel[crater_mask] *= brightness[crater_mask]
                result[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

        return result

    def get_params(self) -> dict:
        n = np.random.randint(self.num_craters_range[0], self.num_craters_range[1] + 1)
        craters = []
        for _ in range(n):
            # Will be placed randomly in apply based on image size
            cx = np.random.randint(50, 430)
            cy = np.random.randint(50, 430)
            r = np.random.randint(self.radius_range[0], self.radius_range[1])
            depth = np.random.uniform(0.3, 2.0)
            craters.append((cx, cy, r, depth))
        return {"craters": craters}

    def get_transform_init_args_names(self) -> tuple:
        return ("num_craters_range", "radius_range")


def get_lunar_augmentations(input_size: int, training: bool) -> A.Compose:
    """Build the full lunar-aware augmentation pipeline.

    Combines standard spatial augmentations with lunar-specific
    shadow, contrast, and BRDF augmentations.

    Args:
        input_size: Target crop size (e.g., 480).
        training: Whether to apply training augmentations.

    Returns:
        Albumentations Compose pipeline.
    """
    if training:
        return A.Compose([
            # Spatial
            A.RandomCrop(height=input_size, width=input_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # Standard photometric
            A.RandomBrightnessContrast(
                brightness_limit=(-0.3, 0.2),
                contrast_limit=(-0.2, 0.4),
                p=0.4,
            ),
            A.GaussNoise(p=0.2),
            # Lunar-specific
            LunarShadowRotation(angle_range=(-45, 45), p=0.3),
            ExtremeContrastAugmentation(p=0.3),
            HapkeBRDFPerturbation(p=0.3),
            SyntheticCraterOverlay(p=0.2),
        ])
    else:
        return A.Compose([
            A.CenterCrop(height=input_size, width=input_size),
        ])
