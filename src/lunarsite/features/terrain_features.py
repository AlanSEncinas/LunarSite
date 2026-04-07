"""Feature engineering for Stage 3 site scoring.

Extracts per-grid-cell features from Stage 1/2 outputs, LOLA GeoTIFFs,
and dark terrain analysis (ShadowCam, Diviner thermal, Mini-RF SAR).

Feature set (22 features):
    LOLA (4):        elevation, slope, roughness, elevation_error
    Stage 1 (3):     crater_density, crater_min_dist, avg_crater_radius
    Stage 2 (3):     rock_coverage_pct, large_rock_count, shadow_coverage_pct
    Illumination (2): avg_illumination_pct, max_continuous_shadow_days
    Other (2):       earth_visibility_pct, distance_to_psr_km
    NEW Dark (6):    psr_fraction, depth_from_shadow, thermal_inertia,
                     rock_abundance_thermal, sar_backscatter, sar_cpd
    NEW Uncertainty (2): segmentation_confidence, depth_uncertainty
"""

import numpy as np
from scipy import ndimage


def compute_rock_coverage(segmentation_mask: np.ndarray) -> dict:
    """Compute rock-related features from a terrain segmentation mask.

    Args:
        segmentation_mask: H x W array with class indices.
            Classes: 0=background, 1=small_rocks, 2=large_rocks, 3=sky

    Returns:
        Dict with rock coverage and count features.
    """
    total_terrain = np.sum(segmentation_mask != 3)  # exclude sky pixels
    if total_terrain == 0:
        return {
            "rock_coverage_pct": 0.0,
            "large_rock_count": 0,
            "small_rock_coverage_pct": 0.0,
            "shadow_coverage_pct": 0.0,
        }

    small_rocks = np.sum(segmentation_mask == 1)
    large_rocks = np.sum(segmentation_mask == 2)

    # Count distinct large rock blobs
    large_rock_mask = (segmentation_mask == 2).astype(np.uint8)
    labeled, n_large = ndimage.label(large_rock_mask)

    return {
        "rock_coverage_pct": float((small_rocks + large_rocks) / total_terrain * 100),
        "large_rock_count": int(n_large),
        "small_rock_coverage_pct": float(small_rocks / total_terrain * 100),
        "shadow_coverage_pct": 0.0,  # computed separately from intensity
    }


def compute_crater_features(crater_mask: np.ndarray, pixel_scale_m: float) -> dict:
    """Compute crater-related features from a binary crater mask.

    Args:
        crater_mask: H x W binary array (1 = crater).
        pixel_scale_m: Meters per pixel.

    Returns:
        Dict with crater density, minimum distance, and average radius.
    """
    labeled, n_craters = ndimage.label(crater_mask.astype(np.uint8))
    h, w = crater_mask.shape
    area_km2 = (h * pixel_scale_m * w * pixel_scale_m) / 1e6

    if n_craters == 0:
        return {
            "crater_density": 0.0,
            "crater_min_dist": float("inf"),
            "avg_crater_radius": 0.0,
        }

    # Compute centroids and radii
    centroids = ndimage.center_of_mass(crater_mask, labeled, range(1, n_craters + 1))
    centroids = np.array(centroids)

    radii = []
    for i in range(1, n_craters + 1):
        area_px = np.sum(labeled == i)
        radius_px = np.sqrt(area_px / np.pi)
        radii.append(radius_px * pixel_scale_m)

    # Minimum distance between crater centroids
    if n_craters >= 2:
        from scipy.spatial.distance import pdist
        dists = pdist(centroids * pixel_scale_m)
        min_dist = float(dists.min())
    else:
        min_dist = float("inf")

    return {
        "crater_density": float(n_craters / area_km2),
        "crater_min_dist": min_dist,
        "avg_crater_radius": float(np.mean(radii)),
    }


def compute_shadow_features(
    image: np.ndarray,
    shadow_threshold: float = 0.05,
) -> dict:
    """Compute shadow-related features from image intensity.

    Shadow analysis provides information about terrain relief and
    permanently shadowed regions that may harbor water ice.

    Args:
        image: Grayscale or RGB image, float [0, 1] or uint8.
        shadow_threshold: Intensity below which pixels are shadow.

    Returns:
        Dict with shadow coverage and PSR fraction estimates.
    """
    if image.dtype == np.uint8:
        gray = np.mean(image.astype(np.float32) / 255.0, axis=-1) if image.ndim == 3 else image / 255.0
    else:
        gray = np.mean(image, axis=-1) if image.ndim == 3 else image

    shadow_mask = gray < shadow_threshold
    total_pixels = gray.size

    # Shadow coverage
    shadow_pct = float(shadow_mask.sum() / total_pixels * 100)

    # Connected shadow regions — large ones may be PSRs
    labeled, n_regions = ndimage.label(shadow_mask)
    region_sizes = ndimage.sum(shadow_mask, labeled, range(1, n_regions + 1))

    # PSR candidates: shadow regions > 5% of image area
    psr_threshold = total_pixels * 0.05
    psr_regions = [s for s in region_sizes if s > psr_threshold]
    psr_fraction = float(sum(psr_regions) / total_pixels) if psr_regions else 0.0

    return {
        "shadow_coverage_pct": shadow_pct,
        "psr_fraction": psr_fraction,
        "n_shadow_regions": int(n_regions),
    }


def compute_depth_from_shadow(
    shadow_mask: np.ndarray,
    sun_elevation_rad: float,
    pixel_scale_m: float,
) -> dict:
    """Estimate terrain depth features from shadow geometry.

    At the lunar south pole, shadow_length * tan(sun_elevation) = depth.
    This provides depth information without any active sensing.

    Args:
        shadow_mask: Binary (H, W) where 1 = shadow.
        sun_elevation_rad: Sun elevation in radians (very small at south pole).
        pixel_scale_m: Ground sample distance in meters/pixel.

    Returns:
        Dict with shadow-derived depth features.
    """
    if sun_elevation_rad <= 0 or shadow_mask.sum() == 0:
        return {
            "depth_from_shadow_mean": 0.0,
            "depth_from_shadow_max": 0.0,
            "shadow_length_mean_m": 0.0,
        }

    labeled, n = ndimage.label(shadow_mask.astype(np.uint8))
    depths = []
    lengths = []

    for i in range(1, n + 1):
        region = labeled == i
        coords = np.argwhere(region)
        if len(coords) < 2:
            continue

        # Shadow length = extent along sun direction (approximate as max extent)
        extent_y = coords[:, 0].max() - coords[:, 0].min()
        extent_x = coords[:, 1].max() - coords[:, 1].min()
        shadow_length_px = max(extent_y, extent_x)
        shadow_length_m = shadow_length_px * pixel_scale_m

        depth = shadow_length_m * np.tan(sun_elevation_rad)
        depths.append(depth)
        lengths.append(shadow_length_m)

    if not depths:
        return {
            "depth_from_shadow_mean": 0.0,
            "depth_from_shadow_max": 0.0,
            "shadow_length_mean_m": 0.0,
        }

    return {
        "depth_from_shadow_mean": float(np.mean(depths)),
        "depth_from_shadow_max": float(np.max(depths)),
        "shadow_length_mean_m": float(np.mean(lengths)),
    }


def compute_thermal_features(thermal_data: np.ndarray) -> dict:
    """Compute terrain features from Diviner thermal observations.

    Diviner measures surface temperature independent of illumination,
    providing terrain characterization in permanently shadowed regions.

    Args:
        thermal_data: Dict or array with thermal inertia and rock abundance.
            If ndarray, assumed to be thermal inertia map.

    Returns:
        Dict with thermal terrain features.
    """
    if isinstance(thermal_data, dict):
        ti = thermal_data.get("thermal_inertia", np.array([0]))
        ra = thermal_data.get("rock_abundance", np.array([0]))
    else:
        ti = thermal_data
        ra = np.array([0])

    ti_valid = ti[ti > 0] if np.any(ti > 0) else np.array([0])

    return {
        "thermal_inertia_mean": float(np.mean(ti_valid)),
        "thermal_inertia_std": float(np.std(ti_valid)),
        "rock_abundance_thermal": float(np.mean(ra[ra >= 0])) if np.any(ra >= 0) else 0.0,
    }


def compute_sar_features(sar_data: np.ndarray, cpd_data: np.ndarray = None) -> dict:
    """Compute terrain features from Mini-RF SAR backscatter.

    SAR provides terrain roughness information independent of illumination,
    crucial for PSR hazard assessment. Circular polarization ratio (CPR/CPD)
    indicates potential water ice deposits.

    Args:
        sar_data: SAR backscatter image (H, W) in dB or linear.
        cpd_data: Circular polarization degree map (H, W), optional.

    Returns:
        Dict with SAR-derived terrain features.
    """
    valid = sar_data[np.isfinite(sar_data)]
    if len(valid) == 0:
        return {
            "sar_backscatter_mean": 0.0,
            "sar_backscatter_std": 0.0,
            "sar_roughness_indicator": 0.0,
            "sar_cpd": 0.0,
        }

    return {
        "sar_backscatter_mean": float(np.mean(valid)),
        "sar_backscatter_std": float(np.std(valid)),
        "sar_roughness_indicator": float(np.percentile(valid, 90) - np.percentile(valid, 10)),
        "sar_cpd": float(np.mean(cpd_data[np.isfinite(cpd_data)])) if cpd_data is not None else 0.0,
    }


def build_feature_vector(
    rock_features: dict,
    crater_features: dict,
    shadow_features: dict,
    depth_features: dict = None,
    thermal_features: dict = None,
    sar_features: dict = None,
    lola_features: dict = None,
    illumination_features: dict = None,
    uncertainty_features: dict = None,
) -> dict:
    """Combine all feature sources into a single feature vector for Stage 3.

    Args:
        All feature dicts from compute_* functions above.

    Returns:
        Combined dict with all features, ready for XGBoost.
    """
    features = {}
    features.update(rock_features)
    features.update(crater_features)
    features.update(shadow_features)

    if depth_features:
        features.update(depth_features)
    if thermal_features:
        features.update(thermal_features)
    if sar_features:
        features.update(sar_features)
    if lola_features:
        features.update(lola_features)
    if illumination_features:
        features.update(illumination_features)
    if uncertainty_features:
        features.update(uncertainty_features)

    return features
