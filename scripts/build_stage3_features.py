"""Build the Stage 3 feature matrix over the lunar south pole.

For each grid cell in the polar stereographic DEM:
  - LOLA features       (elevation / slope / roughness / error: mean/std/min/max)
  - Stage 1 features    (crater_count, crater_mean_diameter_m, nearest_crater_dist_m)
  - Illumination        (annual mean sunlit fraction) -- optional, NaN if missing
  - Earth visibility    (fraction of time Earth above horizon) -- optional, NaN if missing

Outputs a parquet + CSV feature matrix keyed on cell_id with lat/lon columns.
The CASSA pseudo-label is applied separately (labels.py) so this script stays
pure: features only, no label logic.

Usage:
    python scripts/build_stage3_features.py \\
        --dem D:/lola_pgda/product90/LDEM_80S_20MPP_ADJ.TIF \\
        --slope D:/lola_pgda/product90/LDSM_80S_20MPP_ADJ.TIF \\
        --crater-mask outputs/crater_eval_lola/pred.tif \\
        --cell-size-m 100 \\
        --out data/processed/stage3_features.parquet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from scipy import ndimage

from lunarsite.features.grid import iter_grid
from lunarsite.features.lola_features import LolaFeatureExtractor

REPO_ROOT = Path(__file__).resolve().parent.parent


def _crater_features_for_cell(
    crater_mask_tile: np.ndarray,
    pixel_size_m: float,
) -> dict[str, float]:
    """Per-cell crater features from a binary crater mask tile."""
    if crater_mask_tile.size == 0 or crater_mask_tile.sum() == 0:
        return {
            "crater_count": 0,
            "crater_coverage_pct": 0.0,
            "crater_mean_diameter_m": 0.0,
            "crater_max_diameter_m": 0.0,
        }
    labeled, n = ndimage.label(crater_mask_tile.astype(np.uint8))
    if n == 0:
        return {
            "crater_count": 0,
            "crater_coverage_pct": 0.0,
            "crater_mean_diameter_m": 0.0,
            "crater_max_diameter_m": 0.0,
        }
    sizes = ndimage.sum(crater_mask_tile, labeled, range(1, n + 1))
    diameters = 2.0 * np.sqrt(np.asarray(sizes) / np.pi) * pixel_size_m
    return {
        "crater_count": int(n),
        "crater_coverage_pct": float(crater_mask_tile.sum()) / crater_mask_tile.size * 100,
        "crater_mean_diameter_m": float(diameters.mean()),
        "crater_max_diameter_m": float(diameters.max()),
    }


def _sample_at_cell(
    raster: np.ndarray | None,
    row_slice: tuple[int, int],
    col_slice: tuple[int, int],
    name: str,
    scale: float = 1.0,
) -> dict[str, float]:
    """Sample the mean of `raster` inside the cell, optionally multiplied by `scale`."""
    if raster is None:
        return {name: float("nan")}
    r0, r1 = row_slice
    c0, c1 = col_slice
    patch = raster[r0:r1, c0:c1].astype(np.float32)
    valid = patch[np.isfinite(patch)]
    return {name: float(valid.mean() * scale) if valid.size else float("nan")}


# PGDA Product 69 AVGVISIB rasters are int16 with values up to ~25,500 for
# 100% visibility; divide by 255 and multiply by 100 to get percent.
AVGVISIB_TO_PCT_SCALE = 100.0 / 25500.0


def build_features(
    dem_path: Path,
    slope_path: Path | None,
    roughness_path: Path | None,
    error_path: Path | None,
    crater_mask_path: Path | None,
    illumination_path: Path | None,
    earth_vis_path: Path | None,
    cell_size_m: float,
    max_lat_deg: float = -80.0,
) -> pd.DataFrame:
    extractor = LolaFeatureExtractor(
        elevation_path=dem_path,
        slope_path=slope_path,
        roughness_path=roughness_path,
        error_path=error_path,
    )

    with rasterio.open(dem_path) as ds:
        pixel_size_m = abs(ds.transform.a)

    crater_mask = None
    if crater_mask_path and crater_mask_path.exists():
        with rasterio.open(crater_mask_path) as ds:
            crater_mask = ds.read(1).astype(np.uint8)

    # Load DEM once as the reference grid (shape + CRS + transform).
    with rasterio.open(dem_path) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_shape = (ref.height, ref.width)

    def _warp_to_dem(path: Path | None) -> np.ndarray | None:
        """Reproject a raster to the DEM's grid. NaN where the source is out of bounds."""
        if not path or not path.exists():
            return None
        with rasterio.open(path) as src:
            src_data = src.read(1).astype(np.float32)
            src_transform = src.transform
            src_crs = src.crs
            src_nodata = src.nodata
        dst = np.full(ref_shape, np.nan, dtype=np.float32)
        reproject(
            source=src_data,
            destination=dst,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            src_nodata=src_nodata,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        return dst

    illumination = _warp_to_dem(illumination_path)
    earth_vis = _warp_to_dem(earth_vis_path)

    rows = []
    t0 = time.time()
    for i, cell in enumerate(iter_grid(dem_path, cell_size_m, max_lat_deg)):
        feats = {
            "cell_id": cell.cell_id,
            "lon": cell.center_lonlat[0],
            "lat": cell.center_lonlat[1],
            "x_m": cell.center_xy_m[0],
            "y_m": cell.center_xy_m[1],
        }
        feats.update(extractor.features(cell))

        if crater_mask is not None:
            r0, r1 = cell.row_slice
            c0, c1 = cell.col_slice
            feats.update(_crater_features_for_cell(crater_mask[r0:r1, c0:c1], pixel_size_m))
        else:
            feats.update({
                "crater_count": 0,
                "crater_coverage_pct": 0.0,
                "crater_mean_diameter_m": 0.0,
                "crater_max_diameter_m": 0.0,
            })

        feats.update(_sample_at_cell(
            illumination, cell.row_slice, cell.col_slice,
            "avg_illumination_pct", scale=AVGVISIB_TO_PCT_SCALE,
        ))
        feats.update(_sample_at_cell(
            earth_vis, cell.row_slice, cell.col_slice,
            "earth_visibility_pct", scale=AVGVISIB_TO_PCT_SCALE,
        ))
        rows.append(feats)

        if (i + 1) % 50000 == 0:
            print(f"  cells {i+1:,}  ({(time.time() - t0):.0f}s elapsed)")

    df = pd.DataFrame(rows)
    print(f"Built feature matrix: {len(df):,} cells, {len(df.columns)} features")
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dem", type=Path, required=True)
    p.add_argument("--slope", type=Path, default=None)
    p.add_argument("--roughness", type=Path, default=None)
    p.add_argument("--error", type=Path, default=None)
    p.add_argument("--crater-mask", type=Path, default=None)
    p.add_argument("--illumination", type=Path, default=None)
    p.add_argument("--earth-vis", type=Path, default=None)
    p.add_argument("--cell-size-m", type=float, default=100.0)
    p.add_argument("--max-lat-deg", type=float, default=-80.0)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data/processed/stage3_features.parquet")
    args = p.parse_args()

    df = build_features(
        dem_path=args.dem,
        slope_path=args.slope,
        roughness_path=args.roughness,
        error_path=args.error,
        crater_mask_path=args.crater_mask,
        illumination_path=args.illumination,
        earth_vis_path=args.earth_vis,
        cell_size_m=args.cell_size_m,
        max_lat_deg=args.max_lat_deg,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".parquet":
        df.to_parquet(args.out, index=False)
    else:
        df.to_csv(args.out, index=False)

    csv_peek = args.out.with_suffix(".preview.csv")
    df.head(200).to_csv(csv_peek, index=False)

    print(f"Feature matrix: {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")
    print(f"Preview (first 200 rows): {csv_peek}")
    null_pct = df.isna().mean().sort_values(ascending=False).head(10)
    print("\nTop null% columns:")
    print(null_pct.to_string())


if __name__ == "__main__":
    main()
