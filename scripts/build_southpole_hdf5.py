"""Build a DeepMoon-schema HDF5 fine-tuning dataset from the real LOLA south pole DEM.

For Stage 1 v2 (fine-tune on domain):
  - Resample LOLA DEM to 118 m/px (matches DeepMoon training resolution).
  - Tile into non-overlapping 256 x 256 crops.
  - For each tile, build a crater-ring mask from the Robbins 2018 catalog
    projected to the DEM grid, filtered to `--min-diameter-km` (default 3 km
    because the v1 model cannot reliably detect sub-3 km craters at this
    resolution / training distribution).
  - Per-tile 1-99 percentile stretch to uint8 (matches the DeepMoon input
    convention: 8-bit grayscale DEM).
  - Skip tiles with no crater pixels OR all-NaN DEM.
  - Pack everything into a single HDF5 with the DeepMoon schema:
      input_images   (N, 256, 256) uint8
      target_masks   (N, 256, 256) float32 in [0, 1]

Usage:
    python scripts/build_southpole_hdf5.py \\
        --dem D:/lola_pgda/product90/LDEM_80S_20MPP_ADJ.TIF \\
        --craters data/raw/robbins_south_pole.csv \\
        --out data/processed/southpole_finetune_118mpp.hdf5 \\
        --min-diameter-km 3.0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.transform import Affine, rowcol
from rasterio.warp import Resampling, reproject

REPO_ROOT = Path(__file__).resolve().parent.parent
LUNAR_LONLAT = pyproj.CRS("+proj=longlat +a=1737400 +b=1737400 +no_defs")

TILE = 256
RING_THICKNESS_PX = 2


def resample_dem(dem_path: Path, target_px_m: float) -> tuple[np.ndarray, Affine, pyproj.CRS]:
    with rasterio.open(dem_path) as src:
        src_crs = src.crs
        src_t = src.transform
        data = src.read(1).astype(np.float32)
        H, W = src.height, src.width
    scale = abs(src_t.a) / target_px_m
    new_w = int(round(W * scale))
    new_h = int(round(H * scale))
    new_t = Affine(target_px_m, src_t.b, src_t.c, src_t.d, -target_px_m, src_t.f)
    dst = np.full((new_h, new_w), np.nan, dtype=np.float32)
    reproject(
        source=data, destination=dst,
        src_transform=src_t, src_crs=src_crs,
        dst_transform=new_t, dst_crs=src_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan, dst_nodata=np.nan,
    )
    return dst, new_t, src_crs


def rasterize_crater_rings(craters_csv: Path, transform: Affine, shape: tuple[int, int],
                           dem_crs: pyproj.CRS, min_diameter_km: float,
                           ring_thickness_px: int) -> np.ndarray:
    H, W = shape
    pixel_m = abs(transform.a)
    to_dem = pyproj.Transformer.from_crs(LUNAR_LONLAT, dem_crs, always_xy=True)

    df = pd.read_csv(craters_csv)
    df = df[df.DIAM_CIRC_IMG >= min_diameter_km]
    xs, ys = to_dem.transform(df.LON_CIRC_IMG.values, df.LAT_CIRC_IMG.values)
    rows, cols = rowcol(transform, xs, ys)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    radii_px = np.maximum((df.DIAM_CIRC_IMG.values * 1000.0 / pixel_m / 2.0).astype(np.int32), 1)

    mask = np.zeros((H, W), dtype=np.uint8)
    drawn = 0
    for r, c, rad in zip(rows, cols, radii_px):
        if 0 <= r < H and 0 <= c < W:
            cv2.circle(mask, (int(c), int(r)), int(rad), 1, thickness=max(1, ring_thickness_px))
            drawn += 1
    return mask, int(drawn), int(len(df))


def normalize_tile_uint8(tile: np.ndarray) -> np.ndarray:
    """Per-tile 1-99 percentile stretch to uint8. NaN -> midpoint gray."""
    t = np.where(np.isnan(tile), 0.0, tile).astype(np.float32)
    lo, hi = np.percentile(t, [1, 99])
    if hi - lo < 1e-6:
        return np.full(tile.shape, 128, dtype=np.uint8)
    scaled = np.clip((t - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dem", type=Path, required=True)
    p.add_argument("--craters", type=Path, default=REPO_ROOT / "data/raw/robbins_south_pole.csv")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data/processed/southpole_finetune_118mpp.hdf5")
    p.add_argument("--target-px-m", type=float, default=118.0)
    p.add_argument("--min-diameter-km", type=float, default=3.0)
    p.add_argument("--ring-thickness-px", type=int, default=RING_THICKNESS_PX)
    p.add_argument("--tile", type=int, default=TILE)
    p.add_argument("--stride", type=int, default=TILE, help="non-overlapping by default")
    p.add_argument("--min-crater-frac", type=float, default=0.002,
                   help="skip tiles with less than this fraction of crater pixels")
    p.add_argument("--max-nan-frac", type=float, default=0.10,
                   help="skip tiles with more than this fraction of NaN DEM pixels")
    args = p.parse_args()

    t0 = time.time()
    print(f"Resampling DEM to {args.target_px_m} m/px ...")
    dem, transform, crs = resample_dem(args.dem, args.target_px_m)
    print(f"  {dem.shape} @ {args.target_px_m} m/px  ({time.time()-t0:.1f}s)")

    print(f"Rasterizing Robbins craters >= {args.min_diameter_km} km ...")
    ring_mask, drawn, total = rasterize_crater_rings(
        args.craters, transform, dem.shape, crs,
        args.min_diameter_km, args.ring_thickness_px,
    )
    print(f"  drew {drawn:,}/{total:,} craters; mask coverage {ring_mask.mean()*100:.2f}%")

    H, W = dem.shape
    tiles_imgs: list[np.ndarray] = []
    tiles_masks: list[np.ndarray] = []
    skipped_nan = 0
    skipped_empty = 0

    for y0 in range(0, H - args.tile + 1, args.stride):
        for x0 in range(0, W - args.tile + 1, args.stride):
            dem_t = dem[y0:y0+args.tile, x0:x0+args.tile]
            mask_t = ring_mask[y0:y0+args.tile, x0:x0+args.tile]
            nan_frac = np.isnan(dem_t).mean()
            if nan_frac > args.max_nan_frac:
                skipped_nan += 1
                continue
            if mask_t.mean() < args.min_crater_frac:
                skipped_empty += 1
                continue
            tiles_imgs.append(normalize_tile_uint8(dem_t))
            tiles_masks.append(mask_t.astype(np.float32))

    n = len(tiles_imgs)
    print(f"\nKept tiles: {n:,}  |  skipped NaN: {skipped_nan:,}  |  skipped empty: {skipped_empty:,}")
    if n == 0:
        raise RuntimeError("No tiles kept -- check thresholds")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {args.out} ...")
    with h5py.File(args.out, "w") as f:
        f.create_dataset("input_images", data=np.stack(tiles_imgs, axis=0), compression="gzip", compression_opts=4)
        f.create_dataset("target_masks", data=np.stack(tiles_masks, axis=0), compression="gzip", compression_opts=4)
        f.attrs["source"] = str(args.dem)
        f.attrs["min_diameter_km"] = args.min_diameter_km
        f.attrs["ring_thickness_px"] = args.ring_thickness_px
        f.attrs["target_px_m"] = args.target_px_m
        f.attrs["tile_size"] = args.tile
        f.attrs["stride"] = args.stride
        f.attrs["n_tiles"] = n
        f.attrs["crater_coverage_pct"] = float(ring_mask.mean() * 100)
    size_mb = args.out.stat().st_size / 1e6
    print(f"Done. {n:,} tiles, {size_mb:.1f} MB  ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
