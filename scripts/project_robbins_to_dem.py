"""Project the Robbins 2018 south pole crater catalog onto a LOLA DEM grid.

Takes the Robbins south pole subset (lat/lon/diameter_km per crater) and
rasterizes filled crater disks aligned to the LOLA DEM's pixel grid. This
gives us real ground-truth masks to evaluate the DeepMoon-trained Stage 1
U-Net against the lunar south pole — the exact surface Stage 3 will score.

Outputs:
  - {out_mask}             GeoTIFF crater mask aligned to DEM (uint8, 0/1)
  - {out_preview}          PNG quick-look: DEM hillshade + mask overlay

Usage:
    python scripts/project_robbins_to_dem.py \
        --dem data/raw/lola/LDEM_80S_80MPP_ADJ.TIF \
        --craters data/raw/robbins_south_pole.csv \
        --out-mask data/processed/lola_80mpp_crater_mask.tif \
        --out-preview data/processed/lola_80mpp_crater_preview.png
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.transform import rowcol

REPO_ROOT = Path(__file__).resolve().parent.parent

LUNAR_LONLAT = pyproj.CRS("+proj=longlat +a=1737400 +b=1737400 +no_defs")


def rasterize_craters(
    dem_path: Path,
    craters_csv: Path,
    out_mask: Path,
    min_diameter_km: float = 1.0,
    mode: str = "ring",
    ring_thickness_px: int = 2,
) -> dict:
    """Rasterize crater catalog to a mask aligned to the DEM grid.

    mode='ring' draws the rim outline only (matches DeepMoon target_masks).
    mode='disk' draws filled disks (useful for IoU over the crater interior).
    """
    with rasterio.open(dem_path) as ds:
        dem_crs = ds.crs
        transform = ds.transform
        H, W = ds.height, ds.width
        pixel_size_m = abs(transform.a)

    to_dem = pyproj.Transformer.from_crs(LUNAR_LONLAT, dem_crs, always_xy=True)

    df = pd.read_csv(craters_csv)
    need = {"LAT_CIRC_IMG", "LON_CIRC_IMG", "DIAM_CIRC_IMG"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"Missing Robbins columns: {missing}")

    df = df[df.DIAM_CIRC_IMG >= min_diameter_km].copy()

    xs, ys = to_dem.transform(df.LON_CIRC_IMG.values, df.LAT_CIRC_IMG.values)
    rows, cols = rowcol(transform, xs, ys)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)

    radii_px = np.maximum((df.DIAM_CIRC_IMG.values * 1000.0 / pixel_size_m / 2.0).astype(np.int32), 1)

    mask = np.zeros((H, W), dtype=np.uint8)
    drawn = 0
    fill_thickness = -1 if mode == "disk" else max(1, int(ring_thickness_px))
    for r, c, rad in zip(rows, cols, radii_px):
        if 0 <= r < H and 0 <= c < W:
            cv2.circle(mask, (int(c), int(r)), int(rad), 1, thickness=fill_thickness)
            drawn += 1

    out_mask.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_mask, "w",
        driver="GTiff", height=H, width=W, count=1, dtype="uint8",
        crs=dem_crs, transform=transform, compress="lzw", nodata=0,
    ) as dst:
        dst.write(mask, 1)

    coverage = float(mask.sum()) / mask.size
    return {
        "width": W, "height": H, "pixel_size_m": pixel_size_m,
        "mode": mode,
        "ring_thickness_px": ring_thickness_px if mode == "ring" else None,
        "craters_considered": int(len(df)),
        "craters_drawn_in_bounds": drawn,
        "crater_pixel_coverage": coverage,
        "min_diameter_km": min_diameter_km,
    }


def hillshade(arr: np.ndarray, azimuth_deg: float = 315, altitude_deg: float = 45) -> np.ndarray:
    """Simple hillshade of a float DEM array for preview. NaN -> 0."""
    a = np.where(np.isnan(arr), 0, arr).astype(np.float32)
    gy, gx = np.gradient(a)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az = np.deg2rad(360 - azimuth_deg + 90)
    alt = np.deg2rad(altitude_deg)
    shade = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    shade = np.clip(shade, 0, 1)
    return (shade * 255).astype(np.uint8)


def build_preview(dem_path: Path, mask_path: Path, out_png: Path, max_side: int = 2000) -> None:
    with rasterio.open(dem_path) as ds:
        dem = ds.read(1)
    with rasterio.open(mask_path) as ds:
        mask = ds.read(1)

    scale = min(1.0, max_side / max(dem.shape))
    if scale < 1.0:
        dsize = (int(dem.shape[1] * scale), int(dem.shape[0] * scale))
        dem_s = cv2.resize(dem, dsize, interpolation=cv2.INTER_AREA)
        mask_s = cv2.resize(mask, dsize, interpolation=cv2.INTER_NEAREST)
    else:
        dem_s, mask_s = dem, mask

    hs = hillshade(dem_s)
    rgb = cv2.cvtColor(hs, cv2.COLOR_GRAY2RGB)
    # Red translucent mask overlay
    overlay = rgb.copy()
    overlay[mask_s > 0] = [255, 60, 60]
    blended = cv2.addWeighted(rgb, 0.65, overlay, 0.35, 0)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dem", type=Path, default=REPO_ROOT / "data/raw/lola/LDEM_80S_80MPP_ADJ.TIF")
    p.add_argument("--craters", type=Path, default=REPO_ROOT / "data/raw/robbins_south_pole.csv")
    p.add_argument("--out-mask", type=Path, default=REPO_ROOT / "data/processed/lola_80mpp_crater_mask.tif")
    p.add_argument("--out-preview", type=Path, default=REPO_ROOT / "data/processed/lola_80mpp_crater_preview.png")
    p.add_argument("--min-diameter-km", type=float, default=1.0)
    p.add_argument("--mode", choices=["ring", "disk"], default="ring",
                   help="'ring' matches DeepMoon target_masks; 'disk' for interior IoU.")
    p.add_argument("--ring-thickness-px", type=int, default=2)
    args = p.parse_args()

    t0 = time.time()
    stats = rasterize_craters(
        args.dem, args.craters, args.out_mask,
        min_diameter_km=args.min_diameter_km,
        mode=args.mode,
        ring_thickness_px=args.ring_thickness_px,
    )
    t_raster = time.time() - t0

    t0 = time.time()
    build_preview(args.dem, args.out_mask, args.out_preview)
    t_prev = time.time() - t0

    print(f"DEM: {args.dem}")
    print(f"  size: {stats['width']} x {stats['height']} @ {stats['pixel_size_m']:.1f} m/px")
    print(f"Craters rasterized: {stats['craters_drawn_in_bounds']:,} of {stats['craters_considered']:,} (>= {stats['min_diameter_km']} km)")
    print(f"  pixel coverage: {stats['crater_pixel_coverage'] * 100:.2f}%")
    print(f"  mask:    {args.out_mask}")
    print(f"  preview: {args.out_preview}")
    print(f"Done in {t_raster:.1f}s (raster) + {t_prev:.1f}s (preview)")


if __name__ == "__main__":
    main()
