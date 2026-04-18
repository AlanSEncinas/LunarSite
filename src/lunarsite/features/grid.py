"""Grid cell generation for Stage 3 lunar south pole landing site scoring.

Creates a regular grid in the polar stereographic projection of the LOLA DEM
(matching PGDA product 90). Each cell is a square patch (default 100 m x 100 m
= 1250 m x 1250 m at 12 pixels per side when the DEM is 80 m/px, or 50 x 50 pixels
at 20 m/px). We return per-cell metadata that downstream feature extractors
attach to.

Why polar stereographic native: resampling to lat/lon would distort cell areas
near the pole. Keeping the grid in the DEM's native CRS means every cell is
equal-area and feature extraction is a simple `data[row_slice, col_slice]` read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pyproj
import rasterio
from rasterio.transform import rowcol

LUNAR_LONLAT = pyproj.CRS("+proj=longlat +a=1737400 +b=1737400 +no_defs")


@dataclass
class GridCell:
    cell_id: int
    row_slice: tuple[int, int]   # (row_start, row_stop) — use as slice(*row_slice)
    col_slice: tuple[int, int]   # (col_start, col_stop)
    center_xy_m: tuple[float, float]   # polar stereographic x, y in meters
    center_lonlat: tuple[float, float]  # lon, lat in degrees


def iter_grid(
    dem_path: Path,
    cell_size_m: float = 100.0,
    max_lat_deg: float = -80.0,
) -> Iterator[GridCell]:
    """Yield GridCells over the valid polar area of the given DEM.

    Cells are integer-pixel-aligned squares sized `cell_size_m` meters on
    each side. Cells outside the lunar disk (DEM NaN) or whose centroid
    is north of `max_lat_deg` are skipped.
    """
    with rasterio.open(dem_path) as ds:
        transform = ds.transform
        H, W = ds.height, ds.width
        dem = ds.read(1)
        dem_crs = ds.crs

    pixel_size_m = abs(transform.a)
    cell_pixels = max(1, int(round(cell_size_m / pixel_size_m)))
    to_lonlat = pyproj.Transformer.from_crs(dem_crs, LUNAR_LONLAT, always_xy=True)

    cell_id = 0
    for r0 in range(0, H - cell_pixels + 1, cell_pixels):
        for c0 in range(0, W - cell_pixels + 1, cell_pixels):
            patch = dem[r0:r0 + cell_pixels, c0:c0 + cell_pixels]
            if np.isnan(patch).all():
                continue
            # Center of cell in polar stereographic coords
            cx = transform.c + (c0 + cell_pixels / 2) * transform.a
            cy = transform.f + (r0 + cell_pixels / 2) * transform.e
            lon, lat = to_lonlat.transform(cx, cy)
            if lat > max_lat_deg:  # north of the south polar band, skip
                continue
            yield GridCell(
                cell_id=cell_id,
                row_slice=(r0, r0 + cell_pixels),
                col_slice=(c0, c0 + cell_pixels),
                center_xy_m=(cx, cy),
                center_lonlat=(lon, lat),
            )
            cell_id += 1


def grid_stats(dem_path: Path, cell_size_m: float = 100.0, max_lat_deg: float = -80.0) -> dict:
    """Quick counts without materializing all cells (for sanity checks)."""
    cells = list(iter_grid(dem_path, cell_size_m, max_lat_deg))
    lats = [c.center_lonlat[1] for c in cells]
    return {
        "n_cells": len(cells),
        "cell_size_m": cell_size_m,
        "max_lat_deg": max_lat_deg,
        "lat_range": (min(lats), max(lats)) if lats else (None, None),
    }
