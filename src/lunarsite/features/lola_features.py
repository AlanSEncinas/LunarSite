"""Per-grid-cell feature extraction from LOLA GeoTIFFs.

Reads PGDA product 90 rasters (elevation, slope, roughness, error) and
extracts per-cell statistics (mean/std/min/max) aligned to the Stage 3
grid defined in grid.py.

LOLA product conventions:
  - LDEM_*.TIF   : float32 elevation in meters
  - LDSM_*.TIF   : float32 slope in degrees (0-90)
  - LDRM_*.TIF   : float32 roughness (baseline-dependent, see PGDA docs)
  - LDEM_*_ERR.TIF : elevation stddev estimate per pixel

Not all products are available at every resolution. Missing inputs are
handled gracefully with NaN-filled feature values so the feature matrix
stays well-shaped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio

from .grid import GridCell


def _cell_stats(arr: np.ndarray, prefix: str) -> dict[str, float]:
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mean": float(valid.mean()),
        f"{prefix}_std": float(valid.std()),
        f"{prefix}_min": float(valid.min()),
        f"{prefix}_max": float(valid.max()),
    }


class LolaFeatureExtractor:
    """Reads LOLA rasters once, extracts per-cell features on demand.

    All rasters must share the same CRS, transform, and shape (true for PGDA
    products at matching resolution/latitude). The elevation raster is
    required; slope/roughness/error are optional.
    """

    def __init__(
        self,
        elevation_path: Path,
        slope_path: Optional[Path] = None,
        roughness_path: Optional[Path] = None,
        error_path: Optional[Path] = None,
    ) -> None:
        self.elev = self._read(elevation_path)
        self.slope = self._read(slope_path) if slope_path else None
        self.rough = self._read(roughness_path) if roughness_path else None
        self.err = self._read(error_path) if error_path else None

        shapes = {"elev": self.elev.shape}
        for name, arr in [("slope", self.slope), ("rough", self.rough), ("err", self.err)]:
            if arr is not None:
                shapes[name] = arr.shape
        if len({s for s in shapes.values()}) > 1:
            raise ValueError(f"LOLA raster shapes disagree: {shapes}")

    @staticmethod
    def _read(path: Path) -> np.ndarray:
        with rasterio.open(path) as ds:
            return ds.read(1)

    def features(self, cell: GridCell) -> dict[str, float]:
        r0, r1 = cell.row_slice
        c0, c1 = cell.col_slice
        out: dict[str, float] = {}
        out.update(_cell_stats(self.elev[r0:r1, c0:c1], "elevation"))
        if self.slope is not None:
            out.update(_cell_stats(self.slope[r0:r1, c0:c1], "slope"))
        else:
            out.update({f"slope_{k}": float("nan") for k in ("mean", "std", "min", "max")})
        if self.rough is not None:
            out.update(_cell_stats(self.rough[r0:r1, c0:c1], "roughness"))
        else:
            out.update({f"roughness_{k}": float("nan") for k in ("mean", "std", "min", "max")})
        if self.err is not None:
            out.update(_cell_stats(self.err[r0:r1, c0:c1], "elevation_error"))
        else:
            out.update({f"elevation_error_{k}": float("nan") for k in ("mean", "std", "min", "max")})
        return out
