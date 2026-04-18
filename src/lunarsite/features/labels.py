"""Rule-based pseudo-labels for Stage 3 landing site scoring.

Because the south pole has no flown ground-truth landing-suitability labels,
we synthesize labels from physical constraints from the NASA CASSA study:

  - slope_max_deg          : mean slope per cell must be at or below this
  - illumination_min_pct   : mean annual direct solar illumination at or above
  - earth_visibility_min_pct : cell must see Earth this fraction of the time

A cell is "suitable" (label = 1) iff all three thresholds are met. Missing
features (NaN) default to "fail" — conservative for a safety-driven task.

The XGBoost scorer is then trained to predict this rule with access to
additional features (crater density, roughness, rock abundance, etc.). The
added signal is what makes the final model useful — it learns to score
cells on features the rules don't encode directly.

Reference thresholds from configs/stage3_scoring.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class CassaThresholds:
    slope_max_deg: float = 5.0
    illumination_min_pct: float = 33.0
    earth_visibility_min_pct: float = 50.0


def cassa_label(row: pd.Series, t: CassaThresholds) -> int:
    """Return 1 (suitable) if all CASSA thresholds pass, else 0."""
    slope = row.get("slope_mean", float("nan"))
    illum = row.get("avg_illumination_pct", float("nan"))
    earth = row.get("earth_visibility_pct", float("nan"))

    import math
    if any(math.isnan(v) for v in (slope, illum, earth)):
        return 0

    if slope > t.slope_max_deg:
        return 0
    if illum < t.illumination_min_pct:
        return 0
    if earth < t.earth_visibility_min_pct:
        return 0
    return 1


def apply_labels(df: pd.DataFrame, thresholds: CassaThresholds | None = None) -> pd.DataFrame:
    """Add a `suitable` column to the feature dataframe using CASSA rules."""
    if thresholds is None:
        thresholds = CassaThresholds()
    df = df.copy()
    df["suitable"] = df.apply(lambda r: cassa_label(r, thresholds), axis=1)
    return df
