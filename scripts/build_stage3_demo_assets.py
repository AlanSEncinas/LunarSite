"""Generate Streamlit demo assets for Stage 3 + Stage 1.

Produces:
  demo_assets/stage3/top_sites_map.png   - south pole DEM + top-500 cells +
                                           Artemis-III region pins
  demo_assets/stage3/shap_summary.png    - copy of XGBoost SHAP summary plot
  demo_assets/stage3/top10.csv           - human-readable top 10 cells
  demo_assets/stage3/artemis_overlap.json- top-N vs Artemis regions table
  demo_assets/stage1/crater_overlay.png  - south pole DEM + v2 crater mask
                                           (downsampled for fast web load)

Updates demo_assets/manifest.json with stage3 + stage1 blocks.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import pyproj

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = REPO_ROOT / "demo_assets"
LUNAR_LONLAT = pyproj.CRS("+proj=longlat +a=1737400 +b=1737400 +no_defs")


def hillshade(arr: np.ndarray) -> np.ndarray:
    a = np.where(np.isnan(arr), 0, arr).astype(np.float32)
    gy, gx = np.gradient(a)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az = np.deg2rad(360 - 315 + 90)
    alt = np.deg2rad(45)
    shade = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return (np.clip(shade, 0, 1) * 255).astype(np.uint8)


def build_top_sites_map(dem_path: Path, ranked_parquet: Path, artemis_json: Path, out_png: Path) -> None:
    with rasterio.open(dem_path) as ds:
        dem = ds.read(1)
        transform = ds.transform
        dem_crs = ds.crs
        bounds = ds.bounds

    hs = hillshade(dem)
    ranked = pd.read_parquet(ranked_parquet)
    top500 = ranked.head(500)
    regions = json.loads(artemis_json.read_text())["regions"]

    to_dem = pyproj.Transformer.from_crs(LUNAR_LONLAT, dem_crs, always_xy=True)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="#0e0e14")
    ax.set_facecolor("#0e0e14")
    ax.imshow(hs, cmap="gray",
              extent=[bounds.left / 1000, bounds.right / 1000,
                      bounds.bottom / 1000, bounds.top / 1000])
    ax.scatter(top500.x_m / 1000, top500.y_m / 1000,
               c=top500.score, cmap="plasma", s=20, alpha=0.7,
               edgecolors="none", label="Top 500 LunarSite cells")

    # Artemis region pins
    for r in regions:
        rx, ry = to_dem.transform(r["lon_deg"], r["lat_deg"])
        ax.scatter(rx / 1000, ry / 1000, marker="*", s=240,
                   facecolor="#ffdf66", edgecolor="black", linewidth=1.2, zorder=5)
        ax.annotate(r["name"], (rx / 1000, ry / 1000),
                    xytext=(6, 6), textcoords="offset points",
                    color="#fff6a6", fontsize=8, zorder=6)

    ax.set_xlabel("Polar stereographic x (km)", color="#cfd3dc")
    ax.set_ylabel("Polar stereographic y (km)", color="#cfd3dc")
    ax.set_title("LunarSite top-500 scored cells + Artemis III candidates",
                 color="#e8eaf0", pad=12)
    ax.tick_params(colors="#cfd3dc")
    for spine in ax.spines.values():
        spine.set_color("#2a2f3b")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def artemis_overlap_table(ranked_parquet: Path, artemis_json: Path) -> dict:
    ranked = pd.read_parquet(ranked_parquet)
    regions = json.loads(artemis_json.read_text())["regions"]
    result = {"top_n_levels": []}
    for top_n in (100, 500, 1000):
        top = ranked.head(top_n)
        rows = []
        for r in regions:
            rlat, rlon = r["lat_deg"], r["lon_deg"]
            cos_lat = np.cos(np.deg2rad(rlat))
            dlat = top.lat - rlat
            dlon = (top.lon - rlon + 180) % 360 - 180
            dkm = np.sqrt((dlat * 30.3) ** 2 + (dlon * cos_lat * 30.3) ** 2)
            rows.append({
                "region": r["name"],
                "cells_within_15km": int((dkm <= 15).sum()),
                "closest_km": round(float(dkm.min()), 1),
            })
        result["top_n_levels"].append({
            "top_n": top_n,
            "regions_matched": sum(1 for row in rows if row["cells_within_15km"] > 0),
            "regions": rows,
        })
    return result


def build_crater_overlay_demo(dem_path: Path, mask_path: Path, out_png: Path, max_side: int = 1600) -> None:
    with rasterio.open(dem_path) as ds:
        dem = ds.read(1)
    with rasterio.open(mask_path) as ds:
        mask = ds.read(1)

    scale = min(1.0, max_side / max(dem.shape))
    if scale < 1.0:
        dsize = (int(dem.shape[1] * scale), int(dem.shape[0] * scale))
        dem = cv2.resize(dem, dsize, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, dsize, interpolation=cv2.INTER_NEAREST)

    hs = hillshade(dem)
    rgb = cv2.cvtColor(hs, cv2.COLOR_GRAY2RGB)
    over = rgb.copy()
    over[mask > 0] = [80, 200, 255]
    blended = cv2.addWeighted(rgb, 0.55, over, 0.45, 0)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))


def main() -> None:
    dem = REPO_ROOT / "data/raw/lola/LDEM_80S_80MPP_ADJ.TIF"
    ranked = REPO_ROOT / "outputs/stage3_v1/ranked_cells.parquet"
    shap_png = REPO_ROOT / "outputs/stage3_v1/shap_summary.png"
    artemis_json = REPO_ROOT / "data/processed/artemis_regions.json"
    crater_mask = REPO_ROOT / "outputs/crater_eval_lola_v2/pred.tif"

    stage3 = DEMO_DIR / "stage3"
    stage1 = DEMO_DIR / "stage1"
    stage3.mkdir(parents=True, exist_ok=True)
    stage1.mkdir(parents=True, exist_ok=True)

    print("Building top sites map...")
    build_top_sites_map(dem, ranked, artemis_json, stage3 / "top_sites_map.png")
    print("  ->", stage3 / "top_sites_map.png")

    print("Copying SHAP summary...")
    if shap_png.exists():
        shutil.copy(shap_png, stage3 / "shap_summary.png")
        print("  ->", stage3 / "shap_summary.png")

    print("Writing top 10 CSV + Artemis overlap JSON...")
    feats = pd.read_parquet(REPO_ROOT / "data/processed/stage3_features_80mpp_1km.parquet")
    rk = pd.read_parquet(ranked)
    top10 = rk.head(10).merge(feats, on="cell_id", suffixes=("", "_f"))
    top10[["lat", "lon", "score", "elevation_mean", "slope_mean",
           "avg_illumination_pct", "earth_visibility_pct"]].round(3).to_csv(
        stage3 / "top10.csv", index=False)
    (stage3 / "artemis_overlap.json").write_text(
        json.dumps(artemis_overlap_table(ranked, artemis_json), indent=2))

    print("Building Stage 1 crater overlay demo...")
    build_crater_overlay_demo(dem, crater_mask, stage1 / "crater_overlay.png")
    print("  ->", stage1 / "crater_overlay.png")

    # Update manifest
    manifest_path = DEMO_DIR / "manifest.json"
    m = json.loads(manifest_path.read_text())
    m["stage3"] = {
        "top_sites_map": "stage3/top_sites_map.png",
        "shap_summary": "stage3/shap_summary.png",
        "top10_csv": "stage3/top10.csv",
        "artemis_overlap_json": "stage3/artemis_overlap.json",
        "n_cells": int(len(rk)),
        "n_suitable": int(rk["suitable_cassa"].sum()),
        "cassa_thresholds": {"slope_max_deg": 5.0, "illumination_min_pct": 33.0, "earth_visibility_min_pct": 50.0},
    }
    m["stage1"] = {
        "crater_overlay": "stage1/crater_overlay.png",
        "source": "v2 fine-tuned (best_craterunet_v2_southpole_seed1.pt)",
        "metrics": {
            "iou": 0.162, "dice": 0.279, "recall": 0.372, "precision": 0.224,
            "threshold": 0.25, "tta": "flip",
        },
        "v1_vs_v2": {
            "v1_iou": 0.111, "v1_recall": 0.155,
            "v2_iou": 0.162, "v2_recall": 0.372,
        },
    }
    manifest_path.write_text(json.dumps(m, indent=2))
    print("Manifest updated.")


if __name__ == "__main__":
    main()
