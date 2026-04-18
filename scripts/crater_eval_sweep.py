"""Threshold + config sweep for Stage 1 crater U-Net on real LOLA south pole DEM.

Runs a matrix of experiments to diagnose the sim-to-real gap:

  A. Threshold sweep (cheap) on an existing probability GeoTIFF (no re-inference).
  B. Re-inference variants:
      - global-stretch   : fixed elevation range -> [0, 1], same 80 m/px DEM
      - resampled-118    : resample DEM to DeepMoon's 118 m/px training resolution
      - both             : resample + global stretch
  C. Threshold sweep on every generated probability map.

All metrics computed against the Robbins-derived ground truth mask (ring mode).
Outputs a single sweep_results.json + a printed table.

Usage:
    # requires the existing probability map from the baseline run
    PYTHONPATH=src python scripts/crater_eval_sweep.py \\
        --checkpoint outputs/crater_v1_seed1/best_craterunet_seed1.pt \\
        --dem data/raw/lola/LDEM_80S_80MPP_ADJ.TIF \\
        --gt-mask data/processed/lola_80mpp_crater_mask.tif \\
        --baseline-prob outputs/crater_eval_lola/prob.tif \\
        --run-variants global,resampled,both
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject
import torch

from crater_eval_lola import (
    TILE, STRIDE,
    binary_metrics, build_crater_unet, classification_overlay,
    infer_tile, load_checkpoint, tile_positions,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------- Normalization variants ----------

def normalize_per_tile(tile: np.ndarray) -> np.ndarray:
    """1-99 percentile stretch per tile. The original baseline approach."""
    t = np.where(np.isnan(tile), 0.0, tile).astype(np.float32)
    lo, hi = np.percentile(t, [1, 99])
    if hi - lo < 1e-6:
        return np.zeros_like(t)
    return np.clip((t - lo) / (hi - lo), 0.0, 1.0)


def make_global_stretch(dem: np.ndarray) -> tuple[float, float]:
    """Fixed stretch bounds from a single pass over the full DEM (ignoring NaN)."""
    valid = dem[np.isfinite(dem)]
    lo, hi = np.percentile(valid, [1, 99])
    return float(lo), float(hi)


def normalize_global(tile: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Use precomputed global bounds -> [0, 1]."""
    t = np.where(np.isnan(tile), (lo + hi) / 2, tile).astype(np.float32)
    if hi - lo < 1e-6:
        return np.zeros_like(t)
    return np.clip((t - lo) / (hi - lo), 0.0, 1.0)


# ---------- Resampling ----------

def resample_dem_to_target(dem_path: Path, target_px_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Reproject DEM to `target_px_m` meters per pixel, same CRS. Returns (dem, valid)."""
    with rasterio.open(dem_path) as src:
        src_crs = src.crs
        src_transform = src.transform
        src_data = src.read(1).astype(np.float32)
        src_h, src_w = src.height, src.width
        # Compute new transform + shape keeping the same footprint
        scale = abs(src_transform.a) / target_px_m
        new_w = int(round(src_w * scale))
        new_h = int(round(src_h * scale))
        new_transform = rasterio.transform.Affine(
            target_px_m, src_transform.b, src_transform.c,
            src_transform.d, -target_px_m, src_transform.f,
        )
    dst = np.full((new_h, new_w), np.nan, dtype=np.float32)
    reproject(
        source=src_data, destination=dst,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=new_transform, dst_crs=src_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan, dst_nodata=np.nan,
    )
    valid = np.isfinite(dst)
    return dst, valid


def resample_gt_to_match(gt_path: Path, ref_dem_path: Path, target_px_m: float) -> np.ndarray:
    """Resample ground-truth mask to the resampled DEM's grid (nearest neighbor)."""
    with rasterio.open(gt_path) as src:
        src_crs = src.crs
        src_transform = src.transform
        src_data = src.read(1).astype(np.uint8)
    with rasterio.open(ref_dem_path) as ref:
        ref_h, ref_w = ref.height, ref.width
        ref_transform = ref.transform
        scale = abs(ref_transform.a) / target_px_m
        new_w = int(round(ref_w * scale))
        new_h = int(round(ref_h * scale))
        new_transform = rasterio.transform.Affine(
            target_px_m, ref_transform.b, ref_transform.c,
            ref_transform.d, -target_px_m, ref_transform.f,
        )
    dst = np.zeros((new_h, new_w), dtype=np.uint8)
    reproject(
        source=src_data, destination=dst,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=new_transform, dst_crs=src_crs,
        resampling=Resampling.nearest,
    )
    return dst


# ---------- Sliding inference with configurable normalizer ----------

def sliding_inference(
    dem: np.ndarray, valid_mask: np.ndarray,
    model: torch.nn.Module, device: torch.device,
    normalizer,
    tta: bool = True,
) -> np.ndarray:
    H, W = dem.shape
    prob_sum = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    positions = tile_positions(H, W, TILE, STRIDE)
    t0 = time.time()
    for i, (y, x) in enumerate(positions):
        tile = dem[y:y + TILE, x:x + TILE]
        vtile = valid_mask[y:y + TILE, x:x + TILE]
        if vtile.sum() == 0:
            continue
        tile01 = normalizer(tile)
        p = infer_tile(model, tile01, device, tta=tta)
        prob_sum[y:y + TILE, x:x + TILE] += p
        count[y:y + TILE, x:x + TILE] += 1.0
        if (i + 1) % 500 == 0:
            print(f"    tile {i+1}/{len(positions)}  ({time.time()-t0:.0f}s)")
    return np.where(count > 0, prob_sum / np.maximum(count, 1e-6), 0.0)


# ---------- Sweep ----------

def threshold_sweep(prob: np.ndarray, gt: np.ndarray, valid: np.ndarray,
                    thresholds=(0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.075, 0.05)) -> list[dict]:
    rows = []
    for t in thresholds:
        pred = (prob >= t).astype(np.uint8)
        pred[~valid] = 0
        m = binary_metrics(pred, gt, valid)
        rows.append({"threshold": t, **m})
    return rows


def print_table(label: str, rows: list[dict]) -> None:
    print(f"\n=== {label} ===")
    print(f"{'thr':>5} {'IoU':>7} {'Dice':>7} {'prec':>7} {'rec':>7} {'pred%':>7} {'gt%':>6}")
    for r in rows:
        print(f"{r['threshold']:>5.3f} {r['iou']:>7.4f} {r['dice']:>7.4f} "
              f"{r['precision']:>7.4f} {r['recall']:>7.4f} "
              f"{r['pred_coverage']*100:>6.2f} {r['gt_coverage']*100:>5.2f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dem", type=Path, default=REPO_ROOT / "data/raw/lola/LDEM_80S_80MPP_ADJ.TIF")
    p.add_argument("--gt-mask", type=Path, default=REPO_ROOT / "data/processed/lola_80mpp_crater_mask.tif")
    p.add_argument("--baseline-prob", type=Path, default=REPO_ROOT / "outputs/crater_eval_lola/prob.tif")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs/crater_eval_sweep")
    p.add_argument("--target-px-m", type=float, default=118.0, help="Resampling target to match DeepMoon")
    p.add_argument("--run-variants", default="global,resampled,both",
                   help="Comma list: global,resampled,both")
    p.add_argument("--tta", action="store_true", default=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load DEM + GT once.
    with rasterio.open(args.dem) as ds:
        dem = ds.read(1)
    with rasterio.open(args.gt_mask) as ds:
        gt = ds.read(1).astype(np.uint8)
    valid = np.isfinite(dem)
    print(f"DEM: {dem.shape}  valid {valid.sum()/valid.size*100:.1f}%  GT coverage {gt.sum()/gt.size*100:.2f}%")

    results: dict[str, list[dict]] = {}

    # A. Threshold sweep on baseline prob (if provided).
    if args.baseline_prob and args.baseline_prob.exists():
        with rasterio.open(args.baseline_prob) as ds:
            prob_base = ds.read(1)
        rows = threshold_sweep(prob_base, gt, valid)
        print_table("A. Baseline (80 m/px, per-tile norm) -- threshold sweep", rows)
        results["baseline_80mpp_pertile"] = rows

    # Load model for re-inference variants.
    need_inference = any(v in args.run_variants for v in ["global", "resampled", "both"])
    if need_inference:
        model = build_crater_unet()
        meta = load_checkpoint(model, args.checkpoint, device)
        print(f"\nCheckpoint: epoch {meta.get('epoch', '?')}  val_iou {meta.get('val_iou', '?')}")

    # Variant B1: global stretch at 80 m/px.
    if "global" in args.run_variants:
        print("\n>>> Variant: global stretch, 80 m/px")
        lo, hi = make_global_stretch(dem)
        print(f"    global stretch bounds: [{lo:.0f}, {hi:.0f}] m")
        norm = lambda t, lo=lo, hi=hi: normalize_global(t, lo, hi)
        prob = sliding_inference(dem, valid, model, device, norm, tta=args.tta)
        rasterio.open(args.out_dir / "prob_global_80mpp.tif", "w",
                      driver="GTiff", height=dem.shape[0], width=dem.shape[1], count=1, dtype="float32",
                      compress="lzw").write(prob.astype(np.float32), 1)
        rows = threshold_sweep(prob, gt, valid)
        print_table("B1. Global stretch @ 80 m/px", rows)
        results["global_80mpp"] = rows

    # Variant B2: resample to 118 m/px, per-tile norm.
    if "resampled" in args.run_variants or "both" in args.run_variants:
        print(f"\n>>> Resampling DEM to {args.target_px_m} m/px ...")
        t0 = time.time()
        dem_rs, valid_rs = resample_dem_to_target(args.dem, args.target_px_m)
        gt_rs = resample_gt_to_match(args.gt_mask, args.dem, args.target_px_m)
        print(f"    {dem_rs.shape} at {args.target_px_m} m/px  ({time.time()-t0:.1f}s)")

        if "resampled" in args.run_variants:
            print(">>> Variant: resampled 118 m/px, per-tile norm")
            prob = sliding_inference(dem_rs, valid_rs, model, device, normalize_per_tile, tta=args.tta)
            rows = threshold_sweep(prob, gt_rs, valid_rs)
            print_table(f"B2. Resampled @ {args.target_px_m} m/px + per-tile norm", rows)
            results[f"resampled_{int(args.target_px_m)}mpp_pertile"] = rows

        if "both" in args.run_variants:
            print(">>> Variant: resampled 118 m/px + global stretch")
            lo, hi = make_global_stretch(dem_rs)
            print(f"    global stretch bounds: [{lo:.0f}, {hi:.0f}] m")
            norm = lambda t, lo=lo, hi=hi: normalize_global(t, lo, hi)
            prob = sliding_inference(dem_rs, valid_rs, model, device, norm, tta=args.tta)
            rasterio.open(args.out_dir / "prob_both_118mpp.tif", "w",
                          driver="GTiff", height=dem_rs.shape[0], width=dem_rs.shape[1], count=1, dtype="float32",
                          compress="lzw").write(prob.astype(np.float32), 1)
            rows = threshold_sweep(prob, gt_rs, valid_rs)
            print_table(f"B3. Resampled @ {args.target_px_m} m/px + global stretch", rows)
            results[f"both_{int(args.target_px_m)}mpp"] = rows

    # Save sweep results
    out_json = args.out_dir / "sweep_results.json"
    out_json.write_text(json.dumps({
        "dem": str(args.dem),
        "gt_mask": str(args.gt_mask),
        "checkpoint": str(args.checkpoint),
        "target_px_m": args.target_px_m,
        "tta": args.tta,
        "results": results,
    }, indent=2))
    print(f"\nSweep results: {out_json}")


if __name__ == "__main__":
    main()
