"""Full LunarSite end-to-end pipeline over the lunar south pole.

Runs the three production stages in sequence:

  Stage 1  crater_eval_lola.py
           Slide a fine-tuned U-Net (best_craterunet_v2_southpole_seed1.pt)
           over the LOLA DEM, emit a binary crater prediction GeoTIFF.

  Stage 3  build_stage3_features.py + train_scorer.py
           Reproject LOLA slope + PGDA illumination + PGDA Earth visibility
           onto the DEM grid, tile into cells, extract per-cell features,
           apply CASSA rule-based pseudo-labels, train XGBoost + SHAP,
           rank every cell by landing-site suitability.

  Artifacts  Top-ranked cells map + Artemis III overlap + SHAP summary,
             ready for the Streamlit demo.

Stage 2 is optional. It runs per-image on optical photography, not on the
DEM, so it does not feed the south pole scorer directly. This orchestrator
skips Stage 2 by default; pass --run-stage2 to exercise the zero-shot
transfer pipeline on the shipped real moon / south pole orbital images.

Intended as a one-command reproduction path for the Layer 2 ship.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --skip-stage1 --skip-stage3-train
    python scripts/run_pipeline.py --run-stage2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# Default paths (override via CLI flags).
DEFAULTS = {
    "stage1_ckpt":     REPO_ROOT / "outputs/crater_v2_finetune_seed1/best_craterunet_v2_southpole_seed1.pt",
    "dem":             REPO_ROOT / "data/raw/lola/LDEM_80S_80MPP_ADJ.TIF",
    "gt_mask":         REPO_ROOT / "data/processed/lola_80mpp_crater_mask.tif",
    "crater_pred":     REPO_ROOT / "outputs/crater_eval_lola_v2/pred.tif",
    "slope":           Path("D:/lola_pgda/product90/LDSM_80S_20MPP_ADJ.TIF"),
    "illumination":    Path("D:/lola_pgda/product69/AVGVISIB_85S_060M_201608.TIF"),
    "earth_vis":       Path("D:/lola_pgda/product69/AVGVISIB_85S_060M_201608_EARTH.TIF"),
    "features":        REPO_ROOT / "data/processed/stage3_features_80mpp_1km.parquet",
    "stage3_outdir":   REPO_ROOT / "outputs/stage3_v1",
    "demo_dir":        REPO_ROOT / "demo_assets",
}


def _run(label: str, argv: list[str], env: dict | None = None) -> None:
    print(f"\n{'='*72}\n  {label}\n{'='*72}")
    print("  $ " + " ".join(argv))
    t0 = time.time()
    env = {**os.environ, **(env or {})}
    result = subprocess.run(argv, cwd=str(REPO_ROOT), env=env)
    dt = time.time() - t0
    if result.returncode != 0:
        print(f"  FAILED ({dt:.1f}s, exit {result.returncode})")
        sys.exit(result.returncode)
    print(f"  done ({dt:.1f}s)")


def stage1(args) -> None:
    if args.crater_pred.exists() and not args.force_stage1:
        print(f"Stage 1: using existing crater mask {args.crater_pred}  (--force-stage1 to regenerate)")
        return
    _run(
        "Stage 1: crater detection on LOLA south pole DEM",
        [sys.executable, "scripts/crater_eval_lola.py",
         "--checkpoint", str(args.stage1_ckpt),
         "--dem",        str(args.dem),
         "--gt-mask",    str(args.gt_mask),
         "--out-dir",    str(args.crater_pred.parent),
         "--threshold",  "0.25", "--tta"],
        env={"PYTHONPATH": "src"},
    )


def stage3_features(args) -> None:
    if args.features.exists() and not args.force_stage3:
        print(f"Stage 3 features: using existing {args.features}  (--force-stage3 to rebuild)")
        return
    cmd = [sys.executable, "scripts/build_stage3_features.py",
           "--dem",          str(args.dem),
           "--crater-mask",  str(args.crater_pred),
           "--cell-size-m",  str(args.cell_size_m),
           "--out",          str(args.features)]
    if args.slope.exists():        cmd += ["--slope",        str(args.slope)]
    if args.illumination.exists(): cmd += ["--illumination", str(args.illumination)]
    if args.earth_vis.exists():    cmd += ["--earth-vis",    str(args.earth_vis)]
    _run("Stage 3a: build feature matrix", cmd, env={"PYTHONPATH": "src"})


def stage3_train(args) -> None:
    if args.skip_stage3_train:
        print("Stage 3 train skipped (--skip-stage3-train).")
        return
    _run(
        "Stage 3b: XGBoost + SHAP site scorer",
        [sys.executable, "scripts/train_scorer.py",
         "--features", str(args.features),
         "--out-dir",  str(args.stage3_outdir)],
        env={"PYTHONPATH": "src"},
    )


def build_demo_assets(args) -> None:
    if args.skip_demo:
        print("Demo asset build skipped (--skip-demo).")
        return
    _run(
        "Demo assets: top-site map + crater overlay + SHAP for Streamlit",
        [sys.executable, "scripts/build_stage3_demo_assets.py"],
    )


def stage2_optional(args) -> None:
    if not args.run_stage2:
        return
    _run(
        "Stage 2 (optional): terrain segmenter on real south pole orbital imagery",
        [sys.executable, "scripts/segmenter_on_south_pole.py", "--tta"],
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dem", type=Path, default=DEFAULTS["dem"])
    p.add_argument("--gt-mask", type=Path, default=DEFAULTS["gt_mask"])
    p.add_argument("--stage1-ckpt", type=Path, default=DEFAULTS["stage1_ckpt"])
    p.add_argument("--crater-pred", type=Path, default=DEFAULTS["crater_pred"])
    p.add_argument("--slope", type=Path, default=DEFAULTS["slope"])
    p.add_argument("--illumination", type=Path, default=DEFAULTS["illumination"])
    p.add_argument("--earth-vis", type=Path, default=DEFAULTS["earth_vis"])
    p.add_argument("--features", type=Path, default=DEFAULTS["features"])
    p.add_argument("--stage3-outdir", type=Path, default=DEFAULTS["stage3_outdir"])
    p.add_argument("--cell-size-m", type=float, default=1000.0)

    p.add_argument("--skip-stage1", action="store_true")
    p.add_argument("--skip-stage3-features", action="store_true")
    p.add_argument("--skip-stage3-train", action="store_true")
    p.add_argument("--skip-demo", action="store_true")
    p.add_argument("--force-stage1", action="store_true")
    p.add_argument("--force-stage3", action="store_true")
    p.add_argument("--run-stage2", action="store_true", help="Run the optional Stage 2 south pole orbital pass.")
    args = p.parse_args()

    print("LunarSite pipeline — Layer 2 end-to-end")
    print(f"  DEM:           {args.dem}")
    print(f"  Stage 1 ckpt:  {args.stage1_ckpt}")
    print(f"  Cell size:     {args.cell_size_m:.0f} m")

    t_start = time.time()
    stage2_optional(args)
    if not args.skip_stage1:       stage1(args)
    if not args.skip_stage3_features: stage3_features(args)
    stage3_train(args)
    build_demo_assets(args)

    print(f"\nPipeline finished in {(time.time()-t_start)/60:.1f} min.")
    print(f"  Crater mask:    {args.crater_pred}")
    print(f"  Feature matrix: {args.features}")
    print(f"  Scorer output:  {args.stage3_outdir}/")
    print(f"  Demo assets:    {DEFAULTS['demo_dir']}/stage1, stage3")


if __name__ == "__main__":
    main()
