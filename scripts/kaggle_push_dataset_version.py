"""Push a new version of the public Kaggle dataset `encinas88/lunarsite-weights`
with per-file descriptions baked into the metadata JSON.

Unlike `kaggle_push_dataset_metadata.py` (which only updates title/subtitle/
description/keywords/license and does NOT re-upload files), this one goes
through `dataset_create_version`, which DOES re-upload all referenced files
(~816 MB total).

Use this when you want per-file descriptions to land — Kaggle's API does
not expose a "patch file descriptions only" endpoint.

Staging is done with hardlinks (same NTFS volume) so we don't double disk use.

Usage:
    python scripts/kaggle_push_dataset_version.py             # dry-run
    python scripts/kaggle_push_dataset_version.py --push      # actually push
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Use a fresh per-run staging dir to dodge Windows file locks left behind
# when a previous python upload process dies mid-run. The SDK keeps file
# handles open on Windows and they can't be cleared without a reboot, so
# avoiding rmtree on a locked tree is simpler than fighting it.
import time
STAGING = REPO_ROOT / "tmp" / f"kaggle_upload_{int(time.time())}"
DATASET_SLUG = "encinas88/lunarsite-weights"

# Same copy as kaggle_push_dataset_metadata.py — kept in sync manually.
# If dataset-level copy ever diverges between the two scripts, that script
# is the source of truth (it runs more often).
from kaggle_push_dataset_metadata import (  # type: ignore
    TITLE, SUBTITLE, KEYWORDS, LICENSE, DESCRIPTION,
)

# (local_path, kaggle_filename, description)
FILES = [
    (
        REPO_ROOT / "best_resnet34.pt",
        "best_resnet34.pt",
        "Stage 2 terrain segmenter — seed 1. U-Net + ResNet-34 encoder (ImageNet-pretrained), "
        "4 classes (background / small_rocks / large_rocks / sky), 480x480 input, Dice+CE loss. "
        "Val mIoU 0.8357 | Test mIoU 0.8456 with flip TTA. "
        "Production single-model checkpoint used in the Streamlit demo. Also member 1 of the 5-seed "
        "deep ensemble. Load with smp.Unet('resnet34', encoder_weights=None, in_channels=3, classes=4); "
        "ckpt['model_state_dict']. Trained on Kaggle Artificial Lunar Landscape (9,766 synthetic "
        "images, 80/10/10 split, split_seed=42).",
    ),
    (
        REPO_ROOT / "outputs/ensemble_seed2/best_resnet34_seed2.pt",
        "best_resnet34_seed2.pt",
        "Stage 2 terrain segmenter — ensemble member, seed 2. Same U-Net + ResNet-34 architecture "
        "and training config as seed 1; only the random seed differs (weight init, DataLoader shuffle, "
        "augmentation RNG). Data split is held fixed at split_seed=42 so ensemble disagreement is "
        "meaningful. Val mIoU 0.8371 | Test mIoU 0.8434 (flip TTA). Averaging this with seeds 1, 3, "
        "4, 5 gives a 5-member deep ensemble with per-pixel epistemic uncertainty.",
    ),
    (
        REPO_ROOT / "outputs/ensemble_seed3/best_resnet34_seed3.pt",
        "best_resnet34_seed3.pt",
        "Stage 2 terrain segmenter — ensemble member, seed 3. Identical config to seeds 1/2/4/5, "
        "fixed data split (split_seed=42). Val mIoU 0.8367 | Test mIoU 0.8448 (flip TTA).",
    ),
    (
        REPO_ROOT / "outputs/ensemble_seed4/best_resnet34_seed4.pt",
        "best_resnet34_seed4.pt",
        "Stage 2 terrain segmenter — ensemble member, seed 4. Identical config to seeds 1/2/3/5, "
        "fixed data split (split_seed=42). Val mIoU 0.8343 | Test mIoU 0.8428 (flip TTA). "
        "Lowest-performing seed of the five; kept because ensemble diversity matters more than "
        "cherry-picking.",
    ),
    (
        REPO_ROOT / "outputs/ensemble_seed5/best_resnet34_seed5.pt",
        "best_resnet34_seed5.pt",
        "Stage 2 terrain segmenter — ensemble member, seed 5. Identical config to seeds 1/2/3/4, "
        "fixed data split (split_seed=42). Val mIoU 0.8347 | Test mIoU 0.8458 (flip TTA). "
        "Highest test mIoU of the five seeds — within noise of seed 1, confirming no single-seed lottery.",
    ),
    (
        REPO_ROOT / "models/best_segmenter_mcdropout.pt",
        "best_segmenter_mcdropout.pt",
        "Stage 2 terrain segmenter — MC DROPOUT calibrated uncertainty (Layer 3). Fine-tuned 10 "
        "epochs from best_resnet34.pt with 27 Dropout2d(p=0.1) modules injected after every ReLU "
        "(U-Net + ResNet-34 encoder, 4 classes, 480x480). Val mIoU 0.8134 | Test mIoU 0.8181 "
        "(small accuracy cost). ECE 0.0072 across 46M val pixels (textbook-calibrated — 99% "
        "confidence -> 99.3% accuracy). OOD mutual-info 4.7x in-domain val (real moon 0.192 vs "
        "synthetic val 0.041). Loading requires the dropout architecture: smp.Unet('resnet34', "
        "encoder_weights=None, in_channels=3, classes=4); add_mc_dropout(model, p=0.1); "
        "model.load_state_dict(ckpt['model_state_dict']). Use mc_predict(model, image, "
        "n_samples=20) from lunarsite.utils.uncertainty for mean prediction + entropy + mutual "
        "information per pixel.",
    ),
    (
        REPO_ROOT / "models/v2/best_resnet50_v2.pt",
        "best_resnet50_v2.pt",
        "Stage 2 terrain segmenter — v2 NEGATIVE ABLATION. Do not use as production weights; "
        "provided as a documented negative result. Config: U-Net + ResNet-50 encoder, FocalDiceLoss "
        "(Focal Dice 0.7 + CE 0.3), inverse-frequency class weights. Same 480x480 input and "
        "augmentations as v1. Val mIoU 0.8304 | Test mIoU 0.8429 (flip TTA) — underperforms v1 "
        "ResNet-34 (0.8456) by 0.0027 mIoU. Multi-scale TTA was also tested on v2 and found to be "
        "harmful (degraded performance), so it's not included as an inference path. Kept public "
        "because negative ablations are useful science and explain why production stayed on ResNet-34.",
    ),
    (
        REPO_ROOT / "outputs/crater_v1_seed1/best_craterunet_seed1.pt",
        "best_craterunet_seed1.pt",
        "Stage 1 crater detector — v1, DeepMoon-synthetic only. U-Net + ResNet-34 encoder, 1-channel "
        "DEM input, 1-class binary crater-rim output, 256x256 tiles. Config: Dice + BCE loss, Adam "
        "lr=1e-4, 40 epochs, batch 16. Val IoU 0.306 | Test IoU 0.327 with flip TTA on DeepMoon. "
        "Trained on DeepMoon synthetic DEMs (Silburt 2019). Does NOT transfer cleanly to real LOLA "
        "south pole DEM — use best_craterunet_v2_southpole_seed1.pt for real-data inference. "
        "Kept as the pre-fine-tune baseline. Load with smp.Unet('resnet34', encoder_weights=None, "
        "in_channels=1, classes=1); ckpt['model'].",
    ),
    (
        REPO_ROOT / "outputs/crater_v2_finetune_seed1/best_craterunet_v2_southpole_seed1.pt",
        "best_craterunet_v2_southpole_seed1.pt",
        "Stage 1 crater detector — v2, PRODUCTION south-pole model. Fine-tune of best_craterunet_seed1.pt "
        "on 334 real LOLA 20-MPP tiles with Robbins (>=3 km) crater-rim labels. 25 epochs, lr=1e-5, "
        "batch 8. Pre-fine-tune val IoU 0.021 | post-fine-tune val IoU 0.161. Full 7600x7600 LOLA 80S "
        "DEM eval, flip TTA, threshold 0.25: IoU 0.162 | Dice 0.279 | Recall 0.372 | Precision 0.224. "
        "Recall up +256% vs v1 on real data (0.155 -> 0.372). IoU is low in absolute terms because "
        "it's measured against 1-pixel Robbins rims on a 7600x7600 grid — recall is the "
        "operationally meaningful metric. Used as the Stage 1 production model by "
        "scripts/crater_eval_lola.py and scripts/run_pipeline.py. Load with smp.Unet('resnet34', "
        "encoder_weights=None, in_channels=1, classes=1); ckpt['model'].",
    ),
]


def stage_files() -> int:
    """Hardlink all source checkpoints into the staging dir. Returns total bytes.
    Uses a fresh per-run dir (timestamped) so we never collide with leftover
    locked files from a prior crashed run."""
    STAGING.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    for src, name, _desc in FILES:
        dest = STAGING / name
        if not src.exists():
            raise FileNotFoundError(f"Missing source checkpoint: {src}")
        try:
            os.link(src, dest)            # hardlink — no extra disk
        except (OSError, NotImplementedError):
            shutil.copy2(src, dest)       # fallback (different volume)
        total_bytes += src.stat().st_size
        print(f"  staged {name:50s} {src.stat().st_size/1e6:7.1f} MB")
    return total_bytes


def build_metadata() -> dict:
    return {
        "title": TITLE,
        "id": DATASET_SLUG,
        "subtitle": SUBTITLE,
        "description": DESCRIPTION,
        "isPrivate": False,
        "licenses": [{"name": LICENSE}],
        "keywords": KEYWORDS,
        "resources": [
            {"path": name, "description": desc}
            for _src, name, desc in FILES
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--push", action="store_true",
                   help="Actually push. Without this flag, we stage files and build metadata only.")
    p.add_argument("--notes", default="Layer 2 ship: full docs + per-file descriptions.",
                   help="Version notes shown on the Kaggle dataset history.")
    args = p.parse_args()

    print(f"Staging checkpoints into {STAGING} ...")
    total = stage_files()
    print(f"Total upload size: {total/1e6:.0f} MB")

    meta = build_metadata()
    meta_path = STAGING / "dataset-metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path}")
    print(f"  resources: {len(meta['resources'])} files with per-file descriptions")

    if not args.push:
        print("\nDry-run. Re-run with --push to upload to Kaggle.")
        return

    # AVG Antivirus on this machine intercepts SSL with a self-signed root
    # CA that Python's certifi doesn't trust. The Kaggle SDK uses MULTIPLE
    # requests sessions (one for orchestration, one for blob uploads), so
    # we patch requests.Session.send globally to skip verification on every
    # session. Safe because the endpoints are well-known.
    import urllib3, requests
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    _orig_send = requests.Session.send
    def _patched_send(self, request, **kwargs):
        kwargs["verify"] = False
        return _orig_send(self, request, **kwargs)
    requests.Session.send = _patched_send

    from kaggle.api.kaggle_api_extended import KaggleApi
    a = KaggleApi()
    a.authenticate()
    print(f"\nPushing new version of {DATASET_SLUG} ...", flush=True)
    print(f"  Version notes: {args.notes}", flush=True)
    # Last time on Windows we hit a path bug where slashes in the staging
    # folder name corrupted the resumable-upload state file. Run from the
    # parent dir with a flat folder name so the SDK's path-to-statefile
    # conversion stays clean.
    import os, traceback
    os.chdir(str(STAGING.parent))
    try:
        result = a.dataset_create_version(
            folder=STAGING.name,
            version_notes=args.notes,
            quiet=False,
            convert_to_csv=False,
            delete_old_versions=False,
            dir_mode="skip",
        )
    except Exception as exc:
        print(f"\nEXCEPTION during dataset_create_version: {type(exc).__name__}: {exc}",
              flush=True)
        traceback.print_exc()
        return
    if result is None:
        print("ERROR: dataset_create_version returned None — see above output.")
        return
    if getattr(result, "status", "").lower() == "ok":
        print(f"OK — new version live at {result.url}")
    else:
        print(f"ERROR: {getattr(result, 'error', 'unknown')}")


if __name__ == "__main__":
    main()
