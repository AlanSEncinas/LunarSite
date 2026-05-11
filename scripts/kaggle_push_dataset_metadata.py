"""Push dataset-level metadata (title / subtitle / description / keywords / license)
to the public Kaggle dataset `encinas88/lunarsite-weights` via the API.

This does NOT re-upload the .pt checkpoints (no bandwidth cost). It calls
`KaggleApi.dataset_metadata_update`, which only patches metadata.

Per-file descriptions are not touched by this endpoint — they require either
a UI edit per file OR a full dataset-version re-push. See the companion doc
`docs/drafts/kaggle_dataset_copy.md` section 7 for the per-file blurbs.

Usage:
    python scripts/kaggle_push_dataset_metadata.py           # dry-run: prints JSON
    python scripts/kaggle_push_dataset_metadata.py --push    # actually push
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATASET = "encinas88/lunarsite-weights"

TITLE = "LunarSite — Lunar South Pole ML Weights"

SUBTITLE = (
    "Trained weights for lunar south pole terrain & crater detection."
)

KEYWORDS = [
    "computer vision",
    "deep learning",
    "image segmentation",
    "pytorch",
    "pre-trained model",
    "earth and nature",
    "astronomy",
]

LICENSE = "CC-BY-SA-4.0"

DESCRIPTION = """\
# LunarSite — Trained Checkpoints

Pretrained PyTorch weights from **LunarSite**, an end-to-end ML pipeline for lunar south pole landing-site selection. Everything here has been trained, evaluated, and used to produce the results in the project's public case studies.

- **Repo:** https://github.com/AlanSEncinas/LunarSite
- **Project site:** https://alanscottencinas.com
- **Status:** Layer 3 shipped (2026-04-18) — Stage 1 (craters) + Stage 2 (terrain ensemble + MC Dropout calibrated uncertainty) + Stage 3 (PSR-aware XGBoost scorer) complete. Cross-instrument PSR validation against ShadowCam at Cabeus / LCROSS confirms 81–85 % agreement with PGDA-predicted shadow regions.

## What's in this dataset

Nine PyTorch `.pt` checkpoints, grouped into four roles.

### Stage 2 — Terrain Segmentation (5-seed deep ensemble)

U-Net with a ResNet-34 encoder (ImageNet-pretrained), 4-class semantic segmentation: `background`, `small_rocks`, `large_rocks`, `sky`.

| File | Seed | Val mIoU | Test mIoU (flip TTA) | Role |
|---|---|---|---|---|
| `best_resnet34.pt` | 1 | 0.8357 | **0.8456** | Single-model production checkpoint (used in Streamlit demo) |
| `best_resnet34_seed2.pt` | 2 | 0.8371 | 0.8434 | Ensemble member |
| `best_resnet34_seed3.pt` | 3 | 0.8367 | 0.8448 | Ensemble member |
| `best_resnet34_seed4.pt` | 4 | 0.8343 | 0.8428 | Ensemble member |
| `best_resnet34_seed5.pt` | 5 | 0.8347 | **0.8458** | Ensemble member (highest test) |

5-seed ensemble (all five averaged) gives test mIoU = **0.8445 ± 0.0013** and per-pixel epistemic uncertainty from member disagreement.

**Config (identical across all 5 seeds):** 480×480 input · Dice + CE loss · Adam · cosine-annealing LR 1e-4 → 1e-6 · 50 epochs · batch 16 · lunar-specific augmentations (shadow rotation, extreme contrast, Hapke BRDF perturbation). Only the random seed varies (PyTorch, NumPy, DataLoader shuffle, augmentation RNG). `split_seed=42` is held fixed so every member sees the same train/val/test split — this is required for ensemble disagreement to be meaningful.

**Training data:** Kaggle `romainpessia/artificial-lunar-rocky-landscape-dataset` (9,766 synthetic lunar landscape images + 4-class masks), 80/10/10 split.

### Stage 2 — Negative ablation

| File | Val mIoU | Test mIoU (flip TTA) | Notes |
|---|---|---|---|
| `best_resnet50_v2.pt` | 0.8304 | 0.8429 | **Lost vs v1** — kept as a documented negative result |

v2 was a planned upgrade: ResNet-50 encoder + FocalDiceLoss + inverse-frequency class weights. It underperformed v1 by 0.0027 mIoU on test. Multi-scale TTA was also tested and discarded as harmful. Kept public because negative results are useful science and the ablation explains why production stayed on ResNet-34.

### Stage 2 — MC Dropout (calibrated epistemic uncertainty, Layer 3)

| File | Val mIoU | Test mIoU | ECE | OOD mutual-info lift |
|---|---|---|---|---|
| `best_segmenter_mcdropout.pt` | 0.8134 | 0.8181 | **0.0072** | **4.7×** (real moon vs synthetic val) |

Fine-tuned for 10 epochs from `best_resnet34.pt` with 27 `Dropout2d(p=0.1)` modules injected after every ReLU in the U-Net + ResNet-34 encoder. Training-time dropout produces well-calibrated MC sampling at inference: across 46 M val pixels, the model's predicted confidence matches actual accuracy within 0.7 % (when the model says 99 % sure, it's right 99.3 % of the time).

The OOD lift is the operationally useful number for landing-site safety — when shown real moon photographs the model never trained on, MC mutual information jumps **4.7×** vs in-domain validation images (0.192 vs 0.041). The model knows when it's looking at something it doesn't recognize, instead of being confidently wrong on out-of-distribution inputs.

**Inference:** load with `add_mc_dropout(model, p=0.1)` from `lunarsite.utils.uncertainty`, then call `mc_predict(model, image, n_samples=20)` for per-pixel mean prediction + entropy + mutual information.

### Stage 1 — Crater Detection

Binary-segmentation U-Net (ResNet-34 encoder) on 256×256 DEM tiles. Input is a single-channel elevation tile stretched per-tile to 0–1 uint8; output is a binary crater-rim mask.

| File | Training data | Val IoU | Eval target | Eval IoU | Eval recall |
|---|---|---|---|---|---|
| `best_craterunet_seed1.pt` | DeepMoon synthetic | 0.306 | DeepMoon val set (flip TTA) | 0.327 | — |
| `best_craterunet_v2_southpole_seed1.pt` | DeepMoon → fine-tuned on LOLA south pole | 0.161 | LOLA 80S DEM vs Robbins ≥3 km (flip TTA, t=0.25) | **0.162** | **0.372** |

The v1 (DeepMoon-only) model does not transfer cleanly to the real south pole DEM — a documented distribution-within-domain shift, not a modality shift. Fine-tuning on 334 tiles of real LOLA 20-MPP DEM with Robbins-catalog rim masks fixes it: recall on real craters jumps **+256 %** (0.155 → 0.372) for a small precision trade. v2 is the production Stage 1 model.

**Why the "low" IoU is fine:** IoU is computed against 1-pixel rims on a 7600×7600 DEM. Recall (what fraction of real craters the model finds) is the operationally meaningful metric for landing-site hazard feature extraction; IoU is heavily penalized by thin-ring geometry even when the detection is spatially correct.

## Reproducing the results

### Minimal inference example

```python
import torch
import segmentation_models_pytorch as smp

# Stage 2 — terrain segmenter (one ensemble member, deterministic)
model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=4)
ckpt = torch.load("best_resnet34.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Stage 2 — MC Dropout calibrated uncertainty (Layer 3)
from lunarsite.utils.uncertainty import add_mc_dropout, mc_predict
mc_model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=4)
add_mc_dropout(mc_model, p=0.1)  # MUST inject dropout before loading state
ckpt = torch.load("best_segmenter_mcdropout.pt", map_location="cpu")
mc_model.load_state_dict(ckpt["model_state_dict"])
result = mc_predict(mc_model, image_tensor, n_samples=20)
# result contains mean_probs, prediction, entropy, mutual_info, variance

# Stage 1 — crater detector (v2 south-pole fine-tuned)
crater = smp.Unet("resnet34", encoder_weights=None, in_channels=1, classes=1)
ckpt = torch.load("best_craterunet_v2_southpole_seed1.pt", map_location="cpu")
# Stage 1 checkpoints use the 'model' key, not 'model_state_dict'
crater.load_state_dict(ckpt["model"])
crater.eval()
```

### Full pipeline

Clone the repo and run:

```bash
pip install -r requirements.txt
pip install -e .
python scripts/run_pipeline.py   # Stage 1 → Stage 3 → demo assets
```

The repo auto-downloads this dataset via `kagglehub`. See the repo's `README.md` for data-source licensing and the full reproduction path.

## Checkpoint key conventions

All checkpoints are `torch.save(...)` dicts:

- **Stage 2 segmenters** — keys: `model_state_dict`, `optimizer_state_dict`, `epoch`, `best_val_miou`, `config`.
- **Stage 1 crater detectors** — keys: `model`, `optimizer`, `epoch`, `best_val_iou`, `config`.

The key-name difference is historical (different training scripts); code in the repo handles both.

## Data sources & attribution

- **Stage 2 training:** Kaggle `romainpessia/artificial-lunar-rocky-landscape-dataset` (CC BY 4.0).
- **Stage 1 v1 training:** DeepMoon (Silburt et al. 2019, Zenodo 1133969, CC0).
- **Stage 1 v2 fine-tune:** NASA LOLA LDEM 20 MPP south pole (public domain) + Robbins lunar crater catalog ≥ 3 km (public domain, Robbins 2019).
- **Stage 3 features (not in this dataset, downloaded separately):** NASA PGDA slope and illumination products, Mazarico 2011 average-visibility raster.

If you use these checkpoints in academic or professional work, please cite:

```
Encinas, A. (2026). LunarSite — end-to-end ML pipeline for lunar south pole landing site selection. https://github.com/AlanSEncinas/LunarSite
```

## What is NOT in this dataset

- **Training data** — linked above, download from the original sources.
- **LOLA DEMs and PGDA products** — several GB each; fetch from NASA.
- **Stage 3 XGBoost model** — reproducible from the features parquet in under 60 seconds on CPU; not worth shipping a checkpoint.
- **Dark-terrain models** (Depth Anything V2, HORUS) — not trained in LunarSite, use the original repos.

## Versioning

This dataset is versioned as the project evolves. Versions correspond to tagged releases in the GitHub repo when possible. See the repo's commit log for what changed between versions.

---

**Contact:** Alan Encinas · https://alanscottencinas.com · encinas88 on Kaggle and GitHub.
"""


def build_metadata() -> dict:
    return {
        "title": TITLE,
        "id": DATASET,
        "subtitle": SUBTITLE,
        "description": DESCRIPTION,
        "isPrivate": False,
        "licenses": [{"name": LICENSE}],
        "keywords": KEYWORDS,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--push", action="store_true", help="Actually push to Kaggle (otherwise dry-run).")
    p.add_argument("--meta-dir", default="tmp/kaggle_meta",
                   help="Where to write the dataset-metadata.json the API will read.")
    args = p.parse_args()

    meta = build_metadata()
    meta_dir = Path(args.meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    out = meta_dir / "dataset-metadata.json"
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {out}")
    print(f"  title:       {meta['title']}")
    print(f"  subtitle:    {meta['subtitle'][:80]}...")
    print(f"  description: {len(meta['description'])} chars")
    print(f"  keywords:    {meta['keywords']}")
    print(f"  license:     {meta['licenses'][0]['name']}")

    if not args.push:
        print("\nDry-run. Re-run with --push to send to Kaggle.")
        return

    # AVG Antivirus intercepts SSL with a self-signed root CA that Python's
    # certifi doesn't trust. Patch requests.Session.send globally so every
    # session skips verification (the SDK uses multiple sessions internally).
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
    print(f"\nPushing metadata-only update to {DATASET}...")
    a.dataset_metadata_update(DATASET, str(meta_dir))
    print("Done.")


if __name__ == "__main__":
    main()
