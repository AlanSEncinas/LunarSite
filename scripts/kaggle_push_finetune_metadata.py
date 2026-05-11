"""Push refreshed dataset-level metadata to the public Kaggle dataset
`encinas88/lunarsite-southpole-finetune` (the HDF5 fine-tuning subset
used to fine-tune the Stage 1 crater detector on real LOLA south pole).

Companion to scripts/kaggle_push_dataset_metadata.py. Calls the same
metadata-only API path (no file upload), so it works even with AVG
SSL interception (the failing upload path is the file-blob endpoint,
which we don't touch here).

Usage:
    python scripts/kaggle_push_finetune_metadata.py            # dry-run
    python scripts/kaggle_push_finetune_metadata.py --push     # send
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATASET = "encinas88/lunarsite-southpole-finetune"

TITLE = "LunarSite South Pole Crater Fine-Tuning HDF5"
SUBTITLE = "DeepMoon-format tiles from real LOLA south pole + Robbins crater rims"
LICENSE = "CC0-1.0"
KEYWORDS = [
    "computer vision",
    "deep learning",
    "image segmentation",
    "pytorch",
    "earth and nature",
    "astronomy",
]

DESCRIPTION = """\
# LunarSite — South Pole Crater Fine-Tuning HDF5

DeepMoon-format HDF5 used to fine-tune LunarSite's Stage 1 crater detector
on real LOLA south pole DEM tiles. **Companion dataset** to the production
weights at [encinas88/lunarsite-weights](https://www.kaggle.com/datasets/encinas88/lunarsite-weights);
this one holds only the training inputs that produced
`best_craterunet_v2_southpole_seed1.pt`.

- **Repo:** https://github.com/AlanSEncinas/LunarSite
- **Project site:** https://alanscottencinas.com
- **Status:** Used 2026-04-18 to fine-tune Stage 1 v2; v2 is the production
  Stage 1 model in the LunarSite Streamlit demo.

## What's in this dataset

A single HDF5 file:

| File | Format | Tiles | Resolution | Source |
|---|---|---|---|---|
| `southpole_finetune_118mpp.hdf5` | DeepMoon-compatible (input/target groups) | 334 | 256×256 px @ 118 m/px | Real NASA LOLA 20 MPP DEM (resampled to 118 m/px to match v1 training scale) + Robbins 2018 catalog rims ≥3 km |

Built by [scripts/build_southpole_hdf5.py](https://github.com/AlanSEncinas/LunarSite/blob/main/scripts/build_southpole_hdf5.py)
in the LunarSite repo. The script tiles the LDEM_80S_20MPP_ADJ.TIF DEM into
overlapping 256×256 patches, applies a per-tile uint8 stretch, and burns
Robbins rim circles as binary targets.

## Why this exists

The DeepMoon synthetic crater detector (LunarSite Stage 1 v1) **does not
transfer cleanly** to real LOLA south pole DEMs:

| Model | Val IoU | Recall on real LOLA south pole |
|---|---|---|
| v1 (DeepMoon synthetic only) | 0.306 (synthetic val) | **0.155** (real LOLA) |
| v2 (this dataset, fine-tuned from v1) | 0.161 (real val) | **0.372** (real LOLA, **+140 %** vs v1) |

Domain shift between DeepMoon's synthetic procedural craters and real LOLA
geometry was severe. 25 epochs of fine-tuning at lr 1e-5 on these 334 tiles
closed the gap. v2 is the production Stage 1 checkpoint used in
[scripts/crater_eval_lola.py](https://github.com/AlanSEncinas/LunarSite/blob/main/scripts/crater_eval_lola.py)
and [scripts/run_pipeline.py](https://github.com/AlanSEncinas/LunarSite/blob/main/scripts/run_pipeline.py).

## Data sources & attribution

- **DEM:** NASA LOLA `LDEM_80S_20MPP_ADJ.TIF` (public domain, NASA GSFC).
- **Crater catalog:** Robbins, S.J. (2019). *A new global database of lunar
  impact craters >1 km*. Journal of Geophysical Research: Planets, 124(4).
  Public domain via NASA PDS.
- **Tile schema:** Compatible with the DeepMoon HDF5 layout (Silburt et
  al. 2019, Zenodo 1133969) so any DeepMoon-trained checkpoint can be
  fine-tuned directly on this file.

## Loading

```python
import h5py
import numpy as np

with h5py.File("southpole_finetune_118mpp.hdf5", "r") as f:
    # DeepMoon schema: input_images, target_masks (each indexed by tile_id)
    images  = f["input_images"][:]   # (334, 256, 256) uint8
    targets = f["target_masks"][:]   # (334, 256, 256) uint8 (binary rims)
print(images.shape, targets.shape)
```

## Citation

```
Encinas, A. (2026). LunarSite — end-to-end ML pipeline for lunar south
pole landing site selection. https://github.com/AlanSEncinas/LunarSite
```

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
    p.add_argument("--push", action="store_true")
    p.add_argument("--meta-dir", default="tmp/kaggle_finetune_meta")
    args = p.parse_args()

    meta = build_metadata()
    meta_dir = Path(args.meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    out = meta_dir / "dataset-metadata.json"
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {out}")
    print(f"  title:    {meta['title']} ({len(meta['title'])} chars)")
    print(f"  subtitle: {meta['subtitle']} ({len(meta['subtitle'])} chars)")
    print(f"  desc:     {len(meta['description'])} chars")
    print(f"  license:  {meta['licenses'][0]['name']}")
    print(f"  keywords: {meta['keywords']}")

    if not args.push:
        print("\nDry-run. Re-run with --push to send to Kaggle.")
        return

    # AVG SSL bypass — patch all requests sessions globally.
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
