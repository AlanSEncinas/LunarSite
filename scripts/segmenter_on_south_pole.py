"""Run the Stage 2 v1 segmenter on real lunar south pole optical imagery.

Option 1 from the Saturday plan: take the production U-Net+ResNet-34 Dice+CE
checkpoint (best_resnet34.pt, test TTA mIoU 0.8456) and run it zero-shot on
publicly accessible south pole images pulled from NASA Images API into
data/raw/south_pole_optical/.

For each input image this script:
  - center-crops to a square, resizes to 480
  - runs the segmenter with flip TTA
  - writes {stem}_overlay.png and {stem}_sbs.png (input | overlay | mask)

Also builds a contact sheet combining every result into a single PNG and a
summary JSON with class coverage statistics per image.

Usage:
    python scripts/segmenter_on_south_pole.py
    python scripts/segmenter_on_south_pole.py --input-dir data/raw/south_pole_optical --tta
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from sim_to_real_eval import (
    CLASS_COLORS, CLASS_NAMES, INPUT_SIZE, NC,
    build_unet_v1, colorize_mask, compute_class_coverage,
    load_checkpoint, overlay, predict, preprocess,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = REPO_ROOT / "best_resnet34.pt"
DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "south_pole_optical"
DEFAULT_OUT = REPO_ROOT / "outputs" / "segmenter_south_pole"


def build_contact_sheet(rows: list[np.ndarray], cols: int = 2, gap: int = 8) -> np.ndarray:
    if not rows:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    th, tw = rows[0].shape[:2]
    n = len(rows)
    nrows = (n + cols - 1) // cols
    H = nrows * th + (nrows - 1) * gap
    W = cols * tw + (cols - 1) * gap
    sheet = np.full((H, W, 3), 32, dtype=np.uint8)
    for i, row in enumerate(rows):
        r, c = divmod(i, cols)
        y0 = r * (th + gap)
        x0 = c * (tw + gap)
        sheet[y0:y0 + th, x0:x0 + tw] = row
    return sheet


def add_caption(img: np.ndarray, text: str, height: int = 28) -> np.ndarray:
    bar = np.full((height, img.shape[1], 3), 24, dtype=np.uint8)
    cv2.putText(bar, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--size", type=int, default=INPUT_SIZE)
    p.add_argument("--tta", action="store_true", default=True)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = args.out_dir / "overlays"
    overlays_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  TTA: {args.tta}")
    print(f"Checkpoint: {args.checkpoint}")

    model = build_unet_v1()
    meta = load_checkpoint(model, args.checkpoint, device)
    print(f"Model: epoch {meta.get('epoch','?')}  best_metric {meta.get('best_metric','?')}")

    images = sorted(p for p in args.input_dir.glob("*.*")
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    print(f"\nSouth pole images: {len(images)}")
    if not images:
        raise FileNotFoundError(f"No images in {args.input_dir}")

    per_image = []
    contact_rows = []
    t0 = time.time()

    for i, img_path in enumerate(images):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  skip {img_path.name}: unreadable")
            continue

        tensor, img_crop = preprocess(img_bgr, args.size)
        probs = predict(model, tensor, device, args.tta)
        mask = probs.argmax(1).cpu().numpy()[0].astype(np.uint8)

        coverage = compute_class_coverage(mask)
        per_image.append({
            "filename": img_path.name,
            "class_coverage": coverage,
        })

        over = overlay(img_crop, mask)
        pure = colorize_mask(mask)
        sbs = np.concatenate([img_crop, over, pure], axis=1)
        cv2.imwrite(str(overlays_dir / f"{img_path.stem}_overlay.png"),
                    cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(overlays_dir / f"{img_path.stem}_sbs.png"),
                    cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR))

        # Contact sheet row
        caption = (f"{img_path.stem}  "
                   f"bg={coverage['background']*100:.0f}  "
                   f"sr={coverage['small_rocks']*100:.0f}  "
                   f"lr={coverage['large_rocks']*100:.0f}  "
                   f"sky={coverage['sky']*100:.0f}")
        sbs_with_caption = add_caption(sbs, caption)
        contact_rows.append(sbs_with_caption)
        print(f"  [{i+1:>2}/{len(images)}] {img_path.name:<40} "
              f"bg={coverage['background']*100:>4.1f}%  "
              f"sr={coverage['small_rocks']*100:>4.1f}%  "
              f"lr={coverage['large_rocks']*100:>4.1f}%  "
              f"sky={coverage['sky']*100:>4.1f}%")

    elapsed = time.time() - t0
    print(f"\nInference done in {elapsed:.1f}s ({elapsed/max(len(per_image),1)*1000:.0f} ms/image)")

    # Contact sheet
    if contact_rows:
        # Normalize row widths by padding
        max_w = max(r.shape[1] for r in contact_rows)
        padded = []
        for r in contact_rows:
            if r.shape[1] < max_w:
                pad = np.full((r.shape[0], max_w - r.shape[1], 3), 32, dtype=np.uint8)
                r = np.concatenate([r, pad], axis=1)
            padded.append(r)
        sheet = build_contact_sheet(padded, cols=1)
        cv2.imwrite(str(args.out_dir / "contact_sheet.png"),
                    cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        print(f"Contact sheet: {args.out_dir / 'contact_sheet.png'}")

    # Summary
    all_coverage = {c: [p["class_coverage"][c] for p in per_image] for c in CLASS_NAMES}
    summary = {
        "checkpoint": str(args.checkpoint),
        "tta": args.tta,
        "input_size": args.size,
        "n_images": len(per_image),
        "coverage_stats": {
            c: {
                "mean": float(np.mean(vals)) if vals else 0.0,
                "std": float(np.std(vals)) if vals else 0.0,
                "min": float(np.min(vals)) if vals else 0.0,
                "max": float(np.max(vals)) if vals else 0.0,
            }
            for c, vals in all_coverage.items()
        },
        "per_image": per_image,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Summary: {args.out_dir / 'summary.json'}")

    print()
    print("=" * 60)
    print("  SOUTH POLE SIM-TO-REAL COVERAGE (Stage 2 v1, zero-shot)")
    print("=" * 60)
    for c, s in summary["coverage_stats"].items():
        print(f"  {c:<15} mean={s['mean']:.3f}  std={s['std']:.3f}")


if __name__ == "__main__":
    main()
