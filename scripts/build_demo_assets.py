"""Generate demo_assets/ directory for Streamlit Community Cloud deployment.

The Streamlit demo needs preloaded example outputs that are shipped with the
repo, so the app can render instantly on a fresh deploy without downloading
datasets or running cold inference. This script builds those assets once from:

1. Selected real moon images from the Kaggle dataset (sim-to-real outputs)
2. One synthetic benchmark example with the v1 checkpoint

Run this whenever the production model changes to refresh the demo assets.

Usage:
    python scripts/build_demo_assets.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = REPO_ROOT / "best_resnet34.pt"
SIM_OUT = REPO_ROOT / "outputs" / "sim_to_real" / "v1_tta"
DEMO_DIR = REPO_ROOT / "demo_assets"
DATASET_CACHE = Path.home() / ".cache/kagglehub/datasets/romainpessia/artificial-lunar-rocky-landscape-dataset/versions/2"

CLASS_NAMES = ["background", "small_rocks", "large_rocks", "sky"]
CLASS_COLORS = np.array(
    [[0, 0, 0], [255, 165, 0], [255, 0, 0], [135, 206, 235]],
    dtype=np.uint8,
)
NC = 4
INPUT_SIZE = 480

# Curated real moon examples selected for visual diversity and class balance
REAL_MOON_PICKS = [
    ("PCAM5", "Close-up rocky terrain with visible sky"),
    ("PCAM6", "Sun-lit boulder field with sky"),
    ("PCAM8", "Dense scattered rocks on flat regolith"),
    ("PCAM1", "Classic lunar landscape with distant horizon"),
    ("TCAM7", "Close-up rubble field, small rocks dominant"),
    ("TCAM21", "Mixed terrain with rocks and sky"),
]

SYNTHETIC_PICK = "render0042"  # arbitrary deterministic choice


def build_unet():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=NC,
    )


def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    return model.eval()


def center_crop_resize(img_rgb, size=INPUT_SIZE):
    h, w = img_rgb.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    cropped = img_rgb[y0:y0 + side, x0:x0 + side]
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LINEAR)


@torch.no_grad()
def predict_tta(model, img_rgb):
    x = torch.from_numpy(img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0)
    probs = F.softmax(model(x), dim=1)
    probs = probs + torch.flip(F.softmax(model(torch.flip(x, dims=[3])), dim=1), dims=[3])
    probs = probs + torch.flip(F.softmax(model(torch.flip(x, dims=[2])), dim=1), dims=[2])
    probs = probs + torch.flip(F.softmax(model(torch.flip(x, dims=[2, 3])), dim=1), dims=[2, 3])
    probs = probs / 4.0
    return probs.argmax(1).numpy()[0].astype(np.uint8)


def colorize_mask(mask):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NC):
        out[mask == c] = CLASS_COLORS[c]
    return out


def make_overlay(image_rgb, mask, alpha=0.5):
    colored = colorize_mask(mask)
    not_bg = mask != 0
    out = image_rgb.copy()
    out[not_bg] = (alpha * colored[not_bg] + (1 - alpha) * image_rgb[not_bg]).astype(np.uint8)
    return out


def compute_coverage(mask):
    total = mask.size
    return {CLASS_NAMES[c]: float((mask == c).sum()) / total for c in range(NC)}


def save_rgb_png(path, img_rgb):
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    print(f"Checkpoint: {CHECKPOINT}")
    model = build_unet()
    load_checkpoint(model, CHECKPOINT)
    print("Model loaded.")

    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    real_dir = DEMO_DIR / "real_moon"
    synth_dir = DEMO_DIR / "synthetic"
    real_dir.mkdir(parents=True)
    synth_dir.mkdir(parents=True)

    # === Real moon examples === #
    real_src = DATASET_CACHE / "real_moon_images"
    if not real_src.exists():
        raise FileNotFoundError(f"Dataset cache missing: {real_src}. Run the Kaggle training or eval notebook first to populate kagglehub cache.")

    real_manifest = []
    for name, caption in REAL_MOON_PICKS:
        src = real_src / f"{name}.png"
        if not src.exists():
            print(f"  SKIP {name}: not in dataset cache")
            continue

        img_bgr = cv2.imread(str(src))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_crop = center_crop_resize(img_rgb)
        mask = predict_tta(model, img_crop)
        overlay = make_overlay(img_crop, mask)
        coverage = compute_coverage(mask)

        save_rgb_png(real_dir / f"{name}_input.png", img_crop)
        save_rgb_png(real_dir / f"{name}_overlay.png", overlay)
        real_manifest.append({
            "name": name,
            "caption": caption,
            "input": f"real_moon/{name}_input.png",
            "overlay": f"real_moon/{name}_overlay.png",
            "coverage": coverage,
        })
        print(f"  {name:<8} coverage: bg={coverage['background']:.2f} sr={coverage['small_rocks']:.2f} lr={coverage['large_rocks']:.3f} sky={coverage['sky']:.2f}")

    # === Synthetic example === #
    render_src = DATASET_CACHE / "images" / "render" / f"{SYNTHETIC_PICK}.png"
    clean_src = DATASET_CACHE / "images" / "clean" / f"clean{SYNTHETIC_PICK.replace('render', '')}.png"

    synth_manifest = None
    if render_src.exists() and clean_src.exists():
        img_bgr = cv2.imread(str(render_src))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_crop = center_crop_resize(img_rgb)

        gt_bgr = cv2.imread(str(clean_src))
        gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
        gt_crop = center_crop_resize(gt_rgb)

        mask = predict_tta(model, img_crop)
        overlay = make_overlay(img_crop, mask)
        coverage = compute_coverage(mask)

        # Also compute GT coverage for comparison
        # GT is color-encoded: (0,0,0)=bg, (255,0,0)=small_rocks, (0,255,0)=large_rocks, (0,0,255)=sky
        gt_mask = np.zeros(gt_crop.shape[:2], dtype=np.uint8)
        color_to_class = {(0, 0, 0): 0, (255, 0, 0): 1, (0, 255, 0): 2, (0, 0, 255): 3}
        for color, cls in color_to_class.items():
            gt_mask[np.all(gt_crop == color, axis=-1)] = cls
        gt_coverage = compute_coverage(gt_mask)

        # Per-class IoU
        ious = {}
        for c in range(NC):
            pc = (mask == c)
            tc = (gt_mask == c)
            inter = (pc & tc).sum()
            union = (pc | tc).sum()
            ious[CLASS_NAMES[c]] = float(inter) / float(union) if union > 0 else 0.0

        save_rgb_png(synth_dir / "input.png", img_crop)
        save_rgb_png(synth_dir / "overlay.png", overlay)
        save_rgb_png(synth_dir / "ground_truth.png", colorize_mask(gt_mask))

        synth_manifest = {
            "name": SYNTHETIC_PICK,
            "caption": "Synthetic Unreal Engine scene from the Kaggle Artificial Lunar Landscape dataset",
            "input": "synthetic/input.png",
            "overlay": "synthetic/overlay.png",
            "ground_truth": "synthetic/ground_truth.png",
            "predicted_coverage": coverage,
            "ground_truth_coverage": gt_coverage,
            "per_class_iou": ious,
            "mean_iou": float(sum(ious.values()) / len(ious)),
        }
        print(f"  synthetic {SYNTHETIC_PICK}: mean IoU {synth_manifest['mean_iou']:.4f}")
    else:
        print(f"  SKIP synthetic: {render_src} not cached")

    # === Contact sheet === #
    cs_src = SIM_OUT / "contact_sheet.png"
    if cs_src.exists():
        shutil.copy(cs_src, DEMO_DIR / "contact_sheet.png")
        print(f"  copied contact_sheet.png")

    # === Manifest === #
    manifest = {
        "model": "U-Net + ResNet-34 + Dice+CE + flip TTA",
        "checkpoint": CHECKPOINT.name,
        "test_metrics": {
            "mean_iou": 0.8456,
            "per_class_iou": {
                "background": 0.9759,
                "small_rocks": 0.9749,
                "large_rocks": 0.7176,
                "sky": 0.7141,
            },
        },
        "real_moon_examples": real_manifest,
        "synthetic_example": synth_manifest,
    }
    manifest_path = DEMO_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest: {manifest_path}")
    print(f"Demo assets: {DEMO_DIR}")

    # Size summary
    total_bytes = sum(f.stat().st_size for f in DEMO_DIR.rglob("*") if f.is_file())
    print(f"Total size: {total_bytes / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
