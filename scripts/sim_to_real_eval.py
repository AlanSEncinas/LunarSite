"""Sim-to-real qualitative evaluation for Stage 2 terrain segmentation.

Loads the v1 production checkpoint (ResNet-34 + Dice+CE, best_resnet34.pt)
and runs inference on the 74 real moon images included in the Kaggle
Artificial Lunar Landscape dataset. Saves:

1. Per-image side-by-side overlay PNGs to outputs/sim_to_real/
2. A grid contact sheet showing all predictions at once
3. A summary JSON with coverage statistics

Usage:
    python scripts/sim_to_real_eval.py
    python scripts/sim_to_real_eval.py --checkpoint best_resnet34.pt
    python scripts/sim_to_real_eval.py --tta      # apply flip TTA
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = REPO_ROOT / "best_resnet34.pt"
OUTPUT_DIR = REPO_ROOT / "outputs" / "sim_to_real"

CLASS_NAMES = ["background", "small_rocks", "large_rocks", "sky"]
CLASS_COLORS = np.array(
    [
        [0, 0, 0],          # background - black (transparent in overlay)
        [255, 165, 0],      # small_rocks - orange
        [255, 0, 0],        # large_rocks - red
        [135, 206, 235],    # sky - light blue
    ],
    dtype=np.uint8,
)
NC = 4
INPUT_SIZE = 480


def find_real_moon_dir() -> Path:
    """Find real_moon_images dir in kagglehub cache or fall back to dataset download."""
    candidates = [
        Path.home() / ".cache/kagglehub/datasets/romainpessia/artificial-lunar-rocky-landscape-dataset/versions/2/real_moon_images",
        Path.home() / ".cache/kagglehub/datasets/romainpessia/artificial-lunar-rocky-landscape-dataset/versions/1/real_moon_images",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Fall back to kagglehub download
    import kagglehub
    print("Downloading dataset via kagglehub...")
    root = Path(kagglehub.dataset_download("romainpessia/artificial-lunar-rocky-landscape-dataset"))
    return root / "real_moon_images"


def build_unet_v1() -> torch.nn.Module:
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=NC,
    )


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        meta = {k: v for k, v in ckpt.items() if k != "model_state_dict" and not isinstance(v, torch.Tensor)}
    else:
        model.load_state_dict(ckpt)
        meta = {}
    model.to(device).eval()
    return meta


def preprocess(img_bgr: np.ndarray, size: int = INPUT_SIZE) -> tuple[torch.Tensor, np.ndarray]:
    """Center-crop and normalize an image for inference."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    cropped = img_rgb[y0:y0 + side, x0:x0 + side]
    resized = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0)
    return tensor, resized


@torch.no_grad()
def predict(model: torch.nn.Module, x: torch.Tensor, device: torch.device, tta: bool) -> torch.Tensor:
    x = x.to(device)
    if not tta:
        return F.softmax(model(x), dim=1)
    probs = F.softmax(model(x), dim=1)
    probs = probs + torch.flip(F.softmax(model(torch.flip(x, dims=[3])), dim=1), dims=[3])
    probs = probs + torch.flip(F.softmax(model(torch.flip(x, dims=[2])), dim=1), dims=[2])
    probs = probs + torch.flip(F.softmax(model(torch.flip(x, dims=[2, 3])), dim=1), dims=[2, 3])
    return probs / 4.0


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert (H, W) class indices to (H, W, 3) RGB using CLASS_COLORS."""
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NC):
        out[mask == c] = CLASS_COLORS[c]
    return out


def overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Alpha-blend colored mask over original image. Background stays visible."""
    colored = colorize_mask(mask)
    # Background (class 0) stays as original image
    mask_not_bg = mask != 0
    out = image.copy()
    out[mask_not_bg] = (alpha * colored[mask_not_bg] + (1 - alpha) * image[mask_not_bg]).astype(np.uint8)
    return out


def side_by_side(original: np.ndarray, overlay_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Concatenate [original | overlay | pure_mask] horizontally."""
    pure = colorize_mask(mask)
    return np.concatenate([original, overlay_img, pure], axis=1)


def compute_class_coverage(mask: np.ndarray) -> dict:
    """Return fraction of pixels per class."""
    total = mask.size
    return {CLASS_NAMES[c]: float((mask == c).sum()) / total for c in range(NC)}


def build_contact_sheet(thumbs: list[np.ndarray], cols: int = 6, gap: int = 8) -> np.ndarray:
    """Arrange small overlay thumbnails into a grid."""
    if not thumbs:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    th, tw = thumbs[0].shape[:2]
    n = len(thumbs)
    rows = (n + cols - 1) // cols
    H = rows * th + (rows - 1) * gap
    W = cols * tw + (cols - 1) * gap
    sheet = np.full((H, W, 3), 32, dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, cols)
        y0 = r * (th + gap)
        x0 = c * (tw + gap)
        sheet[y0:y0 + th, x0:x0 + tw] = thumb
    return sheet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--tta", action="store_true", help="Apply flip-only TTA")
    parser.add_argument("--size", type=int, default=INPUT_SIZE)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"TTA: {args.tta}")

    real_dir = find_real_moon_dir()
    # Exclude g_*.png — those are ground truth MASK files, not real photos
    all_pngs = sorted(real_dir.glob("*.png"))
    real_imgs = [p for p in all_pngs if not p.name.startswith("g_")]
    excluded = [p for p in all_pngs if p.name.startswith("g_")]
    print(f"Real moon images: {len(real_imgs)} (excluded {len(excluded)} ground-truth mask files)")
    print(f"Source dir: {real_dir}")
    if not real_imgs:
        raise FileNotFoundError(f"No non-mask .png files in {real_dir}")

    model = build_unet_v1()
    meta = load_checkpoint(model, args.checkpoint, device)
    print(f"Model loaded. Epoch: {meta.get('epoch', '?')}, best_metric: {meta.get('best_metric', '?')}")

    out_dir = OUTPUT_DIR / ("v1_tta" if args.tta else "v1_standard")
    out_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = out_dir / "overlays"
    overlays_dir.mkdir(exist_ok=True)

    per_image = []
    contact_thumbs = []
    THUMB_SIZE = 192

    t_start = time.time()
    for i, img_path in enumerate(real_imgs):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  skip {img_path.name}: cannot read")
            continue
        tensor, img_crop = preprocess(img_bgr, args.size)
        probs = predict(model, tensor, device, args.tta)
        mask = probs.argmax(1).cpu().numpy()[0].astype(np.uint8)

        coverage = compute_class_coverage(mask)
        per_image.append({
            "filename": img_path.name,
            "class_coverage": coverage,
        })

        # Save side-by-side PNG
        over = overlay(img_crop, mask)
        sbs = side_by_side(img_crop, over, mask)
        sbs_bgr = cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(overlays_dir / f"{img_path.stem}_sbs.png"), sbs_bgr)

        # Also save raw overlay for demo use
        over_bgr = cv2.cvtColor(over, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(overlays_dir / f"{img_path.stem}_overlay.png"), over_bgr)

        thumb = cv2.resize(over, (THUMB_SIZE, THUMB_SIZE))
        contact_thumbs.append(thumb)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1:3d}/{len(real_imgs)}] {img_path.name} | coverage: bg={coverage['background']:.2f} sr={coverage['small_rocks']:.2f} lr={coverage['large_rocks']:.2f} sky={coverage['sky']:.2f}")

    elapsed = time.time() - t_start
    print(f"Inference done in {elapsed:.1f}s ({elapsed / len(real_imgs) * 1000:.1f} ms/image)")

    # Contact sheet
    sheet = build_contact_sheet(contact_thumbs, cols=6)
    sheet_bgr = cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_dir / "contact_sheet.png"), sheet_bgr)
    print(f"Contact sheet: {out_dir / 'contact_sheet.png'}")

    # Aggregate coverage stats
    all_coverage = {c: [p["class_coverage"][c] for p in per_image] for c in CLASS_NAMES}
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_meta": {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in meta.items()},
        "tta": args.tta,
        "input_size": args.size,
        "n_images": len(per_image),
        "inference_time_s": elapsed,
        "time_per_image_ms": elapsed / max(len(per_image), 1) * 1000,
        "coverage_stats": {
            c: {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
            for c, vals in all_coverage.items()
        },
        "per_image": per_image,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary: {summary_path}")

    print()
    print("=" * 60)
    print("  SIM-TO-REAL COVERAGE SUMMARY (v1 ResNet-34)")
    print("=" * 60)
    for c, stats in summary["coverage_stats"].items():
        print(f"  {c:<15} mean={stats['mean']:.3f}  std={stats['std']:.3f}  min={stats['min']:.3f}  max={stats['max']:.3f}")
    print()
    print(f"Overlays: {overlays_dir}")
    print(f"Contact sheet: {out_dir / 'contact_sheet.png'}")
    print("Open the contact sheet to eyeball sim-to-real quality at a glance.")


if __name__ == "__main__":
    main()
