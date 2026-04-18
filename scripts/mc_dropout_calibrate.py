"""Calibration evaluation for the MC Dropout fine-tuned segmenter.

Runs:
  1. In-domain val-set calibration — reliability diagram + ECE.
     Compares MC-mean confidence to actual per-pixel accuracy across
     probability bins.
  2. In-domain vs OOD uncertainty comparison — mean per-image entropy
     and mutual information on synthetic val images vs real moon
     images. Well-calibrated epistemic uncertainty should be HIGHER on
     OOD data than in-domain data.

Expects:
  models/best_segmenter_mcdropout.pt  (produced by train_segmenter.py
                                       --mc-dropout --resume-from ...)

Outputs:
  outputs/mc_dropout_eval/reliability.png
  outputs/mc_dropout_eval/uncertainty_histogram.png
  outputs/mc_dropout_eval/calibration.json

Usage:
    python scripts/mc_dropout_calibrate.py
    python scripts/mc_dropout_calibrate.py --n-samples 30 --n-val 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import kagglehub
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from PIL import Image
from torch.utils.data import DataLoader, Subset
import albumentations as A

from lunarsite.data.lunar_dataset import LunarTerrainDataset
from lunarsite.utils.uncertainty import add_mc_dropout, enable_mc_dropout

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT = REPO_ROOT / "models" / "best_segmenter_mcdropout.pt"
OUT_DIR = REPO_ROOT / "outputs" / "mc_dropout_eval"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["background", "small_rocks", "large_rocks", "sky"]
N_BINS = 15
INPUT_SIZE = 480
VAL_SPLIT_SEED = 42
VAL_RATIO = 0.1
TRAIN_RATIO = 0.8


def build_model_with_dropout(p: float) -> nn.Module:
    model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=4)
    add_mc_dropout(model, p=p)
    return model


def load_mc_model() -> tuple[nn.Module, float]:
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict") or ckpt.get("model")
    # Dropout p isn't in the checkpoint — default to 0.1 unless reported.
    dropout_p = 0.1
    model = build_model_with_dropout(dropout_p).to(DEVICE)
    model.load_state_dict(state)
    return model, dropout_p


@torch.no_grad()
def mc_probs(model: nn.Module, x: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Returns (n_samples, B, C, H, W) softmax probs."""
    model.eval()
    enable_mc_dropout(model)
    runs = []
    for _ in range(n_samples):
        runs.append(F.softmax(model(x), dim=1).cpu())
    return torch.stack(runs)


def reliability_stats(confidences: np.ndarray, correct: np.ndarray,
                      n_bins: int = N_BINS) -> dict:
    """Compute reliability-diagram bins and ECE."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    total = confidences.size
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else \
               (confidences >= lo) & (confidences <= hi)
        n = int(mask.sum())
        if n == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "n": 0,
                         "mean_conf": None, "accuracy": None})
            continue
        mean_conf = float(confidences[mask].mean())
        acc = float(correct[mask].mean())
        bins.append({"lo": float(lo), "hi": float(hi), "n": n,
                     "mean_conf": mean_conf, "accuracy": acc})
        ece += (n / total) * abs(mean_conf - acc)
    return {"bins": bins, "ece": ece, "total_samples": int(total)}


def draw_reliability(stats: dict, out_png: Path, title: str) -> None:
    bins = stats["bins"]
    conf = [b["mean_conf"] for b in bins if b["mean_conf"] is not None]
    acc = [b["accuracy"] for b in bins if b["accuracy"] is not None]
    widths = [(b["hi"] - b["lo"]) for b in bins if b["mean_conf"] is not None]
    centers = [(b["lo"] + b["hi"]) / 2 for b in bins if b["mean_conf"] is not None]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.bar(centers, acc, width=widths, alpha=0.6, edgecolor="black",
           label="accuracy in bin", color="#00D4FF")
    ax.plot(conf, acc, "o-", color="#FF2D78", label="mean confidence → accuracy")
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{title}\nECE = {stats['ece']:.4f}  "
                 f"(n = {stats['total_samples']:,} pixels)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def per_image_uncertainty(probs_n: torch.Tensor) -> tuple[float, float]:
    """Given (n, C, H, W) softmax probs for ONE image, return (entropy, MI)."""
    mean_probs = probs_n.mean(dim=0)  # (C, H, W)
    entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=0)
    per_sample_H = -(probs_n * torch.log(probs_n + 1e-10)).sum(dim=1)
    mean_H = per_sample_H.mean(dim=0)
    mi = entropy - mean_H
    return float(entropy.mean()), float(mi.mean())


def run_in_domain(model: nn.Module, n_val: int, n_samples: int) -> tuple[dict, list]:
    """Iterate val subset, collect (confidences, correct) for reliability +
    per-image (entropy, MI) for OOD comparison."""
    dataset_path = Path(kagglehub.dataset_download(
        "romainpessia/artificial-lunar-rocky-landscape-dataset"))
    image_dir = dataset_path / "images" / "render"
    mask_dir = dataset_path / "images" / "clean"

    tfm = A.Compose([A.CenterCrop(height=INPUT_SIZE, width=INPUT_SIZE)])
    ds = LunarTerrainDataset(image_dir, mask_dir, transform=tfm)

    n_total = len(ds)
    perm = np.random.RandomState(VAL_SPLIT_SEED).permutation(n_total).tolist()
    n_train = int(n_total * TRAIN_RATIO)
    n_val_all = int(n_total * VAL_RATIO)
    val_idx = perm[n_train:n_train + n_val_all]
    if n_val is not None and n_val < len(val_idx):
        # Deterministic subsample for reproducibility
        val_idx = val_idx[:n_val]

    loader = DataLoader(Subset(ds, val_idx), batch_size=4, shuffle=False, num_workers=0)

    all_confs, all_correct = [], []
    per_image_stats = []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(DEVICE)
            masks = batch["mask"]  # (B, H, W) on CPU
            probs = mc_probs(model, imgs, n_samples)  # (n, B, C, H, W)
            mean_probs = probs.mean(dim=0)  # (B, C, H, W)
            conf, pred = mean_probs.max(dim=1)  # each (B, H, W)
            correct = (pred == masks).float()
            all_confs.append(conf.flatten().numpy())
            all_correct.append(correct.flatten().numpy())

            for bi in range(imgs.shape[0]):
                H, MI = per_image_uncertainty(probs[:, bi])
                per_image_stats.append({"entropy": H, "mutual_info": MI})

    confs = np.concatenate(all_confs)
    correct = np.concatenate(all_correct)
    return reliability_stats(confs, correct), per_image_stats


def run_ood(model: nn.Module, n_samples: int) -> list:
    """MC inference on the 36 real moon images from the dataset."""
    dataset_path = Path(kagglehub.dataset_download(
        "romainpessia/artificial-lunar-rocky-landscape-dataset"))
    real_dir = dataset_path / "real_moon_images"
    imgs = sorted(real_dir.glob("*.png"))

    per_image_stats = []
    with torch.no_grad():
        for p in imgs:
            im = Image.open(p).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
            x = torch.from_numpy(np.array(im)).float().permute(2, 0, 1) / 255.0
            x = x.unsqueeze(0).to(DEVICE)
            probs = mc_probs(model, x, n_samples)  # (n, 1, C, H, W)
            H, MI = per_image_uncertainty(probs[:, 0])
            per_image_stats.append({"name": p.name, "entropy": H, "mutual_info": MI})
    return per_image_stats


def draw_uncertainty_histogram(in_domain: list, ood: list, out_png: Path) -> None:
    in_H = [r["entropy"] for r in in_domain]
    in_MI = [r["mutual_info"] for r in in_domain]
    ood_H = [r["entropy"] for r in ood]
    ood_MI = [r["mutual_info"] for r in ood]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, in_vals, ood_vals, label in [
        (axes[0], in_H, ood_H, "Mean predictive entropy (total)"),
        (axes[1], in_MI, ood_MI, "Mean mutual information (epistemic)"),
    ]:
        ax.hist(in_vals, bins=25, alpha=0.55, color="#00D4FF",
                label=f"in-domain val (n={len(in_vals)})", edgecolor="black")
        ax.hist(ood_vals, bins=25, alpha=0.55, color="#FF2D78",
                label=f"OOD real moon (n={len(ood_vals)})", edgecolor="black")
        ax.set_xlabel(label)
        ax.set_ylabel("count")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title(label.split("(")[0].strip())

    fig.suptitle("Per-image uncertainty: in-domain val vs OOD real moon",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=20, help="MC forward passes per image.")
    p.add_argument("--n-val", type=int, default=200, help="Val subset size for reliability.")
    args = p.parse_args()

    if not CKPT.exists():
        raise FileNotFoundError(
            f"{CKPT} not found. Run the fine-tune first:\n"
            f"  python scripts/train_segmenter.py --mc-dropout --resume-from best_resnet34.pt "
            f"--epochs 10 --lr 2e-5 --tag mcdropout")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CKPT}")

    model, dropout_p = load_mc_model()
    print(f"Loaded MC model with dropout_p={dropout_p}")

    print(f"\n--- In-domain reliability (n_val={args.n_val}, n_samples={args.n_samples}) ---")
    stats, in_domain_per_img = run_in_domain(model, args.n_val, args.n_samples)
    print(f"  ECE: {stats['ece']:.4f}")
    print(f"  pixels evaluated: {stats['total_samples']:,}")
    draw_reliability(stats, OUT_DIR / "reliability.png", "MC Dropout — in-domain calibration")
    print(f"  saved {OUT_DIR / 'reliability.png'}")

    print(f"\n--- OOD uncertainty (real moon images) ---")
    ood_per_img = run_ood(model, args.n_samples)
    mi_in = float(np.mean([r["mutual_info"] for r in in_domain_per_img]))
    mi_ood = float(np.mean([r["mutual_info"] for r in ood_per_img]))
    H_in = float(np.mean([r["entropy"] for r in in_domain_per_img]))
    H_ood = float(np.mean([r["entropy"] for r in ood_per_img]))
    print(f"  mean entropy  in-domain: {H_in:.4f} | OOD: {H_ood:.4f}  "
          f"(OOD higher? {'YES' if H_ood > H_in else 'NO'})")
    print(f"  mean mut info in-domain: {mi_in:.4f} | OOD: {mi_ood:.4f}  "
          f"(OOD higher? {'YES' if mi_ood > mi_in else 'NO'})")
    draw_uncertainty_histogram(in_domain_per_img, ood_per_img, OUT_DIR / "uncertainty_histogram.png")
    print(f"  saved {OUT_DIR / 'uncertainty_histogram.png'}")

    summary = {
        "n_samples_per_image": args.n_samples,
        "dropout_p": dropout_p,
        "ece": stats["ece"],
        "reliability_bins": stats["bins"],
        "pixels_evaluated": stats["total_samples"],
        "in_domain_mean_entropy": H_in,
        "in_domain_mean_mutual_info": mi_in,
        "ood_mean_entropy": H_ood,
        "ood_mean_mutual_info": mi_ood,
        "ood_entropy_lift": H_ood - H_in,
        "ood_mi_lift": mi_ood - mi_in,
        "n_in_domain": len(in_domain_per_img),
        "n_ood": len(ood_per_img),
    }
    out_json = OUT_DIR / "calibration.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary -> {out_json}")


if __name__ == "__main__":
    main()
