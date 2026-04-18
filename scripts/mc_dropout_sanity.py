"""Sanity-check MC Dropout injection on the production ResNet-34 segmenter.

Tests whether:
  1. `add_mc_dropout()` successfully injects Dropout2d into every ReLU
     position in an SMP U-Net + ResNet-34 without breaking forward pass.
  2. With dropout enabled at inference, repeated forward passes on the
     SAME input produce DIFFERENT outputs (confirming stochasticity).
  3. Per-pixel variance is non-trivial but bounded (0 < var < 0.25 for
     probabilities — 0.25 is max variance for Bernoulli).
  4. The dropout-injected network still produces plausible segmentations
     — per-class probability balance shouldn't be wildly different from
     the deterministic forward pass.

This is an INFERENCE-ONLY test. It does NOT retrain the model. A model
that wasn't trained with dropout won't be well-calibrated — that's the
Tue–Wed task. This script only verifies the plumbing works.

Usage:
    python scripts/mc_dropout_sanity.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from lunarsite.utils.uncertainty import add_mc_dropout, enable_mc_dropout, mc_predict

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT = REPO_ROOT / "best_resnet34.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DROPOUT_P = 0.1
N_SAMPLES = 20
INPUT_SIZE = 480


def build_model() -> nn.Module:
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=4,
    )


def count_dropout(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, (nn.Dropout, nn.Dropout2d)))


def count_relu(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, nn.ReLU))


def deterministic_forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(x), dim=1)


def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CKPT}")

    model = build_model().to(DEVICE)
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict") or ckpt.get("model")
    model.load_state_dict(state)

    relu_before = count_relu(model)
    do_before = count_dropout(model)
    print(f"\nBefore injection:  ReLU={relu_before}  Dropout={do_before}")

    add_mc_dropout(model, p=DROPOUT_P)

    relu_after = count_relu(model)
    do_after = count_dropout(model)
    print(f"After injection:   ReLU={relu_after}  Dropout={do_after}")
    assert do_after > 0, "Injection failed — no dropout modules added"
    assert relu_after == relu_before, f"ReLU count changed: {relu_before} → {relu_after}"

    x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=DEVICE)

    det = deterministic_forward(model, x)
    print(f"\nDeterministic (eval, no dropout): shape={tuple(det.shape)}")
    print(f"  per-class mean prob: {det.mean(dim=(0, 2, 3)).tolist()}")

    model.eval()
    enable_mc_dropout(model)
    runs = []
    with torch.no_grad():
        t0 = time.time()
        for _ in range(N_SAMPLES):
            runs.append(torch.softmax(model(x), dim=1).cpu())
        dt = time.time() - t0

    stacked = torch.stack(runs)  # (n, 1, C, H, W)
    stacked = stacked.squeeze(1)  # (n, C, H, W)

    per_run_diff = (stacked[0] - stacked[1]).abs().mean().item()
    per_pixel_var = stacked.var(dim=0).mean().item()
    per_class_var = stacked.var(dim=0).mean(dim=(1, 2)).tolist()
    per_class_mean = stacked.mean(dim=0).mean(dim=(1, 2)).tolist()

    print(f"\nMC sampling: {N_SAMPLES} runs in {dt:.1f}s  ({dt/N_SAMPLES*1000:.0f} ms/run)")
    print(f"  |run0 - run1| mean: {per_run_diff:.5f}   (>0 = stochasticity present)")
    print(f"  Mean per-pixel var: {per_pixel_var:.5f}  (max possible: 0.25)")
    print(f"  Per-class var:      {[round(v, 5) for v in per_class_var]}")
    print(f"  Per-class mean:     {[round(v, 4) for v in per_class_mean]}")

    mc = mc_predict(model, x[0], n_samples=N_SAMPLES, num_classes=4)
    print(f"\nmc_predict() output shapes:")
    for k, v in mc.items():
        print(f"  {k}: {v.shape}  dtype={v.dtype}  "
              f"min={v.min():.4f} max={v.max():.4f} mean={v.mean():.4f}")

    print("\nVERDICT:")
    if per_run_diff < 1e-6:
        print("  FAIL: runs are identical — dropout not active at inference")
    elif per_pixel_var > 0.22:
        print("  WARNING: variance near maximum — dropout is destroying signal")
    elif per_pixel_var < 1e-5:
        print("  WARNING: variance near zero — dropout may be ineffective")
    else:
        print(f"  OK: stochasticity present, variance in reasonable range "
              f"({per_pixel_var:.4f}). Ready to proceed to Tue/Wed fine-tune.")

    det_vs_mc_diff = (det.cpu() - stacked.mean(dim=0).unsqueeze(0)).abs().mean().item()
    print(f"  Deterministic vs MC-mean mean-abs-diff: {det_vs_mc_diff:.5f}")


if __name__ == "__main__":
    main()
