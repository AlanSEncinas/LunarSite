"""Generate notebooks/finetune_crater_southpole_kaggle.ipynb.

Mirrors build_ensemble_notebooks.py in spirit: one Python source of truth that
materializes a Kaggle-runnable .ipynb. Fine-tunes the Stage 1 v1 crater U-Net
on the real LOLA south pole HDF5 we just built, starting from v1's weights.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "finetune_crater_southpole_kaggle.ipynb"


def _md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def _code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


def build() -> dict:
    header = """# LunarSite Stage 1 v2 — Fine-tune crater U-Net on real LOLA south pole

Start from `best_craterunet_seed1.pt` (trained on DeepMoon synthetic DEMs,
test TTA IoU 0.327) and fine-tune on the 334-tile south pole HDF5 we built
from PGDA LOLA 20MPP + Robbins 2018 catalog (>=3 km craters at 118 m/px,
matching DeepMoon training resolution).

Hypothesis: v1 has the right receptive field + crater-ring prior but never
saw the overlapping, rim-on-rim, degraded craters typical of the south pole.
A short fine-tune should close the sim-to-real gap documented in
outputs/crater_eval_sweep/sweep_results.json (v1 caps at ~11% IoU / ~15%
recall on LOLA regardless of normalization / threshold).
"""

    install = """!pip install -q --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchvision==0.20.1
!pip install -q segmentation-models-pytorch h5py albumentations"""

    imports = """import os, sys, json, time, random, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import h5py
import albumentations as A
import segmentation_models_pytorch as smp
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"""

    config = """# ================== CONFIG ==================
SEED = 1
EPOCHS = 25
BATCH_SIZE = 8            # small dataset (334 tiles), keep batches small
LR = 1e-5                 # 10x smaller than v1 training -- we're fine-tuning
WD = 1e-4
ETA_MIN = 1e-7
NUM_WORKERS = 2
VAL_FRAC = 0.10
TEST_FRAC = 0.10

WORK = Path('/kaggle/working')
# v1 checkpoint comes from the checkpoints dataset (attached as input):
V1_CKPT = Path('/kaggle/input/lunarsite-checkpoints/best_craterunet_seed1.pt')
# Fine-tuning HDF5 comes from the southpole dataset (attached as input):
FINETUNE_H5 = Path('/kaggle/input/lunarsite-southpole-finetune/southpole_finetune_118mpp.hdf5')

CKPT_PATH = WORK / f'best_craterunet_v2_southpole_seed{SEED}.pt'
SUMMARY_PATH = WORK / f'summary_v2_seed{SEED}.json'

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

set_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device, 'seed:', SEED)
print('v1 ckpt:', V1_CKPT, 'exists:', V1_CKPT.exists())
print('finetune h5:', FINETUNE_H5, 'exists:', FINETUNE_H5.exists())"""

    dataset = """class SouthPoleCraterDataset(Dataset):
    def __init__(self, hdf5_path, indices=None, transform=None):
        self.hdf5_path = str(hdf5_path)
        self.transform = transform
        self._file = None
        with h5py.File(self.hdf5_path, 'r') as f:
            n_total = f['input_images'].shape[0]
        self.indices = np.arange(n_total) if indices is None else np.asarray(indices)
    def _open(self):
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, 'r', swmr=True)
        return self._file
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        f = self._open()
        idx = int(self.indices[i])
        img = f['input_images'][idx][...]   # uint8 (256,256)
        mask = f['target_masks'][idx][...]  # float32 (256,256) in [0,1]
        mask_bin = (mask > 0.0).astype(np.uint8)
        if self.transform is not None:
            t = self.transform(image=img, mask=mask_bin)
            img, mask_bin = t['image'], t['mask']
        img_t = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        mask_t = torch.from_numpy(mask_bin.astype(np.float32))
        return {'image': img_t, 'mask': mask_t}

train_aug = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)])

with h5py.File(FINETUNE_H5, 'r') as f:
    N_TOTAL = f['input_images'].shape[0]
perm = np.random.RandomState(SEED).permutation(N_TOTAL)
n_test = int(round(N_TOTAL * TEST_FRAC))
n_val  = int(round(N_TOTAL * VAL_FRAC))
test_idx  = perm[:n_test]
val_idx   = perm[n_test:n_test+n_val]
train_idx = perm[n_test+n_val:]
print(f'south pole tiles: total {N_TOTAL}, train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)}')

train_ds = SouthPoleCraterDataset(FINETUNE_H5, indices=train_idx, transform=train_aug)
val_ds   = SouthPoleCraterDataset(FINETUNE_H5, indices=val_idx,   transform=None)
test_ds  = SouthPoleCraterDataset(FINETUNE_H5, indices=test_idx,  transform=None)"""

    loaders = """def worker_init_fn(worker_id):
    s = SEED + worker_id; np.random.seed(s); random.seed(s)
g = torch.Generator(); g.manual_seed(SEED)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          worker_init_fn=worker_init_fn, generator=g, drop_last=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)"""

    model_block = """# ================== MODEL (load v1 weights) ==================
model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights=None,
    in_channels=1,
    classes=1,
).to(device)

ckpt = torch.load(V1_CKPT, map_location=device, weights_only=False)
state_key = 'model' if 'model' in ckpt else ('model_state_dict' if 'model_state_dict' in ckpt else None)
if state_key is None:
    raise RuntimeError(f'unknown ckpt keys: {list(ckpt.keys())[:5]}')
model.load_state_dict(ckpt[state_key])
print(f'loaded v1 weights from epoch {ckpt.get("epoch","?")} val_iou {ckpt.get("val_iou","?")}')

dice_loss = smp.losses.DiceLoss(mode='binary')
bce_loss = nn.BCEWithLogitsLoss()
def criterion(logits, target):
    t = target.unsqueeze(1)
    return 0.5 * dice_loss(logits, t) + 0.5 * bce_loss(logits, t)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=ETA_MIN)
scaler = GradScaler('cuda')"""

    metrics = """@torch.no_grad()
def iou_dice_from_logits(logits, target, thresh=0.5, eps=1e-7):
    pred = (torch.sigmoid(logits).squeeze(1) > thresh).float()
    tgt = target.float()
    inter = (pred * tgt).sum(dim=(1,2))
    union = pred.sum(dim=(1,2)) + tgt.sum(dim=(1,2)) - inter
    iou = (inter + eps) / (union + eps)
    dice = (2 * inter + eps) / (pred.sum(dim=(1,2)) + tgt.sum(dim=(1,2)) + eps)
    return iou, dice

@torch.no_grad()
def evaluate(model, loader, tta=False, thresh=0.5):
    model.eval()
    ious, dices = [], []
    for batch in loader:
        x = batch['image'].to(device, non_blocking=True)
        y = batch['mask'].to(device, non_blocking=True)
        with autocast('cuda'):
            if not tta:
                logits = model(x)
            else:
                probs = torch.zeros_like(x)
                for hflip in (False, True):
                    for vflip in (False, True):
                        xi = x
                        if hflip: xi = torch.flip(xi, dims=[-1])
                        if vflip: xi = torch.flip(xi, dims=[-2])
                        li = model(xi); pi = torch.sigmoid(li)
                        if vflip: pi = torch.flip(pi, dims=[-2])
                        if hflip: pi = torch.flip(pi, dims=[-1])
                        probs = probs + pi
                probs = probs / 4.0
                logits = torch.log(probs.clamp(1e-7, 1-1e-7) / (1 - probs.clamp(1e-7, 1-1e-7)))
        iou, dice = iou_dice_from_logits(logits.float(), y, thresh=thresh)
        ious.append(iou.cpu()); dices.append(dice.cpu())
    return float(torch.cat(ious).mean()), float(torch.cat(dices).mean())

# Pre-finetune baseline on val (v1 weights, fresh eval on south pole)
pre_iou, pre_dice = evaluate(model, val_loader, tta=False, thresh=0.5)
print(f'PRE-FT  val_iou {pre_iou:.4f}  val_dice {pre_dice:.4f}  (v1 zero-shot on south pole)')"""

    train_loop = """best_val_iou = -1.0
best_val_dice = -1.0
best_epoch = -1
history = []
t_train0 = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    ep_loss, n = 0.0, 0
    t0 = time.time()
    for batch in train_loader:
        x = batch['image'].to(device, non_blocking=True)
        y = batch['mask'].to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with autocast('cuda'):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        ep_loss += float(loss.item()) * x.size(0); n += x.size(0)
    sched.step()
    train_loss = ep_loss / max(n, 1)

    val_iou, val_dice = evaluate(model, val_loader, tta=False, thresh=0.5)
    dt = time.time() - t0
    lr_now = opt.param_groups[0]['lr']
    print(f'epoch {epoch:02d}/{EPOCHS}  loss {train_loss:.4f}  val_iou {val_iou:.4f}  val_dice {val_dice:.4f}  lr {lr_now:.2e}  ({dt/60:.1f}m)')
    history.append({'epoch': epoch, 'train_loss': train_loss, 'val_iou': val_iou, 'val_dice': val_dice, 'lr': lr_now})

    if val_iou > best_val_iou:
        best_val_iou = val_iou
        best_val_dice = val_dice
        best_epoch = epoch
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_iou': val_iou, 'val_dice': val_dice, 'seed': SEED,
                    'base': 'v1 crater ft on south pole'}, CKPT_PATH)
        print(f'  -> saved new best to {CKPT_PATH}')

print(f'fine-tune done in {(time.time()-t_train0)/60:.1f} min. best val_iou {best_val_iou:.4f} @ epoch {best_epoch}')"""

    final_eval = """# Final test eval with thresh sweep
best = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model.load_state_dict(best['model'])

thresh_results = {}
for tt, name in [(False, 'no_tta'), (True, 'flip_tta')]:
    for t in (0.5, 0.3, 0.2, 0.1, 0.05):
        iou, dice = evaluate(model, test_loader, tta=tt, thresh=t)
        thresh_results[f'{name}_t{t}'] = {'iou': iou, 'dice': dice}
        print(f'test {name:<8}  thr {t:<5}  iou {iou:.4f}  dice {dice:.4f}')

summary = {
    'seed': SEED,
    'base_model': 'crater_v1_seed1',
    'pre_ft_val_iou': pre_iou,
    'pre_ft_val_dice': pre_dice,
    'best_val_iou': best_val_iou,
    'best_val_dice': best_val_dice,
    'best_epoch': best_epoch,
    'test_thresh_sweep': thresh_results,
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'lr': LR,
    'wd': WD,
    'history': history,
}
with open(SUMMARY_PATH, 'w') as f:
    json.dump(summary, f, indent=2)
print(json.dumps({k: v for k, v in summary.items() if k not in ('history','test_thresh_sweep')}, indent=2))
print('summary ->', SUMMARY_PATH)
print('ckpt    ->', CKPT_PATH)"""

    return {
        "cells": [
            _md(header),
            _code(install),
            _code(imports),
            _md("## Config"),
            _code(config),
            _md("## Dataset (south pole HDF5 + 80/10/10 split)"),
            _code(dataset),
            _md("## DataLoaders"),
            _code(loaders),
            _md("## Load v1 weights + set up optimizer"),
            _code(model_block),
            _md("## Metrics + zero-shot baseline"),
            _code(metrics),
            _md("## Fine-tune training loop"),
            _code(train_loop),
            _md("## Final evaluation (threshold sweep)"),
            _code(final_eval),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    print(f"wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
