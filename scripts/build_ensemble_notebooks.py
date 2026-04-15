"""Generate the ensemble training notebooks for seeds 2-5.

Builds `notebooks/train_ensemble_seed{N}_kaggle.ipynb` for N in {2,3,4,5}.
Also builds a template `notebooks/train_ensemble_kaggle.ipynb` (SEED=2) for reference.

All notebooks are identical except for the SEED constant. Configuration matches
the v1 production winner:
  - U-Net + ResNet-34 encoder (imagenet weights)
  - Dice + CE loss (0.5 * DiceLoss(multiclass) + 0.5 * CE)
  - AdamW, lr=1e-4, wd=1e-4, CosineAnnealingLR(T_max=50)
  - 480x480 input, batch_size=8, 50 epochs
  - v1 lunar augmentations (shadow rotation, extreme contrast, Hapke BRDF)
  - Fixed SPLIT_SEED=42 so the train/val/test split never varies across seeds.

Re-run this script whenever the training recipe changes.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = REPO_ROOT / "notebooks"

SEEDS = [2, 3, 4, 5]


def _code(src: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def _md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def build_cells(seed: int) -> list[dict]:
    header = f"""# LunarSite Stage 2 Ensemble Member - Seed {seed}

Deep ensemble member for epistemic uncertainty on Stage 2 segmentation.

**Config matches v1 production winner** (test mIoU 0.8456):
- U-Net + ResNet-34 (imagenet) encoder
- Dice + CE loss
- AdamW lr=1e-4, wd=1e-4, cosine annealing
- 480x480, batch_size=8, 50 epochs, mixed precision
- v1 lunar augmentations

Only the training stochasticity (weight init, DataLoader shuffle, aug RNG) varies
across ensemble members. Data split is held fixed by SPLIT_SEED=42.

**This notebook uses SEED = {seed}.**

Runs on Kaggle T4 x2. Checkpoint saves to `/kaggle/working/best_resnet34_seed{seed}.pt`.
"""

    install = """# Kaggle assigns P100 (sm_60) by default for `enable_gpu=True`; torch 2.6+ dropped
# sm_60 support -> 'no kernel image available'. Pin torch 2.5 which still has sm_60,
# and that wheel also covers T4 (sm_75) if Kaggle upgrades us.
!pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
!pip install -q segmentation-models-pytorch albumentations kagglehub psutil
import torch; print(f'torch {torch.__version__} | cuda {torch.version.cuda} | arch list: {torch.cuda.get_arch_list() if torch.cuda.is_available() else \"cpu\"}')
"""

    imports = f"""import gc
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import cv2
import kagglehub
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import psutil
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ============================================================
# SEED CONFIG -- the only thing that varies between ensemble members
# ============================================================
SEED = {seed}          # per-member: weight init, DataLoader shuffle, aug RNG
SPLIT_SEED = 42    # FIXED across all members: keeps train/val/test split identical

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {{device}} | SEED={{SEED}} | SPLIT_SEED={{SPLIT_SEED}}')
if device.type == 'cuda':
    print(f'GPU: {{torch.cuda.get_device_name(0)}}')
    print(f'VRAM: {{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}} GB')
print(f'RAM: {{psutil.virtual_memory().total / 1e9:.1f}} GB')

OUT_DIR = '/kaggle/working'
os.makedirs(OUT_DIR, exist_ok=True)

TAG = f'resnet34_seed{{SEED}}'
"""

    safety = """def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, hist, tag):
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_metric': best_metric,
        'history': hist,
        'tag': tag,
        'seed': SEED,
        'split_seed': SPLIT_SEED,
        'timestamp': datetime.now().isoformat(),
    }
    torch.save(ckpt, f'{OUT_DIR}/{tag}_latest.pt')
    with open(f'{OUT_DIR}/{tag}_history.json', 'w') as f:
        json.dump({'epoch': epoch, 'best_metric': best_metric, 'history': hist,
                   'seed': SEED, 'split_seed': SPLIT_SEED}, f, indent=2)


def save_best(model, epoch, best_metric, tag):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'best_metric': best_metric,
        'tag': tag,
        'seed': SEED,
        'split_seed': SPLIT_SEED,
    }, f'{OUT_DIR}/best_{tag}.pt')


def load_checkpoint(tag):
    path = f'{OUT_DIR}/{tag}_latest.pt'
    if os.path.exists(path):
        print(f'Found checkpoint: {path}')
        return torch.load(path, map_location=device, weights_only=False)
    return None


def seed_worker(worker_id):
    # Each DataLoader worker gets a deterministic seed derived from SEED.
    worker_seed = (SEED * 1000 + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


print('Safety utilities ready.')
"""

    dataset = """dataset_path = Path(kagglehub.dataset_download('romainpessia/artificial-lunar-rocky-landscape-dataset'))
image_dir = dataset_path / 'images' / 'render'
mask_dir = dataset_path / 'images' / 'clean'
print(f'Dataset: {dataset_path}')
print(f'Images: {len(list(image_dir.glob("render*.png")))}')
"""

    constants = """COLOR_TO_CLASS = {(0,0,0): 0, (255,0,0): 1, (0,255,0): 2, (0,0,255): 3}
CLASS_COLORS = np.array([[0,0,0],[255,165,0],[255,0,0],[135,206,235]], dtype=np.uint8)
CLASS_NAMES = ['background', 'small_rocks', 'large_rocks', 'sky']
NC = 4
INPUT_SIZE = 480
"""

    augs = """class LunarShadowRotation(ImageOnlyTransform):
    def __init__(self, angle_range=(-45, 45), p=0.5, **kwargs):
        super().__init__(p=p, **kwargs)
        self.angle_range = angle_range
    def apply(self, img, angle=0, **p):
        gray = np.mean(img.astype(np.float32), axis=-1)
        sh = gray < 25
        if sh.sum() < 100: return img
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rot = cv2.warpAffine(sh.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST).astype(bool)
        r = img.copy()
        r[rot] = np.clip(r[rot] * 0.1, 0, 255).astype(np.uint8)
        r[sh & ~rot] = np.clip(r[sh & ~rot] * 3.0 + 30, 0, 200).astype(np.uint8)
        return r
    def get_params(self): return {'angle': np.random.uniform(*self.angle_range)}
    def get_transform_init_args_names(self): return ('angle_range',)


class ExtremeContrast(ImageOnlyTransform):
    def __init__(self, p=0.5, **kwargs):
        super().__init__(p=p, **kwargs)
    def apply(self, img, sf=0.05, bf=1.5, **p):
        g = np.mean(img.astype(np.float32), axis=-1)
        med = np.median(g)
        r = img.astype(np.float32)
        r[g < med * 0.5] *= sf
        r[g > med * 1.5] *= bf
        return np.clip(r, 0, 255).astype(np.uint8)
    def get_params(self):
        return {'sf': np.random.uniform(0.01, 0.15), 'bf': np.random.uniform(1.2, 2.5)}
    def get_transform_init_args_names(self): return ()


class HapkeBRDF(ImageOnlyTransform):
    def __init__(self, p=0.4, **kwargs):
        super().__init__(p=p, **kwargs)
    def apply(self, img, af=1.0, pf=1.0, **p):
        r = img.astype(np.float32) * af
        g = np.mean(r, axis=-1, keepdims=True)
        n = g / (g.max() + 1e-8)
        r *= (1 - n * (1 - n) * 4 * (1 - pf))
        return np.clip(r, 0, 255).astype(np.uint8)
    def get_params(self):
        return {'af': np.random.uniform(0.5, 1.5), 'pf': np.random.uniform(0.7, 1.3)}
    def get_transform_init_args_names(self): return ()


def get_transforms(sz, train):
    if train:
        return A.Compose([
            A.RandomCrop(sz, sz),
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.2), contrast_limit=(-0.2, 0.4), p=0.4),
            A.GaussNoise(p=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.3),
            LunarShadowRotation(p=0.3), ExtremeContrast(p=0.3), HapkeBRDF(p=0.3),
        ])
    return A.Compose([A.CenterCrop(sz, sz)])


class LunarDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(str(self.image_paths[idx])), cv2.COLOR_BGR2RGB)
        msk = cv2.cvtColor(cv2.imread(str(self.mask_paths[idx])), cv2.COLOR_BGR2RGB)
        h, w = msk.shape[:2]
        mask = np.zeros((h, w), dtype=np.int64)
        for color, cls in COLOR_TO_CLASS.items():
            mask[np.all(msk == color, axis=-1)] = cls
        if self.transform:
            t = self.transform(image=img, mask=mask)
            img, mask = t['image'], t['mask']
        return {
            'image': torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255.0),
            'mask': torch.from_numpy(mask.astype(np.int64)),
        }

print('Augmentations + Dataset ready.')
"""

    split = """# FIXED split across all ensemble members -- uses SPLIT_SEED, not SEED
all_imgs = sorted(image_dir.glob('render*.png'))
all_masks = sorted(mask_dir.glob('clean*.png'))
n = len(all_imgs)
perm = np.random.RandomState(SPLIT_SEED).permutation(n).tolist()
nt, nv = int(n * 0.8), int(n * 0.1)
train_idx, val_idx, test_idx = perm[:nt], perm[nt:nt+nv], perm[nt+nv:]
print(f'Total: {n} | Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}')
print(f'Split seed: {SPLIT_SEED} (same for all ensemble members)')
"""

    loss = """class DiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='multiclass')
        self.ce = nn.CrossEntropyLoss()
    def forward(self, p, t):
        return 0.5 * self.dice(p, t) + 0.5 * self.ce(p, t)


def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        imgs = batch['image'].to(device)
        masks = batch['mask'].to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            logits = model(imgs)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, nc):
    model.eval()
    total_loss, n = 0.0, 0
    inter = torch.zeros(nc)
    uni = torch.zeros(nc)
    di = torch.zeros(nc)
    ds = torch.zeros(nc)
    for batch in loader:
        imgs = batch['image'].to(device)
        masks = batch['mask'].to(device)
        with torch.amp.autocast('cuda'):
            logits = model(imgs)
        total_loss += criterion(logits, masks).item() * imgs.size(0)
        n += imgs.size(0)
        preds = logits.argmax(1).cpu()
        mc = masks.cpu()
        for c in range(nc):
            pc, tc = (preds == c), (mc == c)
            inter[c] += (pc & tc).sum().float()
            uni[c] += (pc | tc).sum().float()
            di[c] += (pc.float() * tc.float()).sum()
            ds[c] += pc.float().sum() + tc.float().sum()
    piou = [(inter[c] / uni[c]).item() if uni[c] > 0 else float('nan') for c in range(nc)]
    pdice = [(2 * di[c] / ds[c]).item() if ds[c] > 0 else float('nan') for c in range(nc)]
    vi = [x for x in piou if x == x]
    vd = [x for x in pdice if x == x]
    return {
        'loss': total_loss / n,
        'per_class_iou': piou,
        'mean_iou': sum(vi) / len(vi) if vi else 0,
        'per_class_dice': pdice,
        'mean_dice': sum(vd) / len(vd) if vd else 0,
    }

print('Loss + training functions ready.')
"""

    training = """def run_training(model, train_ds, val_ds, epochs, lr, wd, tag, batch_size=8, num_workers=2):
    criterion = DiceCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    scaler = torch.amp.GradScaler('cuda')

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              generator=g, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            worker_init_fn=seed_worker)

    best = 0.0
    start_epoch = 1
    hist = {'train_loss': [], 'val_loss': [], 'val_miou': [], 'val_dice': [], 'lr': []}

    ckpt = load_checkpoint(tag)
    if ckpt is not None:
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best = ckpt['best_metric']
        hist = ckpt['history']
        print(f'RESUMED {tag} from epoch {ckpt["epoch"]} (best mIoU: {best:.4f})')
        if start_epoch > epochs:
            print(f'{tag} already completed.')
            return hist, best

    print(f'\\n{"="*80}')
    print(f'Training {tag} | epochs {start_epoch}-{epochs} | lr={lr} | batch={batch_size}')
    print(f'Loss: 0.5*Dice + 0.5*CE  (v1 config, ensemble member seed={SEED})')
    print(f'{"-"*80}')

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        tl = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val = evaluate(model, val_loader, criterion, NC)
        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']
        dt = time.time() - t0

        hist['train_loss'].append(tl)
        hist['val_loss'].append(val['loss'])
        hist['val_miou'].append(val['mean_iou'])
        hist['val_dice'].append(val['mean_dice'])
        hist['lr'].append(lr_now)

        iou_str = ' | '.join(
            f"{CLASS_NAMES[i]}: {v:.3f}" if v == v else f"{CLASS_NAMES[i]}: N/A"
            for i, v in enumerate(val['per_class_iou'])
        )
        print(f'Epoch {epoch:3d}/{epochs} | train: {tl:.4f} | val: {val["loss"]:.4f} | '
              f'mIoU: {val["mean_iou"]:.4f} | Dice: {val["mean_dice"]:.4f} | '
              f'lr: {lr_now:.2e} | {dt:.0f}s')
        print(f'         {iou_str}', flush=True)

        if val['mean_iou'] > best:
            best = val['mean_iou']
            save_best(model, epoch, best, tag)
            print(f'  -> new best mIoU {best:.4f} saved')

        save_checkpoint(model, optimizer, scheduler, epoch, best, hist, tag)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return hist, best

print('Training loop ready.')
"""

    build_and_train = """train_set = LunarDataset(
    [all_imgs[i] for i in train_idx], [all_masks[i] for i in train_idx],
    get_transforms(INPUT_SIZE, True))
val_set = LunarDataset(
    [all_imgs[i] for i in val_idx], [all_masks[i] for i in val_idx],
    get_transforms(INPUT_SIZE, False))
test_set = LunarDataset(
    [all_imgs[i] for i in test_idx], [all_masks[i] for i in test_idx],
    get_transforms(INPUT_SIZE, False))

# v1 model: U-Net + ResNet-34 imagenet encoder
model = smp.Unet(
    encoder_name='resnet34', encoder_weights='imagenet',
    in_channels=3, classes=NC,
).to(device)
print(f'ResNet-34 U-Net: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} params')

hist, best = run_training(
    model, train_set, val_set,
    epochs=50, lr=1e-4, wd=1e-4, tag=TAG,
    batch_size=8, num_workers=2,
)
"""

    tta_eval = """# --- Flip TTA evaluation on the test split ---
@torch.no_grad()
def predict_flip_tta(model, img_t):
    model.eval()
    x = img_t.unsqueeze(0).to(device)
    augmented = [x, torch.flip(x, [3]), torch.flip(x, [2]), torch.flip(x, [2, 3])]
    probs_sum = None
    for i, aug_x in enumerate(augmented):
        with torch.amp.autocast('cuda'):
            logits = model(aug_x)
        p = F.softmax(logits, dim=1)
        if i == 1: p = torch.flip(p, [3])
        elif i == 2: p = torch.flip(p, [2])
        elif i == 3: p = torch.flip(p, [2, 3])
        probs_sum = p if probs_sum is None else probs_sum + p
    return (probs_sum / len(augmented)).argmax(1).squeeze(0).cpu().numpy()


@torch.no_grad()
def evaluate_tta(model, loader, nc):
    model.eval()
    inter = torch.zeros(nc)
    uni = torch.zeros(nc)
    for batch in loader:
        imgs, masks = batch['image'], batch['mask']
        for j in range(imgs.size(0)):
            pred = predict_flip_tta(model, imgs[j])
            pred_t = torch.from_numpy(pred)
            gt = masks[j]
            for c in range(nc):
                pc, tc = (pred_t == c), (gt == c)
                inter[c] += (pc & tc).sum().float()
                uni[c] += (pc | tc).sum().float()
    piou = [(inter[c] / uni[c]).item() if uni[c] > 0 else float('nan') for c in range(nc)]
    vi = [x for x in piou if x == x]
    return {'per_class_iou': piou, 'mean_iou': sum(vi) / len(vi) if vi else 0}


# Load best checkpoint, evaluate on test set (standard + flip TTA)
ckpt = torch.load(f'{OUT_DIR}/best_{TAG}.pt', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
print(f'Loaded best checkpoint: epoch {ckpt["epoch"]}, val mIoU {ckpt["best_metric"]:.4f}')

test_loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=0, pin_memory=False)
criterion = DiceCELoss()
test_std = evaluate(model, test_loader, criterion, NC)
print(f'\\nTest (standard) mIoU: {test_std["mean_iou"]:.4f}  Dice: {test_std["mean_dice"]:.4f}')

test_tta = evaluate_tta(model, test_loader, NC)
print(f'Test (flip TTA) mIoU: {test_tta["mean_iou"]:.4f}')

summary = {
    'seed': SEED,
    'split_seed': SPLIT_SEED,
    'tag': TAG,
    'best_val_miou': round(best, 4),
    'best_epoch': ckpt['epoch'],
    'test_standard_miou': round(test_std['mean_iou'], 4),
    'test_standard_dice': round(test_std['mean_dice'], 4),
    'test_tta_miou': round(test_tta['mean_iou'], 4),
    'test_per_class_iou_standard': {CLASS_NAMES[i]: round(v, 4) if v == v else None
                                     for i, v in enumerate(test_std['per_class_iou'])},
    'test_per_class_iou_tta': {CLASS_NAMES[i]: round(v, 4) if v == v else None
                                for i, v in enumerate(test_tta['per_class_iou'])},
}
with open(f'{OUT_DIR}/{TAG}_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\\nSummary saved to {OUT_DIR}/{TAG}_summary.json')
print(f'Best checkpoint: {OUT_DIR}/best_{TAG}.pt')
"""

    cells = [
        _md(header),
        _code(install),
        _code(imports),
        _md("## Safety utilities + deterministic DataLoader worker"),
        _code(safety),
        _md("## Dataset download"),
        _code(dataset),
        _code(constants),
        _md("## Lunar augmentations + Dataset class"),
        _code(augs),
        _md("## Fixed train/val/test split (SPLIT_SEED)"),
        _code(split),
        _md("## Dice + CE loss + eval helpers"),
        _code(loss),
        _md("## Training loop"),
        _code(training),
        _md(f"## Train ensemble member (seed {seed})"),
        _code(build_and_train),
        _md("## Final test evaluation (standard + flip TTA)"),
        _code(tta_eval),
    ]
    return cells


def build_notebook(seed: int) -> dict:
    return {
        "cells": build_cells(seed),
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    # Reference notebook (seed=2) so the CLAUDE.md-specified filename exists.
    ref_path = NOTEBOOK_DIR / "train_ensemble_kaggle.ipynb"
    ref_path.write_text(json.dumps(build_notebook(2), indent=1), encoding="utf-8")
    print(f"wrote {ref_path}")
    for seed in SEEDS:
        path = NOTEBOOK_DIR / f"train_ensemble_seed{seed}_kaggle.ipynb"
        path.write_text(json.dumps(build_notebook(seed), indent=1), encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
