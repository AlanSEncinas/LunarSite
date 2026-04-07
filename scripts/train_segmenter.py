#!/usr/bin/env python3
"""Train Stage 2: Terrain Hazard Segmentation model.

Usage:
    python scripts/train_segmenter.py --config configs/stage2_segmentation.yaml
"""

import argparse
import json
import time
from pathlib import Path

import albumentations as A
import kagglehub
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from lunarsite.data.lunar_dataset import LunarTerrainDataset
from lunarsite.models.unet import build_unet, build_loss
from lunarsite.utils.metrics import iou_score, dice_score

CLASS_NAMES = ["background", "small_rocks", "large_rocks", "sky"]


def get_transforms(input_size: int, training: bool) -> A.Compose:
    """Build albumentations transform pipeline."""
    if training:
        return A.Compose([
            A.RandomCrop(height=input_size, width=input_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
        ])
    else:
        return A.Compose([
            A.CenterCrop(height=input_size, width=input_size),
        ])


def split_indices(
    n: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Split indices into train/val/test."""
    indices = np.random.RandomState(seed).permutation(n).tolist()
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return (
        indices[:n_train],
        indices[n_train : n_train + n_val],
        indices[n_train + n_val :],
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict:
    """Evaluate model, return loss and metrics."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item() * images.size(0)

        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    iou = iou_score(all_preds, all_targets, num_classes)
    dice = dice_score(all_preds, all_targets, num_classes)

    return {
        "loss": total_loss / len(loader.dataset),
        "mean_iou": iou["mean_iou"],
        "per_class_iou": iou["per_class_iou"],
        "mean_dice": dice["mean_dice"],
        "per_class_dice": dice["per_class_dice"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train terrain segmentation model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage2_segmentation.yaml"),
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit total dataset size (for quick CPU experiments).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    data_cfg = config["data"]
    dataset_path = Path(
        kagglehub.dataset_download("romainpessia/artificial-lunar-rocky-landscape-dataset")
    )
    image_dir = dataset_path / "images" / "render"
    mask_dir = dataset_path / "images" / "clean"

    input_size = data_cfg["input_size"]
    train_transforms = get_transforms(input_size, training=True)
    val_transforms = get_transforms(input_size, training=False)

    full_dataset = LunarTerrainDataset(image_dir, mask_dir)

    # Optionally limit dataset size for fast CPU experiments
    n_total = len(full_dataset)
    if args.max_samples and args.max_samples < n_total:
        pool = np.random.RandomState(data_cfg["seed"]).permutation(n_total)[:args.max_samples].tolist()
    else:
        pool = list(range(n_total))

    train_idx, val_idx, test_idx = split_indices(
        len(pool),
        data_cfg["train_split"],
        data_cfg["val_split"],
        data_cfg["seed"],
    )
    # Map back to original dataset indices
    train_idx = [pool[i] for i in train_idx]
    val_idx = [pool[i] for i in val_idx]
    test_idx = [pool[i] for i in test_idx]

    # Create datasets with appropriate transforms
    train_ds = LunarTerrainDataset(image_dir, mask_dir, transform=train_transforms)
    val_ds = LunarTerrainDataset(image_dir, mask_dir, transform=val_transforms)
    test_ds = LunarTerrainDataset(image_dir, mask_dir, transform=val_transforms)
    train_set = Subset(train_ds, train_idx)
    val_set = Subset(val_ds, val_idx)
    test_set = Subset(test_ds, test_idx)

    train_cfg = config["training"]
    batch_size = train_cfg["batch_size"]
    use_cuda = device.type == "cuda"
    num_workers = 2 if use_cuda else 0
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_cuda)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_cuda)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_cuda)

    print(f"Dataset: {len(pool)} images (of {n_total} total)")
    print(f"  Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    # --- Model ---
    model_cfg = config["model"]
    model = build_unet(
        encoder_name=model_cfg["encoder"],
        encoder_weights=model_cfg["encoder_weights"],
        in_channels=model_cfg["in_channels"],
        classes=model_cfg["classes"],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: U-Net + {model_cfg['encoder']} ({param_count:,} params)")

    # --- Loss, Optimizer, Scheduler ---
    loss_cfg = config["loss"]
    criterion = build_loss(loss_cfg["type"], **{k: v for k, v in loss_cfg.items() if k != "type"})

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    scheduler_params = train_cfg.get("scheduler_params", {})
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=scheduler_params.get("T_max", train_cfg["epochs"]),
        eta_min=scheduler_params.get("eta_min", 1e-6),
    )

    # --- Output dirs ---
    output_cfg = config["output"]
    checkpoint_dir = Path(output_cfg["checkpoint_dir"])
    log_dir = Path(output_cfg["log_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    num_classes = data_cfg["num_classes"]
    epochs = train_cfg["epochs"]
    best_metric = 0.0
    training_log = []

    print(f"\nTraining for {epochs} epochs...")
    print("-" * 80)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device, num_classes)
        scheduler.step()

        epoch_time = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_metrics["loss"], 5),
            "val_mean_iou": round(val_metrics["mean_iou"], 4),
            "val_mean_dice": round(val_metrics["mean_dice"], 4),
            "val_per_class_iou": [round(v, 4) if v == v else None for v in val_metrics["per_class_iou"]],
            "lr": lr,
            "time_s": round(epoch_time, 1),
        }
        training_log.append(log_entry)

        # Print progress
        iou_str = " | ".join(
            f"{CLASS_NAMES[i]}: {v:.3f}" if v == v else f"{CLASS_NAMES[i]}: N/A"
            for i, v in enumerate(val_metrics["per_class_iou"])
        )
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_metrics['loss']:.4f} | "
            f"mIoU: {val_metrics['mean_iou']:.4f} | "
            f"Dice: {val_metrics['mean_dice']:.4f} | "
            f"lr: {lr:.2e} | "
            f"{epoch_time:.0f}s"
        )
        print(f"         IoU: {iou_str}")

        # Save best model
        if val_metrics["mean_iou"] > best_metric:
            best_metric = val_metrics["mean_iou"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_metric,
                    "config": config,
                },
                checkpoint_dir / "best_segmenter.pt",
            )
            print(f"         ** New best mIoU: {best_metric:.4f} — saved checkpoint **")

    # --- Save training log ---
    with open(log_dir / "segmenter_training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"\nTraining log saved to {log_dir / 'segmenter_training_log.json'}")

    # --- Final evaluation on test set ---
    print("\n" + "=" * 80)
    print("Evaluating on test set...")
    checkpoint = torch.load(checkpoint_dir / "best_segmenter.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader, criterion, device, num_classes)
    print(f"Test Loss:  {test_metrics['loss']:.4f}")
    print(f"Test mIoU:  {test_metrics['mean_iou']:.4f}")
    print(f"Test Dice:  {test_metrics['mean_dice']:.4f}")
    print("Per-class IoU:")
    for i, v in enumerate(test_metrics["per_class_iou"]):
        val_str = f"{v:.4f}" if v == v else "N/A"
        print(f"  {CLASS_NAMES[i]:15s}: {val_str}")

    # Save test results
    test_results = {
        "test_loss": round(test_metrics["loss"], 5),
        "test_mean_iou": round(test_metrics["mean_iou"], 4),
        "test_mean_dice": round(test_metrics["mean_dice"], 4),
        "test_per_class_iou": {
            CLASS_NAMES[i]: round(v, 4) if v == v else None
            for i, v in enumerate(test_metrics["per_class_iou"])
        },
        "test_per_class_dice": {
            CLASS_NAMES[i]: round(v, 4) if v == v else None
            for i, v in enumerate(test_metrics["per_class_dice"])
        },
        "best_epoch": checkpoint["epoch"],
    }
    with open(log_dir / "segmenter_test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"\nTest results saved to {log_dir / 'segmenter_test_results.json'}")


if __name__ == "__main__":
    main()
