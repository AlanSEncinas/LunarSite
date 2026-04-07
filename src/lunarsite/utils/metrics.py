"""Segmentation metrics: IoU, Dice, and per-class variants."""

import torch


def iou_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> dict:
    """Compute per-class and mean Intersection over Union.

    Args:
        pred: Predicted class indices (N, H, W).
        target: Ground truth class indices (N, H, W).
        num_classes: Total number of classes.

    Returns:
        Dict with 'per_class_iou' (list of floats) and 'mean_iou' (float).
    """
    per_class_iou = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        if union == 0:
            per_class_iou.append(float("nan"))
        else:
            per_class_iou.append((intersection / union).item())

    valid = [v for v in per_class_iou if v == v]  # filter NaN
    mean_iou = sum(valid) / len(valid) if valid else 0.0
    return {"per_class_iou": per_class_iou, "mean_iou": mean_iou}


def dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> dict:
    """Compute per-class and mean Dice coefficient.

    Args:
        pred: Predicted class indices (N, H, W).
        target: Ground truth class indices (N, H, W).
        num_classes: Total number of classes.

    Returns:
        Dict with 'per_class_dice' (list of floats) and 'mean_dice' (float).
    """
    per_class_dice = []
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        total = pred_c.sum() + target_c.sum()
        if total == 0:
            per_class_dice.append(float("nan"))
        else:
            per_class_dice.append((2.0 * intersection / total).item())

    valid = [v for v in per_class_dice if v == v]
    mean_dice = sum(valid) / len(valid) if valid else 0.0
    return {"per_class_dice": per_class_dice, "mean_dice": mean_dice}
