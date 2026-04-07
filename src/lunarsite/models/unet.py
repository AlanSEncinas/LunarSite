"""U-Net model wrapper around segmentation_models_pytorch.

Used by both Stage 1 (crater detection) and Stage 2 (terrain segmentation).
Loads pretrained encoder weights and exposes a consistent interface for
training scripts.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_unet(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    classes: int = 4,
) -> smp.Unet:
    """Build a U-Net model with a pretrained encoder.

    Args:
        encoder_name: Backbone architecture (e.g., 'resnet34', 'efficientnet-b0').
        encoder_weights: Pretrained weights to load (e.g., 'imagenet', None).
        in_channels: Number of input channels (3 for RGB, 1 for DEM).
        classes: Number of output segmentation classes.

    Returns:
        A segmentation_models_pytorch.Unet instance.
    """
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )


class DiceCELoss(nn.Module):
    """Combined Dice + Cross-Entropy loss for multi-class segmentation."""

    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice = smp.losses.DiceLoss(mode="multiclass")
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.

        Args:
            pred: Model output logits (N, C, H, W).
            target: Ground truth class indices (N, H, W).
        """
        return self.dice_weight * self.dice(pred, target) + self.ce_weight * self.ce(pred, target)


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss for binary segmentation."""

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = smp.losses.DiceLoss(mode="binary")
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.

        Args:
            pred: Model output logits (N, 1, H, W).
            target: Ground truth binary mask (N, 1, H, W) float.
        """
        return self.dice_weight * self.dice(pred, target) + self.bce_weight * self.bce(pred, target)


def build_loss(loss_type: str = "dice_ce", **kwargs) -> nn.Module:
    """Build a combined segmentation loss function.

    Args:
        loss_type: One of 'dice_ce', 'dice_bce'.
        **kwargs: Parameters passed to the loss (dice_weight, ce_weight/bce_weight).

    Returns:
        A callable loss module.
    """
    if loss_type == "dice_ce":
        return DiceCELoss(
            dice_weight=kwargs.get("dice_weight", 0.5),
            ce_weight=kwargs.get("ce_weight", 0.5),
        )
    elif loss_type == "dice_bce":
        return DiceBCELoss(
            dice_weight=kwargs.get("dice_weight", 0.5),
            bce_weight=kwargs.get("bce_weight", 0.5),
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
