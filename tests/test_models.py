"""Tests for model construction, loss functions, and metrics."""

import torch
import pytest

from lunarsite.models.unet import build_unet, build_loss
from lunarsite.utils.metrics import iou_score, dice_score


class TestUNet:
    """Test U-Net model building and inference."""

    def test_unet_builds_with_defaults(self):
        model = build_unet()
        assert model is not None
        assert sum(p.numel() for p in model.parameters()) > 0

    def test_unet_forward_shape(self):
        model = build_unet(classes=4)
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        assert out.shape == (2, 4, 64, 64)

    def test_unet_single_channel_input(self):
        model = build_unet(in_channels=1, classes=1)
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        assert out.shape == (2, 1, 64, 64)


class TestLoss:
    """Test loss function construction."""

    def test_dice_ce_loss(self):
        loss_fn = build_loss("dice_ce")
        pred = torch.randn(2, 4, 32, 32, requires_grad=True)
        target = torch.randint(0, 4, (2, 32, 32))
        loss = loss_fn(pred, target)
        assert loss.item() > 0

    def test_dice_bce_loss(self):
        loss_fn = build_loss("dice_bce")
        pred = torch.randn(2, 1, 32, 32)
        target = torch.rand(2, 1, 32, 32)
        loss = loss_fn(pred, target)
        assert loss.item() > 0

    def test_invalid_loss_type(self):
        with pytest.raises(ValueError):
            build_loss("invalid")


class TestMetrics:
    """Test IoU and Dice metrics."""

    def test_perfect_iou(self):
        pred = torch.tensor([[[0, 1], [2, 3]]])
        target = torch.tensor([[[0, 1], [2, 3]]])
        result = iou_score(pred, target, 4)
        assert result["mean_iou"] == 1.0

    def test_zero_iou(self):
        pred = torch.zeros(1, 4, 4, dtype=torch.long)
        target = torch.ones(1, 4, 4, dtype=torch.long)
        result = iou_score(pred, target, 4)
        assert result["per_class_iou"][0] == 0.0
        assert result["per_class_iou"][1] == 0.0

    def test_perfect_dice(self):
        pred = torch.tensor([[[0, 1], [2, 3]]])
        target = torch.tensor([[[0, 1], [2, 3]]])
        result = dice_score(pred, target, 4)
        assert result["mean_dice"] == 1.0

    def test_iou_with_missing_class(self):
        pred = torch.zeros(1, 4, 4, dtype=torch.long)
        target = torch.zeros(1, 4, 4, dtype=torch.long)
        result = iou_score(pred, target, 4)
        # Class 0 has IoU=1.0, classes 1-3 are NaN (absent from both)
        assert result["per_class_iou"][0] == 1.0
        assert result["mean_iou"] == 1.0  # mean of valid only
