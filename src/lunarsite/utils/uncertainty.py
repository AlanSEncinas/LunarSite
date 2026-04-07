"""Uncertainty estimation via MC Dropout for segmentation and scoring.

Provides per-pixel epistemic uncertainty maps that highlight regions where
the model is least confident — typically dark/shadowed areas, ambiguous
terrain boundaries, and out-of-distribution inputs.

References:
    - MC Frequency Dropout (2025): arxiv.org/abs/2501.11258
    - Uncertainty Meets Diversity (CVPR 2025)
    - Bayesian Deep Learning for Landing Safety: arxiv.org/abs/2102.10545
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def enable_mc_dropout(model: nn.Module) -> None:
    """Enable dropout layers during inference for MC sampling.

    By default, dropout is disabled during model.eval(). This function
    re-enables dropout layers while keeping batch norm in eval mode.

    Args:
        model: PyTorch model with dropout layers.
    """
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            module.train()


def add_mc_dropout(model: nn.Module, p: float = 0.1) -> nn.Module:
    """Add Dropout2d layers after ReLU activations in a model.

    For models without built-in dropout (like segmentation_models_pytorch),
    this inserts dropout layers to enable MC Dropout inference.

    Args:
        model: Model to modify in-place.
        p: Dropout probability.

    Returns:
        The modified model.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            # Replace ReLU with ReLU + Dropout2d
            setattr(model, name, nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=p),
            ))
        else:
            add_mc_dropout(module, p)
    return model


@torch.no_grad()
def mc_predict(
    model: nn.Module,
    image: torch.Tensor,
    n_samples: int = 20,
    num_classes: int = 4,
) -> dict:
    """Run MC Dropout inference for uncertainty estimation.

    Performs multiple stochastic forward passes with dropout enabled,
    then computes mean prediction and uncertainty metrics.

    Args:
        model: Segmentation model with dropout layers.
        image: Input tensor (1, C, H, W) or (C, H, W).
        n_samples: Number of MC forward passes.
        num_classes: Number of segmentation classes.

    Returns:
        Dict with:
            'mean_probs': (num_classes, H, W) mean predicted probabilities
            'prediction': (H, W) most likely class
            'entropy': (H, W) predictive entropy (total uncertainty)
            'mutual_info': (H, W) mutual information (epistemic uncertainty)
            'variance': (H, W) mean variance across classes
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)

    device = next(model.parameters()).device
    image = image.to(device)

    # Enable MC dropout
    model.eval()
    enable_mc_dropout(model)

    # Collect stochastic predictions
    all_probs = []
    for _ in range(n_samples):
        logits = model(image)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs.cpu())

    all_probs = torch.stack(all_probs)  # (n_samples, 1, C, H, W)
    all_probs = all_probs.squeeze(1)    # (n_samples, C, H, W)

    # Mean prediction
    mean_probs = all_probs.mean(dim=0)  # (C, H, W)
    prediction = mean_probs.argmax(dim=0)  # (H, W)

    # Predictive entropy: H[y|x] = -sum(p * log(p))
    # Total uncertainty = aleatoric + epistemic
    entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=0)

    # Per-sample entropy: E[H[y|x, w]]
    per_sample_entropy = -(all_probs * torch.log(all_probs + 1e-10)).sum(dim=1)  # (n, H, W)
    mean_sample_entropy = per_sample_entropy.mean(dim=0)

    # Mutual information: I[y, w|x] = H[y|x] - E[H[y|x,w]]
    # Epistemic uncertainty (model uncertainty)
    mutual_info = entropy - mean_sample_entropy

    # Variance of predictions across samples
    variance = all_probs.var(dim=0).mean(dim=0)  # mean across classes

    return {
        "mean_probs": mean_probs.numpy(),
        "prediction": prediction.numpy(),
        "entropy": entropy.numpy(),
        "mutual_info": mutual_info.numpy(),
        "variance": variance.numpy(),
    }


def uncertainty_map_to_rgb(
    uncertainty: np.ndarray,
    colormap: str = "hot",
) -> np.ndarray:
    """Convert uncertainty map to RGB visualization.

    Args:
        uncertainty: (H, W) uncertainty values.
        colormap: Matplotlib colormap name.

    Returns:
        (H, W, 3) uint8 RGB image.
    """
    import matplotlib.cm as cm

    # Normalize to [0, 1]
    u_min, u_max = uncertainty.min(), uncertainty.max()
    if u_max > u_min:
        normalized = (uncertainty - u_min) / (u_max - u_min)
    else:
        normalized = np.zeros_like(uncertainty)

    cmap = cm.get_cmap(colormap)
    colored = (cmap(normalized)[:, :, :3] * 255).astype(np.uint8)
    return colored
