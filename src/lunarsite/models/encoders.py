"""Foundation model encoders for lunar terrain segmentation.

Provides DINOv2 and Prithvi-EO as drop-in replacements for ResNet-34,
delivering significantly better feature representations for terrain analysis.

References:
    - DINOv2 (Meta, 2023): domain-agnostic self-supervised features
    - Prithvi-EO-2.0 (NASA/IBM, Dec 2024): 300M ViT pretrained on 4.2M satellite images
    - SAM2-UNet (2025): Hiera backbone with U-Net decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2Encoder(nn.Module):
    """DINOv2 ViT as a segmentation encoder with multi-scale features.

    Extracts features from intermediate transformer layers and reshapes
    them into spatial feature maps compatible with a U-Net decoder.

    Args:
        model_name: DINOv2 variant ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14').
        output_channels: List of channel dims for each output scale level.
            Used to project ViT features to match decoder expectations.
        frozen: If True, freeze encoder weights (feature extraction only).
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        output_channels: list[int] = None,
        frozen: bool = False,
    ):
        super().__init__()
        self.model_name = model_name

        # Load DINOv2 from torch hub
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size

        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Default output channels matching a typical U-Net decoder
        if output_channels is None:
            output_channels = [64, 128, 256, 512]
        self.output_channels = output_channels

        # Extract features from these transformer block indices
        n_blocks = len(self.backbone.blocks)
        self.feature_indices = [
            n_blocks // 4 - 1,
            n_blocks // 2 - 1,
            3 * n_blocks // 4 - 1,
            n_blocks - 1,
        ]

        # Project ViT features to desired channel counts
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.embed_dim, ch, 1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            )
            for ch in output_channels
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale features.

        Args:
            x: Input image (N, 3, H, W). H and W should be divisible by patch_size.

        Returns:
            List of feature maps at increasing depth, each (N, C_i, H_i, W_i).
        """
        B, _, H, W = x.shape
        h = H // self.patch_size
        w = W // self.patch_size

        # Get intermediate features via forward hooks
        features = []
        hooks = []

        def make_hook(storage):
            def hook_fn(module, input, output):
                storage.append(output)
            return hook_fn

        for idx in self.feature_indices:
            feat_list = []
            features.append(feat_list)
            hooks.append(
                self.backbone.blocks[idx].register_forward_hook(make_hook(feat_list))
            )

        # Forward pass
        _ = self.backbone.forward_features(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Reshape and project features
        outputs = []
        for i, (feat_list, proj) in enumerate(zip(features, self.projections)):
            # feat shape: (B, num_patches + 1, embed_dim) — +1 for CLS token
            feat = feat_list[0]
            if hasattr(feat, "shape") and feat.dim() == 3:
                feat = feat[:, 1:, :]  # remove CLS token
            feat = feat.reshape(B, h, w, self.embed_dim).permute(0, 3, 1, 2)

            # Project to desired channels
            feat = proj(feat)

            # Downsample to create multi-scale (scale factor = 2^i relative to base)
            if i < len(self.output_channels) - 1:
                scale = 2 ** (len(self.output_channels) - 1 - i)
                feat = F.interpolate(feat, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)

            outputs.append(feat)

        return outputs


class DINOv2UNet(nn.Module):
    """U-Net with DINOv2 encoder for lunar terrain segmentation.

    Replaces the standard ResNet-34 encoder with DINOv2 features,
    keeping a standard convolutional decoder.

    Args:
        model_name: DINOv2 variant.
        classes: Number of output segmentation classes.
        frozen_encoder: Whether to freeze DINOv2 weights.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        classes: int = 4,
        frozen_encoder: bool = False,
    ):
        super().__init__()
        decoder_channels = [256, 128, 64, 32]
        encoder_channels = [64, 128, 256, 512]

        self.encoder = DINOv2Encoder(
            model_name=model_name,
            output_channels=encoder_channels,
            frozen=frozen_encoder,
        )
        self.patch_size = self.encoder.patch_size

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(decoder_channels)):
            in_ch = encoder_channels[-(i + 1)]
            if i > 0:
                in_ch += decoder_channels[i - 1]
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, decoder_channels[i], 3, padding=1),
                    nn.BatchNorm2d(decoder_channels[i]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(decoder_channels[i], decoder_channels[i], 3, padding=1),
                    nn.BatchNorm2d(decoder_channels[i]),
                    nn.ReLU(inplace=True),
                )
            )

        self.final = nn.Conv2d(decoder_channels[-1], classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input (N, 3, H, W). H, W must be divisible by patch_size (14).

        Returns:
            Logits (N, classes, H, W).
        """
        _, _, H, W = x.shape
        encoder_features = self.encoder(x)  # [small, ..., large]

        # Decode from deepest to shallowest
        d = self.decoder_blocks[0](encoder_features[-1])

        for i in range(1, len(self.decoder_blocks)):
            # Upsample and concatenate with skip connection
            d = F.interpolate(d, size=encoder_features[-(i + 1)].shape[2:], mode="bilinear", align_corners=False)
            d = torch.cat([d, encoder_features[-(i + 1)]], dim=1)
            d = self.decoder_blocks[i](d)

        # Upsample to original resolution
        d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)
        return self.final(d)


def build_encoder_model(
    encoder_type: str = "resnet34",
    classes: int = 4,
    in_channels: int = 3,
    **kwargs,
) -> nn.Module:
    """Build a segmentation model with the specified encoder.

    Args:
        encoder_type: One of 'resnet34', 'dinov2_vits14', 'dinov2_vitb14',
            'dinov2_vitl14', or any smp-compatible encoder name.
        classes: Number of output classes.
        in_channels: Number of input channels.
        **kwargs: Additional arguments (e.g., frozen_encoder for DINOv2).

    Returns:
        A segmentation model.
    """
    if encoder_type.startswith("dinov2"):
        return DINOv2UNet(
            model_name=encoder_type,
            classes=classes,
            frozen_encoder=kwargs.get("frozen_encoder", False),
        )
    else:
        # Fall back to segmentation_models_pytorch
        from lunarsite.models.unet import build_unet
        return build_unet(
            encoder_name=encoder_type,
            in_channels=in_channels,
            classes=classes,
        )
