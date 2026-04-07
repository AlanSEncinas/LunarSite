#!/usr/bin/env python3
"""Train Stage 1: Crater Detection model.

Usage:
    python scripts/train_crater_detector.py --config configs/stage1_crater_detection.yaml
"""

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Train crater detection model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage1_crater_detection.yaml"),
        help="Path to config YAML.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # TODO: Implement training loop
    # 1. Build datasets + dataloaders
    # 2. Build model via build_unet(in_channels=1, classes=1)
    # 3. Build loss via build_loss("dice_bce")
    # 4. Build optimizer + scheduler
    # 5. Training loop with validation
    # 6. Save best checkpoint
    pass


if __name__ == "__main__":
    main()
