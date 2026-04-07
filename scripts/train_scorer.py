#!/usr/bin/env python3
"""Train Stage 3: XGBoost Landing Site Scorer.

Usage:
    python scripts/train_scorer.py --config configs/stage3_scoring.yaml
"""

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost site scorer.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage3_scoring.yaml"),
        help="Path to config YAML.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # TODO: Implement
    # 1. Load feature matrix from data/processed/
    # 2. Generate rule-based pseudo-labels
    # 3. Train XGBoost classifier
    # 4. SHAP analysis
    # 5. Save model + SHAP plots
    pass


if __name__ == "__main__":
    main()
