#!/usr/bin/env python3
"""Download datasets for the LunarSite pipeline.

Usage:
    python scripts/download_data.py --stage 2          # Kaggle lunar landscape
    python scripts/download_data.py --stage 1          # Kaggle crater dataset
    python scripts/download_data.py --stage 3          # LOLA GeoTIFFs
    python scripts/download_data.py --all               # Everything
"""

import argparse
from pathlib import Path

import kagglehub

# Project root is one level up from scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"


def download_stage2_data() -> Path:
    """Download Kaggle Artificial Lunar Rocky Landscape dataset.

    Returns:
        Path to the downloaded dataset directory.
    """
    print("Downloading Stage 2: Artificial Lunar Rocky Landscape dataset...")
    path = kagglehub.dataset_download("romainpessia/artificial-lunar-rocky-landscape-dataset")
    print(f"Stage 2 dataset available at: {path}")
    return Path(path)


def download_stage1_data() -> Path:
    """Download crater detection datasets."""
    print("Downloading Stage 1: Crater Detection dataset...")
    path = kagglehub.dataset_download("lincolnzh/martianlunar-crater-detection-dataset")
    print(f"Stage 1 dataset available at: {path}")
    return Path(path)


def download_stage3_data() -> None:
    """Download LOLA GeoTIFFs from NASA PGDA."""
    # TODO: Download from pgda.gsfc.nasa.gov/products/90
    print("Stage 3 LOLA data download not yet implemented.")
    print("Download manually from: https://pgda.gsfc.nasa.gov/products/90")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LunarSite datasets.")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], help="Stage to download data for.")
    parser.add_argument("--all", action="store_true", help="Download all datasets.")
    args = parser.parse_args()

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    if args.all or args.stage == 2:
        download_stage2_data()
    if args.all or args.stage == 1:
        download_stage1_data()
    if args.all or args.stage == 3:
        download_stage3_data()


if __name__ == "__main__":
    main()
