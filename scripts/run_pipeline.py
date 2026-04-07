#!/usr/bin/env python3
"""Run the full LunarSite end-to-end pipeline.

Executes all three stages in sequence:
    Stage 2 (or load checkpoint) -> Stage 1 (or load) -> Stage 3 -> output map
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full LunarSite pipeline.")
    parser.add_argument("--skip-training", action="store_true", help="Use saved checkpoints.")
    args = parser.parse_args()

    # TODO: Implement pipeline orchestration
    # 1. Run/load Stage 2 segmentation
    # 2. Run/load Stage 1 crater detection
    # 3. Extract features for Stage 3
    # 4. Run Stage 3 scoring
    # 5. Generate output maps
    pass


if __name__ == "__main__":
    main()
