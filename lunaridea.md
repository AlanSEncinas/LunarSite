# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LunarSite** — an end-to-end ML pipeline for lunar south pole landing site selection. Three-stage architecture: (1) Crater Detection CNN, (2) Terrain Hazard Segmentation, (3) XGBoost Landing Site Scorer with SHAP explainability. Python-only. Portfolio project by Alan.

**Current status:** Pre-scaffold. Build order starts at Phase 0 (scaffold), then Phase 1 (Stage 2 — terrain segmentation first). See the full project specification in `Claude.md`.

## Commands

```bash
# Environment setup
pip install -r requirements.txt

# Training scripts (once implemented)
python scripts/train_segmenter.py        # Stage 2: terrain segmentation
python scripts/train_crater_detector.py  # Stage 1: crater detection
python scripts/train_scorer.py           # Stage 3: XGBoost site scorer
python scripts/run_pipeline.py           # Full end-to-end pipeline

# Data download
python scripts/download_data.py

# Tests
pytest tests/
pytest tests/test_models.py -k "test_unet"  # Single test
```

## Architecture

Three stages, each feeding into the next:

1. **Stage 1 — Crater Detection (PyTorch):** U-Net on 256x256/512x512 DEM tiles → crater masks/detections with position + radius. Feeds `crater_density`, `crater_min_dist`, `avg_crater_radius` into Stage 3.

2. **Stage 2 — Terrain Segmentation (PyTorch):** U-Net + ResNet-34 encoder on 480x480 RGB lunar images → per-pixel classification {background, small rocks, large rocks, sky}. Dice+CE loss, Adam with cosine annealing. Feeds `rock_coverage_pct`, `large_rock_count`, `shadow_coverage_pct` into Stage 3.

3. **Stage 3 — Site Scorer (XGBoost):** 100mx100m grid cells over 80S-90S. 16 features from LOLA products + Stage 1/2 outputs. Rule-based pseudo-labels from NASA CASSA thresholds (slope <=5deg, illumination >=33%, Earth visibility >=50%). SHAP for explainability.

## Key Technical Decisions

- **Build order:** Stage 2 first (Kaggle dataset is ready, simpler pipeline), then Stage 1, then Stage 3.
- **Segmentation library:** `segmentation_models_pytorch` for pretrained U-Net encoders.
- **Geospatial stack:** rasterio + GDAL + pyproj for LOLA GeoTIFFs (polar stereographic, MOON_ME frame).
- **Config:** YAML files in `configs/` per stage.
- **Data dirs:** `data/raw/` and `models/` and `outputs/` are gitignored. `data/processed/` is tracked.

## Data Sources

- **Stage 2 training:** Kaggle Artificial Lunar Landscape (9,766 images + masks) — `romainpessia/artificial-lunar-rocky-landscape-dataset`
- **Stage 1 training:** Kaggle Crater Detection (`lincolnzh/martianlunar-crater-detection-dataset`), DeepMoon synthetic DEMs, Kaggle LU3M6TGT
- **Stage 3 features:** NASA PGDA LOLA products at pgda.gsfc.nasa.gov/products/90 (20m/px elevation, slope, roughness, error GeoTIFFs), NASA SVS illumination data at svs.gsfc.nasa.gov/5027/

## Domain Context

- This project fills the gap between NASA's deterministic SPLICE flight system and academic ML research. It's a **pre-mission analysis** tool, not a real-time descent system.
- The IM-2 south pole crash (March 2025) validated the need for ML approaches — geometric algorithms failed under extreme south pole lighting.
- **Validation target:** Top-ranked sites should overlap with NASA's 9 Artemis candidate regions (Cabeus B, Haworth, Malapert Massif, Mons Mouton Plateau, Mons Mouton, Nobile Rim 1, Nobile Rim 2, de Gerlache Rim 2, Slater Plain).
- **Benchmark:** ResGAT-F found 7.81% of south pole area suitable — our model should find similar.
- Slope is universally the #1 predictive feature across all published studies.

## Build Plan

### Phase 0: Scaffold
- [x] CLAUDE.md spec
- [ ] Repo structure, requirements.txt, .gitignore, configs
- [ ] Data download scripts
- [ ] README.md (initial)

### Phase 1: Stage 2 — Terrain Segmentation (START HERE)
- [ ] Download + explore Kaggle landscape dataset
- [ ] Preprocess, split, PyTorch Dataset
- [ ] U-Net + ResNet-34 encoder, Dice+CE loss
- [ ] Train, evaluate (mIoU, per-class IoU)
- [ ] Sim-to-real qualitative comparison on LRO NAC images

### Phase 2: Stage 1 — Crater Detection
- [ ] Download crater datasets + LOLA DEMs
- [ ] DEM tile extraction pipeline
- [ ] Crater U-Net (or YOLO variant)
- [ ] Train, evaluate, run inference on south pole DEM

### Phase 3: Stage 3 — Feature Engineering & Scoring
- [ ] Download PGDA products (slope, roughness, error, K-means, illumination)
- [ ] Compute Stage 1/2 derived features
- [ ] Build feature matrix, define labels
- [ ] Train XGBoost, SHAP analysis
- [ ] Compare against NASA's 9 Artemis regions

### Phase 4: Integration & Polish
- [ ] End-to-end pipeline script
- [ ] Full demo notebook
- [ ] Final README with results
- [ ] Tests, cleanup
