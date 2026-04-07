# LunarSite

**ML pipeline for lunar south pole landing site selection.**

Three-stage architecture that combines deep learning for hazard detection with gradient-boosted scoring for site ranking, validated against NASA's Artemis candidate regions.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LunarSite Pipeline                           │
│                                                                 │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  Stage 1      │  │  Stage 2          │  │  LOLA GeoTIFFs  │  │
│  │  Crater Det.  │  │  Terrain Seg.     │  │  (slope, elev,  │  │
│  │  U-Net/DEM    │  │  U-Net/RGB        │  │   roughness)    │  │
│  └──────┬───────┘  └────────┬──────────┘  └───────┬─────────┘  │
│         │ crater_density     │ rock_coverage        │ slope     │
│         │ crater_min_dist    │ large_rock_count     │ elevation │
│         │ avg_crater_radius  │ shadow_coverage      │ roughness │
│         └────────┬───────────┴──────────┬───────────┘           │
│                  │                      │                       │
│           ┌──────▼──────────────────────▼──────┐                │
│           │       Stage 3: XGBoost Scorer      │                │
│           │   16 features → site suitability   │                │
│           │        + SHAP explainability        │                │
│           └────────────────┬───────────────────┘                │
│                            │                                    │
│                    ┌───────▼────────┐                           │
│                    │  Ranked Sites  │                           │
│                    │  80°S – 90°S   │                           │
│                    └────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

## Setup

```bash
# Clone and enter
git clone https://github.com/YOUR_USERNAME/lunarsite.git
cd lunarsite

# Create environment (recommended: conda for GDAL)
conda create -n lunarsite python=3.11
conda activate lunarsite
conda install gdal

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Download data
python scripts/download_data.py --stage 2
```

## Usage

```bash
# Train terrain segmentation (Stage 2)
python scripts/train_segmenter.py --config configs/stage2_segmentation.yaml

# Train crater detection (Stage 1)
python scripts/train_crater_detector.py --config configs/stage1_crater_detection.yaml

# Train site scorer (Stage 3)
python scripts/train_scorer.py --config configs/stage3_scoring.yaml

# Run full pipeline
python scripts/run_pipeline.py
```

## Project Status

- [x] Phase 0: Scaffold
- [ ] Phase 1: Terrain Segmentation (Stage 2)
- [ ] Phase 2: Crater Detection (Stage 1)
- [ ] Phase 3: Site Scoring (Stage 3)
- [ ] Phase 4: Integration & Polish

## Data Sources

| Stage | Dataset | Source |
|-------|---------|--------|
| 2 | Artificial Lunar Rocky Landscape | [Kaggle](https://www.kaggle.com/datasets/romainpessia/artificial-lunar-rocky-landscape-dataset) |
| 1 | Crater Detection Dataset | [Kaggle](https://www.kaggle.com/datasets/lincolnzh/martianlunar-crater-detection-dataset) |
| 3 | LOLA Gridded Products | [NASA PGDA](https://pgda.gsfc.nasa.gov/products/90) |

## License

MIT
