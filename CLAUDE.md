# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LunarSite** — an end-to-end ML pipeline for lunar south pole landing site selection. Three-stage architecture: (1) Crater Detection CNN, (2) Terrain Hazard Segmentation, (3) XGBoost Landing Site Scorer with SHAP explainability. Python-only. Portfolio project by Alan.

**Current status:** Phase 1 in progress. Stage 2 baseline trained (U-Net + ResNet-34, val best mIoU 0.8357 / test mIoU 0.8425). v2 training in progress on Kaggle T4 x2 (ResNet-50 + FocalDiceLoss + class weights + TTA).

## Ship Definition

LunarSite has three defined ship tiers. **Tier 2 is the real ship.** Everything else is either pre-ship infrastructure (Tier 1) or post-ship expansion (Tier 3). Pick one explicitly; do not let scope drift.

**Tier 1 — Minimum Viable (weeks away):** Stage 2 segmentation only. Trained segmenter (v1 or v2 winner) + sim-to-real qualitative eval + Streamlit v0 demo with preloaded example outputs. Portfolio-credible. Not "LunarSite" — it's "a good lunar terrain segmenter."

**Tier 2 — Core (the real ship):** Tier 1 + Stage 1 crater detection + Stage 3 XGBoost scorer with LOLA features + deep ensemble uncertainty on Stage 2 + Streamlit v3 demo (coordinates → full site score + SHAP). This is LunarSite as pitched: input south pole coordinates, output safety score with explainability. No dark terrain module, no arXiv paper. **End-to-end pipeline working is the definition of done.**

**Tier 3 — Full Vision (post-ship, earned by shipping Tier 2 first):** Tier 2 + Dark Terrain module (ShadowCam, HORUS denoising, shadow-depth validation against LOLA) + MC Dropout uncertainty + arXiv paper + commercial outreach.

### Uncertainty strategy
Deep ensembles in Tier 2, not MC Dropout. 4-5 independent ResNet-50 runs with varied random seeds (weight init, DataLoader shuffle, augmentation RNG), **identical data split and config across all members**, all siblings of the v2 winning config. MC Dropout stays in Tier 3 — the current implementation is dead code (`src/lunarsite/utils/uncertainty.py` is not imported anywhere; notebook copies only work on DINOv2, not the production ResNet).

### Streamlit demo progression
Build early, upgrade incrementally. Each version is independently shareable.
- **v0** (as soon as v2 test is done): Stage 2 only. Preloaded synthetic + real moon example with cached prediction artifacts — first paint shows finished overlay + per-class IoU table, not an empty upload box. Upload input below the example.
- **v1:** + real moon image gallery, sim-to-real comparison.
- **v2:** + crater detection output once Stage 1 lands.
- **v3:** + coordinate input → full site score + SHAP waterfall plot once Stage 3 lands.

The demo must exist at every stage of the build. Never be in a "can't show this yet" position.

## Commands

```bash
# Environment setup
pip install -r requirements.txt
pip install -e .  # editable install for lunarsite package

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

> **Scope note:** The architecture described here reflects the full Tier 3 vision. The Tier 2 ship excludes the Dark Terrain module (§3) and uses deep ensembles instead of MC Dropout for uncertainty. See Ship Definition above for what's in-scope for the real ship.

Four-module pipeline with dark terrain analysis:

1. **Stage 1 — Crater Detection (PyTorch):** U-Net on 256x256/512x512 DEM tiles → crater masks/detections with position + radius. Feeds `crater_density`, `crater_min_dist`, `avg_crater_radius` into Stage 3.

2. **Stage 2 — Terrain Segmentation (PyTorch):** DINOv2 encoder (or ResNet-34 baseline) on 480x480 RGB lunar images → per-pixel classification {background, small rocks, large rocks, sky}. Dice+CE loss, AdamW with cosine annealing. Lunar-specific augmentations (shadow rotation, extreme contrast, Hapke BRDF). Feeds `rock_coverage_pct`, `large_rock_count`, `shadow_coverage_pct` into Stage 3. MC Dropout uncertainty maps for confidence-weighted features.

3. **Dark Terrain Module — Depth & Enhancement:** Depth Anything V2 for monocular depth, shadow geometry for physics-based depth (`depth = shadow_length * tan(sun_elevation)`), HORUS-style denoising for PSR imagery, illumination decomposition (albedo/shading separation). Feeds `depth_from_shadow`, `psr_fraction`, `segmentation_confidence` into Stage 3.

4. **Stage 3 — Site Scorer (XGBoost):** 100mx100m grid cells over 80S-90S. 22+ features from LOLA + Diviner thermal + Mini-RF SAR + Stage 1/2 outputs + dark terrain features. Rule-based pseudo-labels from NASA CASSA thresholds (slope <=5deg, illumination >=33%, Earth visibility >=50%). SHAP for explainability.

## Key Technical Decisions

- **Build order:** Stage 2 first (Kaggle dataset is ready, simpler pipeline), then Stage 1, then Stage 3.
- **Package layout:** `src/lunarsite/` with editable install via `setup.py`.
- **Segmentation encoders:** DINOv2 (primary, domain-agnostic ViT) or ResNet-34 via `segmentation_models_pytorch` (baseline).
- **Dark terrain analysis:** Depth Anything V2 for monocular depth, shadow geometry for physics-based depth, HORUS-style denoising for PSR imagery.
- **Uncertainty (Tier 2 ship):** Deep ensembles — 4-5 ResNet-50 runs with varied random seeds (weight init, DataLoader shuffle, augmentation RNG), identical data split and config. Epistemic uncertainty from ensemble disagreement.
- **Uncertainty (Tier 3 future):** MC Dropout with mutual information for epistemic uncertainty, flagging low-confidence predictions in shadowed regions. Currently dead code; see Ship Definition.
- **Augmentations:** Lunar-specific shadow rotation, extreme contrast, Hapke BRDF perturbation, synthetic crater overlay.
- **Geospatial stack:** rasterio + GDAL + pyproj for LOLA GeoTIFFs (polar stereographic, MOON_ME frame).
- **Config:** YAML files in `configs/` per stage.
- **Data dirs:** `data/raw/` and `models/` and `outputs/` are gitignored. `data/processed/` is tracked.

## Data Sources

- **Stage 2 training (primary):** Kaggle Artificial Lunar Landscape (9,766 images + masks) — `romainpessia/artificial-lunar-rocky-landscape-dataset`
- **Stage 2 training (supplementary):** LuSNAR (108GB, 5-class, 9 UE scenes) — HuggingFace `JeremyLuo/LuSNAR`
- **Stage 2 validation (real):** 74 real moon images from Kaggle dataset + ShadowCam PSR imagery from KPLO (data.ser.asu.edu)
- **Stage 1 training:** Kaggle Crater Detection (`lincolnzh/martianlunar-crater-detection-dataset`), DeepMoon synthetic DEMs, Kaggle LU3M6TGT
- **Stage 3 features:** NASA PGDA LOLA products at pgda.gsfc.nasa.gov/products/90 (20m/px elevation, slope, roughness, error GeoTIFFs), NASA SVS illumination data at svs.gsfc.nasa.gov/5027/
- **Dark terrain:** ShadowCam (200x LROC NAC sensitivity, ~2m PSR resolution), Diviner thermal (rock abundance, thermal inertia), Mini-RF SAR (backscatter, CPD ice indicator)
- **Validation:** Global Lunar Boulder Map (94M boulders, Zenodo 14751586), ResGAT-F benchmark (7.81% suitable area)

## Domain Context

- This project fills the gap between NASA's deterministic SPLICE flight system and academic ML research. It's a **pre-mission analysis** tool, not a real-time descent system.
- The IM-2 south pole crash (March 2025) validated the need for ML approaches — geometric algorithms failed under extreme south pole lighting.
- **Validation target:** Top-ranked sites should overlap with NASA's 9 Artemis candidate regions (Cabeus B, Haworth, Malapert Massif, Mons Mouton Plateau, Mons Mouton, Nobile Rim 1, Nobile Rim 2, de Gerlache Rim 2, Slater Plain).
- **Benchmark:** ResGAT-F found 7.81% of south pole area suitable — our model should find similar.
- Slope is universally the #1 predictive feature across all published studies.

## Build Plan

### Phase 0: Scaffold
- [x] CLAUDE.md spec
- [x] Repo structure, requirements.txt, .gitignore, configs
- [x] Data download scripts
- [x] README.md (initial)

### Phase 1: Stage 2 — Terrain Segmentation  _(Tier 1/2)_
- [x] Download + explore Kaggle landscape dataset (9,766 images, 4 classes)
- [x] Preprocess, split, PyTorch Dataset (`LunarTerrainDataset`)
- [x] U-Net + ResNet-34 baseline, Dice+CE loss — **val best mIoU 0.8357, test mIoU 0.8425** (A100, 50 epochs, 480px)
- [x] Colab notebook for GPU training (`notebooks/train_segmenter_colab.ipynb`)
- [x] Lunar-specific augmentations (shadow rotation, extreme contrast, Hapke BRDF)
- [ ] v2 training: ResNet-50 + FocalDiceLoss + class weights + TTA (Kaggle T4 x2)
- [ ] Pick winner between v1 and v2, lock production config
- [ ] Deep ensemble: 4-5 runs of winning config with varied seeds
- [ ] Sim-to-real evaluation on real moon images
- [ ] Streamlit v0 demo — Stage 2 only, preloaded examples

### Phase 2: Stage 1 — Crater Detection  _(Tier 2)_
- [ ] Download crater datasets + LOLA DEMs
- [ ] DEM tile extraction pipeline
- [ ] Crater U-Net (or YOLO variant)
- [ ] Train, evaluate, run inference on south pole DEM
- [ ] Streamlit v2 demo upgrade — add crater output

### Phase 3: Stage 3 — Feature Engineering & Scoring  _(Tier 2)_
- [ ] Download PGDA products (slope, roughness, error, K-means, illumination)
- [ ] Download Diviner thermal (thermal inertia, rock abundance) + Mini-RF SAR
- [ ] Compute Stage 1/2 derived features
- [ ] Build feature matrix, define labels
- [ ] Train XGBoost, SHAP analysis
- [ ] Compare against NASA's 9 Artemis regions + ResGAT-F benchmark
- [ ] Streamlit v3 demo upgrade — coordinates → site score + SHAP

### Phase 4: Integration & Polish  _(Tier 2 — SHIP)_
- [ ] End-to-end pipeline script
- [ ] Full demo notebook
- [ ] Final README with results
- [ ] Tests, cleanup
- [ ] **Tier 2 ship checkpoint — LunarSite is "done enough to show people"**

---

_Everything below is Tier 3 (post-ship). Do not start until Tier 2 is shipped._

### Phase 5: Dark Terrain Analysis Module  _(Tier 3, deferred)_
- [x] Depth estimation module (Depth Anything V2 wrapper, shadow-based depth) — code exists, not validated
- [x] HORUS-style dark image enhancement (DestripeNet + PhotonNet) — code exists, not validated
- [x] Illumination decomposition (albedo/shading separation) — code exists, not validated
- [ ] ShadowCam data download + preprocessing
- [ ] Shadow-to-depth pipeline on south pole imagery
- [ ] Validate depth estimates against LOLA DEM ground truth
- [ ] Add dark terrain features (PSR fraction, shadow depth, SAR backscatter, thermal inertia) to Stage 3 feature matrix

### Phase 6: Advanced Uncertainty  _(Tier 3, deferred)_
- [ ] MC Dropout proper implementation on production ResNet (inject dropout, retrain/fine-tune, validate calibration)
- [ ] LuSNAR supplementary data integration
- [ ] DINOv2 encoder — revisit if baseline results warrant

### Phase 7: Impact & Release  _(Tier 3, deferred)_
- [ ] **Open-source release** — clean README, Docker container, reproducible results, MIT license
- [ ] **arXiv paper** — novel contributions: DINOv2 lunar augmentations, shadow-depth validation against LOLA, MC Dropout uncertainty for landing site confidence
- [ ] **Interactive web demo** — Streamlit/Gradio app: input south pole coordinates → safety score + terrain overlay + uncertainty map + SHAP explanation
- [ ] **Commercial outreach** — share results with Intuitive Machines, Firefly, Astrobotic, ispace via LinkedIn with demo link
- [ ] **Community launch** — post demo video to Twitter/X space community, Reddit r/space, Hacker News
