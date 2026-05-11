# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LunarSite** — an end-to-end ML pipeline for lunar south pole landing site selection. Three-stage architecture: (1) Crater Detection CNN, (2) Terrain Hazard Segmentation, (3) XGBoost Landing Site Scorer with SHAP explainability. Python-only. Portfolio project by Alan.

- **Repo:** https://github.com/AlanSEncinas/LunarSite
- **Live demo:** https://lunarsite.streamlit.app
- **Kaggle weights:** https://www.kaggle.com/datasets/encinas88/lunarsite-weights
- **Kaggle fine-tune data:** https://www.kaggle.com/datasets/encinas88/lunarsite-southpole-finetune
- **Project site:** https://alanscottencinas.com

**Current status:** **Layer 3 engineering shipped (2026-04-18); project finalized (2026-05-11).** Stage 2: U-Net + ResNet-34 + Dice+CE + flip TTA, test mIoU **0.8456**; 5-seed deep ensemble at 0.8445 ± 0.0013. MC Dropout fine-tune: val mIoU 0.8134, **ECE 0.0072** across 46 M val pixels, **4.7× OOD mutual-info lift** on real moon photos ([outputs/mc_dropout_eval/](outputs/mc_dropout_eval/)). Stage 1: crater v2 fine-tuned on real LOLA south pole (recall **0.372, +140%** over DeepMoon-only). Stage 3: XGBoost scorer over 315k 1-km cells × 29 features (incl. PSR-aware `psr_fraction` and `illumination_min_pct` — the latter in top-7 SHAP); **0/100** top cells contain PSR ground. Cross-instrument validation: **81–85%** of ShadowCam's deepest-observed-shadow pixels at Cabeus / LCROSS fall inside PGDA-predicted PSRs ([outputs/shadowcam_validation/](outputs/shadowcam_validation/)). Streamlit demo at [lunarsite.streamlit.app](https://lunarsite.streamlit.app). Remaining work is content-only (blog, case study, Kaggle per-file description polish via UI); no further ML/code work planned.

## Ship Definition

LunarSite has three build layers, each with a distinct role. **Layer 2 is the real ship.** Layer 1 is the foundation it stands on; Layer 3 is what validates it in the world. Pick one explicitly; do not let scope drift.

**Layer 1 — Foundation (weeks away):** Stage 2 segmentation only. Trained segmenter (v1 winner) + sim-to-real qualitative eval + Streamlit v0 demo with preloaded example outputs. *The core ML capability works and you can show it to someone.* The segmenter is the primary technical risk, the demo is the delivery channel, and the data → train → eval → deploy pattern gets established here. Everything in Layers 2 and 3 depends on this foundation being solid. Not "LunarSite" — it's "a good lunar terrain segmenter." If this doesn't work, nothing else matters.

**Layer 2 — Deepening (the real ship):** Layer 1 + Stage 1 crater detection + Stage 3 XGBoost scorer with LOLA features + deep ensemble uncertainty on Stage 2 + Streamlit v3 demo (coordinates → full site score + SHAP). *The foundation is proven; now you deepen it into the actual LunarSite pipeline.* Ensemble adds uncertainty rigor to Stage 2. Stage 1 adds crater detection as a parallel capability. Stage 3 brings Stages 1 and 2 together with LOLA features into the actual scorer the project is *for*. No dark terrain module, no arXiv paper. **This is LunarSite. End-to-end pipeline working is the definition of done.**

**Layer 3 — Validation (engineering shipped 2026-04-18) + End Game (deferred):** Layer 2 + MC Dropout calibrated uncertainty (ECE 0.0072, 4.7× OOD lift) + Stage 3 PSR-aware features (psr_fraction, illumination_min_pct from PGDA — top 100 cells contain 0 PSR ground) + cross-instrument PSR validation against ShadowCam at Cabeus / LCROSS (81–85 % agreement on deepest shadow). Dark Terrain module shipped via PGDA-derived PSR detection rather than full ShadowCam HORUS pipeline (skipped HORUS denoising + shadow-from-depth as low credibility-per-effort). End-game items deferred: arXiv paper + commercial outreach + community launch + blog/case study writeup.

### Uncertainty strategy
Both shipped: deep ensembles for Layer 2 aleatoric coverage (4-5 ResNet-34 seeds with identical data split + config) + MC Dropout for Layer 3 calibrated epistemic uncertainty (27 Dropout2d modules injected after every ReLU, fine-tuned 10 epochs from `best_resnet34.pt`, ECE 0.0072 across 46 M val pixels, 4.7× OOD mutual-info lift on real moon photos). MC Dropout was originally dead code in `src/lunarsite/utils/uncertainty.py` — now imported by `streamlit_app.py`, `scripts/mc_dropout_calibrate.py`, and `scripts/train_segmenter.py --mc-dropout`.

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

# === Kaggle automation (no browser needed) === #
# Prereq: pip install kaggle; place kaggle.json at ~/.kaggle/kaggle.json
python scripts/kaggle_run.py list                    # Show registered kernels
python scripts/kaggle_run.py push eval_v1_vs_v2      # Push notebook to Kaggle (queues run)
python scripts/kaggle_run.py status eval_v1_vs_v2    # Check kernel status
python scripts/kaggle_run.py wait eval_v1_vs_v2      # Poll until complete (60s interval)
python scripts/kaggle_run.py pull eval_v1_vs_v2      # Download outputs to outputs/<name>/
python scripts/kaggle_run.py run eval_v1_vs_v2       # Full loop: push + wait + pull

# Dataset versioning (e.g. updating checkpoint dataset)
kaggle datasets version -p tmp_upload/ -m "version notes"
kaggle datasets files encinas88/lunarsite-weights
```

## Kaggle Workflow

LunarSite uses Kaggle's free T4 GPUs for training and eval. The workflow is automated via `scripts/kaggle_run.py`:

1. **Register a new kernel:** add an entry to the `KERNELS` dict in `scripts/kaggle_run.py` with slug, notebook path, accelerator, and dataset dependencies.
2. **One-command run:** `python scripts/kaggle_run.py run <name>` pushes the notebook, polls until complete, and pulls outputs to `outputs/<name>/`.
3. **Checkpoints dataset:** trained model weights live in the private `encinas88/lunarsite-weights` Kaggle dataset. Update with `kaggle datasets version` to publish new versions — the eval notebook auto-downloads the latest via `kagglehub`.
4. **No browser clicks for reruns:** once a kernel is registered, re-running it is one command. The metadata (GPU, internet, datasets) is baked into `KERNELS`, not the Kaggle UI.

**Why this matters:** the manual browser flow (import notebook → attach dataset → set GPU → Run All → download outputs) takes ~5 minutes of clicks per run and is error-prone (easy to forget to attach a dataset). The script eliminates all of that. For ensemble training (5 seeds × future configs) this is the difference between an hour of clicking and a single script invocation.

## Architecture

> **Scope note (as shipped 2026-04-18):** All three layers shipped. Stage 2 has BOTH deep ensemble (5 seeds) AND MC Dropout calibrated uncertainty. Stage 3 includes PSR-aware features derived from PGDA. Dark Terrain shipped via PGDA + ShadowCam Cabeus cross-validation rather than the full HORUS / shadow-from-depth pipeline.

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
- **Uncertainty (Layer 3, shipped 2026-04-18):** MC Dropout with mutual information for epistemic uncertainty, flagging low-confidence predictions in shadowed and out-of-distribution regions. Production checkpoint: `models/best_segmenter_mcdropout.pt`. Validated calibration: ECE 0.0072 across 46 M val pixels, 4.7× OOD MI lift on real moon photos.
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

### Phase 1: Stage 2 — Terrain Segmentation  _(Layer 1 Foundation + Layer 2 Deepening)_
- [x] Download + explore Kaggle landscape dataset (9,766 images, 4 classes)
- [x] Preprocess, split, PyTorch Dataset (`LunarTerrainDataset`)
- [x] U-Net + ResNet-34 baseline, Dice+CE loss — **val best mIoU 0.8357, test mIoU 0.8425** (A100, 50 epochs, 480px)
- [x] Colab notebook for GPU training (`notebooks/train_segmenter_colab.ipynb`)
- [x] Lunar-specific augmentations (shadow rotation, extreme contrast, Hapke BRDF)
- [x] v2 training: ResNet-50 + FocalDiceLoss + class weights (Kaggle T4 x2) — **val best 0.8304, test+TTA 0.8429** (lost vs v1)
- [x] v1 vs v2 comparison with flip TTA + multi-scale TTA — **v1 ResNet-34 + flip TTA wins at test mIoU 0.8456** ([outputs/v1_vs_v2_eval/test_comparison.json](outputs/v1_vs_v2_eval/test_comparison.json))
- [x] **Production config locked:** U-Net + ResNet-34 + Dice+CE + flip TTA. v2 kept as documented negative ablation.
- [x] Sim-to-real qualitative evaluation on 36 real moon images — coherent transfer, class balance preserved (bg 75% vs training 76%, sr 19% vs 19%), known failure mode on bright-rocks-as-sky. See [outputs/sim_to_real/v1_tta/contact_sheet.png](outputs/sim_to_real/v1_tta/contact_sheet.png).
- [x] Streamlit v0 demo — [streamlit_app.py](streamlit_app.py) with preloaded real moon + synthetic examples, model card, upload box. Smoke tested locally.
- [x] **Deep ensemble** (5 seeds of v1 config, identical data split with `split_seed=42`). Test TTA mIoU 0.8445 ± 0.0013. Member 5 highest at 0.8458; member 4 lowest at 0.8428.

### Phase 2: Stage 1 — Crater Detection  _(Layer 2, shipped 2026-04-18)_
- [x] Download crater datasets + LOLA DEMs (DeepMoon synthetic, LOLA `LDEM_80S_80MPP_ADJ.TIF`, Robbins ≥3 km catalog)
- [x] DEM tile extraction pipeline ([scripts/build_southpole_hdf5.py](scripts/build_southpole_hdf5.py))
- [x] Crater U-Net (binary segmentation, SMP `Unet("resnet34")`)
- [x] **v1** trained on DeepMoon synthetic — test IoU 0.327 (flip TTA) on DeepMoon val; fails on real LOLA south pole (recall 0.155)
- [x] **v2** fine-tuned on 334 real LOLA south pole tiles + Robbins rim labels — test IoU **0.162**, recall **0.372** (+140 % over v1). Production Stage 1.
- [x] Streamlit Stage 1 section + crater overlay on south pole DEM

### Phase 3: Stage 3 — Feature Engineering & Scoring  _(Layer 2, shipped 2026-04-18)_
- [x] Download PGDA Product 90 (slope, elevation) + Product 69 (illumination, Earth visibility)
- [x] Compute Stage 1 derived features (crater density per cell)
- [x] Grid generator + LOLA feature extractor over 80°S–90°S
- [x] Feature matrix: **315,034 cells × 29 features** ([data/processed/stage3_features_80mpp_1km.parquet](data/processed/))
- [x] CASSA rule-based pseudo-labels (slope ≤5°, illumination ≥33%, Earth visibility ≥50%)
- [x] XGBoost + SHAP analysis ([scripts/train_scorer.py](scripts/train_scorer.py))
- [x] **Artemis III overlap validation**: 5/9 NASA candidate regions matched at top 1000 cells (Mons Mouton dominates)
- [x] Streamlit Stage 3 section: top-sites map + Artemis overlap table + SHAP summary + per-cell PSR exposure

### Phase 4: Integration & Polish  _(Layer 2 SHIP, 2026-04-18)_
- [x] End-to-end pipeline script ([scripts/run_pipeline.py](scripts/run_pipeline.py))
- [x] README v2 with headline results, architecture diagram, reproduction commands
- [x] **Layer 2 ship checkpoint reached** — LunarSite end-to-end pipeline working, demo live, repo public.

---

### Phase 5: Dark Terrain Analysis  _(Layer 3, shipped 2026-04-18)_
- [x] Stage 3 PSR features from PGDA AVGVISIB (Mazarico 2011): `psr_fraction` and `illumination_min_pct`. `illumination_min_pct` lands top-7 SHAP. **Top 100 ranked cells contain 0 PSR ground.**
- [x] ShadowCam Cabeus / LCROSS data ingestion (Zenodo DOI 10.5281/zenodo.11175455, 10.3 GB CC-BY-4.0). Extracted composites + DEM-and-ortho subsets (~543 MB) to `D:/shadowcam/extracted/`.
- [x] Cross-instrument PSR validation ([scripts/shadowcam_psr_validate.py](scripts/shadowcam_psr_validate.py)): **81–85 % of ShadowCam's deepest-observed-shadow pixels at Cabeus fall inside PGDA-predicted PSRs**. Outputs: `outputs/shadowcam_validation/`.
- _Scaffolding kept, not validated:_ `src/lunarsite/models/depth.py` (Depth Anything V2 wrapper + shadow-based depth), `src/lunarsite/models/enhancement.py` (HORUS-style denoising), illumination decomposition (albedo/shading split).

### Phase 6: Advanced Uncertainty  _(Layer 3, shipped 2026-04-18)_
- [x] MC Dropout sanity check ([scripts/mc_dropout_sanity.py](scripts/mc_dropout_sanity.py)) — 27 Dropout2d modules inject cleanly into U-Net + ResNet-34, MC sampling produces non-trivial variance.
- [x] `--mc-dropout`, `--dropout-p`, `--resume-from`, `--epochs`, `--lr`, `--tag` flags added to [scripts/train_segmenter.py](scripts/train_segmenter.py).
- [x] **Fine-tune** (10 epochs, lr 2e-5, p=0.1) from `best_resnet34.pt` on local RTX 4070. Val mIoU 0.8134 (-0.02 vs non-dropout baseline), test mIoU 0.8181. Checkpoint: `models/best_segmenter_mcdropout.pt`.
- [x] **Calibration eval** ([scripts/mc_dropout_calibrate.py](scripts/mc_dropout_calibrate.py)): **ECE = 0.0072** across 46 M val pixels (textbook-calibrated). OOD real-moon mutual info **4.7× higher** than in-domain val (0.192 vs 0.041). Outputs: [outputs/mc_dropout_eval/](outputs/mc_dropout_eval/).
- [x] Streamlit demo integration — MC Dropout mode + entropy/MI heatmaps in the upload box.

### Explicitly skipped (low credibility-per-effort)
- Shadow-from-depth physics validation against LOLA
- HORUS dark-image enhancement validation (would need raw 19 GB ShadowCam cubes)
- LuSNAR supplementary training data integration
- DINOv2 encoder revisit
