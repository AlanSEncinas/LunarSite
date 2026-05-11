# LunarSite Timeline

Working plan with dates. Updated as milestones land or slip.

**Current date:** 2026-05-11
**Current phase:** Project finalized. Layer 3 engineering shipped 2026-04-18 in a single-day sprint. Remaining work is content (blog, case study) and Kaggle UI polish.

**Final pace:** Layer 2 ship + Layer 3 engineering both shipped 2026-04-18, ~3 weeks ahead of original Layer 2 estimate.

---

## Shipped

| Date | Milestone |
|---|---|
| 2026-04-11 | **Layer 1 Foundation ship** — Stage 2 segmenter (test mIoU 0.8456), sim-to-real eval, Streamlit demo live, Medium case study, LinkedIn + X + Threads + HN posts, GitHub public, Kaggle notebook + dataset public |
| 2026-04-15 | Layer 2 deep ensemble (5 seeds), per-pixel uncertainty maps, Streamlit v1 with uncertainty toggle, Kaggle dataset updated with all 5 checkpoints |
| 2026-04-15 | Stage 1 prep: Robbins south pole subset, LOLA 80MPP DEM, geospatial stack verified, `DeepMoonCraterDataset` class, Kaggle training notebook + kernel registered, Robbins→LOLA ground-truth ring mask |
| 2026-04-18 | Stage 1 eval script on real LOLA DEM (`scripts/crater_eval_lola.py`), sliding 256x256 tiles + 50% overlap + flip TTA + classification overlay |
| 2026-04-18 | Stage 3 full scaffolding: grid generator, LOLA feature extractor, CASSA pseudo-labels, `build_stage3_features.py` orchestrator, `train_scorer.py` XGBoost + SHAP. Smoke-tested: Haworth 0% illum ✓, Shackleton 28.8% ✓ |
| 2026-04-18 | PGDA Product 69 illumination + Earth visibility GeoTIFFs (60 m/px @ 85°S-90°S) downloaded. Artemis 9 candidate region coordinates committed. |
| 2026-04-18 | Stage 1 **crater v1** trained on DeepMoon (Kaggle P100, 40 epochs, test TTA IoU 0.327). Eval on real LOLA south pole produced large sim-to-real gap (best IoU 0.111, recall 0.15). Ran diagnostic sweep (4 configs × 9 thresholds) to root-cause. |
| 2026-04-18 | Stage 1 **crater v2** fine-tuned on LOLA south pole HDF5 (334 tiles, 118 m/px, Robbins ≥3 km). Kaggle P100, 25 epochs, pre-FT val IoU 0.021 → post-FT best val IoU 0.161. Head-to-head on full DEM: **v2 IoU 0.162, recall 0.372 vs v1 IoU 0.111, recall 0.155** (+23% IoU, **+140% recall**). Production Stage 1 checkpoint ready for Stage 3. |
| 2026-04-18 | Stage 2 **south pole orbital transfer** — ran v1 segmenter zero-shot on 4 real NASA south pole orbital images. Coverage distribution coherent with training. Shipped new "Real south pole orbital imagery" section in Streamlit demo. |
| 2026-04-18 | **Local CUDA enabled** — switched torch from CPU to CUDA 12.4 on RTX 4070 Laptop GPU. All inference now 9-10× faster locally (crater_eval_lola.py: 11.5 min → 1.1 min). |

## Layer 2 — Deepening (shipped 2026-04-18)

All milestones completed in the single-day Layer 2 sprint:
- Stage 1 crater detection (v1 + v2 fine-tune on real LOLA south pole)
- Full Stage 3 feature matrix (315 k cells × 27 features at the time)
- Stage 3 XGBoost + SHAP scorer with **5/9 Artemis III** region overlap at top-1000
- End-to-end `run_pipeline.py` orchestrator
- README v2 refresh with headline metrics
- Streamlit v2 with crater overlay + Stage 3 landing-site map
- All committed and pushed to GitHub

## Layer 3 — Engineering (shipped 2026-04-18, same day)

Pivoted off the original 4-week dark-terrain plan; collapsed the engineering scope into a single-day sprint by leveraging PGDA + a Zenodo ShadowCam subset rather than waiting on the broken ASU archive UI.

| Date | Milestone |
|---|---|
| 2026-04-18 | **MC Dropout calibrated uncertainty** — fine-tune from `best_resnet34.pt` with 27 Dropout2d(p=0.1) modules. Val mIoU 0.8134, **ECE 0.0072** across 46 M val pixels, **4.7× OOD mutual-info lift** on real moon photos. Streamlit demo gains MC Dropout mode + entropy/MI heatmaps. |
| 2026-04-18 | **Stage 3 PSR features** — `psr_fraction` and `illumination_min_pct` from PGDA AVGVISIB raster. `illumination_min_pct` lands in top-7 SHAP. **0/100** top ranked cells contain any PSR ground. |
| 2026-04-18 | **Cross-instrument PSR validation** — downloaded Fassett 2024 ShadowCam Cabeus / LCROSS archive (10.3 GB) from Zenodo. **81–85 %** of ShadowCam's deepest-observed-shadow pixels fall inside PGDA-predicted PSRs. Outputs in `outputs/shadowcam_validation/`. |

Explicitly skipped (would add little credibility relative to effort):
- Shadow-from-depth physics validation
- HORUS dark-image enhancement validation (needs raw 19 GB ShadowCam cubes)
- LuSNAR supplementary training data integration
- DINOv2 encoder revisit

---

## Known risks

- **Kaggle working-dir limit (20 GB)** burned us once on DeepMoon's 30 GB HDF5 set. Fixed with internal split. If v2 also fails, fallback is Google Colab (~100 GB runtime disk). No code changes needed beyond download path.
- **LOLA GeoTIFF stack** (GDAL + rasterio + pyproj polar stereographic) was the #1 flagged risk — **eliminated 4/15**. Clean install, lunar CRS validated, reprojection working.
- **PGDA Product 69 scale factor** — we assume int16 values scaled by 25500 → percent. Verified visually against known Haworth (0% PSR) and Shackleton (~30% illum) literature; refine if top-ranked sites don't match Artemis 9 regions.
- **Crater training time on P100** — DeepMoon's 30k tiles at 256² batch=16 × 40 epochs estimates 4-5 hrs. If wall-clock exceeds Kaggle's 12 hr limit, reduce to 30 epochs.
- **Evenings-and-weekends pace** — roadmap above is a target, not a commitment. Slip milestones proportionally if life gets in the way.
