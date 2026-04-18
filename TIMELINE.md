# LunarSite Timeline

Working plan with dates. Updated as milestones land or slip.

**Current date:** 2026-04-18 (Saturday, 7am)
**Current phase:** Layer 2 — Deepening (crater training in flight, Stage 3 scaffolding complete)

**Current pace:** ~2 weeks ahead of original estimate. Target Layer 2 ship pulled from late May → **early-to-mid May**.

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

## Layer 2 — Deepening (remaining)

| Target | Milestone | Status |
|---|---|---|
| ~~4/18 today~~ | Crater training (v1 + v2 fine-tune) | **Done 4/18** |
| ~~4/18 today~~ | PGDA 20 MPP DEM + slope download to D: | **Done 4/18** |
| ~~4/19 Sun~~ | Stage 1 eval + crater prediction mask | **Done 4/18 (v2)** |
| 4/18-4/19 | Full Stage 3 feature matrix (20 MPP DEM + slope + illum + Earth vis + v2 crater mask) | Unblocked, ready to run |
| 4/18-4/19 | Stage 3 XGBoost + SHAP, top-ranked sites, Artemis region overlap check | Next up |
| 4/20–4/26 | Streamlit v2 (crater overlay on south pole DEM) + v3 (coordinate input → full pipeline + SHAP) | Not started |
| 4/27–5/3 | End-to-end `run_pipeline.py`, README v2 refresh, TIMELINE-final polish | Not started |
| ~5/3–5/10 | **Layer 2 ship** — "LunarSite end-to-end works" launch post across LinkedIn / Medium / HN / Kaggle / X | Not started |

**New target Layer 2 ship: early-to-mid May 2026** (~2-3 weeks from today, pulled ~2-3 weeks forward from original estimate).

## Layer 3 — Validation & End Game (post-Layer-2)

Not scheduled. Separate sprint after Layer 2 ships. Includes:
- Dark Terrain module (ShadowCam, HORUS denoising, shadow-depth validation)
- MC Dropout proper implementation on production ResNet
- arXiv paper on novel contributions (sim-to-real transfer, Stage 1 crater U-Net + LU5M812TGT densified labels, deep ensemble uncertainty, Stage 3 CASSA scorer)
- Commercial outreach (Intuitive Machines / Firefly / Astrobotic / ispace)
- Community launch (Reddit, r/space, Hacker News)

Candidate Layer 1.5 idea to evaluate post-Layer-2: **Stage 1 ensemble** (4 more seeds on Kaggle + Colab in parallel for wall-clock speedup) to add epistemic uncertainty to crater predictions the same way Stage 2 has it.

---

## Known risks

- **Kaggle working-dir limit (20 GB)** burned us once on DeepMoon's 30 GB HDF5 set. Fixed with internal split. If v2 also fails, fallback is Google Colab (~100 GB runtime disk). No code changes needed beyond download path.
- **LOLA GeoTIFF stack** (GDAL + rasterio + pyproj polar stereographic) was the #1 flagged risk — **eliminated 4/15**. Clean install, lunar CRS validated, reprojection working.
- **PGDA Product 69 scale factor** — we assume int16 values scaled by 25500 → percent. Verified visually against known Haworth (0% PSR) and Shackleton (~30% illum) literature; refine if top-ranked sites don't match Artemis 9 regions.
- **Crater training time on P100** — DeepMoon's 30k tiles at 256² batch=16 × 40 epochs estimates 4-5 hrs. If wall-clock exceeds Kaggle's 12 hr limit, reduce to 30 epochs.
- **Evenings-and-weekends pace** — roadmap above is a target, not a commitment. Slip milestones proportionally if life gets in the way.
