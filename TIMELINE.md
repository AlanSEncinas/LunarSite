# LunarSite Timeline

Working plan with dates. Updated as milestones land or slip.

**Current date:** 2026-04-15
**Current phase:** Layer 2 — Deepening (ensemble in flight)

---

## Shipped

| Date | Milestone |
|---|---|
| 2026-04-11 | **Layer 1 Foundation ship** — Stage 2 segmenter (test mIoU 0.8456), sim-to-real eval, Streamlit demo live, Medium case study, LinkedIn + X + Threads + HN posts, GitHub public, Kaggle notebook + dataset public |

## Layer 2 — Deepening

| Target week | Milestone | Status |
|---|---|---|
| 4/14 – 4/19 | Deep ensemble: 5 seeds of v1 config, aggregate script, per-pixel uncertainty map, Streamlit v1 with uncertainty toggle, upload checkpoints to Kaggle dataset | In progress (seeds 2/3 done, 4/5 running) |
| 4/20 – 5/3 | **Stage 1 Crater Detection** — data download, DEM tile pipeline, U-Net train+eval, Streamlit v2 | Not started |
| 5/4 – 5/17 | **Stage 3 XGBoost Scorer** — LOLA PGDA features, feature matrix, train, SHAP, Artemis region validation | Not started |
| 5/18 – 5/24 | Integration — end-to-end pipeline script, Streamlit v3 (coordinate input → full output), README refresh | Not started |
| ~5/24 – 5/31 | **Layer 2 ship** — "LunarSite end-to-end works" launch post | Not started |

**Target Layer 2 ship: late May / early June 2026** (~6 weeks from today).

## Layer 3 — Validation & End Game (post-Layer-2)

Not scheduled. Separate sprint after Layer 2 ships. Includes Dark Terrain module, MC Dropout proper implementation, arXiv paper, commercial outreach (IM / Firefly / Astrobotic / ispace), community launch.

---

## Known risks

- **LOLA GeoTIFF stack** (GDAL + rasterio + pyproj polar stereographic, MOON_ME frame) is the #1 timeline risk in Layer 2 — first time touching it. Budget extra time in the 5/4 week.
- **Stage 1 crater dataset** is less familiar than Stage 2. Budget a day of data exploration before committing to a config.
- **Evenings-and-weekends pace** assumes no extended breaks. Slip each milestone proportionally if life gets in the way — the roadmap above is a target, not a commitment.
