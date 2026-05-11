"""LunarSite Streamlit Demo v0 - Stage 2 terrain segmentation.

Self-contained demo that ships with the repo. Preloaded examples live in
demo_assets/ so first-paint is instant even on a cold Streamlit Community
Cloud deploy. Upload box is below the examples for visitors who want to try
their own images.

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = REPO_ROOT / "best_resnet34.pt"
MC_CHECKPOINT_PATH = REPO_ROOT / "models" / "best_segmenter_mcdropout.pt"
DEMO_DIR = REPO_ROOT / "demo_assets"
MANIFEST_PATH = DEMO_DIR / "manifest.json"

CLASS_NAMES = ["background", "small_rocks", "large_rocks", "sky"]
CLASS_COLORS = np.array(
    [[0, 0, 0], [255, 165, 0], [255, 0, 0], [135, 206, 235]],
    dtype=np.uint8,
)
NC = 4
INPUT_SIZE = 480


# ----- Cached resources -----

@st.cache_resource
def load_model():
    import segmentation_models_pytorch as smp
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=NC,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    return model, device, ckpt.get("epoch", "?"), ckpt.get("best_metric", None)


@st.cache_resource
def load_mc_model():
    """Load the MC-Dropout fine-tuned model. Returns None if the checkpoint
    doesn't exist (so the UI can hide the MC Dropout option gracefully)."""
    if not MC_CHECKPOINT_PATH.exists():
        return None, None, None
    import segmentation_models_pytorch as smp
    try:
        from lunarsite.utils.uncertainty import add_mc_dropout  # type: ignore
    except ImportError:
        return None, None, None
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=NC)
    add_mc_dropout(model, p=0.1)  # Must match training-time dropout_p
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MC_CHECKPOINT_PATH, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    return model, device, ckpt.get("best_metric", None)


@st.cache_data
def load_manifest() -> Optional[dict]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return None


# ----- Inference helpers -----

def preprocess(img_rgb: np.ndarray, size: int = INPUT_SIZE) -> tuple[torch.Tensor, np.ndarray]:
    h, w = img_rgb.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    cropped = img_rgb[y0:y0 + side, x0:x0 + side]
    resized = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0)
    return tensor, resized


@torch.no_grad()
def predict_with_tta(model, x: torch.Tensor, device: torch.device) -> np.ndarray:
    x = x.to(device)
    probs = F.softmax(model(x), dim=1)
    probs = probs + torch.flip(F.softmax(model(torch.flip(x, dims=[3])), dim=1), dims=[3])
    probs = probs + torch.flip(F.softmax(model(torch.flip(x, dims=[2])), dim=1), dims=[2])
    probs = probs + torch.flip(F.softmax(model(torch.flip(x, dims=[2, 3])), dim=1), dims=[2, 3])
    probs = probs / 4.0
    return probs.argmax(1).cpu().numpy()[0].astype(np.uint8)


@torch.no_grad()
def predict_with_mc_dropout(
    model, x: torch.Tensor, device: torch.device, n_samples: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (pred_mask, entropy_map, mutual_info_map) — all (H, W).

    - pred_mask: argmax of MC-mean probs, uint8
    - entropy_map: predictive entropy (total uncertainty), float32 in [0, log(C)]
    - mutual_info_map: epistemic uncertainty from dropout disagreement, float32
    """
    from lunarsite.utils.uncertainty import enable_mc_dropout  # type: ignore
    model.eval()
    enable_mc_dropout(model)
    x = x.to(device)
    runs = []
    for _ in range(n_samples):
        runs.append(F.softmax(model(x), dim=1))
    stacked = torch.stack(runs).squeeze(1)  # (n, C, H, W)
    mean_probs = stacked.mean(dim=0)
    pred = mean_probs.argmax(0).cpu().numpy().astype(np.uint8)
    H = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(0)
    per_sample_H = -(stacked * torch.log(stacked + 1e-10)).sum(1)
    mi = H - per_sample_H.mean(0)
    return pred, H.cpu().numpy().astype(np.float32), mi.cpu().numpy().astype(np.float32)


def uncertainty_heatmap(uncertainty: np.ndarray) -> np.ndarray:
    """Normalize (H, W) to uint8 RGB heatmap (hot colormap)."""
    lo, hi = uncertainty.min(), uncertainty.max()
    norm = (uncertainty - lo) / max(hi - lo, 1e-8)
    norm = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_HOT)[..., ::-1]  # BGR → RGB


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NC):
        out[mask == c] = CLASS_COLORS[c]
    return out


def overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    colored = colorize_mask(mask)
    mask_not_bg = mask != 0
    out = image.copy()
    out[mask_not_bg] = (alpha * colored[mask_not_bg] + (1 - alpha) * image[mask_not_bg]).astype(np.uint8)
    return out


def compute_class_coverage(mask: np.ndarray) -> dict:
    total = mask.size
    return {CLASS_NAMES[c]: float((mask == c).sum()) / total for c in range(NC)}


def render_coverage_bars(coverage: dict):
    for c in CLASS_NAMES:
        st.progress(coverage[c], text=f"{c}: {coverage[c] * 100:.1f}%")


# ----- UI -----

st.set_page_config(
    page_title="LunarSite - Terrain Segmentation",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("LunarSite")
st.markdown("### End-to-end ML pipeline for lunar south pole landing site selection")

st.markdown(
    """
    **Why this exists:** The lunar south pole is the highest-stakes landing region in modern
    spaceflight — Intuitive Machines IM-2 crashed there in March 2025 under conditions that
    broke classical geometric algorithms. LunarSite is a pre-mission analysis tool that uses
    ML to screen candidate landing cells across 80°S–90°S, filling the gap between NASA's
    flight-time SPLICE system and academic terrain-analysis research.
    """
)

st.markdown(
    """
    The full [LunarSite](https://github.com/AlanSEncinas/LunarSite) pipeline running on the lunar south pole.
    Three stages working together:

    - **Stage 2 — Terrain Segmentation:** U-Net + ResNet-34 segments lunar imagery into four classes
      (**background**, **small rocks**, **large rocks**, **sky**), trained exclusively on 9,766 synthetic
      Unreal Engine scenes. See how it transfers to real moon photography below with **zero domain
      adaptation**, and try MC Dropout calibrated uncertainty on your own image at the bottom.
    - **Stage 1 — Crater Detection:** binary U-Net on the LOLA south pole DEM, fine-tuned on real
      Robbins crater rims for the production model.
    - **Stage 3 — XGBoost Site Scorer:** ranks 315,034 candidate 1 km landing cells over 80°S–90°S
      using 29 features (LOLA topography + PGDA illumination + crater density + PSR exposure), with
      SHAP explainability and validation against NASA's 9 Artemis III candidate regions.
    """
)

manifest = load_manifest()
model, device, epoch, best_metric = load_model()

# --- Model card ---
with st.expander("Model details and test set performance", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Architecture**")
        st.markdown(
            """
            - U-Net + ResNet-34 encoder (`segmentation_models_pytorch`)
            - Dice + Cross-Entropy loss
            - AdamW + cosine annealing, 50 epochs @ 480px
            - Inference: flip-only test-time augmentation (4-way)
            """
        )
        st.text(f"Checkpoint: epoch {epoch}, val best mIoU {best_metric:.4f}" if best_metric else f"Checkpoint: epoch {epoch}")
        st.text(f"Device: {device}")
    with col2:
        st.markdown("**Test set performance (with flip TTA)**")
        if manifest and "test_metrics" in manifest:
            st.metric("Mean IoU", f"{manifest['test_metrics']['mean_iou']:.4f}")
            for c, v in manifest["test_metrics"]["per_class_iou"].items():
                st.text(f"  {c:<14}{v:.4f}")
        else:
            st.text("Test mIoU: 0.8456")
        if manifest and "ensemble" in manifest:
            ens = manifest["ensemble"]
            st.markdown("**Deep ensemble (5 seeds)**")
            st.text(
                f"{ens['n_members']} members | "
                f"mean test TTA mIoU: {ens['test_tta_mean']:.4f} | "
                f"stdev: {ens['test_tta_stdev']:.4f}"
            )

st.divider()

# --- Real moon examples (gallery) ---
st.header("Real moon photography")
st.markdown(
    "The model was trained **only on synthetic data**. Here's what it predicts on actual moon photos "
    "— no fine-tuning, no domain adaptation, no real data in the training set."
)

if manifest and manifest.get("real_moon_examples"):
    examples = manifest["real_moon_examples"]
    # Initial example selection
    selected_idx = st.selectbox(
        "Select example",
        range(len(examples)),
        format_func=lambda i: f"{examples[i]['name']} — {examples[i]['caption']}",
    )
    ex = examples[selected_idx]
    show_uncertainty = st.toggle(
        "Show ensemble uncertainty",
        value=False,
        help="Per-pixel disagreement across a 5-model deep ensemble (brighter = less confident).",
    )
    if show_uncertainty and ex.get("uncertainty"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(str(DEMO_DIR / ex["input"]), caption="Real moon photo (center-cropped)", use_container_width=True)
        with col2:
            st.image(str(DEMO_DIR / ex["overlay"]), caption="Prediction (seed 1)", use_container_width=True)
        with col3:
            st.image(str(DEMO_DIR / ex["uncertainty"]), caption="Ensemble uncertainty (5 seeds)", use_container_width=True)
        if "ensemble_mean_uncertainty" in ex:
            st.caption(
                f"Mean uncertainty: {ex['ensemble_mean_uncertainty']:.3f}  |  "
                f"Max: {ex['ensemble_max_uncertainty']:.3f}  "
                f"(bright pixels = ensemble disagreement, usually on class boundaries)"
            )
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(str(DEMO_DIR / ex["input"]), caption="Real moon photo (center-cropped)", use_container_width=True)
        with col2:
            st.image(str(DEMO_DIR / ex["overlay"]), caption="Model prediction overlay", use_container_width=True)
    render_coverage_bars(ex["coverage"])

    with st.expander("Legend and known failure modes"):
        st.markdown(
            """
            **Color legend (overlay colors on the prediction image above):**
            - **background** (lunar regolith) — unmodified / black
            - **small_rocks** — orange
            - **large_rocks** — red
            - **sky** — light blue

            **Known failure mode:** Bright sun-lit rocks can be misclassified as sky.
            The synthetic training set has bright, uniform sky in every frame, so the model
            learned a "bright + uniform = sky" shortcut that occasionally fires on sun-lit boulders.
            Visible in some examples above where light-colored rocks show blue overlay.
            """
        )
else:
    st.info("Demo assets not found. Run `python scripts/build_demo_assets.py` to generate them.")

# --- Real south pole orbital imagery ---
if manifest and manifest.get("south_pole_examples"):
    st.header("Real south pole orbital imagery")
    st.markdown(
        "Zero-shot transfer to **real LRO and NASA south pole orbital imagery**. The model "
        "was trained only on synthetic Unreal Engine lunar *surface* renders — it has never seen "
        "orbital views and never seen the south pole. These predictions are what it does with no "
        "fine-tuning, no domain adaptation. Class distributions remain roughly consistent with "
        "training (bg ~85% on orbital vs 76% on surface, small_rocks proportional)."
    )
    sp_examples = manifest["south_pole_examples"]
    sp_idx = st.selectbox(
        "Select south pole example",
        range(len(sp_examples)),
        format_func=lambda i: f"{sp_examples[i]['name']} — {sp_examples[i]['caption']}",
        key="sp_select",
    )
    sp = sp_examples[sp_idx]
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(str(DEMO_DIR / sp["input"]),
                 caption=f"Real south pole image (center-cropped): {sp['name']}",
                 use_container_width=True)
    with col2:
        st.image(str(DEMO_DIR / sp["overlay"]),
                 caption="Stage 2 prediction (zero-shot, no fine-tuning)",
                 use_container_width=True)
    render_coverage_bars(sp["coverage"])

# --- Stage 1: crater detection ---
if manifest and manifest.get("stage1"):
    s1 = manifest["stage1"]
    st.header("Stage 1 — Crater Detection on LOLA south pole DEM")
    st.markdown(
        f"Binary U-Net + ResNet-34 trained on DeepMoon synthetic DEM tiles (Silburt 2019), "
        f"then **fine-tuned on 334 real LOLA south pole tiles** (118 m/px) with "
        f"Robbins 2018 crater-ring ground truth ≥3 km. The fine-tune closed a 7× sim-to-real gap "
        f"(v1 IoU 0.021 → v2 IoU 0.161 on held-out val)."
    )
    st.image(str(DEMO_DIR / s1["crater_overlay"]),
             caption="LOLA south pole DEM hillshade with v2 crater predictions (cyan). "
                     "608 × 608 km, 80°S–90°S, downsampled for web.",
             use_container_width=True)
    m = s1["metrics"]; v = s1["v1_vs_v2"]
    cA, cB, cC, cD = st.columns(4)
    cA.metric("v2 IoU",         f"{m['iou']:.3f}", f"{m['iou'] - v['v1_iou']:+.3f} vs v1")
    cB.metric("v2 Recall",      f"{m['recall']:.3f}", f"{(m['recall'] / v['v1_recall'] - 1) * 100:+.0f}% vs v1")
    cC.metric("Precision",      f"{m['precision']:.3f}")
    cD.metric("Threshold",      f"{m['threshold']:.2f}",  "flip TTA")
    st.caption(
        "Metrics computed on the full 7600×7600 LOLA south pole DEM vs Robbins 2018 crater rings "
        "(14,406 craters ≥1 km)."
    )

    st.divider()

# --- Stage 3: landing site scoring ---
if manifest and manifest.get("stage3"):
    s3 = manifest["stage3"]
    st.header("Stage 3 — XGBoost Landing Site Scorer")
    n_feat_str = s3.get("n_features", 27)
    st.markdown(
        f"**{s3['n_cells']:,} grid cells** (1 km each) over 80°S–90°S scored on a "
        f"{n_feat_str}-feature vector combining LOLA topography, 60 m/px illumination + Earth "
        f"visibility (Mazarico 2011, PGDA Product 69), Stage 1 crater predictions, and per-cell "
        f"PSR exposure statistics. Labels are **rule-based** from NASA's CASSA thresholds "
        f"(slope ≤5°, illumination ≥33%, Earth visibility ≥50%). XGBoost learns a soft score "
        f"that generalizes to features the hard rules don't encode."
    )

    st.subheader("Top-500 scored cells vs NASA Artemis III candidate regions")
    st.image(str(DEMO_DIR / s3["top_sites_map"]),
             caption="LOLA south pole hillshade. Color-graded dots are LunarSite's top-500 "
                     "ranked cells; gold stars are NASA's 9 Artemis III candidates.",
             use_container_width=True)

    # Artemis overlap table
    overlap_path = DEMO_DIR / s3.get("artemis_overlap_json", "")
    if overlap_path.exists():
        overlap = json.loads(overlap_path.read_text())
        st.subheader("Artemis III overlap — LunarSite top-N vs 9 NASA candidate regions")
        mA, mB, mC = st.columns(3)
        for col, level in zip((mA, mB, mC), overlap["top_n_levels"]):
            col.metric(
                f"Top {level['top_n']}",
                f"{level['regions_matched']}/9 matched",
                help="regions with >=1 LunarSite top-N cell within 15 km of the published center",
            )
        with st.expander("Per-region detail (top 500)"):
            for lvl in overlap["top_n_levels"]:
                if lvl["top_n"] == 500:
                    import pandas as pd
                    df = pd.DataFrame(lvl["regions"])
                    df = df.rename(columns={"region": "NASA region",
                                            "cells_within_15km": "cells in 15 km",
                                            "closest_km": "closest (km)"})
                    st.dataframe(df, use_container_width=True, hide_index=True)

    # PSR exposure check — Layer 3 Dark Terrain quantitative piece
    if s3.get("psr_map") and "psr_exposure" in s3:
        st.subheader("Permanently Shadowed Region (PSR) exposure")
        psr = s3["psr_exposure"]
        st.markdown(
            f"Cells where the minimum solar illumination drops below "
            f"**{psr['psr_threshold_pct']}%** are tagged as permanently shadowed "
            f"(Mazarico et al. 2011, Icarus threshold for PSR identification). "
            f"{psr['total_psr_cells_in_grid']:,} of {s3['n_cells']:,} grid cells "
            f"contain PSR ground. The plot below shows those PSR cells in red and "
            f"LunarSite's top 100 ranked cells in yellow."
        )
        st.image(str(DEMO_DIR / s3["psr_map"]),
                 caption="Top 100 ranked cells (yellow) vs PSR cells (red) on LOLA hillshade. "
                         "Zero overlap: the scorer is already avoiding permanently shadowed "
                         "ground without being told to.",
                 use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Top 10 cells with any PSR",  f"{psr['top_10']['cells_with_any_psr']}/10")
        c2.metric("Top 100 cells with any PSR", f"{psr['top_100']['cells_with_any_psr']}/100")
        c3.metric("Top 500 cells with any PSR", f"{psr['top_500']['cells_with_any_psr']}/500")
        st.caption(
            f"Top 100 mean minimum illumination: {psr['top_100']['mean_illum_min_pct']}% "
            f"(safely above the 0.5% PSR threshold). This is empirical backup for the "
            f"scorer's dark-terrain-awareness claim without requiring ShadowCam imagery — "
            f"the PGDA illumination map (LOLA + 10 years of sun-angle simulation) is the "
            f"authoritative PSR source used by NASA and every peer-reviewed landing-site study."
        )

    st.subheader("SHAP explainability — which features drive the score?")
    st.image(str(DEMO_DIR / s3["shap_summary"]),
             caption="SHAP summary on 5,000 sampled cells. Top 3 features are the 3 CASSA "
                     "threshold inputs (sanity check). `elevation_std` and `illumination_min_pct` "
                     "emerge in the top 7 — signals the hard rules don't encode.",
             use_container_width=True)

    with st.expander("Top 10 cells (direct numeric output)"):
        import pandas as pd
        top10 = pd.read_csv(DEMO_DIR / s3["top10_csv"])
        st.dataframe(top10, use_container_width=True, hide_index=True)
        st.caption(
            "All top 10 cells sit at elevations +4.5 to +4.9 km (Mons Mouton massif), "
            "slopes 2.7–4.6°, illumination 35–40%, Earth visibility 60–85%. "
            "The Mons Mouton region is the peak-of-eternal-light area NASA independently "
            "identified as the top Artemis III candidate."
        )

    st.divider()

# --- Synthetic benchmark ---
st.header("Synthetic benchmark")
if manifest and manifest.get("synthetic_example"):
    synth = manifest["synthetic_example"]
    st.markdown(
        f"Synthetic Unreal Engine scene from the training distribution. "
        f"**Per-image mIoU: {synth['mean_iou']:.4f}** (test set average: 0.8456)."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(str(DEMO_DIR / synth["input"]), caption="Synthetic input", use_container_width=True)
    with col2:
        st.image(str(DEMO_DIR / synth["overlay"]), caption="Model prediction", use_container_width=True)
    with col3:
        st.image(str(DEMO_DIR / synth["ground_truth"]), caption="Ground truth", use_container_width=True)

    st.markdown("**Per-class IoU on this image:**")
    iou_cols = st.columns(4)
    for i, (c, v) in enumerate(synth["per_class_iou"].items()):
        with iou_cols[i]:
            st.metric(c, f"{v:.3f}")

st.divider()

# --- Upload box ---
st.header("Try your own image")
mc_model, mc_device, mc_best = load_mc_model()
mc_available = mc_model is not None
if mc_available:
    st.markdown(
        "Upload any lunar-looking image (synthetic, rover photo, telescope shot). "
        "Choose an inference mode — flip TTA for the fastest/best single prediction, or "
        "MC Dropout for per-pixel epistemic uncertainty (20 stochastic forward passes). "
        "Images are center-cropped to 480×480."
    )
else:
    st.markdown(
        "Upload any lunar-looking image (synthetic, rover photo, telescope shot). "
        "The model applies flip TTA during inference. Images are center-cropped to 480×480."
    )

uploaded = st.file_uploader("Upload PNG / JPG", type=["png", "jpg", "jpeg"])

if mc_available:
    inference_mode = st.radio(
        "Inference mode",
        options=["Flip TTA (deterministic)", "MC Dropout (20 samples + uncertainty)"],
        horizontal=True,
    )
else:
    inference_mode = "Flip TTA (deterministic)"

if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("Could not decode image.")
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor, img_crop = preprocess(img_rgb)

        if inference_mode.startswith("MC Dropout"):
            with st.spinner("Running 20 stochastic forward passes (MC Dropout)..."):
                mask, entropy, mi = predict_with_mc_dropout(mc_model, tensor, mc_device, n_samples=20)
            over = overlay(img_crop, mask)
            coverage = compute_class_coverage(mask)
            entropy_rgb = uncertainty_heatmap(entropy)
            mi_rgb = uncertainty_heatmap(mi)

            col1, col2 = st.columns(2)
            with col1:
                st.image(img_crop, caption="Your input (480×480)", use_container_width=True)
            with col2:
                st.image(over, caption="MC-mean prediction overlay", use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                st.image(entropy_rgb,
                         caption=f"Predictive entropy (total) — mean {float(entropy.mean()):.3f}",
                         use_container_width=True)
            with col4:
                st.image(mi_rgb,
                         caption=f"Mutual info (epistemic) — mean {float(mi.mean()):.3f}",
                         use_container_width=True)

            st.caption(
                "Brighter pixels = higher uncertainty. "
                "Mutual information isolates epistemic (model) uncertainty and should be highest "
                "on out-of-distribution inputs or ambiguous class boundaries."
            )

            st.subheader("Per-class coverage (from MC-mean prediction)")
            render_coverage_bars(coverage)
        else:
            with st.spinner("Running inference with flip TTA..."):
                mask = predict_with_tta(model, tensor, device)
            over = overlay(img_crop, mask)
            coverage = compute_class_coverage(mask)

            col1, col2 = st.columns(2)
            with col1:
                st.image(img_crop, caption="Your input (480×480)", use_container_width=True)
            with col2:
                st.image(over, caption="Prediction overlay", use_container_width=True)

            st.subheader("Per-class coverage")
            render_coverage_bars(coverage)

st.divider()

# --- Footer ---
st.markdown(
    """
    ---
    **LunarSite** is an end-to-end ML pipeline for lunar south pole landing site selection.
    This demo walks all three stages: **Stage 2** terrain segmentation (with MC Dropout calibrated
    uncertainty), **Stage 1** crater detection on the LOLA south pole DEM, and **Stage 3** XGBoost
    site scoring with PSR-aware features and SHAP explainability. Top-ranked cells overlap NASA's
    independently selected Artemis III candidate regions.

    Built by [Alan Scott Encinas](https://github.com/AlanSEncinas) · [Source on GitHub](https://github.com/AlanSEncinas/LunarSite) · MIT License
    """
)
