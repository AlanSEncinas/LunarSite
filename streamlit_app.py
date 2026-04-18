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
st.markdown("### Lunar terrain semantic segmentation — sim-to-real transfer from synthetic training data")

st.markdown(
    """
    Stage 2 of the [LunarSite](https://github.com/AlanSEncinas/LunarSite) ML pipeline for lunar south pole
    landing site selection. This model segments lunar surface imagery into four classes —
    **background**, **small rocks**, **large rocks**, **sky** — and was trained exclusively on
    9,766 synthetic Unreal Engine lunar scenes. The examples below show how it transfers to
    real lunar photography with **zero domain adaptation**.
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
            st.markdown("**Deep ensemble (Layer 2)**")
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
    st.markdown(
        f"**{s3['n_cells']:,} grid cells** (1 km each) over 80°S–90°S scored on a 27-feature vector "
        f"combining LOLA topography, 60 m/px illumination + Earth visibility (Mazarico 2011, "
        f"PGDA Product 69), and Stage 1 crater predictions. Labels are **rule-based** from NASA's "
        f"CASSA thresholds (slope ≤5°, illumination ≥33%, Earth visibility ≥50%). XGBoost learns "
        f"a soft score that generalizes to features the hard rules don't encode."
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

    st.subheader("SHAP explainability — which features drive the score?")
    st.image(str(DEMO_DIR / s3["shap_summary"]),
             caption="SHAP summary on 5,000 sampled cells. "
                     "Top 3 features are the 3 CASSA threshold inputs (sanity check). "
                     "`elevation_std` emerges as #4 — a signal the hard rules don't encode.",
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
st.markdown(
    "Upload any lunar-looking image (synthetic, rover photo, telescope shot). "
    "The model applies flip TTA during inference. Images are center-cropped to 480×480."
)

uploaded = st.file_uploader("Upload PNG / JPG", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("Could not decode image.")
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor, img_crop = preprocess(img_rgb)
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
    **LunarSite** is a three-stage ML pipeline for lunar south pole landing site selection.
    This demo shows **Stage 2** (terrain segmentation). The full pipeline adds **Stage 1**
    (crater detection) and **Stage 3** (XGBoost site scorer with SHAP explainability).

    Built by [Alan Scott Encinas](https://github.com/AlanSEncinas) · [Source on GitHub](https://github.com/AlanSEncinas/LunarSite) · MIT License
    """
)
