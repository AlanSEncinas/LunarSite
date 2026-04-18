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
