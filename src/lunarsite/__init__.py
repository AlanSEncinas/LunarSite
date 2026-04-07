"""
LunarSite — ML pipeline for lunar south pole landing site selection.

Three-stage architecture:
    Stage 1: Crater Detection CNN
    Stage 2: Terrain Hazard Segmentation (U-Net)
    Stage 3: XGBoost Landing Site Scorer with SHAP explainability
"""

__version__ = "0.1.0"
