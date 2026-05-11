"""LunarSite — end-to-end ML pipeline for lunar south pole landing site selection."""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
LONG_DESCRIPTION = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="lunarsite",
    version="0.1.0",
    description=(
        "End-to-end ML pipeline for lunar south pole landing site selection. "
        "Terrain segmentation + crater detection + XGBoost site scorer with "
        "MC Dropout calibrated uncertainty and PSR-aware features. Validated "
        "against NASA Artemis III candidate regions."
    ),
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Alan Scott Encinas",
    url="https://github.com/AlanSEncinas/LunarSite",
    project_urls={
        "Homepage":      "https://github.com/AlanSEncinas/LunarSite",
        "Source":        "https://github.com/AlanSEncinas/LunarSite",
        "Live Demo":     "https://lunarsite.streamlit.app",
        "Author":        "https://alanscottencinas.com",
        "Bug Tracker":   "https://github.com/AlanSEncinas/LunarSite/issues",
        "Kaggle Weights":           "https://www.kaggle.com/datasets/encinas88/lunarsite-weights",
        "Kaggle Fine-Tune Data":    "https://www.kaggle.com/datasets/encinas88/lunarsite-southpole-finetune",
    },
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords=[
        "machine-learning",
        "deep-learning",
        "computer-vision",
        "semantic-segmentation",
        "pytorch",
        "xgboost",
        "shap",
        "uncertainty-quantification",
        "mc-dropout",
        "nasa",
        "moon",
        "lunar",
        "planetary-science",
        "landing-site-selection",
    ],
)
