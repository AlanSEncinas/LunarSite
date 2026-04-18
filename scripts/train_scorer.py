"""Stage 3: Train the XGBoost landing site scorer with SHAP explainability.

Takes the feature matrix built by `build_stage3_features.py`, applies CASSA
rule-based pseudo-labels (labels.py), trains an XGBoost classifier, runs
SHAP, and saves the model + per-cell scores + SHAP summary plot.

The CASSA rule-labels are not the end goal -- they are a supervision signal
the tree model uses to learn a soft score that generalizes to features the
rules ignore (crater density, roughness, Stage-1/Stage-2 derived signals).
The *score* is what we rank sites by; the *label* is the scaffold.

Usage:
    python scripts/train_scorer.py \\
        --features data/processed/stage3_features.parquet \\
        --out-dir outputs/stage3 \\
        --test-size 0.2 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


META_COLS = {"cell_id", "lon", "lat", "x_m", "y_m"}


def load_features(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, average_precision_score

    from lunarsite.features.labels import CassaThresholds, apply_labels

    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs/stage3")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--slope-max-deg", type=float, default=5.0)
    p.add_argument("--illum-min-pct", type=float, default=33.0)
    p.add_argument("--earth-min-pct", type=float, default=50.0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_features(args.features)
    print(f"Loaded {len(df):,} cells  x  {len(df.columns)} cols")

    thresholds = CassaThresholds(
        slope_max_deg=args.slope_max_deg,
        illumination_min_pct=args.illum_min_pct,
        earth_visibility_min_pct=args.earth_min_pct,
    )
    df = apply_labels(df, thresholds)
    pos = int(df["suitable"].sum())
    print(f"CASSA pseudo-labels: {pos:,} suitable ({pos / len(df) * 100:.2f}%)")

    feature_cols = [c for c in df.columns if c not in META_COLS and c != "suitable"]
    X = df[feature_cols].to_numpy()
    y = df["suitable"].to_numpy()

    # XGBoost handles native NaN. But drop columns that are entirely NaN (e.g.
    # illumination not yet downloaded) to keep the model clean.
    all_nan_cols = [c for c, col in zip(feature_cols, X.T) if np.isnan(col).all()]
    if all_nan_cols:
        print(f"Dropping {len(all_nan_cols)} all-NaN columns: {all_nan_cols}")
        keep = [c for c in feature_cols if c not in all_nan_cols]
        X = df[keep].to_numpy()
        feature_cols = keep

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed,
        stratify=y if 0 < pos < len(y) else None,
    )

    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="auc",
        tree_method="hist",
        random_state=args.seed,
        n_jobs=-1,
    )
    t0 = time.time()
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    print(f"Train time: {time.time() - t0:.1f}s")

    prob_te = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, prob_te) if 0 < y_te.sum() < len(y_te) else float("nan")
    ap = average_precision_score(y_te, prob_te) if y_te.sum() > 0 else float("nan")
    print(f"Test AUC: {auc:.4f}  |  AP: {ap:.4f}")

    # Score ALL cells for ranking
    scores_all = model.predict_proba(X)[:, 1]
    ranked = df[["cell_id", "lon", "lat", "x_m", "y_m"]].copy()
    ranked["score"] = scores_all
    ranked["suitable_cassa"] = y
    ranked = ranked.sort_values("score", ascending=False)
    ranked.to_parquet(args.out_dir / "ranked_cells.parquet", index=False)
    ranked.head(500).to_csv(args.out_dir / "top500_cells.csv", index=False)

    # SHAP
    try:
        import shap
        print("Computing SHAP on 5000-cell sample ...")
        sample = np.random.RandomState(args.seed).choice(len(X), size=min(5000, len(X)), replace=False)
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X[sample])

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        shap.summary_plot(shap_vals, X[sample], feature_names=feature_cols, show=False)
        plt.tight_layout()
        plt.savefig(args.out_dir / "shap_summary.png", dpi=150)
        plt.close()

        mean_abs = np.abs(shap_vals).mean(axis=0)
        feat_imp = sorted(zip(feature_cols, mean_abs), key=lambda kv: kv[1], reverse=True)
        (args.out_dir / "feature_importance.json").write_text(
            json.dumps([{"feature": f, "mean_abs_shap": float(v)} for f, v in feat_imp], indent=2)
        )
        print("Top 10 features by |SHAP|:")
        for f, v in feat_imp[:10]:
            print(f"  {f:<30} {v:.4f}")
    except ImportError:
        print("shap not installed, skipping explainability. pip install shap")

    model.save_model(str(args.out_dir / "site_scorer.xgb"))
    summary = {
        "n_cells": int(len(df)),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "cassa_thresholds": thresholds.__dict__,
        "cassa_positive_rate": float(pos / len(df)),
        "test_auc": float(auc) if not np.isnan(auc) else None,
        "test_average_precision": float(ap) if not np.isnan(ap) else None,
        "seed": args.seed,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nArtifacts in {args.out_dir}/")


if __name__ == "__main__":
    main()
