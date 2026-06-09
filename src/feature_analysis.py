# feature_analysis.py
# Identifies and evaluates feature contributions using the best saved model.
# Combines RF feature importance and permutation importance.

# Usage:
#   python src/feature_analysis.py
#   python src/feature_analysis.py --model-dir <path>

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODEL_SAVE_DIR

CLASS_NAMES = ["low", "moderate", "high"]


# ── 1. Load artefacts ─────────────────────────────────────────────────────────
def load_artefacts(model_dir: str):
    """Load best model, test data, and metadata from saved_model/."""

    # Identify best model from saved JSON
    best_path = os.path.join(model_dir, "best_model.json")
    with open(best_path) as f:
        best_info = json.load(f)
    model_name = best_info["best_model"]
    print(f"[load] Best model: {model_name} (CV f1_macro = {best_info['cv_score']:.4f})")

    # Load model
    model = joblib.load(os.path.join(model_dir, f"{model_name}.joblib"))

    # Load feature names and activity map
    with open(os.path.join(model_dir, "feature_names.json")) as f:
        feature_names = json.load(f)
    with open(os.path.join(model_dir, "activity_map.json")) as f:
        activity_map = json.load(f)

    # Load test data — use scaled if model needs it, unscaled otherwise
    needs_scaled = (model_name == "logistic_regression")
    x_file = "X_test_scaled.parquet" if needs_scaled else "X_test.parquet"
    X_test = pd.read_parquet(os.path.join(model_dir, x_file))[feature_names]
    y_test = np.load(os.path.join(model_dir, "y_test.npy"))
    print(f"[load] Test set: {X_test.shape[0]:,} samples | {'scaled' if needs_scaled else 'unscaled'}")

    return model, model_name, feature_names, activity_map, X_test, y_test


# ── 2. RF Feature Importance ──────────────────────────────────────────────────
def plot_rf_importance(model, feature_names: list, model_name: str,
                       model_dir: str) -> pd.Series:
    """
    Extracts built-in feature_importances_ to measure the mean decrease in impurity across all trees. 
    This identifies which features contribute most to model's decisions.

    Automatically flags and highlights top-performing variables (above the 75th percentile) using 
    distinct color coding and references a baseline mean threshold for clear scannability.
    Note: built-in importance can overvalue features with many unique values (high cardinality).
    Permutation Importance—is required to confirm the true impact of each feature without bias.
    """
    if not hasattr(model, "feature_importances_"):
        print("[rf_importance] Model has no feature_importances_ — skipping")
        return None

    imp = pd.Series(model.feature_importances_, index=feature_names)
    imp = imp.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#2E86AB" if v >= imp.quantile(0.75) else "#A8DADC" for v in imp]
    imp.plot(kind="barh", ax=ax, color=colors)
    ax.set_xlabel("Mean Decrease in Impurity (Feature Importance)")
    ax.set_title(f"Feature Importance — {model_name.replace('_', ' ').title()}")
    ax.axvline(imp.mean(), color="red", linestyle="--", linewidth=1, label=f"Mean ({imp.mean():.3f})")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(model_dir, "feature_importance_rf.png")
    plt.savefig(out, dpi=120)
    print(f"[rf_importance] Saved → {out}")

    print(f"\n[rf_importance] Rankings:")
    for feat, val in imp.sort_values(ascending=False).items():
        print(f"  {feat:<45} {val:.6f}")

    return imp.sort_values(ascending=False)


# ── 3. Permutation Importance ─────────────────────────────────────────────────
def plot_permutation_importance(model, X_test: pd.DataFrame, y_test: np.ndarray,
                                feature_names: list, model_name: str,
                                model_dir: str) -> pd.Series:
    """
    Permutation importance on the held-out test set.

    More reliable than built-in importance:
    - Measures actual drop in macro F1 when each feature is randomly shuffled
    - Not biased toward high-cardinality features
    - Computed on test data — reflects generalisation, not training fit
    """
    print("\n[permutation] Computing permutation importance on test set...")
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,
        scoring="f1_macro",
        random_state=42,
        n_jobs=-1
    )

    perm = pd.Series(result.importances_mean, index=feature_names)
    perm_std = pd.Series(result.importances_std, index=feature_names)
    perm = perm.sort_values(ascending=True)
    perm_std = perm_std[perm.index]

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#E63946" if v >= perm.quantile(0.75) else "#F4A261" for v in perm]
    ax.barh(perm.index, perm.values, xerr=perm_std.values,
            color=colors, capsize=3)
    ax.set_xlabel("Mean Decrease in Macro F1 (Permutation Importance)")
    ax.set_title(f"Permutation Importance — {model_name.replace('_', ' ').title()}")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    out = os.path.join(model_dir, "feature_importance_permutation.png")
    plt.savefig(out, dpi=120)
    print(f"[permutation] Saved → {out}")

    print(f"\n[permutation] Rankings:")
    for feat, val in perm.sort_values(ascending=False).items():
        print(f"  {feat:<45} {val:.6f} (+/- {perm_std[feat]:.6f})")

    return perm.sort_values(ascending=False)


# ── 4. Comparison Table ───────────────────────────────────────────────────────
def print_comparison(rf_imp: pd.Series, perm_imp: pd.Series) -> None:
    """
    Side-by-side ranking comparison between RF importance and permutation
    importance. Agreement across both methods strengthens the conclusion.
    """
    if rf_imp is None or perm_imp is None:
        return

    rf_rank   = {f: i + 1 for i, f in enumerate(rf_imp.index)}
    perm_rank = {f: i + 1 for i, f in enumerate(perm_imp.index)}

    print(f"\n{'='*65}")
    print(f"  FEATURE RANKING COMPARISON")
    print(f"  {'Feature':<45} {'RF Rank':>8} {'Perm Rank':>10}")
    print(f"  {'-'*63}")
    for feat in rf_imp.index:
        print(f"  {feat:<45} {rf_rank[feat]:>8} {perm_rank.get(feat, '-'):>10}")
    print(f"{'='*65}")


# ── Main ──────────────────────────────────────────────────────────────────────
def run_feature_analysis(model_dir: str = MODEL_SAVE_DIR) -> None:
    print(f"\n{'='*50}\n  FEATURE ANALYSIS — START\n{'='*50}")

    model, model_name, feature_names, activity_map, X_test, y_test = \
        load_artefacts(model_dir)

    rf_imp   = plot_rf_importance(model, feature_names, model_name, model_dir)
    perm_imp = plot_permutation_importance(model, X_test, y_test,
                                           feature_names, model_name, model_dir)
    print_comparison(rf_imp, perm_imp)

    print(f"\n{'='*50}\n  FEATURE ANALYSIS — COMPLETE")
    print(f"  Plots saved to: {model_dir}/\n{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature importance analysis")
    parser.add_argument("--model-dir", type=str, default=MODEL_SAVE_DIR)
    args = parser.parse_args()
    run_feature_analysis(args.model_dir)