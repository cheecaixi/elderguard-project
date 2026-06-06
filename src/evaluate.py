# evaluate.py
# Loads saved models and computes test-set metrics on held-out data.
# Shows confusion matrix, classification report, and per-class metrics.

# Usage:
#   python src/evaluate.py                        # evaluate all models
#   python src/evaluate.py --model random_forest  # evaluate one model

import os
import sys
import json
import argparse
import joblib
import matplotlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODEL_SAVE_DIR


# ── 1. Load Test Data ─────────────────────────────────────────────────────────
def load_test_data(model_dir: str, use_scaled: bool = True):
    """Load test data saved by train.py."""
    X_path = (os.path.join(model_dir, "X_test_scaled.parquet") if use_scaled
              else os.path.join(model_dir, "X_test.parquet"))
    y_path = os.path.join(model_dir, "y_test.npy")
    X_test = pd.read_parquet(X_path)
    y_test = np.load(y_path)
    print(f"[load] {X_test.shape[0]:,} test samples, {'scaled' if use_scaled else 'unscaled'}")
    return X_test, y_test


# ── 2. Load Train Data ────────────────────────────────────────────────────────
def load_train_data(model_dir: str, use_scaled: bool = True):
    """Load train data saved by train.py."""
    X_path = (os.path.join(model_dir, "X_train_scaled.parquet") if use_scaled
              else os.path.join(model_dir, "X_train.parquet"))
    y_path = os.path.join(model_dir, "y_train.npy")
    X_train = pd.read_parquet(X_path)
    y_train = np.load(y_path)
    print(f"[load] {X_train.shape[0]:,} train samples, {'scaled' if use_scaled else 'unscaled'}")
    return X_train, y_train


# ── 3. Load Model ─────────────────────────────────────────────────────────────
def load_model(model_dir: str, model_name: str):
    """Load trained model and artefacts."""
    model = joblib.load(os.path.join(model_dir, f"{model_name}.joblib"))
    with open(os.path.join(model_dir, "feature_names.json")) as f:
        features = json.load(f)
    with open(os.path.join(model_dir, "activity_map.json")) as f:
        rev_map = {v: k for k, v in json.load(f).items()}
    return model, features, rev_map


# ── 4. Print Metrics ──────────────────────────────────────────────────────────
def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    """Print accuracy, macro F1, classification report and confusion matrix."""
    acc      = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")

    print(f"\n  ── {label} ──────────────────────────────────")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {f1_macro:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=["low", "moderate", "high"],
                                digits=3))
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion Matrix:\n{cm}")
    return {"accuracy": acc, "f1_macro": f1_macro, "cm": cm}


# ── 5. Evaluate Single Model ──────────────────────────────────────────────────
def evaluate_model(model_dir: str, model_name: str) -> dict:
    """Evaluate a single model — prints TRAIN results then TEST results."""
    print(f"\n{'='*50}\n  {model_name.upper()}\n{'='*50}")

    # Load model
    model, features, rev_map = load_model(model_dir, model_name)

    # ── BEFORE: training set ──────────────────────────────────────────────────
    needs_scaled = (model_name == "logistic_regression")

    X_train, y_train = load_train_data(model_dir, use_scaled=needs_scaled)
    X_train          = X_train[features]
    y_train_pred     = model.predict(X_train)
    train_res = print_metrics(y_train, y_train_pred, label="TRAIN SET (after tuning)")

    plt.figure(figsize=(7, 5))
    sns.heatmap(train_res["cm"], annot=True, fmt="d", cmap="Greens",
                xticklabels=["Low", "Mod", "High"],
                yticklabels=["Low", "Mod", "High"])
    plt.title(f'{model_name.replace("_", " ").title()} — Train')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f"cm_{model_name}_train.png"), dpi=120)

    # ── AFTER: test set ───────────────────────────────────────────────────────
    X_test, y_test = load_test_data(model_dir, use_scaled=needs_scaled)
    X_test         = X_test[features]
    y_test_pred    = model.predict(X_test)
    test_res = print_metrics(y_test, y_test_pred, label="TEST SET  (after tuning)")

    plt.figure(figsize=(7, 5))
    sns.heatmap(test_res["cm"], annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low", "Mod", "High"],
                yticklabels=["Low", "Mod", "High"])
    plt.title(f'{model_name.replace("_", " ").title()} — Test')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f"cm_{model_name}_test.png"), dpi=120)

    return {
        "name":           model_name,
        "train_accuracy": train_res["accuracy"],
        "train_f1_macro": train_res["f1_macro"],
        "test_accuracy":  test_res["accuracy"],
        "test_f1_macro":  test_res["f1_macro"],
        "cm_train":       train_res["cm"],
        "cm_test":        test_res["cm"],
    }


# ── 6. Evaluate All ───────────────────────────────────────────────────────────
def evaluate_all(model_dir: str) -> None:
    """Evaluate all saved models and print a ranked summary."""
    models = [f.replace(".joblib", "") for f in os.listdir(model_dir)
              if f.endswith(".joblib") and f != "scaler.joblib"]

    results = {}
    for name in models:
        results[name] = evaluate_model(model_dir, name)

    # Summary table
    print(f"\n{'='*50}\n  SUMMARY\n{'='*50}")
    print(f"  {'Model':<25} {'Train Acc':>10} {'Train F1':>10} {'Test Acc':>10} {'Test F1':>10}")
    print(f"  {'-'*65}")
    for name, res in sorted(results.items(),
                            key=lambda x: x[1]["test_f1_macro"], reverse=True):
        print(f"  {name:<25} {res['train_accuracy']:>10.4f} {res['train_f1_macro']:>10.4f} "
              f"{res['test_accuracy']:>10.4f} {res['test_f1_macro']:>10.4f}")

    # Comparison bar chart
    if len(results) > 1:
        raw_names   = list(results.keys())
        clean_names = [n.replace("_", " ").title() for n in raw_names]
        
        train_f1s   = [results[n]["train_f1_macro"] for n in raw_names]
        test_f1s    = [results[n]["test_f1_macro"]  for n in raw_names]
        x           = np.arange(len(raw_names))
        width       = 0.35

        fig, ax = plt.subplots(figsize=(9, 5))
        bars1 = ax.bar(x - width / 2, train_f1s, width, label="Train F1", color="#4CAF50")
        bars2 = ax.bar(x + width / 2, test_f1s,  width, label="Test F1",  color="#2E86AB")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Macro F1 Score")
        ax.set_title("Model Comparison — Train vs Test Macro F1")
        ax.set_xticks(x)
        ax.set_xticklabels(clean_names, rotation=15)  # FIXED: Uses clean formatting on chart labels
        ax.legend()
        for bar, score in zip(bars1, train_f1s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{score:.3f}", ha="center", fontweight="bold", fontsize=9)
        for bar, score in zip(bars2, test_f1s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{score:.3f}", ha="center", fontweight="bold", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "comparison.png"), dpi=120)


# ── 7. Main ───────────────────────────────────────────────────────────────────
def run_evaluation(model_dir: str = MODEL_SAVE_DIR, model_name: str = None) -> None:
    print(f"\n{'='*50}\n  EVALUATION PIPELINE — START\n{'='*50}")
    if model_name:
        evaluate_model(model_dir, model_name)
    else:
        evaluate_all(model_dir)
    print(f"\n{'='*50}\n  EVALUATION PIPELINE — COMPLETE\n{'='*50}")

    plt.show(block=False)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=MODEL_SAVE_DIR)
    parser.add_argument("--model",     default=None)
    args = parser.parse_args()
    run_evaluation(args.model_dir, args.model)