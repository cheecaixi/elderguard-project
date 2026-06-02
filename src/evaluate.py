# evaluate.py
# Loads saved models and computes test-set metrics on held-out data.
# Shows confusion matrix, classification report, and per-class metrics.

# Usage:
#   python src/evaluate.py                    # evaluate all models
#   python src/evaluate.py --model random_forest  # evaluate specific model

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODEL_SAVE_DIR


# ── 1. Load Test Data ─────────────────────────────────────────────────────────
def load_test_data(model_dir: str, use_scaled: bool = True):
    """Load test data saved by train.py."""
    X_path = os.path.join(model_dir, "X_test_scaled.parquet") if use_scaled else os.path.join(model_dir, "X_test.parquet")
    y_path = os.path.join(model_dir, "y_test.npy")
    
    X_test = pd.read_parquet(X_path)
    y_test = np.load(y_path)
    print(f"[load] {X_test.shape[0]:,} test samples, {'scaled' if use_scaled else 'unscaled'}")
    return X_test, y_test


# ── 2. Load Model ─────────────────────────────────────────────────────────────
def load_model(model_dir: str, model_name: str):
    """Load trained model and artefacts."""
    model = joblib.load(os.path.join(model_dir, f"{model_name}.joblib"))
    
    with open(os.path.join(model_dir, "feature_names.json"), "r") as f:
        features = json.load(f)
    with open(os.path.join(model_dir, "activity_map.json"), "r") as f:
        rev_map = {v: k for k, v in json.load(f).items()}
    
    return model, features, rev_map


# ── 3. Evaluate ───────────────────────────────────────────────────────────────
def evaluate_model(model_dir: str, model_name: str) -> dict:
    """Evaluate a single model on test data."""
    print(f"\n{'='*50}\n  {model_name.upper()}\n{'='*50}")
    
    # Load data (LR needs scaling, tree models don't)
    needs_scaling = model_name == "logistic_regression"
    X_test, y_test = load_test_data(model_dir, use_scaled=needs_scaling)
    
    # Load model
    model, features, rev_map = load_model(model_dir, model_name)
    X_test = X_test[features]  # ensure correct column order
    
    # Predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    
    print(f"Accuracy:    {acc:.4f}")
    print(f"Macro F1:    {f1_macro:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=["low", "moderate", "high"], 
                                digits=3))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low','Mod','High'], 
                yticklabels=['Low','Mod','High'])
    plt.title(f'{model_name.replace("_"," ").title()}')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f"cm_{model_name}.png"), dpi=120)
    plt.show()
    
    return {"name": model_name, "accuracy": acc, "f1_macro": f1_macro, "cm": cm}


# ── 4. Evaluate All ───────────────────────────────────────────────────────────
def evaluate_all(model_dir: str) -> pd.DataFrame:
    """Evaluate all saved models."""
    models = [f.replace(".joblib", "") for f in os.listdir(model_dir) 
              if f.endswith(".joblib") and f != "scaler.joblib"]
    
    results = {}
    for name in models:
        results[name] = evaluate_model(model_dir, name)
    
    # Summary
    print(f"\n{'='*50}\n  SUMMARY\n{'='*50}")
    for name, res in sorted(results.items(), key=lambda x: x[1]["f1_macro"], reverse=True):
        print(f"  {name:<25} Acc={res['accuracy']:.4f}  F1={res['f1_macro']:.4f}")
    
    # Comparison plot
    if len(results) > 1:
        plt.figure(figsize=(8, 5))
        names = list(results.keys())
        scores = [results[n]["f1_macro"] for n in names]
        bars = plt.bar(names, scores, color=['#2E86AB', '#A23B72', '#F18F01'])
        plt.ylim(0, 1)
        plt.ylabel("Macro F1 Score")
        plt.title("Model Comparison")
        plt.xticks(rotation=15)
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "comparison.png"), dpi=120)
        plt.show()
    
    return pd.DataFrame(results).T


# ── 5. Main ───────────────────────────────────────────────────────────────────
def run_evaluation(model_dir: str = MODEL_SAVE_DIR, model_name: str = None) -> None:
    print(f"\n{'='*50}\n  EVALUATION PIPELINE — START\n{'='*50}")
    
    if model_name:
        evaluate_model(model_dir, model_name)
    else:
        evaluate_all(model_dir)
    
    print(f"\n{'='*50}\n  EVALUATION PIPELINE — COMPLETE\n{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=MODEL_SAVE_DIR)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    run_evaluation(args.model_dir, args.model)