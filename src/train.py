# train.py
# Training pipeline: clean → features → split → scale → tune → train → save

# Usage:
#   python src/train.py
#   python src/train.py --no-tune          (skip GridSearchCV, use default params)
#   python src/train.py --db <path>        (custom db path)
#   python src/train.py --save-dir <path>  (custom save directory)

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    DB_PATH, TEST_SIZE, RANDOM_STATE, MODEL_SAVE_DIR,
    TUNE_MODELS,
    RF_PARAMS,  RF_PARAM_GRID,
    GB_PARAMS,  GB_PARAM_GRID,
    LR_PARAMS,  LR_PARAM_GRID,
    CV_FOLDS, CV_SCORING,
)
from src.cleaning import clean_data
from src.features import build_features, scale_features


# ── Models ────────────────────────────────────────────────────────

def get_models() -> dict:
    """
    Define the three models used in the pipeline.

    - LogisticRegression          : linear baseline, interpretable, needs scaling
    - RandomForestClassifier      : robust to outliers and nonlinear interactions
    - HistGradientBoostingClassifier : histogram-based boosting, supports class_weight,
                                      generally strongest on tabular data
    All use class_weight=balanced to handle the ~58/28/14% class imbalance.
    """
    return {
        "logistic_regression": {
            "model": LogisticRegression(**LR_PARAMS),
            "param_grid": LR_PARAM_GRID,
            "needs_scaling": True,
        },
        "random_forest": {
            "model": RandomForestClassifier(**RF_PARAMS),
            "param_grid": RF_PARAM_GRID,
            "needs_scaling": False,
        },
        "gradient_boosting": {
            "model": HistGradientBoostingClassifier(**GB_PARAMS),
            "param_grid": GB_PARAM_GRID,
            "needs_scaling": False,
        },
    }

# ── Split ─────────────────────────────────────────────────────────────────────
def split_data(X, y, save_dir: str):
    """
    Stratified train/test split. Test split saved to disk for evaluate.py.
    stratify=y preserves class distribution across both splits.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[split] Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")

    os.makedirs(save_dir, exist_ok=True)
    X_test.to_parquet(os.path.join(save_dir, "X_test.parquet"), index=False)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test)
    return X_train, X_test, y_train, y_test


# ── Tune ──────────────────────────────────────────────────────────

def tune_model(model, param_grid: dict, X_train, y_train, name: str):
    """
    Tuning using GridSearchCV with StratifiedKFold.

    Justification:
    - GridSearchCV exhaustively searches the param_grid combinations
    - StratifiedKFold preserves class distribution in each fold
      — important given class imbalance (~58% Low Activity)
    - Macro F1 used as scoring metric — consistent with evaluation
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(model, param_grid, cv=cv, scoring=CV_SCORING, n_jobs=-1, verbose=0)
    
    gs.fit(X_train, y_train)
    print(f"[tune] {name} best params : {gs.best_params_}")
    print(f"[tune] {name} best CV {CV_SCORING}: {gs.best_score_:.4f}")
    return gs.best_estimator_


# ── Train ─────────────────────────────────────────────────────────
def train_models(X_train, X_train_scaled, y_train, tune: bool) -> dict:
    """
    Train all models. If tune=True, run GridSearchCV first.
    Reports 5-fold CV score after training to check for overfitting.
    Returns trained model configs ranked by CV macro F1.
    """
    models = get_models()
    trained = {}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, cfg in models.items():
        X = X_train_scaled if cfg["needs_scaling"] else X_train
        print(f"\n{'='*50}\n  {name.upper().replace('_', ' ')}\n{'='*50}")

        if tune:
            # ── BEFORE tuning ─────────────────────────────
            print(f"\n  --- BEFORE TUNING ---")
            before_scores = cross_val_score(cfg["model"], X, y_train, cv=cv, scoring=CV_SCORING, n_jobs=-1)
            cfg["model"].fit(X, y_train)
            y_pred = cfg["model"].predict(X)
            print(f"[before] CV {CV_SCORING}    : {before_scores.mean():.4f} (+/- {before_scores.std():.4f})")
            print(f"[before] Train accuracy  : {accuracy_score(y_train, y_pred):.4f}")
            print(f"[before] Train macro F1  : {f1_score(y_train, y_pred, average='macro'):.4f}")
            print(f"[before] Train precision : {precision_score(y_train, y_pred, average='macro', zero_division=0):.4f}")
            print(f"[before] Train recall    : {recall_score(y_train, y_pred, average='macro', zero_division=0):.4f}")

            # ── AFTER tuning ──────────────────────────────
            fitted = tune_model(cfg["model"], cfg["param_grid"], X, y_train, name)
            print(f"\n  --- AFTER TUNING ---")
            after_scores = cross_val_score(fitted, X, y_train, cv=cv, scoring=CV_SCORING, n_jobs=-1)
            y_pred = fitted.predict(X)
            print(f"[tune] CV {CV_SCORING}    : {after_scores.mean():.4f} (+/- {after_scores.std():.4f})")
            print(f"[tune] Train accuracy  : {accuracy_score(y_train, y_pred):.4f}")
            print(f"[tune] Train macro F1  : {f1_score(y_train, y_pred, average='macro'):.4f}")
            print(f"[tune] Train precision : {precision_score(y_train, y_pred, average='macro', zero_division=0):.4f}")
            print(f"[tune] Train recall    : {recall_score(y_train, y_pred, average='macro', zero_division=0):.4f}")

            # ── Improvement ───────────────────────────────
            print(f"\n  --- IMPROVEMENT ---")
            print(f"[tune] CV {CV_SCORING} delta : {after_scores.mean() - before_scores.mean():+.4f}")

        else:
            print("[train] Tuning skipped — using default params")
            fitted = cfg["model"].fit(X, y_train)
            after_scores = cross_val_score(fitted, X, y_train, cv=cv, scoring=CV_SCORING)
            y_pred = fitted.predict(X)
            print(f"[train] CV {CV_SCORING}    : {after_scores.mean():.4f} (+/- {after_scores.std():.4f})")
            print(f"[train] Train accuracy  : {accuracy_score(y_train, y_pred):.4f}")
            print(f"[train] Train macro F1  : {f1_score(y_train, y_pred, average='macro'):.4f}")
            print(f"[train] Train precision : {precision_score(y_train, y_pred, average='macro', zero_division=0):.4f}")
            print(f"[train] Train recall    : {recall_score(y_train, y_pred, average='macro', zero_division=0):.4f}")

        print(f"\n[train] Note: train metrics will be optimistic — use evaluate.py for true test-set performance")

        trained[name] = {
            "model":         fitted,
            "needs_scaling": cfg["needs_scaling"],
            "cv_mean":       float(after_scores.mean()),
        }

    # Print ranking
    print(f"\n{'='*50}\n  MODEL RANKING\n{'='*50}")
    for rank, (name, cfg) in enumerate(
        sorted(trained.items(), key=lambda x: x[1]["cv_mean"], reverse=True), 1
    ):
        print(f"  {rank}. {name:<25} CV F1 = {cfg['cv_mean']:.4f}")

    return trained


# ── Save ──────────────────────────────────────────────────────────────────────

def save_artefacts(trained: dict, save_dir: str, scaler, feature_names: list, activity_map: dict) -> None:
    """
    Save all models and pipeline artefacts to disk.

    Files saved:
    - <model_name>.joblib  : trained sklearn model
    - scaler.joblib        : fitted StandardScaler (for LR inference)
    - feature_names.json   : column order for consistent inference
    - activity_map.json    : class encoding (for decoding predictions)
    - best_model.json      : best model name by CV macro F1
    """
    os.makedirs(save_dir, exist_ok=True)

    for name, cfg in trained.items():
        path = os.path.join(save_dir, f"{name}.joblib")
        joblib.dump(cfg["model"], path)
        print(f"[save] {name} → {path}")

    joblib.dump(scaler, os.path.join(save_dir, "scaler.joblib"))

    with open(os.path.join(save_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    with open(os.path.join(save_dir, "activity_map.json"), "w") as f:
        json.dump(activity_map, f, indent=2)

    best = max(trained, key=lambda n: trained[n]["cv_mean"])
    with open(os.path.join(save_dir, "best_model.json"), "w") as f:
        json.dump({"best_model": best, "cv_score": trained[best]["cv_mean"]}, f, indent=2)

    print(f"[save] artefacts → {save_dir}/")
    print(f"[save] best model: {best} (CV {CV_SCORING} = {trained[best]['cv_mean']:.4f})")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_training(db_path: str = DB_PATH, save_dir: str = MODEL_SAVE_DIR, tune: bool = TUNE_MODELS) -> None:
    sep = "=" * 50
    print(f"\n{sep}\n  TRAINING PIPELINE — START")
    print(f"  Tuning: {'ON (GridSearchCV)' if tune else 'OFF (default params)'}\n{sep}")

    # 1. Clean + features
    X, y, activity_map, feature_names = build_features(clean_data(db_path))

    # 2. Split
    X_train, X_test, y_train, y_test = split_data(X, y, save_dir)


    # 3. Scale — fit on train only, transform both
    X_train_scaled, scaler = scale_features(X_train)
    X_test_scaled,  _      = scale_features(X_test, scaler=scaler)
    X_train_scaled = X_train_scaled.fillna(0)
    X_test_scaled  = X_test_scaled.fillna(0)
    X_test_scaled.to_parquet(os.path.join(save_dir, "X_test_scaled.parquet"), index=False)

    # 4. Train
    trained = train_models(X_train, X_train_scaled, y_train, tune)

    # 5. Save
    save_artefacts(trained, save_dir, scaler, feature_names, activity_map)

    print(f"\n{sep}\n  TRAINING PIPELINE — COMPLETE")
    print(f"  Artefacts saved to : {save_dir}")
    print(f"  Run evaluate.py to compute test-set metrics.\n{sep}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML models on gas monitoring data")
    parser.add_argument("--db",       
                        type=str, default=DB_PATH,        
                        help="Path to gas_monitoring.db")
    parser.add_argument("--save-dir", 
                        type=str, 
                        default=MODEL_SAVE_DIR, 
                        help="Directory to save artefacts")
    parser.add_argument("--no-tune",
                        action="store_true",              
                        help="Skip GridSearchCV")
    args = parser.parse_args()
    run_training(db_path=args.db, 
                 save_dir=args.save_dir, 
                 tune=not args.no_tune)

