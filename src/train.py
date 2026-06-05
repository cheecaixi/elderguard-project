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
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
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
    Define the three models used in the pipeline with algorithmic class balancing.
    """
    return {
        "logistic_regression": {
            "model":            LogisticRegression(**LR_PARAMS),
            "param_grid":       LR_PARAM_GRID,
            "needs_scaling":    True,   
        },
        "random_forest": {
            "model":            RandomForestClassifier(**RF_PARAMS),
            "param_grid":       RF_PARAM_GRID,
            "needs_scaling":    False,
        },
        "gradient_boosting": {
            "model":            HistGradientBoostingClassifier(**GB_PARAMS), 
            "param_grid":       GB_PARAM_GRID,
            "needs_scaling":    False,
        },
    }


# ── Split ─────────────────────────────────────────────────────────────────────
def split_data(X: pd.DataFrame, y: np.ndarray, save_dir: str):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[split] Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")

    os.makedirs(save_dir, exist_ok=True)
    X_test.to_parquet(os.path.join(save_dir, "X_test.parquet"), index=False)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test)
    X_train.to_parquet(os.path.join(save_dir, "X_train.parquet"), index=False)
    np.save(os.path.join(save_dir, "y_train.npy"), y_train)

    return X_train, X_test, y_train, y_test


# ── Tune ──────────────────────────────────────────────────────────
def tune_model(model_name: str, base_model, param_grid: dict, X_train: pd.DataFrame, y_train: np.ndarray, needs_scaling: bool):
    """
    Hyperparameter tuning using standard scikit-learn Pipeline with class weights.
    """
    steps = []
    if needs_scaling:
        from sklearn.preprocessing import StandardScaler
        steps.append(("scaler", StandardScaler()))
        
    steps.append(("model", base_model))
    pipeline = Pipeline(steps)
    
    pipe_param_grid = {}
    for k, v in param_grid.items():
        if k.startswith("model__"):
            pipe_param_grid[k] = v
        else:
            pipe_param_grid[f"model__{k}"] = v
            
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(pipeline, pipe_param_grid, cv=cv, scoring=CV_SCORING, n_jobs=-1, verbose=0)
    
    X_train_float = X_train.astype({col: "float64" for col in X_train.columns})
    gs.fit(X_train_float, y_train)
    
    print(f"[tune] {model_name} best params : {gs.best_params_}")
    print(f"[tune] {model_name} best CV {CV_SCORING}: {gs.best_score_:.4f}")
    
    return gs.best_estimator_.named_steps["model"], gs.best_score_, gs.best_estimator_


# ── Train ─────────────────────────────────────────────────────────
def train_models(X_train: pd.DataFrame, X_train_scaled: pd.DataFrame,
                 y_train: np.ndarray, tune: bool) -> dict:
    """
    Train models using Cost-Sensitive algorithms. Reports matching metrics.
    """
    models = get_models()
    trained = {}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    X_train_float = X_train.astype({col: "float64" for col in X_train.columns})
    X_train_scaled_float = X_train_scaled.astype({col: "float64" for col in X_train_scaled.columns})

    for name, cfg in models.items():
        print(f"\n{'='*50}\n  {name.upper().replace('_', ' ')}\n{'='*50}")

        X_base_raw = X_train_scaled_float if cfg["needs_scaling"] else X_train_float

        # ── BEFORE TUNING ──
        print(f"\n  --- BEFORE TUNING ---")
        before_cv_scores = []
        for train_idx, val_idx in cv.split(X_base_raw, y_train):
            X_tr_cv, X_val_cv = X_base_raw.iloc[train_idx], X_base_raw.iloc[val_idx]
            y_tr_cv, y_val_cv = y_train[train_idx], y_train[val_idx]
            
            fold_model = clone(cfg["model"])
            fold_model.fit(X_tr_cv, y_tr_cv)
            fold_pred = fold_model.predict(X_val_cv)
            before_cv_scores.append(f1_score(y_val_cv, fold_pred, average="macro"))

        before_cv_mean = np.mean(before_cv_scores)
        before_cv_std  = np.std(before_cv_scores)

        base_model_fitted = clone(cfg["model"])
        base_model_fitted.fit(X_base_raw, y_train)
        y_pred_before = base_model_fitted.predict(X_base_raw)

        print(f"[before] CV {CV_SCORING}    : {before_cv_mean:.4f} (+/- {before_cv_std:.4f})")
        print(f"[before] Train accuracy  : {accuracy_score(y_train, y_pred_before):.4f}")
        print(f"[before] Train macro F1  : {f1_score(y_train, y_pred_before, average='macro'):.4f}")

        # ── TUNING & AFTER METRICS ──
        if tune:
            fitted_estimator, best_cv_score, full_pipeline = tune_model(
                name, cfg["model"], cfg["param_grid"], X_train, y_train, cfg["needs_scaling"]
            )

            if cfg["needs_scaling"]:
                # Use full pipeline — scaler + model together
                full_pipeline.fit(X_train_float, y_train)
                fitted = full_pipeline
            else:
                # Tree models — refit bare model with tuned params on unscaled data
                fitted_estimator.fit(X_base_raw, y_train)
                fitted = fitted_estimator

            print(f"\n  --- AFTER TUNING ---")
            y_pred_after = fitted.predict(X_base_raw if not cfg["needs_scaling"] else X_train_float)
            print(f"[tune] CV {CV_SCORING}    : {best_cv_score:.4f}")
            print(f"[tune] Train accuracy  : {accuracy_score(y_train, y_pred_after):.4f}")
            print(f"[tune] Train macro F1  : {f1_score(y_train, y_pred_after, average='macro'):.4f}")
            print(f"\n  --- IMPROVEMENT ---")
            print(f"[tune] CV {CV_SCORING} delta : {best_cv_score - before_cv_mean:+.4f}")
            after_cv_mean = best_cv_score
        else:
            print("[train] Tuning skipped — using default params")
            fitted = base_model_fitted
            after_cv_mean = before_cv_mean

        trained[name] = {
            "model":         fitted,
            "needs_scaling": cfg["needs_scaling"],
            "cv_mean":       float(after_cv_mean),
        }

    print(f"\n{'='*50}\n  MODEL RANKING\n{'='*50}")
    for rank, (name, cfg) in enumerate(
        sorted(trained.items(), key=lambda x: x[1]["cv_mean"], reverse=True), 1
    ):
        print(f"  {rank}. {name:<25} CV F1 = {cfg['cv_mean']:.4f}")

    return trained

# ── Save ──────────────────────────────────────────────────────────────────────
def save_artefacts(trained: dict, save_dir: str, scaler,
                   feature_names: list, activity_map: dict) -> None:
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
def run_training(db_path: str = DB_PATH,
                 save_dir: str = MODEL_SAVE_DIR,
                 tune: bool = TUNE_MODELS) -> None:
    sep = "=" * 50
    print(f"\n{sep}\n  TRAINING PIPELINE — START")
    print(f"  Tuning: {'ON (GridSearchCV)' if tune else 'OFF (default params)'}\n{sep}")

    X, y, activity_map, feature_names = build_features(clean_data(db_path))
    X_train, X_test, y_train, y_test = split_data(X, y, save_dir)

    X_train_scaled, scaler = scale_features(X_train)
    X_test_scaled, _       = scale_features(X_test, scaler=scaler)
    X_train_scaled = X_train_scaled.fillna(0)
    X_test_scaled  = X_test_scaled.fillna(0)

    X_train_scaled.to_parquet(os.path.join(save_dir, "X_train_scaled.parquet"), index=False)
    X_test_scaled.to_parquet(os.path.join(save_dir,  "X_test_scaled.parquet"),  index=False)

    trained = train_models(X_train, X_train_scaled, y_train, tune)
    save_artefacts(trained, save_dir, scaler, feature_names, activity_map)

    print(f"\n{sep}\n  TRAINING PIPELINE — COMPLETE")
    print(f"  Artefacts saved to : {save_dir}")
    print(f"  Run evaluate.py to compute test-set metrics.\n{sep}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML models on gas monitoring data")
    parser.add_argument("--db",       type=str, default=DB_PATH, help="Path to gas_monitoring.db")
    parser.add_argument("--save-dir", type=str, default=MODEL_SAVE_DIR, help="Directory to save artefacts")
    parser.add_argument("--no-tune",  action="store_true", help="Skip GridSearchCV")
    args = parser.parse_args()
    run_training(db_path=args.db, save_dir=args.save_dir, tune=not args.no_tune)