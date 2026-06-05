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
warnings.filterwarnings("ignore", message="`sklearn.utils.parallel.delayed` should be used with")
warnings.filterwarnings("ignore", module="sklearn.utils.parallel")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    DB_PATH, TEST_SIZE, RANDOM_STATE, MODEL_SAVE_DIR,
    TUNE_MODELS,
    RF_PARAMS,  RF_PARAM_GRID,
    XGB_PARAMS, XGB_PARAM_GRID,
    LR_PARAMS,  LR_PARAM_GRID,
    CV_FOLDS, CV_SCORING,
)
from src.cleaning import clean_data
from src.features import build_features, scale_features

# ── Models ────────────────────────────────────────────────────────
def get_models() -> dict:
    """
    Define the three models used in the pipeline.

    - LogisticRegression      : linear model for binary classification; requires scaling.
    - RandomForestClassifier  : ensemble of trees, robust to outliers and
                                non-linear interactions; no scaling needed.
    - XGBClassifier           : gradient boosting with sample_weight for class
                                imbalance handling; generally strongest on tabular data.

    All tree models use class_weight=balanced or sample_weight to handle the
    ~58/28/14% class imbalance across low/moderate/high activity.
    """
    return {
        "logistic_regression": {
        "model":            LogisticRegression(**LR_PARAMS),
        "param_grid":       LR_PARAM_GRID,
        "needs_scaling":    True,   # uses X_train_scaled
        "use_sample_weight": False,
    },
        "random_forest": {
            "model": RandomForestClassifier(**RF_PARAMS),
            "param_grid": RF_PARAM_GRID,
            "needs_scaling": False,
            "use_sample_weight": False,
        },
        "xgboost": {
            "model": XGBClassifier(**XGB_PARAMS),
            "param_grid": XGB_PARAM_GRID,
            "needs_scaling": False,
            "use_sample_weight": True,
        },
    }


# ── Split ─────────────────────────────────────────────────────────────────────
def split_data(X: pd.DataFrame, y: np.ndarray, save_dir: str):
    """
    Stratified train/test split. Both splits saved to disk for evaluate.py.
    stratify=y preserves class distribution across both splits.
    """
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
def tune_model(model, param_grid: dict, X_train: pd.DataFrame,
               y_train: np.ndarray, name: str, fit_kwargs: dict = {}):
    """
    Hyperparameter tuning using GridSearchCV with StratifiedKFold.

    SMOTE is applied inside each CV fold via ImbPipeline to prevent
    data leakage. Applying SMOTE before CV inflates scores by ~0.21
    because synthetic validation samples contaminate training folds.

    Justification:
    - GridSearchCV exhaustively searches all param_grid combinations.
    - StratifiedKFold preserves class distribution in each fold.
    - Macro F1 used as scoring metric — consistent with evaluation.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    if SMOTE_AVAILABLE:
        pipe = ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
            ("clf",   model)
        ])
        pipe_grid = {f"clf__{k}": v for k, v in param_grid.items()}
        gs = GridSearchCV(pipe, pipe_grid, cv=cv, scoring=CV_SCORING,
                          n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train)
        clean_params = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
        print(f"[tune] {name} best params : {clean_params}")
        print(f"[tune] {name} best CV {CV_SCORING}: {gs.best_score_:.4f}")
        return gs.best_estimator_.named_steps["clf"], gs.best_score_
    else:
        gs = GridSearchCV(model, param_grid, cv=cv, scoring=CV_SCORING,
                          n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train, **fit_kwargs)
        print(f"[tune] {name} best params : {gs.best_params_}")
        print(f"[tune] {name} best CV {CV_SCORING}: {gs.best_score_:.4f}")
        return gs.best_estimator_, gs.best_score_


# ── Train ─────────────────────────────────────────────────────────
def train_models(X_train: pd.DataFrame, X_train_scaled: pd.DataFrame,
                 y_train: np.ndarray, tune: bool) -> dict:
    """
    Train all models. SMOTE is applied inside each CV fold via tune_model
    to prevent leakage. The final model is fit on the full training set
    without SMOTE — class_weight='balanced'/'balanced_subsample' handles
    imbalance at final fit time.
    """
    models  = get_models()
    trained = {}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, cfg in models.items():
        X      = X_train_scaled if cfg["needs_scaling"] else X_train
        print(f"\n{'='*50}\n  {name.upper().replace('_', ' ')}\n{'='*50}")

        # ── BEFORE tuning: honest CV score with SMOTE inside folds ──
        print(f"\n  --- BEFORE TUNING ---")
        before_scores = []
        for tr_idx, val_idx in cv.split(X, y_train):
            from sklearn.base import clone
            X_tr, X_val = X.iloc[tr_idx].copy(), X.iloc[val_idx].copy()
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
            fold_model = clone(cfg["model"])
            if SMOTE_AVAILABLE:
                X_tr_f = X_tr.astype({c: "float64" for c in X_tr.columns})
                X_tr_f, y_tr_f = SMOTE(random_state=RANDOM_STATE, k_neighbors=5).fit_resample(X_tr_f, y_tr)
                X_tr_f = pd.DataFrame(X_tr_f, columns=X_tr.columns)
            else:
                X_tr_f, y_tr_f = X_tr, y_tr
            fold_model.fit(X_tr_f, y_tr_f)
            before_scores.append(f1_score(y_val, fold_model.predict(X_val), average="macro"))

        before_cv_mean = np.mean(before_scores)
        before_cv_std  = np.std(before_scores)

        # Train on full X (no SMOTE) — class_weight handles imbalance
        cfg["model"].fit(X, y_train)
        y_pred = cfg["model"].predict(X)
        print(f"[before] CV {CV_SCORING}    : {before_cv_mean:.4f} (+/- {before_cv_std:.4f})")
        print(f"[before] Train accuracy  : {accuracy_score(y_train, y_pred):.4f}")
        print(f"[before] Train macro F1  : {f1_score(y_train, y_pred, average='macro'):.4f}")
        print(f"[before] Train precision : {precision_score(y_train, y_pred, average='macro', zero_division=0):.4f}")
        print(f"[before] Train recall    : {recall_score(y_train, y_pred, average='macro', zero_division=0):.4f}")

        # ── AFTER tuning ─────────────────────────────────────────────
        if tune:
            fitted, best_cv_score = tune_model(cfg["model"], cfg["param_grid"],
                                               X, y_train, name)
            print(f"\n  --- AFTER TUNING ---")
            y_pred = fitted.predict(X)
            print(f"[tune] CV {CV_SCORING}    : {best_cv_score:.4f}")
            print(f"[tune] Train accuracy  : {accuracy_score(y_train, y_pred):.4f}")
            print(f"[tune] Train macro F1  : {f1_score(y_train, y_pred, average='macro'):.4f}")
            print(f"[tune] Train precision : {precision_score(y_train, y_pred, average='macro', zero_division=0):.4f}")
            print(f"[tune] Train recall    : {recall_score(y_train, y_pred, average='macro', zero_division=0):.4f}")
            print(f"\n  --- IMPROVEMENT ---")
            print(f"[tune] CV {CV_SCORING} delta : {best_cv_score - before_cv_mean:+.4f}")
            after_cv_mean = best_cv_score
        else:
            print("[train] Tuning skipped — using default params")
            fitted        = cfg["model"]
            after_cv_mean = before_cv_mean

        print(f"\n[train] Note: train metrics will be optimistic — use evaluate.py for true test-set performance")

        trained[name] = {
            "model":         fitted,
            "needs_scaling": cfg["needs_scaling"],
            "cv_mean":       float(after_cv_mean),
        }

    # Ranking
    print(f"\n{'='*50}\n  MODEL RANKING\n{'='*50}")
    for rank, (name, cfg) in enumerate(
        sorted(trained.items(), key=lambda x: x[1]["cv_mean"], reverse=True), 1
    ):
        print(f"  {rank}. {name:<25} CV F1 = {cfg['cv_mean']:.4f}")

    return trained


# ── Save ──────────────────────────────────────────────────────────────────────
def save_artefacts(trained: dict, save_dir: str, scaler,
                   feature_names: list, activity_map: dict) -> None:
    """
    Save all models and pipeline artefacts to disk.

    Files saved:
    - <model_name>.joblib  : trained sklearn/xgb model
    - scaler.joblib        : fitted StandardScaler
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
def run_training(db_path: str = DB_PATH,
                 save_dir: str = MODEL_SAVE_DIR,
                 tune: bool = TUNE_MODELS) -> None:
    sep = "=" * 50
    print(f"\n{sep}\n  TRAINING PIPELINE — START")
    print(f"  Tuning: {'ON (GridSearchCV)' if tune else 'OFF (default params)'}\n{sep}")

    # 1. Clean + features
    X, y, activity_map, feature_names = build_features(clean_data(db_path))

    # 2. Split (saves unscaled parquet files for evaluate.py)
    X_train, X_test, y_train, y_test = split_data(X, y, save_dir)
    # Cast to float64 so SMOTE can generate synthetic samples without dtype conflicts.
    # CO_GasSensor uses pandas Int64 (nullable) which SMOTE cannot cast back from float64.
    X_train = X_train.astype("float64")
    X_test  = X_test.astype("float64")

    # 3. Scale — Fit on train only, transform both (Kept for downstream file compatibility)
    X_train_scaled, scaler = scale_features(X_train)
    X_test_scaled, _       = scale_features(X_test, scaler=scaler)
    X_train_scaled = X_train_scaled.fillna(0)
    X_test_scaled  = X_test_scaled.fillna(0)

    # Save scaled frames for evaluate.py file-existence checks
    X_train_scaled.to_parquet(os.path.join(save_dir, "X_train_scaled.parquet"), index=False)
    X_test_scaled.to_parquet(os.path.join(save_dir,  "X_test_scaled.parquet"),  index=False)

    # 4. Train (Now routes unscaled matrices to your Tree/Boosting architectures)
    trained = train_models(X_train, X_train_scaled, y_train, tune)

    # 5. Save
    save_artefacts(trained, save_dir, scaler, feature_names, activity_map)

    print(f"\n{sep}\n  TRAINING PIPELINE — COMPLETE")
    print(f"  Artefacts saved to : {save_dir}")
    print(f"  Run evaluate.py to compute test-set metrics.\n{sep}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML models on gas monitoring data")
    parser.add_argument("--db",       type=str, default=DB_PATH,
                        help="Path to gas_monitoring.db")
    parser.add_argument("--save-dir", type=str, default=MODEL_SAVE_DIR,
                        help="Directory to save artefacts")
    parser.add_argument("--no-tune",  action="store_true",
                        help="Skip GridSearchCV")
    args = parser.parse_args()
    run_training(db_path=args.db, save_dir=args.save_dir, tune=not args.no_tune)