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
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
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

    - LogisticRegression    : Linear model; requires scaled continuous values.
    - RandomForestClassifier: Ensemble of trees; handles non-linear interactions;
                              does not strictly require scaling.
    - XGBClassifier         : Gradient boosted trees optimized for tabular layouts.

    Imbalance Management Execution:
    - This model registry acts as the entry point for configuration variables.
    - Logistic Regression and Random Forest pass downstream parameters 
      (class_weight='balanced' and 'balanced_subsample') which mathematically 
      penalize class underrepresentation during training. (when missclassify high_activity, 
      the model receives a stronger gradient signal to correct this error)
    - XGBoost does not have a built-in class_weight parameter, so we pass
      sample weights to the fit method based on class frequencies
    - To handle the ~58/28/14% class imbalance across low/moderate/high activity.
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
    Stratified 80/20 train/test split. Stratified means each split preserves the 
    same class proportions, so the minority High Activity class doesn't accidentally 
    end up mostly in one split. Both splits saved to disk for evaluate.py.

    Justification:
    - Prevents Data Leakage: Splitting data before scaling or tuning creates a strict firewall, 
      keeping the test set entirely hidden from the training process for unbiased evaluation.
    - Preserves Class Balance: Using stratify=y ensures both training and test sets have 
      identical class proportions, guaranteeing stable and reliable validation.
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

    Justification:
    - If SMOTE is available, an ImbPipeline wraps the SMOTE resampling step and 
      the classifier together. This ensures SMOTE runs dynamically only withon training folds.
      This prevents synthetic data from leaking into validation folds, ensuring unbiased scoring.
    - GridSearchCV optimizes explicitly for Macro F1 scores, prioritizing minority class 
      performance rather than overall accuracy on imbalanced data.
    - The final returned best estimator is automatically refit by 
      scikit-learn on the whole dataset using the ImbPipeline structure, embedding the SMOTE-augmented 
      architecture directly into the finalized model.
    - GridSearchCV tries every combination of model settings to find the one that gets the highest score.
    - StratifiedKFold splits the data into equal parts for testing, making sure every part has the 
      exact same balance of categories so the test is fair. They find the absolute best settings for the 
      model while making sure it doesn't get lucky or cheat during the test.
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
        gs = GridSearchCV(model, param_grid, cv=cv, scoring=CV_SCORING, n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train, **fit_kwargs)
        print(f"[tune] {name} best params : {gs.best_params_}")
        print(f"[tune] {name} best CV {CV_SCORING}: {gs.best_score_:.4f}")
        return gs.best_estimator_, gs.best_score_


# ── Sample Weights ─────────────────────────────────────────────────────────────
def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """
    Compute per-sample weights inversely proportional to class frequency.

    Used for XGBoost which has no built-in class_weight parameter.
    Equivalent effect to class_weight='balanced' in sklearn models:
    minority classes (e.g. high_activity ~14%) receive higher weight
    so the model does not optimise purely for the majority class.

    Formula: weight = n_samples / (n_classes * class_count)
    """
    return compute_sample_weight(class_weight="balanced", y=y)

# ── Train ──────────────────────────────────────────────────────────────────────
def train_models(X_train: pd.DataFrame, X_train_scaled: pd.DataFrame,
                 y_train: np.ndarray, tune: bool) -> dict:
    """
    Orchestrates model baseline loops, hyperparameter optimization, and tracking.

    Actual execution logic breakdown:
    - Baseline Evaluation ("Before Tuning"): Computes baseline Macro F1 scores via 
      manual cross-validation, applying SMOTE strictly inside individual training folds to eliminate leakage.
    - Right after scoring inside the baseline step, the raw base model is fit directly onto the full training 
      array. Instead of SMOTE, class imbalance is handled algorithmically via internal class weights 
      (for RF/LR) or dynamically computed sample weights (for XGBoost).
    - Hyperparameter Optimization ("After Tuning"): Re-routes the pipeline into GridSearchCV with StratifiedKFold, 
      exhaustively searching for the best hyperparameters using Macro F1 as the guiding metric.
    """
    models  = get_models()
    trained = {}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, cfg in models.items():
        X = X_train_scaled if cfg["needs_scaling"] else X_train
        print(f"\n{'='*50}\n  {name.upper().replace('_', ' ')}\n{'='*50}")

        # Pre-compute sample weights for models that need them (XGBoost)
        sample_weights = (
            compute_sample_weights(y_train) if cfg["use_sample_weight"] else None
        )

        # ── BEFORE tuning: honest CV score with SMOTE inside folds ──────────
        print(f"\n  --- BEFORE TUNING ---")
        before_scores = []
        for tr_idx, val_idx in cv.split(X, y_train):
            from sklearn.base import clone
            X_tr, X_val = X.iloc[tr_idx].copy(), X.iloc[val_idx].copy()
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
            fold_model  = clone(cfg["model"])

            if SMOTE_AVAILABLE:
                X_tr_f = X_tr.astype({c: "float64" for c in X_tr.columns})
                X_tr_f, y_tr_f = SMOTE(random_state=RANDOM_STATE, k_neighbors=5).fit_resample(X_tr_f, y_tr)
                X_tr_f = pd.DataFrame(X_tr_f, columns=X_tr.columns)
                # After SMOTE the class distribution is balanced, so sample
                # weights are recomputed on the resampled labels, not the
                # original fold weights — prevents double-penalising minority class.
                fold_fit_kwargs = (
                    {"sample_weight": compute_sample_weights(y_tr_f)}
                    if cfg["use_sample_weight"] else {}
                )
                fold_model.fit(X_tr_f, y_tr_f, **fold_fit_kwargs)
            else:
                fold_fit_kwargs = (
                    {"sample_weight": compute_sample_weights(y_tr)}
                    if cfg["use_sample_weight"] else {}
                )
                fold_model.fit(X_tr, y_tr, **fold_fit_kwargs)

            before_scores.append(f1_score(y_val, fold_model.predict(X_val), average="macro"))

        before_cv_mean = np.mean(before_scores)
        before_cv_std  = np.std(before_scores)

        # Baseline training on full training partition
        baseline_fit_kwargs = {"sample_weight": sample_weights} if cfg["use_sample_weight"] else {}
        cfg["model"].fit(X, y_train, **baseline_fit_kwargs)
        y_pred = cfg["model"].predict(X)
        print(f"[before] CV {CV_SCORING}    : {before_cv_mean:.4f} (+/- {before_cv_std:.4f})")
        print(f"[before] Train accuracy  : {accuracy_score(y_train, y_pred):.4f}")
        print(f"[before] Train macro F1  : {f1_score(y_train, y_pred, average='macro'):.4f}")
        print(f"[before] Train precision : {precision_score(y_train, y_pred, average='macro', zero_division=0):.4f}")
        print(f"[before] Train recall    : {recall_score(y_train, y_pred, average='macro', zero_division=0):.4f}")

        # ── AFTER tuning ──────────────────────────────────────────────────────
        if tune:
            fitted, best_cv_score = tune_model(
                cfg["model"], cfg["param_grid"], X, y_train, name,
                fit_kwargs=baseline_fit_kwargs   # passes sample_weight into GridSearchCV
            )
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
    for rank, (n, c) in enumerate(
        sorted(trained.items(), key=lambda x: x[1]["cv_mean"], reverse=True), 1
    ):
        print(f"  {rank}. {n:<25} CV F1 = {c['cv_mean']:.4f}")

    return trained


# ── Save ──────────────────────────────────────────────────────────────────────
def save_artefacts(trained: dict, save_dir: str, scaler,
                   feature_names: list, activity_map: dict) -> None:
    """
    Save all models and pipeline artefacts to disk.

    Files saved:
    - <model_name>.joblib  : the saved, fully-trained AI. You load it to make instant 
                             predictions without needing to retrain it from scratch.
    - scaler.joblib        : Stores the exact math used to resize your original data so 
                             you can shrink or normalize new data the exact same way.
    - feature_names.json   : locks down required matrix dimensions for evaluation
    - activity_map.json    : Translates the model's numeric outputs back into human-readable text
    - best_model.json      : logs the best model based on cross-validation macro F1
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

    # 3. Scale — applied on training data only, then the same scaler is used to transform 
    # the test data. This prevents data leakage.
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