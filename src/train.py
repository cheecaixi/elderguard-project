# train.py
# Training pipeline: loads data → builds features → splits → tunes → trains → saves.
#
# Usage:
#   python src/train.py
#   python src/train.py --db path/to/gas_monitoring.db --save-dir saved_model
#   python src/train.py --no-tune    (skip GridSearchCV, use default params)

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
)

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


# ── Tuning Functions ──────────────────────────────────────────────────────────

def tune_model(model, param_grid: dict, X_train, y_train, name: str):
    """
    Tune a model using GridSearchCV with StratifiedKFold.

    Justification:
    - GridSearchCV exhaustively searches the param_grid
    - StratifiedKFold preserves class distribution in each fold
      — important given class imbalance (~58% Low Activity)
    - Macro F1 used as scoring metric — consistent with evaluation
    - Results in best estimator fitted on full training set
    """
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )
    gs = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=CV_SCORING,
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X_train, y_train)
    print(f"[tune] {name} best params : {gs.best_params_}")
    print(f"[tune] {name} best CV {CV_SCORING}: {gs.best_score_:.4f}")
    return gs.best_estimator_


# ── ModelTrainer Class ────────────────────────────────────────────────────────

class ModelTrainer:
    """
    Handles tuning, training, cross-validation, and saving of ML models.

    Separates tree-based models (no scaling needed) from linear models
    (scaling required) so each model receives the appropriate feature set.
    """

    def __init__(self, random_state: int = RANDOM_STATE, tune: bool = TUNE_MODELS):
        self.random_state   = random_state
        self.tune           = tune
        self.trained_models = {}
        self.cv_results = {}
        self.best_model_name = None
        self.best_score = -1

    def _get_base_models(self) -> dict:
        """
        Define base models with default parameters from config.py.

        Model selection justification:
        - LogisticRegression  : Linear baseline. Fast and interpretable.
                                Uses class_weight=balanced to handle imbalance.
                                Requires scaled features.
        - RandomForest        : Ensemble of decision trees. Robust to outliers
                                and nonlinear sensor interactions. No scaling needed.
                                class_weight=balanced handles minority High Activity class.
        - GradientBoosting    : Sequential boosting on structured/tabular data.
                                Iteratively corrects errors on minority class.
                                Generally achieves best accuracy on tabular data.
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
                "model": GradientBoostingClassifier(**GB_PARAMS),
                "param_grid": GB_PARAM_GRID,
                "needs_scaling": False,
            },
        }

    def train_all(
        self,
        X_train: pd.DataFrame,
        X_train_scaled: pd.DataFrame,
        y_train: np.ndarray,
    ) -> None:
        """
        Tune (optional) and train all models.

        If TUNE_MODELS = True  → GridSearchCV finds best hyperparameters
        If TUNE_MODELS = False → Default params from config.py used directly
        After tuning/training, 5-fold cross-validation score is printed
        to verify model is not overfitting on training data.
        """
        models = self._get_base_models()

        for name, config in models.items():
            X = X_train_scaled if config["needs_scaling"] else X_train

            print(f"\n{'='*55}")
            print(f"  {name.upper().replace('_', ' ')}")
            print(f"{'='*55}")

            if self.tune:
                print(f"[train] Tuning {name} with GridSearchCV...")
                best_model = tune_model(
                    config["model"],
                    config["param_grid"],
                    X, y_train, name
                )
            else:
                print(f"[train] Tuning skipped — using default params from config.py")
                best_model = config["model"]
                best_model.fit(X, y_train)

            # Cross-validation to check for overfitting
            cv = StratifiedKFold(
                n_splits=CV_FOLDS,
                shuffle=True,
                random_state=self.random_state
            )
            cv_scores = cross_val_score(
                best_model,
                X,
                y_train,
                cv=cv,
                scoring=CV_SCORING,
            )

            mean_score = cv_scores.mean()
            std_score = cv_scores.std()

            print(
                f"[train] {name} CV {CV_SCORING}: "
                f"{mean_score:.4f} (+/- {std_score:.4f})"
            )

            self.cv_results[name] = {
                "mean": float(mean_score),
                "std": float(std_score),
            }

            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_model_name = name

            self.trained_models[name] = {
                "model":        best_model,
                "needs_scaling": config["needs_scaling"],
            }

        print(f"\n{'='*55}")
        print("  MODEL RANKING")
        print(f"{'='*55}")

        ranking = sorted(
            self.cv_results.items(),
            key=lambda x: x[1]["mean"],
            reverse=True,
        )

        for rank, (name, scores) in enumerate(ranking, start=1):
            print(
                f"{rank}. {name:<20} "
                f"{scores['mean']:.4f}"
            )

        print(
            f"\n[best] {self.best_model_name} "
            f"({self.best_score:.4f})"
        )    

    def save_all(
        self,
        save_dir: str,
        scaler,
        feature_names: list,
        activity_map: dict,
    ) -> None:
        """
        Save all trained models and pipeline artefacts to disk.

        Saved files:
        - <model_name>.joblib  : trained sklearn model
        - scaler.joblib        : fitted StandardScaler
        - feature_names.json   : ordered feature columns (ensures consistent order)
        - activity_map.json    : class encoding (for decoding predictions)
        """
        os.makedirs(save_dir, exist_ok=True)

        for name, config in self.trained_models.items():
            path = os.path.join(save_dir, f"{name}.joblib")
            joblib.dump(config["model"], path)
            print(f"[save] {name} → {path}")

        joblib.dump(scaler, os.path.join(save_dir, "scaler.joblib"))
        print(f"[save] scaler → {save_dir}/scaler.joblib")

        with open(os.path.join(save_dir, "feature_names.json"), "w") as f:
            json.dump(feature_names, f, indent=2)
        print(f"[save] feature_names → {save_dir}/feature_names.json")

        with open(os.path.join(save_dir, "activity_map.json"), "w") as f:
            json.dump(activity_map, f, indent=2)
        print(f"[save] activity_map → {save_dir}/activity_map.json")

        with open(
            os.path.join(save_dir, "best_model.json"),
            "w"
        ) as f:
            json.dump(
                {
                    "best_model": self.best_model_name,
                    "cv_score": float(self.best_score),
                },
                f,
                indent=2,
            )

        print(f"[save] best_model → {save_dir}/best_model.json")


# ── Training Pipeline ─────────────────────────────────────────────────────────
def run_training(
    db_path: str  = DB_PATH,
    save_dir: str = MODEL_SAVE_DIR,
    tune: bool    = TUNE_MODELS,
) -> None:
    """
    Full end-to-end training pipeline.
    """
    sep = "=" * 55
    print(f"\n{sep}")
    print("  TRAINING PIPELINE — START")
    print(f"  Tuning: {'ON (GridSearchCV)' if tune else 'OFF (default params)'}")
    print(sep)

    # ── Step 1-2: Load, clean, build features ────────────────────────────────
    df_clean = clean_data(db_path)
    X_train, y_train, activity_map, feature_names = build_features(df_clean)

    print(f"\n[pipeline] Activity map: {activity_map}")

    # ── Step 3: Stratified split ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    print(f"\n[split] Train : {X_train.shape[0]:,} rows")
    print(f"[split] Test  : {X_test.shape[0]:,} rows")

    # Save test split for evaluate.py
    os.makedirs(save_dir, exist_ok=True)
    X_test.to_parquet(os.path.join(save_dir, "X_test.parquet"), index=False)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test)
    print(f"[split] Test split saved to {save_dir}/")

    # ── Step 4: Scale — fit on train only (no data leakage) ──────────────────
    X_train_scaled, scaler = scale_features(X_train, scaler=None)
    X_test_scaled,  _       = scale_features(X_test,  scaler=scaler)

    # Save test scaled split for evaluate.py
    X_test_scaled.to_parquet(
        os.path.join(save_dir, "X_test_scaled.parquet"), index=False
    )

    # ── Step 5: Train ─────────────────────────────────────────────────────────
    trainer = ModelTrainer(random_state=RANDOM_STATE, tune=tune)
    trainer.train_all(X_train, X_train_scaled, y_train)

    # ── Step 6: Save ─────────────────────────────────────────────────────────
    # Passing list(X_train.columns) directly to prevent the NameError 
    trainer.save_all(save_dir,scaler,feature_names,activity_map,)

    print(
        f"\n[summary] Best model: "
        f"{trainer.best_model_name} "
        f"(CV {CV_SCORING}={trainer.best_score:.4f})"
    )
    print(f"\n{sep}")
    print("  TRAINING PIPELINE — COMPLETE")
    print(f"  Artefacts saved to : {save_dir}")
    print(f"  Run evaluate.py to compute test-set metrics.")
    print(f"{sep}\n")

# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ML models on gas monitoring data"
    )
    parser.add_argument(
        "--db",
        type=str,
        default=DB_PATH,
        help=f"Path to gas_monitoring.db (default: {DB_PATH})",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=MODEL_SAVE_DIR,
        help="Directory to save the trained model"
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip GridSearchCV and use default params from config.py",
    )
    args = parser.parse_args()
    run_training(
        db_path=args.db,
        save_dir=args.save_dir,
        tune=not args.no_tune,
    )
