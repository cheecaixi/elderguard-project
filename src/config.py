# config.py
# Central configuration for the ML pipeline

import os

# ── Paths ────────────────────────────────────────────────────
DB_PATH = os.path.join("data", "gas_monitoring.db")
MODEL_SAVE_DIR = "saved_model"

# ── Data Cleaning ─────────────────────────────────────────────
CONTAMINATED_SESSIONS = [2586]   # identified in EDA
MIN_TEMPERATURE = 15.0           # realistic indoor min (°C)
MAX_TEMPERATURE = 40.0           # realistic indoor max (°C)
MIN_HUMIDITY = 0.0
MAX_HUMIDITY = 100.0

# ── Features ──────────────────────────────────────────────────
TARGET_COLUMN = "Activity Level"
DROP_COLUMNS = ["Session ID"]

# ── Train/Test Split ────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ── Tuning ────────────────────────────────────────────────────
# Set TUNE_MODELS = True  to run GridSearchCV (slower, finds best params)
# Set TUNE_MODELS = False to skip tuning and use default params below (faster)
TUNE_MODELS = True

# ── Default Model Parameters (used when TUNE_MODELS = False) ──
# Random Forest
RF_PARAMS = {
    "n_estimators":     200,
    "min_samples_leaf": 2,
    "class_weight":     "balanced_subsample",
    "random_state":     RANDOM_STATE,
    "n_jobs":           -1,
}

# XGBoost
XGB_PARAMS = {
    "n_estimators":  200,
    "max_depth":     5,
    "learning_rate": 0.1,
    "tree_method":   "hist",
    "eval_metric":   "mlogloss",
    "random_state":  RANDOM_STATE,
    "n_jobs":        -1,
}

# Decision Tree 
DT_PARAMS = {
    "max_depth":         6,              
    "min_samples_leaf":  15,
    "class_weight":      "balanced",     
    "random_state":      RANDOM_STATE,
}

# ── Tuning Grids (used when TUNE_MODELS = True) ────────────────
# Only the most impactful parameters are tuned to keep runtime reasonable.
# 5-fold StratifiedKFold CV is used to preserve class distribution.
# Macro F1 is the scoring metric — consistent with evaluation.

RF_PARAM_GRID = {
    "n_estimators":     [150, 300],
    "max_depth":        [10, 12, 14],
    "min_samples_leaf": [10, 15, 25],     
    "max_features":     ["sqrt"]      
}

XGB_PARAM_GRID = {
    "n_estimators":     [150, 200],
    "max_depth":        [5, 6],       
    "learning_rate":    [0.04, 0.06],
    "subsample":        [0.8],
    "colsample_bytree": [0.8]
}

DT_PARAM_GRID = {
    "max_depth":        [4, 6, 8, 12],
    "min_samples_leaf": [5, 10, 20, 50],
    "criterion":        ["gini", "entropy"],
    "class_weight": ["balanced"]
}

# ── Cross Validation ───────────────────────────────────────────
CV_FOLDS   = 3      # number of folds for StratifiedKFold
CV_SCORING = "f1_macro"  # primary metric — treats all classes equally
