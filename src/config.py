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

# Gradient Boosting
GB_PARAMS = {
    "max_iter":     200,
    "learning_rate": 0.1,
    "max_depth":    5,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

# Logistic Regression
LR_PARAMS = {
    "max_iter":      1000,
    "class_weight":  "balanced",
    "random_state":  RANDOM_STATE,
}

# ── Tuning Grids (used when TUNE_MODELS = True) ────────────────
# Only the most impactful parameters are tuned to keep runtime reasonable.
# 5-fold StratifiedKFold CV is used to preserve class distribution.
# Macro F1 is the scoring metric — consistent with evaluation.

RF_PARAM_GRID = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [None, 10, 20],
    "min_samples_leaf": [1, 2, 4],     
    "max_features":     ["sqrt", "log2"], 
}

GB_PARAM_GRID = {
    "max_iter":      [100, 200],    # drop 300
    "learning_rate": [0.05, 0.1],   # drop 0.01
    "max_depth":     [3, 5],        # keep as is
}

LR_PARAM_GRID = {
    "C":      [0.01, 0.1, 1.0, 10.0],   
    "solver": ["lbfgs", "saga"],          
}

# ── Cross Validation ───────────────────────────────────────────
CV_FOLDS   = 5      # number of folds for StratifiedKFold
CV_SCORING = "f1_macro"  # primary metric — treats all classes equally