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
    "random_state":     RANDOM_STATE,
    "class_weight":     "balanced",
    "max_depth":        8,
    "min_samples_leaf": 16,
    "n_estimators":     200,
    "max_features":     "sqrt",
}

# HistGradientBoosting Classifier (Replaces XGBoost)
GB_PARAMS = {
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",
    "max_iter": 300,           
    "learning_rate": 0.05,
    "max_depth": 5,              
    "early_stopping": True,          
    "n_iter_no_change": 10,        
    "validation_fraction": 0.1        
}

# Logistic Regression
LR_PARAMS = {
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",
    "max_iter": 1000,
    "solver": "lbfgs"
}

# ── Tuning Grids (used when TUNE_MODELS = True) ────────────────
RF_PARAM_GRID = {
    "model__max_depth":        [10, 12, 14],
    "model__min_samples_leaf": [24, 32, 48],      
    "model__n_estimators":     [200, 300],
    "model__max_features":     [0.5],
    "model__class_weight":     ["balanced", "balanced_subsample"],
}

GB_PARAM_GRID = {
    "model__learning_rate":    [0.03, 0.05],
    "model__max_depth":        [4, 5, 6],
    "model__max_iter":         [100, 150, 200],
    "model__min_samples_leaf": [16, 20, 30],
}

LR_PARAM_GRID = {
    "model__C": [0.01, 0.1, 1.0, 10.0]
}

# ── Cross Validation ───────────────────────────────────────────
CV_FOLDS   = 5          
CV_SCORING = "f1_macro"