# src/config.py
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

# ── Train/Test Split & CV ─────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5                     

# ── Tuning Flag ───────────────────────────────────────────────
TUNE_MODELS = True

# ── Default Model Parameters (used when TUNE_MODELS = False) ──
# Random Forest
RF_PARAMS = {
    "n_estimators": 150,
    "max_depth": 8,                   
    "min_samples_split": 15,           
    "min_samples_leaf": 8,
    "max_features": "sqrt",
    "class_weight": "balanced",       
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# XGBoost
# In config.py
XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_weight": 5,
    "tree_method": "hist",
    "eval_metric": "mlogloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Logistic Regression
LR_PARAMS = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "solver": "saga"
}

# ── Tuning Grids (used when TUNE_MODELS = True) ────────────────
RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [6, 8, 10],    
    "min_samples_leaf": [4, 8, 16],      
    "max_features": ["sqrt"],
}

XGB_PARAM_GRID = {
    "n_estimators": [50, 150, 250],
    "max_depth": [4, 5, 6],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "gamma": [0, 0.1, 0.3]            # Added gamma to control tree complexity
}

LR_PARAM_GRID = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "solver": ["saga"]
}
