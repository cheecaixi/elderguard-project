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

# ── Models ────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Random Forest
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "random_state": RANDOM_STATE,
}

# XGBoost
XGB_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 6,
    "random_state": RANDOM_STATE,
}

# Logistic Regression
LR_PARAMS = {
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
}