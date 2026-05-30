# cleaning.py
# Loads raw data from SQLite and applies all cleaning steps identified in EDA

# Import Libraries
import sqlite3
import pandas as pd
import numpy as np
import sys
import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    DB_PATH, CONTAMINATED_SESSIONS,
    MAX_TEMPERATURE, MIN_TEMPERATURE, HUMIDITY_MIN, HUMIDITY_MAX
)

# ── Data Loading ─────────────────────────────────────────────
def load_data(db_path: str) -> pd.DataFrame:
    """Load raw data from SQLite database."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM gas_monitoring", conn)
    conn.close()
    print(f"[load_data] Loaded {len(df):,} rows x {df.shape[1]} columns")
    return df

# ── Remove Duplicate ─────────────────────────────────────────────
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the dataset."""
    before = len(df)
    df = df.drop_duplicates()
    print(f"[remove_duplicates] Removed {before - len(df)} duplicate rows")
    return df   

# ──Handle Missing Values ─────────────────────────────────────────────

# ── Fix Invalid Values ─────────────────────────────────────────────

# ── Standardise Categorical Labels ─────────────────────────────────────────────
def clean_activity_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise inconsistent Activity Level labels."""
    label_map = {
        "LowActivity": "low_activity",
        "Low_Activity": "low_activity",
        "HighActivity": "high_activity",
        "High_Activity": "high_activity",
        "ModerateActivity": "moderate_activity",
        "Moderate_Activity": "moderate_activity",
    }
    df["Activity Level"] = df["Activity Level"].replace(label_map).str.strip()
    print(f"[clean_activity_labels] Unique labels: {df['Activity Level'].unique()}")
    return df


# ── Correct Data Types ─────────────────────────────────────────────

# ── Detect and Handle Outliers ─────────────────────────────────────────────
