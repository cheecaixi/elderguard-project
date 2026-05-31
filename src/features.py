# features.py
# Encodes, scales, and engineers features from the cleaned dataset.
# Designed to run after cleaning.py — call build_features() directly,
# or run as a standalone script to write features.csv to disk.

import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DB_PATH
from src.cleaning import clean_data


# ── 1. Load Data ─────────────────────────────────────────────────────────────
# Data is loaded by calling clean_data() from cleaning.py inside build_features().
# No separate load function is needed here — cleaning.py handles the full
# loading + cleaning pipeline and returns a ready DataFrame.


# ── 2. Drop Columns Not Needed for Modelling ─────────────────────────────────
def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop Session ID before modelling.

    Justification:
    - Session ID is an arbitrary identifier retained through cleaning for
      session-based imputation. It carries no predictive signal and must
      be removed to prevent data leakage.
    """
    cols_to_drop = [col for col in ["Session ID"] if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"[drop_unused_columns] Dropped columns: {cols_to_drop}")
    return df


# ── 7. Encode Categorical Features ───────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode nominal categorical columns.

    Justification:
    - Time of Day, HVAC Operation Mode, and Activity Level have no natural
      ordinal ordering that a model should exploit numerically.
    - drop_first=True drops one dummy per group to avoid perfect
      multicollinearity (the dummy-variable trap).
    - Ambient Light Level is intentionally excluded here; if it is treated
      as ordinal elsewhere it should be label-encoded instead.
    """
    categorical_cols = [
        col for col in ["Time of Day", "HVAC Operation Mode", "Activity Level"]
        if col in df.columns
    ]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    print(f"[encode_categoricals] One-hot encoded: {categorical_cols}")
    print(f"[encode_categoricals] Dataset shape after encoding: {df.shape}")
    return df


# ── 8. Scale Numerical Features ──────────────────────────────────────────────
def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Standardise continuous sensor readings with StandardScaler.

    Returns:
    - df_scaled : copy of df with scaled columns (ready for distance/gradient
                  models such as Logistic Regression, KNN, SVM).
    - scaler    : fitted StandardScaler (save this to inverse-transform or
                  apply consistently to test data).

    Note:
    - The original df (unscaled) is still available from the caller for
      tree-based models (Random Forest, XGBoost) that do not require scaling.
    - CO_GasSensor and High_CO are excluded: discrete ordinal / binary flags.
    - Engineered features (MOS_Mean, CO2_Disagreement, Comfort_Index) are
      included because they are continuous and on heterogeneous scales.
    """
    candidate_scale_cols = [
        "Temperature",
        "Humidity",
        "CO2_InfraredSensor",
        "CO2_ElectroChemicalSensor",
        "MetalOxideSensor_Unit1",
        "MetalOxideSensor_Unit2",
        "MetalOxideSensor_Unit3",
        "MetalOxideSensor_Unit4",
        "MOS_Mean",
        "CO2_Disagreement",
        "Comfort_Index",
    ]
    scale_cols = [col for col in candidate_scale_cols if col in df.columns]

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[scale_cols] = scaler.fit_transform(df[scale_cols])

    # Sanity check: scaled columns should have mean ≈ 0, std ≈ 1
    stats = df_scaled[scale_cols].agg(["mean", "std"]).round(4)
    print("[scale_features] Post-scaling stats (mean ≈ 0, std ≈ 1):")
    print(stats.to_string())

    return df_scaled, scaler


# ── 9. Feature Engineering ────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-informed features from existing sensor columns.

    New features:
    - MOS_Mean           : Average of all four Metal Oxide Sensor units.
                           Reduces per-sensor noise into a single air quality signal.
    - CO2_Disagreement   : Absolute difference between the two CO₂ sensors.
                           Large values flag sensor drift or localised CO₂ pockets.
    - Comfort_Index      : Mean of Temperature and Humidity.
                           Combines the two primary occupant-comfort variables.
    - High_CO            : Binary flag (1 = CO_GasSensor >= 3).
                           Highlights elevated carbon-monoxide readings that may
                           indicate poor ventilation or unsafe conditions.
    """
    mos_cols = [
        "MetalOxideSensor_Unit1",
        "MetalOxideSensor_Unit2",
        "MetalOxideSensor_Unit3",
        "MetalOxideSensor_Unit4",
    ]
    if all(c in df.columns for c in mos_cols):
        df["MOS_Mean"] = df[mos_cols].mean(axis=1)
        print("[engineer_features] Created MOS_Mean")

    if "CO2_InfraredSensor" in df.columns and "CO2_ElectroChemicalSensor" in df.columns:
        df["CO2_Disagreement"] = (
            df["CO2_InfraredSensor"] - df["CO2_ElectroChemicalSensor"]
        ).abs()
        print("[engineer_features] Created CO2_Disagreement")

    if "Temperature" in df.columns and "Humidity" in df.columns:
        df["Comfort_Index"] = (df["Temperature"] + df["Humidity"]) / 2
        print("[engineer_features] Created Comfort_Index")

    if "CO_GasSensor" in df.columns:
        df["High_CO"] = (df["CO_GasSensor"] >= 3).astype(int)
        print("[engineer_features] Created High_CO")

    return df


# ── 10. Final Validation ──────────────────────────────────────────────────────
def validate(df: pd.DataFrame, label: str = "df") -> None:
    """Print a concise quality summary of the final feature set."""
    print(f"\n[validate] ── {label} ──────────────────────────")
    print(f"  Shape            : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Missing values   : {df.isnull().sum().sum()}")

    dupes = df.duplicated().sum()
    print(f"  Duplicate rows   : {dupes}")
    if dupes:
        print(f"  → removing {dupes} duplicates")
        df = df.drop_duplicates()

    inf_count = np.isinf(df.select_dtypes(include=np.number)).sum().sum()
    print(f"  Infinite values  : {inf_count}")

    invalid_temp_low  = (df["Temperature"] < 18).sum() if "Temperature" in df.columns else "N/A"
    invalid_temp_high = (df["Temperature"] > 40).sum() if "Temperature" in df.columns else "N/A"
    invalid_hum_low   = (df["Humidity"] < 0).sum()    if "Humidity"    in df.columns else "N/A"
    invalid_hum_high  = (df["Humidity"] > 100).sum()  if "Humidity"    in df.columns else "N/A"
    print(f"  Temperature < 18 : {invalid_temp_low}")
    print(f"  Temperature > 40 : {invalid_temp_high}")
    print(f"  Humidity < 0     : {invalid_hum_low}")
    print(f"  Humidity > 100   : {invalid_hum_high}")
    print(f"  Dtypes:\n{df.dtypes.value_counts().to_string()}")


# ── Master Pipeline ───────────────────────────────────────────────────────────
def build_features(
    db_path: str = DB_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Run the full feature-engineering pipeline.

    Steps:
        1.  Load + clean raw data via clean_data() from cleaning.py
        2.  Drop unused columns (Session ID)
        7.  One-hot encode categorical features
        8.  Scale numerical features
        9.  Feature engineering
        10. Final validation

    Returns:
        df        — unscaled feature set (use with tree-based models)
        df_scaled — standardised feature set (use with Logistic Regression / KNN / SVM)
        scaler    — fitted StandardScaler instance
    """
    # Steps 1–6 handled by cleaning.py
    df = clean_data(db_path)

    # Step 2 — drop identifiers not needed for modelling
    df = drop_unused_columns(df)

    # Step 7 — encode categoricals
    df = encode_categoricals(df)

    # Step 8 — scale numerical features
    df_scaled, scaler = scale_features(df)

    # Step 9 — engineer new features (on unscaled df so values stay interpretable)
    df        = engineer_features(df)
    df_scaled = engineer_features(df_scaled)

    # Step 10 — final validation
    validate(df,        label="df (unscaled)")
    validate(df_scaled, label="df_scaled")

    print(f"\n[build_features] Done.")
    print(f"  df        — original scale : {df.shape}")
    print(f"  df_scaled — standardised   : {df_scaled.shape}")

    return df, df_scaled, scaler


# ── Run as standalone script ──────────────────────────────────────────────────
if __name__ == "__main__":
    df, df_scaled, scaler = build_features()

    # Write both outputs to CSV for inspection / downstream use
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(out_dir, exist_ok=True)

    df.to_csv(os.path.join(out_dir, "features.csv"), index=False)
    df_scaled.to_csv(os.path.join(out_dir, "features_scaled.csv"), index=False)

    print(f"\n[__main__] Saved features.csv and features_scaled.csv to {out_dir}")
    print(df.head())
