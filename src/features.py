# features.py
# Encodes, scales, and engineers features from the cleaned dataset.
# Designed to run after cleaning.py — call build_features() directly,
# or run as a standalone script to write features.csv to disk.

# NOTE: Scaling is intentionally excluded from build_features().
#       Call scale_features() in train.py AFTER train/test split
#       to ensure the scaler is fit on training data only (no data leakage).

import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import TARGET_COLUMN, DROP_COLUMNS


# ── 1. Drop Columns Not Needed for Modelling ─────────────────────────────────
def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop Session ID before modelling.

    Justification:
    - Session ID is just an identifier retained through cleaning for
      session-based imputation. The number itself contains no useful information.
      Keeping it could allow the model to memorise sessions rather than learn patterns.
    """
    cols_to_drop = [col for col in ["Session ID"] if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"[drop_unused_columns] Dropped columns: {cols_to_drop}")
    return df


# ── 2. Feature Engineering ────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-informed features from existing sensor columns.

    Features created:
    - CO2_Disagreement     : |Infrared CO2 - ElectroChemical CO2|
                             Large disagreement signals sensor drift or rapid
                             CO2 flux during high physical activity.
    - CO2_Mean             : Mean of both CO2 sensors.
                             Reduces per-sensor noise into a single CO2 signal
                             and is more robust than using either sensor alone.
    - MOS_Core_Active_Mean : Mean of the highly predictive Metal Oxide Sensor
                             Unit 2 and Unit 4. Eliminates the baseline dilution noise of Units 1 and 3.
    - MOS_Core_Active_Range: max - min between MOS Unit 2 and Unit 4. Captures
                             sharp localized volatility drops or spikes.
    - Ambient_Light_Ordinal: Ordinal encoding of Ambient Light Level (0-4).
                             Preserves natural order for linear models.
    """

    # ── 2a. CO2 Features ──────────────────────────────────────────────────────
    # Requires both CO2 sensor columns to be present.

    if "CO2_InfraredSensor" in df.columns and "CO2_ElectroChemicalSensor" in df.columns:

        # Absolute difference between the two CO2 sensors.
        # A large gap suggests sensor drift or a sudden CO2 spike (e.g. heavy exercise).
        df["CO2_Disagreement"] = (
            df["CO2_InfraredSensor"] - df["CO2_ElectroChemicalSensor"]
        ).abs()
        print("[engineer_features] Created CO2_Disagreement")

        # Row-wise mean of both CO2 sensors.
        # Averaging cancels out per-sensor noise for a cleaner overall CO2 signal.
        df["CO2_Mean"] = df[["CO2_InfraredSensor", "CO2_ElectroChemicalSensor"]].mean(axis=1)
        print("[engineer_features] Created CO2_Mean")

    # ── 2b. Metal Oxide Sensor (MOS) Features ────────────────────────────────
    # Only uses Units 2 and 4 — identified as the strongest predictors.
    # Units 1 and 3 are excluded as they dilute the signal with baseline noise.

    core_mos_cols = ["MetalOxideSensor_Unit2", "MetalOxideSensor_Unit4"]

    if all(c in df.columns for c in core_mos_cols):

        # Average of the two core MOS units — a smoothed VOC/activity proxy.
        df["MOS_Core_Active_Mean"] = df[core_mos_cols].mean(axis=1)
        print("[engineer_features] Created MOS_Core_Active_Mean (Units 2 & 4)")

        # Max minus min of the two core MOS units.
        # Captures localised volatility: sharp spikes or drops between the pair.
        df["MOS_Core_Active_Range"] = df[core_mos_cols].max(axis=1) - df[core_mos_cols].min(axis=1)
        print("[engineer_features] Created MOS_Core_Active_Range (Units 2 & 4)")

    # ── 2c. Ambient Light Ordinal Encoding ───────────────────────────────────
    # Converts the string light level into an ordered integer (0–4).
    # Ordinal encoding is chosen over one-hot because the levels have a
    # clear natural order (very_dim < dim < moderate < bright < very_bright).

    if "Ambient Light Level" in df.columns:
        light_order = {
            "very_dim":    0,
            "dim":         1,
            "moderate":    2,
            "bright":      3,
            "very_bright": 4,
        }
        df["Ambient_Light_Ordinal"] = df["Ambient Light Level"].map(light_order)
        df["Ambient_Light_Ordinal"] = df["Ambient_Light_Ordinal"].astype(int)
        print("[engineer_features] Created Ambient_Light_Ordinal")
        print("[engineer_features] Light level mapping applied:")
        for label, code in light_order.items():
            print(f"    {label} → {code}")


    # Remove the original text column — the ordinal version created in
    # engineer_features() is what the model will use instead.
    if "Ambient Light Level" in df.columns:
        df = df.drop(columns=["Ambient Light Level"])
        print("[encode_categorical] Dropped Ambient Light Level (ordinal version kept)")

    return df

# ── 3. Encode Categorical Features ───────────────────────────────────────────
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode nominal categorical columns for model training.

    Strategy:
    - Time of Day, HVAC Operation Mode: One-Hot Encoding.
      Both are nominal with no natural numeric ordering.
      drop_first=True removes one dummy per group to avoid
      perfect multicollinearity (dummy variable trap).
    - Ambient Light Level: dropped here — ordinal version already
      created in engineer_features().
    - CO_GasSensor: kept as integer (ordinal 0-4, no encoding needed).
    - Activity Level: handled separately in encode_target().
    """
    ohe_cols = ["Time of Day", "HVAC Operation Mode"]

    # Fixed category lists ensure the same dummy columns are always produced,
    # regardless of which values happen to appear in a given data split.
    categories_dict = {
        "Time of Day": ["morning", "afternoon", "evening", "night"],
        "HVAC Operation Mode": [
            "off",
            "heating_low",
            "heating_high",
            "cooling_low",
            "cooling_high",
            "ventilation_only",
        ],
    }

    # Cast each column to pd.Categorical with the locked category list
    # before calling get_dummies — this guarantees column consistency.
    for col in ohe_cols:
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=categories_dict[col])

    existing_nominal = [col for col in ohe_cols if col in df.columns]
    df = pd.get_dummies(df, columns=existing_nominal, drop_first=True, dtype=int)
    print(f"[encode_categorical] Standardized One-hot encoded: {existing_nominal}")

    return df

# ── 4. Encode Target ──────────────────────────────────────────────────────────
def encode_target(df: pd.DataFrame) -> tuple:
    """
    Separate and encode the target column with a fixed ordinal mapping.

    A manual mapping is used instead of LabelEncoder to guarantee
    consistent class ordering regardless of data sort order.
    LabelEncoder sorts alphabetically, which gives:
        high_activity=0, low_activity=1, moderate_activity=2
    — inconsistent with the natural low→moderate→high ordering.

    Fixed mapping:
        low_activity      → 0
        moderate_activity → 1
        high_activity     → 2

    Returns:
    - df          : feature DataFrame with target column removed
    - y           : encoded target as numpy array
    - activity_map: encoding dict (invert for decoding predictions)
    """
    activity_map = {
        "low_activity":      0,
        "moderate_activity": 1,
        "high_activity":     2,
    }

    if df[TARGET_COLUMN].isnull().any():
        raise ValueError(
            f"[encode_target] Target column '{TARGET_COLUMN}' contains NaN. "
            "Check cleaning.py."
        )

    y = df[TARGET_COLUMN].map(activity_map).values
    df = df.drop(columns=[TARGET_COLUMN])

    unique, counts = np.unique(y, return_counts=True)
    dist = {activity_map_inv(activity_map, int(k)): int(v)
            for k, v in zip(unique, counts)}
    print(f"[encode_target] Mapping applied: {activity_map}")
    print(f"[encode_target] Class distribution: {dist}")

    return df, y, activity_map

"""
This helper function converts encoded numbers back into activity labels. 
It is mainly used for displaying readable class names."""

def activity_map_inv(activity_map: dict, code: int) -> str:
    """Return the class name for a given encoded integer."""
    return {v: k for k, v in activity_map.items()}.get(code, str(code))


# ── 5. Scale Numerical Features ──────────────────────────────────────────────
def scale_features(df: pd.DataFrame, scaler: StandardScaler = None) -> tuple:
    """
    Standardise continuous sensor readings using StandardScaler.

    Justification:
    - Required for Logistic Regression — sensitive to feature scale.
    - Not required for Random Forest / Gradient Boosting (rank-based splits).
    - Both scaled and unscaled versions are saved so each model
      uses the appropriate input.
    """
    scale_cols = [col for col in df.columns if col in [
        "Temperature", "Humidity",
        "CO2_InfraredSensor", "CO2_ElectroChemicalSensor",
        "MetalOxideSensor_Unit1", "MetalOxideSensor_Unit2",
        "MetalOxideSensor_Unit3", "MetalOxideSensor_Unit4",
        "CO2_Disagreement", "CO2_Mean",
        "MOS_Core_Active_Mean", "MOS_Core_Active_Range",
    ]]

    df_scaled = df.copy()
    if len(scale_cols) > 0:
        if scaler is None:
            # No scaler provided — fit a new one on this data (training split only).
            scaler = StandardScaler()
            df_scaled[scale_cols] = scaler.fit_transform(df[scale_cols])
            print(f"[scale_features] Fit and scaled {len(scale_cols)} continuous features.")
        else:
            # Scaler already fitted on training data — apply transform only.
            df_scaled[scale_cols] = scaler.transform(df[scale_cols])
            print(f"[scale_features] Transformed {len(scale_cols)} continuous features using fitted scaler.")
    else:
        if scaler is None:
            scaler = StandardScaler()  # Fallback to prevent downstream initialization crashes

    return df_scaled, scaler


# ── 6. Validate ───────────────────────────────────────────────────────────────
def validate(X: pd.DataFrame, y: np.ndarray, label: str = "") -> None:
    """
    Sanity check the final feature set before returning to train.py.
    Catches silent data issues — missing values, infinite values,
    unexpected class counts — before model training begins.
    """
    print(f"\n[validate] ── {label} ────────────────────────────")
    print(f"  X shape         : {X.shape[0]:,} rows × {X.shape[1]} columns")
    print(f"  y shape         : {y.shape[0]:,} labels")
    print(f"  Missing values  : {X.isnull().sum().sum()}")
    inf_count = np.isinf(X.select_dtypes(include=np.number)).sum().sum()
    print(f"  Infinite values : {inf_count}")
    unique, counts = np.unique(y, return_counts=True)
    print(f"  y class counts  : {dict(zip(unique.tolist(), counts.tolist()))}")
    print(f"  Feature list    : {list(X.columns)}")

    if X.isnull().sum().sum() > 0:
        missing_cols = X.columns[X.isnull().any()].tolist()
        print(f"  WARNING — columns with missing values: {missing_cols}")
    if inf_count > 0:
        print(f"  WARNING — infinite values detected, check engineer_features()")


# ── Master Pipeline ───────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> tuple:
    """
    Full feature-engineering pipeline.

    Steps:
        1. Drop unused columns (Session ID)
        2. Engineer new features (CO2_Disagreement, CO2_Mean,
           MOS_Core_Active_Mean, MOS_Core_Active_Range, Ambient_Light_Ordinal)
        3. One-hot encode nominal categoricals
        4. Separate and encode target column
        5. Validate final feature set

    NOTE: Scaling is NOT done here. Call scale_features() in train.py
          AFTER train/test split so the scaler is fit on training data only.

    Returns:
        X            — unscaled feature DataFrame
        y            — encoded target labels (numpy array)
        activity_map — encoding dict {class_name: int}
        feature_names— ordered list of final feature column names
    """
    df = drop_unused_columns(df)
    df = engineer_features(df)
    df = encode_categorical(df)
    df, y, activity_map = encode_target(df)

    # Drop features identified as low-importance during feature importance analysis.
    # These HVAC dummies and the noisy MOS unit add variance without predictive value.
    weak_features = [
        "HVAC Operation Mode_heating_low",
        "HVAC Operation Mode_cooling_low",
        "HVAC Operation Mode_cooling_high",
        "HVAC Operation Mode_heating_high",
        " MetalOxideSensor_Unit2"
    ]

    df = df.drop(
        columns=[c for c in weak_features if c in df.columns],
        errors="ignore"
    )

    print(f"[feature_selection] Removed {len(weak_features)} weak features")

    X = df.copy()

    validate(X, y, label="Final feature set")

    feature_names = list(X.columns)
    print(f"\n[build_features] Done — {len(feature_names)} features, {len(y):,} samples\n")

    return X, y, activity_map, feature_names


# ── Run as standalone script ──────────────────────────────────────────────────
if __name__ == "__main__":
    from src.cleaning import clean_data
    df_clean = clean_data()
    X, y, activity_map, feature_names = build_features(df_clean)
    print(f"X shape       : {X.shape}")
    print(f"y shape       : {y.shape}")
    print(f"Activity map  : {activity_map}")
    print(f"Feature names : {feature_names}")
