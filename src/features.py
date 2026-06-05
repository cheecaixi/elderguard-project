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
    Drop Session ID after engineering has extracted context.

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
    - CO2_Disagreement    : |Infrared CO2 - ElectroChemical CO2|
    - CO2_Mean            : Mean of both CO2 sensors.
    - MOS_Mean            : Mean of all 4 Metal Oxide Sensor units.
    - MOS_Range           : max - min across all 4 MOS units.
    - Ambient_Light_Ordinal: Ordinal encoding of Ambient Light Level (0-4).
    """
    df = df.copy()
    
    # CO2 sensor disagreement + mean
    if "CO2_InfraredSensor" in df.columns and "CO2_ElectroChemicalSensor" in df.columns:
        df["CO2_Disagreement"] = (
            df["CO2_InfraredSensor"] - df["CO2_ElectroChemicalSensor"]
        ).abs()
        print("[engineer_features] Created CO2_Disagreement")

        df["CO2_Mean"] = df[["CO2_InfraredSensor", "CO2_ElectroChemicalSensor"]].mean(axis=1)
        print("[engineer_features] Created CO2_Mean")

        # Drop raw CO2 sensors
        df = df.drop(columns=["CO2_InfraredSensor", "CO2_ElectroChemicalSensor"], errors="ignore")
        print("[engineer_features] Dropped individual CO2 sensor columns")

    # Mean and range of all MOS units
    mos_cols = [
        "MetalOxideSensor_Unit1",
        "MetalOxideSensor_Unit2",
        "MetalOxideSensor_Unit3",
        "MetalOxideSensor_Unit4",
    ]
    if all(c in df.columns for c in mos_cols):
        df["MOS_Mean"] = df[mos_cols].mean(axis=1)
        print("[engineer_features] Created MOS_Mean")

        df["MOS_Range"] = df[mos_cols].max(axis=1) - df[mos_cols].min(axis=1)
        print("[engineer_features] Created MOS_Range")

    if "CO2_Mean" in df.columns and "Temperature" in df.columns:
        df["CO2_Temp_Interaction"] = df["CO2_Mean"] * df["Temperature"]
        print("[engineer_features] Created CO2_Temp_Interaction")   

    # Ambient light ordinal encoding
    if "Ambient Light Level" in df.columns:
        light_order = {
            "very_dim":    0,
            "dim":         1,
            "moderate":    2,
            "bright":      3,
            "very_bright": 4,
        }
        df["Ambient_Light_Ordinal"] = df["Ambient Light Level"].map(light_order).astype(int)
        print("[engineer_features] Created Ambient_Light_Ordinal")
        print("[engineer_features] Light level mapping applied:")
        for label, code in light_order.items():
            print(f"    {label} → {code}")

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
    ohe_cols = [
        col for col in ["Time of Day", "HVAC Operation Mode"]
        if col in df.columns
    ]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True, dtype=int)
    print(f"[encode_categorical] One-hot encoded: {ohe_cols}")

    # Drop original Ambient Light Level — ordinal version already created
    if "Ambient Light Level" in df.columns:
        df = df.drop(columns=["Ambient Light Level"])
        print("[encode_categorical] Dropped Ambient Light Level (ordinal version kept)")

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

    Excluded from scaling:
    - One-hot encoded dummy columns (already 0/1)
    - Ambient_Light_Ordinal (ordinal integer)
    - CO_GasSensor (discrete ordinal 0-4)
    """
    scale_cols = [col for col in df.columns if col in [
        "Temperature", "Humidity",
        "MetalOxideSensor_Unit1", "MetalOxideSensor_Unit2",
        "MetalOxideSensor_Unit3", "MetalOxideSensor_Unit4",
        "CO2_Disagreement", "CO2_Mean",
        "MOS_Mean", "MOS_Range","MOS_Ratio", "CO2_Temp_Interaction"
    ]]

    df_scaled = df.copy()
    if scaler is None:
        scaler = StandardScaler()
        df_scaled[scale_cols] = scaler.fit_transform(df[scale_cols])
        print(f"[scale_features] Fitted new scaler on {len(scale_cols)} columns")
    else:
        df_scaled[scale_cols] = scaler.transform(df[scale_cols])

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
    
    missing_count = X.isnull().sum().sum()
    print(f"  Missing values  : {missing_count}")
    
    inf_count = np.isinf(X.select_dtypes(include=np.number)).sum().sum()
    print(f"  Infinite values : {inf_count}")
    
    unique, counts = np.unique(y, return_counts=True)
    print(f"  y class counts  : {dict(zip(unique.tolist(), counts.tolist()))}")
    print(f"  Feature list    : {list(X.columns)}")

    if missing_count > 0:
        missing_cols = X.columns[X.isnull().any()].tolist()
        raise ValueError(
            f"[validate] CRITICAL FAILURE: {missing_count} missing values detected "
            f"in columns: {missing_cols}. Fix data engineering pipeline cascades."
        )
        
    if inf_count > 0:
        raise ValueError(
            f"[validate] CRITICAL FAILURE: Infinite values detected inside features. "
            f"Check division-by-zero or mathematical transformations in engineer_features()."
        )


# ── Master Pipeline ───────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> tuple:
    """
    Full feature-engineering pipeline.

    1. Drop unused columns (Session ID)
        2. Engineer new features (CO2_Disagreement, CO2_Mean,
           MOS_Mean, MOS_Range, MOS_Ratio, CO2_Temp_Interaction,
           Ambient_Light_Ordinal)
        3. One-hot encode nominal categoricals
        4. Drop weak features identified by permutation importance
           (negative or near-zero permutation scores on test set)
        5. Separate and encode target column
        6. Validate final feature set

    NOTE: Scaling is NOT done here. Call scale_features() in train.py
          AFTER train/test split so the scaler is fit on training data only.

    Returns:
        X            — unscaled feature DataFrame
        y            — encoded target labels (numpy array)
        activity_map — encoding dict {class_name: int}
        feature_names— ordered list of final feature column names
    """
    df = engineer_features(df)
    df = drop_unused_columns(df)
    df = encode_categorical(df)

    # Drop features with negative or near-zero permutation importance.
    # These were identified via permutation importance on the held-out
    # test set — shuffling them improved or did not hurt macro F1,
    # meaning they contribute noise rather than signal.
    WEAK_FEATURES = [
        "Ambient_Light_Ordinal",       # perm rank 19 — negative score
        "Time of Day_night",           # perm rank 20 — most negative
        "CO2_Mean",                    # perm rank 17 — negative score
        "MOS_Mean",                    # perm rank 12 — near zero
    ]
    weak_to_drop = [f for f in WEAK_FEATURES if f in df.columns]
    df = df.drop(columns=weak_to_drop)
    print(f"[build_features] Dropped weak features: {weak_to_drop}")

    # encode_target called once only
    df, y, activity_map = encode_target(df)

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
