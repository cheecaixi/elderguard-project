# cleaning.py
# Loads raw data from SQLite and applies all cleaning steps identified in EDA

# Import Libraries
import sqlite3
import pandas as pd
import numpy as np
import sys
import os

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    DB_PATH, CONTAMINATED_SESSIONS,
    MAX_TEMPERATURE, MIN_TEMPERATURE, MIN_HUMIDITY, MAX_HUMIDITY
)

# ── 1. Data Loading ─────────────────────────────────────────────
def load_data(DB_PATH: str) -> pd.DataFrame:
    """Load raw data from SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM gas_monitoring", conn)
    conn.close()
    print(f"[load_data] Loaded {len(df):,} rows x {df.shape[1]} columns")
    return df

# ── 2. Remove Duplicates ─────────────────────────────────────────────
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the dataset."""
    before = len(df)
    df = df.drop_duplicates()
    print(f"[remove_duplicates] Removed {before - len(df)} duplicate rows")
    return df   

# ── 3. Standardise Categorical Labels ─────────────────────────────────────────────
def clean_activity_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise all Activity Level variants to lowercase_underscore format."""
    df["Activity Level"] = (
        df["Activity Level"]
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    label_map = {
        "lowactivity":      "low_activity",
        "highactivity":     "high_activity",
        "moderateactivity": "moderate_activity",
    }
    df['Activity Level'] = df['Activity Level'].str.strip().str.lower().apply(
    lambda x: 'low_activity' if 'low' in str(x)
    else 'moderate_activity' if 'moderate' in str(x)
    else 'high_activity' if 'high' in str(x)
    else None
)

def clean_hvac_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise HVAC Operation Mode labels to lowercase."""
    df["HVAC Operation Mode"] = df["HVAC Operation Mode"].str.strip().str.lower()
    print(f"[clean_hvac_labels] Unique HVAC modes: {df['HVAC Operation Mode'].nunique()}")
    return df

# ── 4. Fix Invalid Values ─────────────────────────────────────────────
def remove_contaminated_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Remove sessions flagged as contaminated in EDA.
    Assumption: Session 2586 is removed entirely despite CO2 and humidity
    readings appearing normal. 
    Justification: mean temperature of 89.9C is
    physically impossible indoors, indicating faulty sensor hardware. Since
    all readings come from the same hardware, other sensor values from this
    session cannot be fully trusted even if they appear plausible. Removing
    the entire session is the conservative and safer choice for model integrity.
    """
    before = len(df)
    df = df[~df["Session ID"].isin(CONTAMINATED_SESSIONS)]
    print(f"[remove_contaminated_sessions] Removed {before - len(df)} rows from sessions: {CONTAMINATED_SESSIONS}")
    return df

def fix_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace physically impossible sensor readings with NaN.

    Justification:
    - Capping (e.g. 292°C → 40°C) introduces artificial values that could mislead model training
    - Removing rows loses 24% of data and introduces selection bias
    - Converting to NaN allows session-level median imputation to fill with realistic 
    values from the same environmental context
    - This is the most statistically honest approach
    """
    # Temperature: impossible values → NaN
    invalid_temp = (
        (df["Temperature"] < MIN_TEMPERATURE) |
        (df["Temperature"] > MAX_TEMPERATURE)
    ).sum()
    df.loc[
        (df["Temperature"] < MIN_TEMPERATURE) |
        (df["Temperature"] > MAX_TEMPERATURE),
        "Temperature"
    ] = np.nan
    print(f"[fix_invalid_values] Temperature: marked {invalid_temp} impossible values as NaN")

    # Humidity: impossible values → NaN
    invalid_hum = (
        (df["Humidity"] < MIN_HUMIDITY) |
        (df["Humidity"] > MAX_HUMIDITY)
    ).sum()
    df.loc[
        (df["Humidity"] < MIN_HUMIDITY) |
        (df["Humidity"] > MAX_HUMIDITY),
        "Humidity"
    ] = np.nan
    print(f"[fix_invalid_values] Humidity: marked {invalid_hum} impossible values as NaN")

    # CO2 sensors: negative values → NaN
    for col in ["CO2_InfraredSensor", "CO2_ElectroChemicalSensor"]:
        invalid = (df[col] < 0).sum()
        df.loc[df[col] < 0, col] = np.nan
        print(f"[fix_invalid_values] {col}: marked {invalid} negative values as NaN")

    # CO_GasSensor: negative values → NaN
    invalid_co = (df["CO_GasSensor"] < 0).sum()
    df.loc[df["CO_GasSensor"] < 0, "CO_GasSensor"] = np.nan
    print(f"[fix_invalid_values] CO_GasSensor: marked {invalid_co} negative values as NaN")

    return df

# ── 5. Handle Missing Values ─────────────────────────────────────────────
def impute_numeric_session_median(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute using session-level median for high-variance columns.
    Now includes Temperature since fix_invalid_values() marks
    impossible readings as NaN instead of capping or removing.
    """
    session_median_cols = [
        "Temperature",         
        "Humidity",
        "MetalOxideSensor_Unit2",
        "CO2_InfraredSensor",
        "CO2_ElectroChemicalSensor"
    ]

    for col in session_median_cols:
        missing_before = df[col].isnull().sum()

        # Step 1: session-level median
        df[col] = df.groupby("Session ID")[col].transform(
            lambda x: x.fillna(x.median())
        )

        # Step 2: global median fallback
        remaining = df[col].isnull().sum()
        if remaining > 0:
            df[col] = df[col].fillna(df[col].median())

        print(f"[impute_numeric_session_median] {col}: filled {missing_before} missing "
              f"(session median: {missing_before - remaining}, "
              f"global fallback: {remaining})")

    return df

def impute_co_global_median(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute CO_GasSensor using global median.

    Justification:
    - CO_GasSensor only has 5 discrete values (0-4)
    - Low variance across sessions makes session-level imputation unnecessary
    - Global median is simpler and equally valid for low-variance columns
    """
    missing_before = df["CO_GasSensor"].isnull().sum()
    df["CO_GasSensor"] = df["CO_GasSensor"].fillna(df["CO_GasSensor"].median())
    print(f"[impute_co_global_median] CO_GasSensor: filled {missing_before} "
          f"missing values with global median")
    return df


def impute_ambient_light_global_mode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute Ambient Light Level using global mode.

    Justification:
    - All 38 sessions contain all 5 light levels - session mode not meaningful
    - Time of Day shows near-identical light distribution across all periods
      (very_bright ~37-40%, bright ~30% regardless of time)
    - No feature provides a meaningful basis for smarter imputation
    - Global mode is the most defensible fallback given the data evidence
    """
    missing_before = df["Ambient Light Level"].isnull().sum()
    df["Ambient Light Level"] = df["Ambient Light Level"].fillna(
        df["Ambient Light Level"].mode()[0]
    )
    print(f"[impute_ambient_light_global_mode] Ambient Light Level: filled "
          f"{missing_before} missing values with global mode "
          f"({df['Ambient Light Level'].mode()[0]})")
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master imputation function — calls all three imputation strategies.
    Each strategy is justified by EDA findings.
    """
    df = impute_numeric_session_median(df)
    df = impute_co_global_median(df)
    df = impute_ambient_light_global_mode(df)
    return df

# ── 6. Correct Data Types ─────────────────────────────────────────────
def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to appropriate data types.

    Justification:
    - Time of Day, HVAC Operation Mode, Ambient Light Level, Activity Level
      are nominal/ordinal categories  (category dtype)
    - CO_GasSensor has only 5 discrete values (0-4) (integer dtype)
      to reflect its ordinal scale nature rather than continuous measurement
    - Session ID is retained here for session-based imputation but will
      be dropped in features.py before model training
    """
    # Nominal and ordinal text columns
    categorical_cols = [
        "Time of Day",
        "HVAC Operation Mode",
        "Ambient Light Level",
        "Activity Level"
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # CO_GasSensor — discrete ordinal scale (0-4), store as integer
    if "CO_GasSensor" in df.columns:
            df["CO_GasSensor"] = pd.to_numeric(df["CO_GasSensor"], errors="coerce").astype("Int64")
            print(f"[fix_data_types] CO_GasSensor successfully converted to int.")

    print(f"[fix_data_types] Converted {len(categorical_cols)} columns to category dtype")
    return df

# ── 7. Detect and Handle Outliers ─────────────────────────────────────────────
def cap_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap statistical outliers using IQR Winsorization (no data lost).

    Justification:
    - Logistic Regression is sensitive to extreme values as outliers
      distort the decision boundary
    - Tree-based models (Random Forest, XGBoost) are naturally robust
      to outliers but capping does not hurt them either
    - CO_GasSensor excluded - it is a discrete ordinal integer (0-4),
      capping would break its discrete nature
    """
    outlier_columns = [
        "Humidity",
        "CO2_InfraredSensor",
        "CO2_ElectroChemicalSensor",
        "MetalOxideSensor_Unit1",
        "MetalOxideSensor_Unit2",
        "MetalOxideSensor_Unit3",
        "MetalOxideSensor_Unit4"
    ]

    # Only cap columns that exist in the DataFrame
    outlier_columns = [col for col in outlier_columns if col in df.columns]

    for col in outlier_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"[cap_outliers_iqr] {col}: capped {before} outliers")

    return df


def clean_data(db_path: str = DB_PATH) -> pd.DataFrame:
    df = load_data(db_path)
    df = remove_duplicates(df)
    print(f"shape: {df.shape}")
    df = clean_activity_labels(df)
    df = clean_hvac_labels(df)
    df = remove_contaminated_sessions(df)
    print(f"shape: {df.shape}")
    df = fix_invalid_values(df)
    df = impute_missing(df)
    print(f"shape: {df.shape}")
    df = cap_outliers_iqr(df)
    df = fix_data_types(df)
    # Second pass: imputation and IQR capping can create new duplicate rows
    # (e.g. two rows with different NaN patterns filled with the same session median).
    df = remove_duplicates(df)
    print(f"\n[clean_data] Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"[clean_data] Remaining missing values: {df.isnull().sum().sum()}")
    print(f"[clean_data] Remaining duplicate rows: {df.duplicated().sum()}")
    print(f"[clean_data] Temperature out-of-range remaining: {((df['Temperature'] < 15) | (df['Temperature'] > 40)).sum()}")
    print(f"[clean_data] Humidity out-of-range remaining: {((df['Humidity'] < 0) | (df['Humidity'] > 100)).sum()}")
    return df

if __name__ == "__main__":
    df_clean = clean_data()
    print(df_clean.head())
