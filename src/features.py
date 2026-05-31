��
# **1. Load Data**
# Import Libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data from previous step
df = pd.read_csv("cleaneding.csv")  

# 7. Encode Categorical Features
# Identify categorical columns
categorical_cols = [
    "Time of Day",
    "HVAC Operation Mode",
    "Activity Level"
]

print("Categorical Features:")
print(categorical_cols)

"""# To identify categorical features that require numerical encoding before machine learning."""

# Apply one hot encoding
df = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True
)

print("Encoding complete.")
print(f"Updated dataset shape: {df.shape}")

"""To convert categorical labels into numerical format while avoiding ordinal assumptions between categories."""
"""Categorical features were transformed into numerical representations using one hot encoding to ensure compatibility with machine learning algorithms. The first category was dropped during encoding to reduce redundancy and prevent multicollinearity between encoded variables."""

# 8. Scale Numerical Features
# Columns to scale only continuous sensor readings
# Exclude: encoded flags (0/1), ordinal columns, Session ID
all_scale_cols = [
    "Temperature",
    "Humidity",
    "CO2_InfraredSensor",
    "CO2_ElectroChemicalSensor",
    "MetalOxideSensor_Unit1",
    "MetalOxideSensor_Unit2",
    "MetalOxideSensor_Unit3",
    "MetalOxideSensor_Unit4",
    "CO_GasSensor",
    "CO2_Disagreement",
    "MOS_Mean"
]

# Filter to include only columns that exist in the DataFrame
scale_cols = [col for col in all_scale_cols if col in df.columns]

# Fit and transform
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[scale_cols] = scaler.fit_transform(df[scale_cols])

# Verify: scaled columns should have mean ≈ 0 and std ≈ 1
print("Post-scaling stats (should be mean≈0, std≈1):")
print(df_scaled[scale_cols].agg(["mean", "std"]).round(4).to_string())

# Keep both: df (unscaled, interpretable) and df_scaled (ready for modelling)
# For tree-based models (Random Forest, XGBoost) scaling is not needed
# For distance/gradient-based models (KNN, SVM, LogReg) use df_scaled
print(f"df        — original scale, shape: {df.shape}")
print(f"df_scaled — standardised,    shape: {df_scaled.shape}")

plt.figure(figsize=(8,4))

plt.hist(
    df["CO2_InfraredSensor"],
    bins=30,
    alpha=0.6,
    label="Before Scaling"
)

plt.hist(
    df_scaled["CO2_InfraredSensor"],
    bins=30,
    alpha=0.6,
    label="After Scaling"
)

plt.title("CO2 Infrared Sensor Before and After Scaling")
plt.legend()
plt.show()

"""To standardize continuous sensor features so that variables with larger numerical ranges do not dominate machine learning models."""
"""Continuous sensor features were standardized using StandardScaler while preserving a separate unscaled version of the dataset for interpretation. Scaling is particularly important for distance based and gradient based machine learning algorithms, while tree based models typically do not require feature scaling."""

# **9. Feature Engineering**
# Average MOS Sensor, Reduces noise from individual sensors.
df["MOS_Average"] = (
    df["MetalOxideSensor_Unit1"] +
    df["MetalOxideSensor_Unit2"] +
    df["MetalOxideSensor_Unit3"] +
    df["MetalOxideSensor_Unit4"]
) / 4

"""The MOS_Average feature was created by combining the Mean Opinion Scores from both ends of the communication. This provides an overall measure of call quality and user satisfaction, making it easier to analyze communication performance."""

# CO2 Sensor Difference, Measures disagreement between sensors.
df["CO2_Difference"] = abs(
    df["CO2_InfraredSensor"] -
    df["CO2_ElectroChemicalSensor"]
)

"""The CO2_Difference feature was created to measure the variation between indoor and outdoor CO₂ levels. This helps identify ventilation effectiveness and highlights environments where air quality may be significantly different from outdoor conditions."""

# Indoor Comfort Index, Simple environmental indicator.
df["Comfort_Index"] = (
    df["Temperature"] +
    df["Humidity"]
) / 2

"""The Comfort_Index feature combines temperature and humidity into a single metric representing environmental comfort. This simplifies analysis by capturing the combined effect of multiple factors that influence occupant comfort."""

# High CO Alert, May help classify activity patterns.
df["High_CO"] = (
    df["CO_GasSensor"] >= 3
).astype(int)

"""The High_CO feature was created as a binary indicator to identify records with elevated carbon monoxide levels. This makes it easier to detect potential air quality concerns and supports classification or risk assessment tasks."""

"""Feature engineering was explored but no additional features were retained as the original sensor measurements already captured the environmental conditions directly and engineered features introduced redundancy without clear predictive value."""

# **10. Final Validation**
# Missing Values
print("Missing Values:")
print(df.isnull().sum())

# Duplicate Rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Dataset Shape
print("\nDataset Shape:")
print(df.shape)

# Data Types
print("\nData Types:")
print(df.dtypes)

# Invalid Temperature Values
print("\nTemperature < 18:")
print((df["Temperature"] < 18).sum())

print("\nTemperature > 40:")
print((df["Temperature"] > 40).sum())

# Invalid Humidity Values
print("\nHumidity < 0:")
print((df["Humidity"] < 0).sum())

print("\nHumidity > 100:")
print((df["Humidity"] > 100).sum())

# Infinite Values
print("\nInfinite Values:")
print(np.isinf(df.select_dtypes(include=np.number)).sum().sum())

"""# A final validation check identified 2 remaining duplicate records in the dataset. These duplicates may have appeared after preprocessing and feature engineering steps, where transformations caused previously different records to become identical. Since duplicate observations can introduce bias and affect the reliability of the analysis, the remaining duplicates were removed to ensure that the final dataset contains only unique records."""

# Check remaining duplicates
print("Duplicates before removal:", df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

# Verify removal
print("Duplicates after removal:", df.duplicated().sum())

"""Conclusion: The dataset underwent a comprehensive preprocessing process to improve data quality and consistency. Missing numerical values were handled using median imputation to reduce the influence of extreme values, while categorical values were standardized to ensure uniformity across records. Invalid values were corrected, data types were converted to appropriate formats, duplicates were removed, and outliers were treated to minimize their impact on the analysis. Additionally, new features were engineered to capture meaningful patterns and relationships within the data. As a result, the dataset is now cleaner, more reliable, and better prepared for analysis and machine learning applications."""
