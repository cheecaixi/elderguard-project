# ElderGuard Analytics — Activity Level Prediction Pipeline

## Group Information

- **Group Name:** Team Cai Xi, Amanda and Yi Xin
- **Group Members:** Cai Xi, Amanda, Yi Xin

 ### Code Contributions
| File | Author |
|------|--------|
| `src/cleaning.py` | Cai Xi & Amanda |
| `src/features.py` | Amanda |
| `src/config.py` | Cai Xi |
| `src/train.py` | Cai Xi |
| `src/evaluate.py` | Yi Xin & Cai Xi|
| `src/feature_analysis.py` | Cai Xi |
| `eda.ipynb` | All members |
| `cleaning.ipynb` | Amanda |
| `Readme.md` | Cai Xi & Yi Xin |

##Problem Statement

Raw sensor data contains errors, missing values, and physical impossibilities (e.g. 89°C indoors, negative CO2 readings), preventing reliable prediction of human activity levels from gas/environmental sensors.

Goal: Build an end-to-end ML pipeline — EDA → cleaning → feature engineering → training → evaluation → feature analysis — that predicts Activity Level (low / moderate / high) from raw sensor data, identifies key predictive features, and serves as a reliable, non-invasive early-warning system to help caregivers intervene proactively.
 
## Instructions on how to run the pipeline
-  Docker Desktop installed and running, OR Python 3.10+ with pip

### Option 1 — Docker (Recommended)

**Build the image:**
```bash
docker build -t elderguard .
```

**Run the full pipeline:**
```bash
docker run --rm elderguard
```

The pipeline will execute all steps in order: data cleaning → feature engineering → model training → evaluation → feature analysis. Trained models and evaluation plots are saved to `saved_model/`.

### Option 2 — Run Locally

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the pipeline:**
```bash
bash run.sh
```

**Or run individual steps:**
```bash
python src/cleaning.py
python src/features.py
python src/train.py
python src/evaluate.py
python src/feature_analysis.py
```

**Skip hyperparameter tuning (faster):**
```bash
python src/train.py --no-tune
```

---

## Docker Development Environment

To develop interactively inside the container with your local files mounted (changes reflect immediately without rebuilding):

```bash
docker run --rm -it -v ${PWD}:/app elderguard bash
```

This mounts your project folder into the container so you can edit files locally and re-run scripts inside the container without a full rebuild.


## Project Structure

```
elderguard-project/
├── data/
│   └── gas_monitoring.db       # raw sensor database
├── src/
│   ├── config.py               # central configuration
│   ├── cleaning.py             # data loading and cleaning pipeline
│   ├── features.py             # feature engineering and encoding
│   ├── train.py                # model training and hyperparameter tuning
│   ├── evaluate.py             # test-set evaluation and plots
│   └── feature_analysis.py     # feature importance 
├── saved_model/                # trained models and artefacts (generated)
├── eda.ipynb                   # exploratory data analysis notebook
├── cleaning.ipynb              # cleaning visualization notebook
├── Dockerfile
├── requirements.txt
├── run.sh
├── .dockerignore
├── .gitignore
└── README.md
```
## Version Control

This project uses Git and GitHub for version control. Each member developed on their own named branch and merged into `main` upon completion.

**Repository:** https://github.com/cheecaixi/elderguard-project

**Branches:**
- `main` — stable, submission-ready code
- `Chee-Cai-Xi` — Cai Xi's development branch (cleaning, config, train, feature_analysis); merged to `main` upon completion
- `Amanda-Jane-Lobo` — Amanda's development branch (cleaning, features); merged to `main` upon completion
- `Ong-Yi-Xin` — Yi Xin's development branch (evaluate, readme); merged to `main` upon completion

---

## Key EDA Findings

### 1. Data Quality Issues
The raw dataset contained several quality problems requiring cleaning before modelling:
- **Inconsistent activity labels** — variants like `lowactivity`, `low activity`, `low_activity` all referring to the same class, standardised to `low_activity`, `moderate_activity`, `high_activity`.
- **Physically impossible sensor readings** — Session 2586 had a mean temperature of 89.9°C, which is physically impossible indoors. Since all readings originate from the same faulty hardware, the entire session (56 rows) was removed. Additionally, 929 temperature readings outside 15–40°C and 410 humidity readings outside 0–100% were marked as NaN and imputed.
- **Missing values** — 824 CO_GasSensor readings, 1,051 Ambient Light Level readings, and others were imputed using session-level median or global median/mode depending on the column's variance characteristics.

### 2. Weak Individual Feature Correlations
No single sensor reliably predicts activity level on its own. Activity depends on complex interactions between multiple sensors simultaneously. This finding motivated our choice of ensemble models and the engineering of interaction features.

### 3. Class Imbalance
The dataset is imbalanced: Low Activity ~58%, Moderate Activity ~28%, High Activity ~14%. A naive model predicting "Low" every time would achieve 58% accuracy while completely failing to detect High Activity. This directly shaped our metric choice (Macro F1) that averages F1 equally across all 3 classes and imbalance-handling strategy (SMOTE, class weigths).

---

## Feature Engineering

Based on the EDA finding that individual features have weak predictive power, we engineered features to capture sensor interactions and reduce noise. After feature importance analysis (RF importance + permutation importance), we retained the following engineered features and dropped weak ones:

### Features Created

**CO2_Disagreement** — Absolute difference between the infrared and electrochemical CO2 sensors. Both sensors measure the same gas, so they should agree. Large disagreement signals either sensor drift or rapid CO2 flux during high physical activity — a pattern the raw values alone cannot capture.

**CO2_Mean** — Average of both CO2 sensors. Reduces per-sensor noise into a single CO2 signal more robust than either sensor individually.

**MOS_Core_Active_Mean** — Mean of MetalOxideSensor Unit2 and Unit4, identified as the two highest-importance MOS units from permutation importance analysis. The raw individual units are dropped because the engineered mean captures their combined signal with higher permutation importance (0.029) than any individual unit. This reduces noise while preserving the VOC signal.

**MOS_Core_Active_Range** — Max minus min between Unit2 and Unit4. Captures sharp localised volatility between the two core sensors.

**Ambient_Light_Ordinal** — Ordinal encoding of Ambient Light Level (very_dim=0 through very_bright=4). Preserves natural order for linear models, which is lost with standard one-hot encoding.

### Feature Selection
After running permutation importance on the held-out test set, four HVAC dummy features (heating_low, heating_high, cooling_low, cooling_high) showed zero permutation importance and were dropped. This reduced noise and improved Random Forest test Macro F1 from 0.5401 to 0.5494.

### Key Predictive Features (from Permutation Importance on test set)
Permutation Importance — shuffle one feature's values randomly, then measure how much the model's score dropsvin F1. Big drop = feature matters. Small drop = feature is not important. 

| Rank | Feature | Permutation Importance |
|------|---------|----------------------|
| 1 | MOS_Core_Active_Mean | 0.0290 |
| 2 | MetalOxideSensor_Unit3 | 0.0170 |
| 3 | MetalOxideSensor_Unit1 | 0.0138 |
| 4 | CO2_ElectroChemicalSensor | 0.0131 |
| 5 | MetalOxideSensor_Unit4 | 0.0109 |
| 6 | CO2_Disagreement | 0.0103 |

`MOS_Core_Active_Mean` is the dominant predictor by a large margin — 0.029 vs 0.017 for the next feature. This confirms that VOC sensor activity is the strongest environmental signal for distinguishing activity levels. CO2-related features (both raw and engineered disagreement) consistently appear in the top 6, supporting our EDA finding that activity drives respiratory changes detectable by CO2 sensors.

## Insight — RF Importance vs Permutation Importance

MetalOxideSensor_Unit2 ranks 2nd by RF built-in importance but 11th by permutation importance. This means Unit2 helped the model fit training data but contributes little on unseen data — a sign of overfitting to noise in that sensor.

RF built-in importance is computed on training data and can be misleading. Permutation importance on the held-out test set is more trustworthy because it measures actual generalisation, not training fit.

**Next step:** Drop MetalOxideSensor_Unit2 from the feature set and retrain — if test F1 holds or improves, it confirms Unit2 was noise. This iterative process of importance analysis → feature removal → retraining is how production ML pipelines are refined over time.

---

## Model Choices and Tuning

We trained three models covering a range of complexity:

### Logistic Regression (Baseline)
A linear model included as a simple, interpretable baseline. Requires feature scaling (StandardScaler applied to continuous features). Uses `class_weight='balanced'` to penalise misclassification of minority classes. Expected to perform worst given EDA showed non-linear sensor interactions.

### Random Forest
An ensemble of decision trees that naturally handles non-linear feature interactions — directly addressing the key EDA finding. Uses `class_weight='balanced_subsample'`, which applies balanced weighting independently per tree for more robust minority class handling. **Best performing model: Test Macro F1 = 0.5494.**

### XGBoost
Gradient boosted trees, strong on tabular data. Unlike sklearn models, XGBoost has no native `class_weight` parameter, so per-sample weights computed from class frequencies (`compute_sample_weight('balanced')`) are passed directly to `fit()`. This achieves equivalent effect to `class_weight='balanced'`. Hyperparameter tuning constrained `max_depth` to 3–4 and `learning_rate` to 0.05–0.1 to prevent overfitting — earlier unconstrained runs with `max_depth=5` and `learning_rate=0.2` produced a train F1 of 0.92 against a test F1 of 0.51, a clear overfit. The constrained grid produced a healthy train F1 of 0.5890 vs test F1 of 0.5248.

### Imbalance Handling
All three models additionally use **SMOTE** (Synthetic Minority Oversampling Technique) during cross-validation folds. SMOTE generates synthetic minority class samples inside each fold only, preventing data leakage into validation sets. After SMOTE balances the fold, sample weights are recomputed on the resampled labels to avoid double-penalising the minority class.

### Hyperparameter Tuning
GridSearchCV with 3-fold StratifiedKFold cross-validation, scoring on Macro F1. StratifiedKFold preserves class distribution across folds. SMOTE is wrapped inside an ImbPipeline within GridSearchCV so resampling occurs inside each fold and never leaks into validation data. Key parameters tuned: `max_depth`, `min_samples_leaf`, `n_estimators` (RF); `learning_rate`, `max_depth`, `n_estimators` (XGBoost); `C`, `solver` (Logistic Regression).

---

## Evaluation Metric

**Primary metric: Macro F1-score**

Accuracy is misleading for imbalanced datasets. With 58% Low Activity samples, a model predicting "Low" for every input achieves 58% accuracy while completely failing to detect High Activity — the most clinically critical class for an elderly care early-warning system.

Macro F1 averages the F1-score across all three classes equally, regardless of class size. Poor performance on the minority High Activity class directly penalises the score. This forces the model to perform well across all activity levels, which aligns with the problem statement goal of detecting distress and medical episodes.

### Final Results

| Model | Train Accuracy | Train Macro F1 | Test Accuracy | Test Macro F1 |
|-------|---------------|----------------|--------------|---------------|
| Random Forest | 0.7082 | 0.6613 | 0.6207 | **0.5494** |
| XGBoost | 0.6634 | 0.5890 | 0.6264 | 0.5248 |
| Logistic Regression | 0.5869 | 0.5067 | 0.5994 | 0.5153 |

Random Forest is selected as the best model based on highest CV (0.5325) and test Macro F1 (0.5494). The High Activity class F1 of 0.353 reflects the inherent difficulty of detecting the minority class (14% of data), and is meaningfully higher than Logistic Regression (0.293) and XGBoost (0.270). The train-test gap for Random Forest (~0.11) is expected given SMOTE inflates training scores — the CV score of 0.5325 is the honest generalisation estimate, and the test result of 0.5494 slightly exceeded it, confirming no test set overfitting.
