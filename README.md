- Group name : Team Cai Xi , Amanda and Yi Xin 
- Group members name : Cai Xi , Amanda and Yi Xin
  
- Cai Xi wrote cleaning, config and train files
- Amanda wrote features and cleaning files
- Yixin wrote evaluate and readme files
  
- Instructions on how to run the pipeline: It takes raw sensor data and produces a trained model in 4 steps : cleaning, feature, training and evaluation. The entire process is automated with this shell script.
  
- docker .
  
- Our EDA revealed three main findings. First, data quality issues - inconsistent labels, impossible readings like 89°C temperatures, and contaminated session 2586 which we'll remove entirely. Secondly, weak individual feature correlations - no single sensor predicts activity alone; activity depends on complex sensor interactions. Thirdly, class imbalance - High Activity is the minority class. Therefore, we'll use macro F1-score as our primary metric and ensemble models like Random Forest or Gradient Boosting that handle both feature interactions and class imbalance well.
  
- Based on our EDA finding that individual features have weak relationships with activity, we engineered three new features to capture sensor interactions and reduce noise.
Firstly, CO2 Disagreement - the absolute difference between infrared and electrochemical CO2 sensors. These two sensors measure the same thing, so they should agree. Large disagreement signals either sensor drift OR rapid CO2 flux during high physical activity. This captures something the raw values alone cannot.
Secondly, MOS Mean - the average of all four metal oxide sensors. These VOC sensors are individually noisy. The mean reduces per-sensor noise into a single 'overall VOC' signal. Think of it like averaging expert opinions - it's more reliable than any single sensor. We're keeping both the individual units AND the mean, then letting the model decide which is more useful.
Thirdly, Ambient Light Ordinal - converting light levels from categories like 'bright' and 'very_bright' to numbers 0 through 4. This preserves natural order for linear models - very_bright is clearly more than bright - which gets lost with standard categorical encoding.
Why these three? CO2 Disagreement captures sensor interactions. MOS Mean reduces noise. Light Ordinal enables linear modeling. Together, they address the key limitation from our EDA - weak individual features - by creating features that capture richer signals."

-We chose three models. Logistic Regression is our simple baseline. Random Forest handles the non-linear interactions our EDA showed are important. HistGradientBoosting is our strongest model for tabular data. All three use class_weight=balanced because our classes are uneven - 58% Low, 28% Moderate, 14% High. This tells the model to care more about getting High Activity right. For tuning, we use grid search with cross-validation - it tries different parameter combinations and validates on multiple data splits to avoid overfitting.

-We use macro F1-score as our primary metric. Accuracy would be misleading because our classes are imbalanced - 58% Low, 28% Moderate, 14% High. A dummy model predicting 'Low' every time would achieve 58% accuracy but completely fail at detecting High Activity. Macro F1 averages performance across all three classes equally, so poor performance on the minority High Activity class directly hurts the score. This forces our model to perform well on ALL activity levels, not just the majority class. 


