# Modelling Traffic Patterns in Bamberg

An ensemble modeling approach to predict traffic congestion (`jamFactor`) using weather, time, street data, POI, and traffic speed.
This project utilizes three powerful ensemble models: RandomForest, XGBoost, and GradientBoosting, to predict traffic congestion (jamFactor). These models are trained using historical traffic data, weather conditions, and additional features like Points of Interest (POI) and traffic speed. Each model independently predicts traffic congestion based on the input features, offering a diverse range of perspectives for accurate forecasting.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Model Architecture](#model-architecture)
5. [Training the Model](#training-the-model)
6. [Making Predictions](#making-predictions)
7. [Evaluation Metrics](#evaluation-metrics)

---

## Project Overview

### Objective

Predict the `jamFactor` (a metric indicating traffic congestion severity) using:

- **Day Type**: Weekday or weekend.
- **Time**: 09:00, 12:00, 15:00, 18:00, or 21:00.
- **Street**: Street name (e.g., Feldkirchenstraße).
- **Temperature**: Real-time temperature (°C).
- **Speed**: Average vehicle speed (km/h).

### Workflow

1. **Data Merging**: Load and merge multiple JSON files containing traffic and weather data from weekdays and weekends, including features like number of nearby POIs, speed, and temperature.
2. **Feature Engineering**: Extract time-based features such as hour and weekday. Aggregate data by averaging speed and jamFactor for each combination of features.
3. **Preprocessing**: Handle missing values with mean imputation and apply One-Hot Encoding to the `street_name` column.
4. **Model Training**: Train ensemble models to predict traffic congestion (`jamFactor`):
   - **RandomForestRegressor**
   - **XGBRegressor** (XGBoost)
   - **GradientBoostingRegressor**
5. **Evaluation**: Evaluate model performance using K-fold cross-validation with metrics like **Mean Absolute Error (MAE)** and **R² score**, and visualize results using scatter and residual plots.
6. **Prediction**: Use the trained models to predict traffic congestion for new inputs such as street, POIs, time, weather, and speed.

---

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/traffic-jam-predictor.git
   cd traffic-jam-predictor
   ```

2. Install dependencies:
   ```bash
   pip install requirements.txt

   ```
   *(requirements includes: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`)*

3. Ensure the following files are present:
   -  `Predictive_Models/data/` (Contains JSON files for weekdays and weekends.)
   - `model_training.ipynb/` Script to train the model  and also for testing

---

## Data Preparation

### 1. Loading JSON Files

The dataset consists of separate JSON files for:

- **Weekdays**: `merged_traffic_and_weather_9.json`, `merged_traffic_and_weather_12.json`, etc.
- **Weekends**: `merged_traffic_and_weather_weekend_9.json`, `merged_traffic_and_weather_weekend_12.json`, etc.


### 2. Preprocessing
- **Categorical Encoding**:
  - **Street Names**: One-hot encoded (e.g., `Feldkirchenstraße` → `[1, 0, 0]`, `Brennerstraße` → `[0, 1, 0]`).
  - **Time Features**:
    - **Hour**: Used as a numerical feature.
    - **Day of the Week**: Encoded as an integer (`0 = Monday`, `6 = Sunday`).
    - **Is Weekend**: Binary feature (`0 = weekday`, `1 = weekend`).

- **Numerical Features**:
  - **Number of Nearby POIs**: Represents the count of nearby points of interest.
  - **Temperature**: Imputed using the **mean strategy** if missing.
  - **Speed**: Imputed using the **mean strategy** if missing.

- **Numerical Scaling**:
  - `temperature` and `speed` are standardized using `StandardScaler`.


---


## Model Architecture

### Machine Learning Models
The following ensemble models are used to predict **Jam Factor**:

```python
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}
```

--- 

## Training the Model

### K-Fold Cross-Validation

To evaluate model performance, **5-Fold Cross-Validation** is used. This technique splits the dataset into five equal-sized folds, where:

- The model is trained on **4 folds** and tested on **1 fold**.
- This process repeats **5 times**, ensuring that each fold serves as a test set once.
- The results from all folds are averaged to provide a more reliable performance estimate.
```python
from sklearn.model_selection import KFold

# Define K-Fold Cross-Validation (5 Folds)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

```

**Metrics Tracked**:
  - Mean MAE
  - Mean R² Score
 

---

## Making Predictions

### Using the `model_training_and_analysis.py` Script
```python
# Prediction with GradientBoosting model
selected_model = models["GradientBoosting"] 

predicted_jam_factor = predict_jam_factor(
    street="Feldkirchenstraße", 
    number_of_nearby_pois=5,
    hour=9,
    weekday=2,  # Tuesday
    is_weekend=0,
    temperature=5.0,
    speed=10.0,
    model=selected_model
)

print(f"Predicted Jam Factor is: {predicted_jam_factor:.2f}")
```

## Prediction Workflow

1. **Encode Street**: Convert street name using one-hot encoding.
2. **Encode Day Type**: Convert to `0` (weekday) or `1` (weekend).
3. **Encode Time**: Use `hour` and `weekday` as input features.
4. **Scale Features**: Handle missing values using `SimpleImputer`.
5. **Predict**: Feed the processed features into the selected  model (`RandomForestRegressor`, `XGBRegressor`, or `GradientBoostingRegressor`) to estimate the `jamFactor`.


---

## Evaluation Metrics

### Metrics
1. **Mean Absolute Error (MAE)**: Measures the average absolute difference between actual and predicted values.
   ```python
   mae_scores = cross_val_score(model, X, y, cv=kf, scoring=make_scorer(mean_absolute_error, greater_is_better=False))
 
   ```
2. **R² Score (Coefficient of Determination)**: Indicates how well the model explains the variance in the target variable.
   ```python
   mse = mean_squared_error(y_true, y_pred)
   ```

---

### Visualization
---
1. **Actual vs Predicted Plot**:
   ```python
   sns.scatterplot(x=y, y=y_pred, alpha=0.6, color="blue", label="Predicted vs Actual")

   plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle="--", label="Perfect Prediction Line")
   ```
   ![Actual vs Predicted](Predictive_Models\model_output_visuals\gradient_boosting_actual_vs_predicted_jamfactor.png)
2. **Residual Plot**:
   ```python
   sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, color="green", label="Residuals vs Predicted")

   plt.axhline(y=0, color="red", linestyle="--", label="Zero Residuals Line")
   ```
   ![Residual Plot](Predictive_Models\model_output_visuals\gradient_boosting_residuals_vs_predicted_jamfactor.png)
---