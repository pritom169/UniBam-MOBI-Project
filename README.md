# Modelling Traffic Patterns in Bamberg

A regression-based system to predict traffic congestion (`jamFactor`) using weather, time, and street data. This project processes JSON files containing historical traffic and weather data, trains a **Ridge Regression (L2 Regularization)** model, and allows predictions for specific conditions.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Model Training](#model-training)
5. [Making Predictions](#making-predictions)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Visualization](#visualization)

---

## Project Overview

### Objective

Predict the `jamFactor` (a metric indicating traffic congestion severity) using:

- **Day Type**: Weekday or weekend.
- **Time Slots**: 09:00, 12:00, 15:00, 18:00, 21:00.
- **Street**: Different locations in Bamberg.
- **Weather**: Temperature (°C).
- **Traffic Speed**: Average vehicle speed (km/h).
- **Nearby POIs**: Number of Points of Interest within 650m.

### Workflow

1. **Data Merging**: JSON files are merged into a structured dataset.
2. **Preprocessing**: Encoding categorical features, handling missing values, and splitting datasets.
3. **Model Training**: Using **Ridge Regression** to predict `jamFactor`.
4. **Prediction**: Using the trained model for new data.
5. **Evaluation**: Using RMSE and MAE to assess model performance.
6. **Visualization**: Generating plots for **SHAP feature importance** and model accuracy.

---

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/dev-shakil/traffic-modelling.git
   cd traffic-modelling
   ```

2. Install dependencies:

   ```bash
   pip install numpy pandas scikit-learn shap matplotlib
   ```

3. Ensure the following directories exist:

   - `[07]-traffic-and-weather-data/` (Contains JSON files for weekdays and weekends.)
   - `visual/` (Stores generated plots for analysis.)

---

## Data Preparation

### 1. Loading JSON Files

The dataset consists of separate JSON files for:

- **Weekdays**: `merged_traffic_and_weather_9.json`, `merged_traffic_and_weather_12.json`, etc.
- **Weekends**: `merged_traffic_and_weather_weekend_9.json`, `merged_traffic_and_weather_weekend_12.json`, etc.

### 2. Feature Engineering

- **Categorical Features**:
  - `day_of_week`: Encoded as an integer (0 = Monday, 6 = Sunday).
  - `is_weekend`: Binary feature (0 = weekday, 1 = weekend).
- **Numerical Features**:
  - `speed`, `temperature`, `number_of_nearby_pois`.

### 3. Train-Test Split

- **Training Data**: 60% of the dataset.
- **Testing Data**: 40% of the dataset.

---

## Model Training

### Ridge Regression (L2 Regularization)

```python
from sklearn.linear_model import Ridge
ridge_model = Ridge(alpha=1.0)  # Regularization strength
ridge_model.fit(train_weekday[features], train_weekday[target])
```

### Training Process

1. Train **Ridge Regression** model on training data.
2. Predict `jamFactor` on test data.
3. Evaluate using RMSE and MAE.

---

## Making Predictions

Use the trained Ridge Regression model to predict traffic congestion.

```python
new_data = pd.DataFrame({
    "speed": [25],
    "temperature": [12.5],
    "number_of_nearby_pois": [5],
    "day_of_week": [2],
    "is_weekend": [0]
})

predicted_jamFactor = ridge_model.predict(new_data)
print(f"Predicted jamFactor: {predicted_jamFactor[0]:.2f}")
```

---

## Evaluation Metrics

### 1. Root Mean Squared Error (RMSE)

```python
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
```

Measures how far the predictions deviate from actual values.

### 2. Mean Absolute Error (MAE)

```python
mae = mean_absolute_error(y_true, y_pred)
```

Gives the average absolute error in prediction.

---

## Visualization

### 1. Actual vs. Predicted jamFactor

```python
plt.scatter(y_actual, y_predicted, color='blue', alpha=0.5)
plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], 'r--')
plt.xlabel("Actual jamFactor")
plt.ylabel("Predicted jamFactor")
plt.title("Actual vs. Predicted jamFactor")
plt.savefig("visual/actual_vs_predicted.png")
```

### 2. SHAP Feature Importance

```python
explainer = shap.Explainer(ridge_model, train_weekday[features])
shap_values = explainer(train_weekday[features])
shap.summary_plot(shap_values, train_weekday[features])
plt.savefig("visual/shap_summary.png")
```

---

## Future Improvements

✅ **Try Lasso Regression for feature selection.** ✅ **Use Deep Learning Approaches for complex relationships.** ✅ **Integrate Real-Time Traffic Data for live predictions.**

---

