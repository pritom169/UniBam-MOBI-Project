# Import necessary libraries
import json
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Get the absolute path to the modeling directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths using os.path.join() to locate JSON data files correctly
weekday_files = [
    os.path.join(BASE_DIR, "[07]-traffic-and-weather-data", "[00]-weekdays", f"merged_traffic_and_weather_{hour}.json")
    for hour in ["9", "12", "15", "18", "21"]
]

weekend_files = [
    os.path.join(BASE_DIR, "[07]-traffic-and-weather-data", "[01]-weekends",
                 f"merged_traffic_and_weather_weekend_{hour}.json")
    for hour in ["09", "12", "15", "18", "21"]
]


# Function to load and preprocess data from JSON files
def load_data(file_paths, is_weekend):
    data_list = []
    for file_path in file_paths:
        with open(file_path, "r") as file:
            data = json.load(file)  # Load JSON file

        for street in data:
            street_name = street["street_name"]
            num_pois = street["number_of_nearby_pois"]
            for entry in street["traffic"]:
                record = {
                    "street_name": street_name,
                    "number_of_nearby_pois": num_pois,
                    "date": entry["date"],
                    "speed": entry["speed"],
                    "jamFactor": entry["jamFactor"],  # Target variable
                    "temperature": entry.get("temperature", np.nan),  # Handle missing temperature values
                    "is_weekend": is_weekend  # Assign weekend flag (0 for weekdays, 1 for weekends)
                }
                # Convert date to datetime format and extract day of the week
                record["date"] = pd.to_datetime(record["date"], errors="coerce")
                if pd.notna(record["date"]):
                    record["day_of_week"] = record["date"].dayofweek
                else:
                    continue  # Skip invalid date entries

                data_list.append(record)

    return pd.DataFrame(data_list)  # Return structured DataFrame


# Load weekday and weekend data
df_weekday = load_data(weekday_files, is_weekend=0)
df_weekend = load_data(weekend_files, is_weekend=1)

# Drop original date column (not needed for modeling)
df_weekday.drop(columns=["date"], inplace=True)
df_weekend.drop(columns=["date"], inplace=True)

# Handle missing temperature values by filling with the mean temperature
df_weekday["temperature"].fillna(df_weekday["temperature"].mean(), inplace=True)
df_weekend["temperature"].fillna(df_weekend["temperature"].mean(), inplace=True)

# Split data into training (60%) and testing (40%) sets
train_weekday, test_weekday = train_test_split(df_weekday, test_size=0.4, random_state=42)
train_weekend, test_weekend = train_test_split(df_weekend, test_size=0.4, random_state=42)

# Define features and target variable
features = ["speed", "temperature", "number_of_nearby_pois", "day_of_week", "is_weekend"]
target = "jamFactor"

# Train Linear Regression Model for Weekdays
lr_weekday = LinearRegression()
lr_weekday.fit(train_weekday[features], train_weekday[target])

# Predict on Weekday Test Set
pred_weekday = lr_weekday.predict(test_weekday[features])

# Calculate RMSE & MAE for Weekdays
rmse_weekday = np.sqrt(mean_squared_error(test_weekday[target], pred_weekday))
mae_weekday = mean_absolute_error(test_weekday[target], pred_weekday)

# Train Linear Regression Model for Weekends
lr_weekend = LinearRegression()
lr_weekend.fit(train_weekend[features], train_weekend[target])

# Predict on Weekend Test Set
pred_weekend = lr_weekend.predict(test_weekend[features])

# Calculate RMSE & MAE for Weekends
rmse_weekend = np.sqrt(mean_squared_error(test_weekend[target], pred_weekend))
mae_weekend = mean_absolute_error(test_weekend[target], pred_weekend)

# Create "visual" folder if it doesn't exist
VISUAL_DIR = os.path.join(BASE_DIR, "visual")
os.makedirs(VISUAL_DIR, exist_ok=True)


# Function to plot and save actual vs. predicted jamFactor with explanation
def plot_actual_vs_predicted(actual, predicted, title, filename):
    plt.figure(figsize=(8, 6))
    plt.scatter(actual, predicted, alpha=0.5, color="blue", label="Predictions")
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], linestyle="--", color="red",
             label="Perfect Fit (y=x)")
    plt.xlabel("Actual jamFactor")
    plt.ylabel("Predicted jamFactor")
    plt.title(title)
    plt.legend()

    # Add explanation inside the image
    explanation = "🔵 Predictions vs. Actual\n🔴 Red Line = Perfect Fit (Ideal Prediction)\n" \
                  "Points below the line → Underestimation\nPoints above the line → Overestimation"
    plt.text(0.1, max(actual) * 0.9, explanation, fontsize=10, bbox=dict(facecolor="white", alpha=0.8))

    plt.savefig(os.path.join(VISUAL_DIR, filename))
    plt.close()


# Function to save SHAP summary plot with proper text placement
def save_shap_summary_plot(explainer, data, title, filename):
    shap_values = explainer(data)

    # Further increase figure size for more space on the right
    plt.figure(figsize=(14, 6))  # Extra-wide image

    shap.summary_plot(shap_values, data, show=False)  # Generate SHAP summary plot

    # Get x and y limits to place text properly
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()

    explanation = ("🔴 Red = Higher Feature Value\n🔵 Blue = Lower Feature Value\n"
                   "➡️ Right = Increases Congestion (jamFactor)\n"
                   "⬅️ Left = Decreases Congestion\n"
                   "📊 Top features have the most impact on traffic.")

    # Move text even further right to fully clear the feature value color bar
    plt.text(x_max + (x_max - x_min) * 0.5, y_max * 0.8, explanation, fontsize=12,
             bbox=dict(facecolor="white", alpha=0.8), ha='left')

    # Save the SHAP summary image
    plt.savefig(os.path.join(VISUAL_DIR, filename), bbox_inches="tight")
    plt.close()


# Generate all plots
plot_actual_vs_predicted(test_weekday[target], pred_weekday, "Weekday Model: Actual vs. Predicted",
                         "weekday_actual_vs_predicted.png")
plot_actual_vs_predicted(test_weekend[target], pred_weekend, "Weekend Model: Actual vs. Predicted",
                         "weekend_actual_vs_predicted.png")

# Save SHAP summary plots with explanations
shap.initjs()

# SHAP for Weekday Model
explainer_weekday = shap.Explainer(lr_weekday, train_weekday[features])
save_shap_summary_plot(explainer_weekday, train_weekday[features], "Weekday SHAP Summary", "weekday_shap_summary.png")

# SHAP for Weekend Model
explainer_weekend = shap.Explainer(lr_weekend, train_weekend[features])
save_shap_summary_plot(explainer_weekend, train_weekend[features], "Weekend SHAP Summary", "weekend_shap_summary.png")

# Display Results
results_df = pd.DataFrame({
    "Model": ["Weekday Model", "Weekend Model"],
    "RMSE": [rmse_weekday, rmse_weekend],
    "MAE": [mae_weekday, mae_weekend]
})

print(results_df)
results_df.to_csv("model_results.csv", index=False)
