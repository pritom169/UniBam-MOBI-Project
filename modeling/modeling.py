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

# Load and combine data
df_weekday = load_data(weekday_files, is_weekend=0)
df_weekend = load_data(weekend_files, is_weekend=1)
df_combined = pd.concat([df_weekday, df_weekend], ignore_index=True)

# Drop original date column (not needed for modeling)
df_combined.drop(columns=["date"], inplace=True)

# Handle missing temperature values by filling with the mean temperature
df_combined["temperature"].fillna(df_combined["temperature"].mean(), inplace=True)

# Split data into training (60%) and testing (40%) sets
train, test = train_test_split(df_combined, test_size=0.4, random_state=42)

# Define features and target variable
features = ["speed", "temperature", "number_of_nearby_pois", "day_of_week", "is_weekend"]
target = "jamFactor"

# Train Linear Regression Model
lr = LinearRegression()
lr.fit(train[features], train[target])

# Predict on Test Set
pred = lr.predict(test[features])

# Calculate RMSE & MAE
rmse = np.sqrt(mean_squared_error(test[target], pred))
mae = mean_absolute_error(test[target], pred)

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
    plt.savefig(os.path.join(VISUAL_DIR, filename))
    plt.close()

# Generate actual vs predicted plot
plot_actual_vs_predicted(test[target], pred, "Combined Model: Actual vs Predicted", "combined_actual_vs_predicted.png")

# Create residuals plot
residuals = test[target] - pred
plt.figure(figsize=(8, 6))
plt.scatter(pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.savefig(os.path.join(VISUAL_DIR, "residuals_plot.png"))
plt.close()


# Function to save SHAP summary plot with proper text placement
def save_shap_summary_plot(explainer, data, title, filename):
    shap_values = explainer(data)

    # Increase figure size for better visibility
    plt.figure(figsize=(12, 6))  # Wider figure to accommodate text

    shap.summary_plot(shap_values, data, show=False)
    plt.title(title)
    plt.savefig(os.path.join(VISUAL_DIR, filename), bbox_inches='tight')
    plt.close()


# Then perform SHAP analysis
shap.initjs()
explainer = shap.Explainer(lr, train[features])
save_shap_summary_plot(explainer, train[features], "Combined SHAP Summary", "combined_shap_summary.png")
# Display Results
results_df = pd.DataFrame({
    "Model": ["Combined Model"],
    "RMSE": [rmse],
    "MAE": [mae]
})

print(results_df)
results_df.to_csv("model_results.csv", index=False)