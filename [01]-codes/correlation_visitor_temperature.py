import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# File Paths
visitor_file = r"D:\Bamberg University\winter24_25\project\UniBam-MOBI-Project\data\Visitor_Traffic_Data_by_Zone.csv"
weather_file = r"D:\Bamberg University\winter24_25\project\UniBam-MOBI-Project\data\bamberg_weather.csv"
figures_path = r"D:\Bamberg University\winter24_25\project\UniBam-MOBI-Project\figures"

# Ensure figures directory exists
os.makedirs(figures_path, exist_ok=True)

# Load Visitor Data
visitor_df = pd.read_csv(visitor_file)
visitor_df["timestamp"] = pd.to_datetime(visitor_df["timestamp"])

# Aggregate Visitor Data to Daily Total
visitor_df["Date"] = visitor_df["timestamp"].dt.date  # Extract date
visitor_daily = visitor_df.groupby("Date").sum(numeric_only=True).reset_index()  # Exclude datetime column
visitor_daily["Total Visitors"] = visitor_daily.iloc[:, 1:].sum(axis=1)  # Sum across all zones
visitor_daily["Date"] = pd.to_datetime(visitor_daily["Date"])  # Convert Date to datetime

# Load Weather Data
weather_df = pd.read_csv(weather_file)
weather_df["Date"] = pd.to_datetime(weather_df["Date"])  # Convert Date to datetime

# Merge Data on Date
merged_df = pd.merge(visitor_daily, weather_df, on="Date")

# Compute Correlation
correlation_matrix = merged_df.corr(numeric_only=True)
print("Correlation Matrix:\n", correlation_matrix)

# Plot and Save Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Visitor Traffic and Weather Variables", fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "correlation_heatmap.png"))
plt.show()

# Scatter Plot: Total Visitors vs Average Temperature
plt.figure(figsize=(8, 5))
sns.scatterplot(data=merged_df, x="Average Temperature (°C)", y="Total Visitors")
plt.title("Visitor Count vs Average Temperature", fontsize=14)
plt.xlabel("Average Temperature (°C)", fontsize=12)
plt.ylabel("Total Visitors", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "visitor_vs_temperature.png"))
plt.show()

# Scatter Plot: Total Visitors vs Wind Speed
plt.figure(figsize=(8, 5))
sns.scatterplot(data=merged_df, x="Max Wind Speed (m/s)", y="Total Visitors")
plt.title("Visitor Count vs Wind Speed", fontsize=14)
plt.xlabel("Max Wind Speed (m/s)", fontsize=12)
plt.ylabel("Total Visitors", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "visitor_vs_wind_speed.png"))
plt.show()
