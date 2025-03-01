import json
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests, adfuller

#Load POI Data
with open("../Causal_Analysis/data/poi_data.json", "r", encoding="utf-8") as f:
    poi_data = json.load(f)

zone_poi_counts = {}
zone_address_mapping = {
    "bz2454": "Mußstraße",
    "bz2453": "Obere Sandstraße",
    "bz2452": "gabelmann",
    "bz2457": "Domplatz",
    "bz2458": "Geyerswörthstraße",
    "bz2460": "Ob. Brücke",
    "bz2462": "Maximilianspl"
}

for zone_code, address in zone_address_mapping.items():
    count = sum(len(entry["pois"]) for entry in poi_data if entry["address"] == address)
    zone_poi_counts[zone_code] = count

#Load Visitor Data (with timestamps)
visitor_data = pd.read_csv("../[02]-source-files/[01]-population-data/mobithek_data_with_rssi_60min.xls", sep=";", skiprows=3, parse_dates=["epocutc"])
visitor_data.set_index("epocutc", inplace=True)  # Ensure timestamp is the index

# Prepare Visitor Count Per Zone Over Time
time_series_data = {}
for zone_code in zone_poi_counts.keys():
    matching_columns = [col for col in visitor_data.columns if zone_code in col]
    if matching_columns:
        time_series_data[zone_code] = visitor_data[matching_columns].sum(axis=1)  # Sum over all sub-columns
    else:
        print(f"No matching columns for {zone_code}. Skipping.")

# Convert to DataFrame
visitor_df = pd.DataFrame(time_series_data)
visitor_df = visitor_df.rename_axis("timestamp")  # Ensure proper time index

print('visitor_df', visitor_df)

# Check Stationarity & Apply Differencing

def check_stationarity(series, label):
    result = adfuller(series.dropna())  # Drop NaN for ADF test
    print(f"{label}: ADF Statistic = {result[0]}, p-value = {result[1]}")
    if result[1] > 0.05:
        print(f"{label} is NOT stationary. Differencing applied.")
        return series.diff().dropna()
    else:
        print(f"{label} is stationary.")
        return series

# Apply to each zone's visitor count
for zone in visitor_df.columns:
    visitor_df[zone] = check_stationarity(visitor_df[zone], f"Visitor Count ({zone})")

#Merge POI Count (Static) With Visitor Data (Time Series)
for zone_code in zone_poi_counts.keys():
    # Treat POI count as a constant over time for each zone
    visitor_df[f"{zone_code}_poi_count"] = zone_poi_counts[zone_code]

#Apply Granger Causality Test
for zone in zone_poi_counts.keys():
    if zone in visitor_df.columns:
        # Create a DataFrame with both visitor count (time series) and POI count (constant)
        data_for_test = visitor_df[[zone, f"{zone}_poi_count"]].dropna()
        
        # Check if the POI count column is constant (i.e., it has no variation)
        if data_for_test[f"{zone}_poi_count"].nunique() == 1:
            print(f"Skipping Granger Causality Test for {zone} because POI count is constant.")
            continue
        
        num_obs = len(data_for_test)
        maxlag = min(2, num_obs - 1)
        
        if maxlag >= 1:
            print(f"\nGranger Causality Test for {zone}:")
            grangercausalitytests(data_for_test, maxlag=maxlag, verbose=True)
        else:
            print(f"Not enough data for {zone}.")




