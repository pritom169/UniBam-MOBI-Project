import json
import pandas as pd
from scipy.stats import pearsonr

# Load POI Data
with open("data/poi_data.json", "r", encoding="utf-8") as f:
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

print("POI Counts per Zone:", zone_poi_counts)

# Load Visitor Data
visitor_data = pd.read_csv("../[02]-source-files/[01]-population-data/mobithek_data_with_rssi_60min.xls", sep=";", skiprows=3)

# Identify visitor data columns corresponding to the zones
zone_visitor_counts = {}
for zone_code in zone_poi_counts.keys():
    matching_columns = [col for col in visitor_data.columns[1:] if zone_code in col]
    if matching_columns:
        zone_visitor_counts[zone_code] = visitor_data[matching_columns].sum().sum()
    else:
        print(f"No matching columns for {zone_code}. Skipping this zone.")

# Convert np.int64 to native Python int
zone_visitor_counts = {zone: int(visitor_count) for zone, visitor_count in zone_visitor_counts.items()}
print("Visitor Counts per Zone:", zone_visitor_counts)

# Combine Data and Perform Pearson Correlation
poi_counts = []
visitor_counts = []
for zone in zone_poi_counts.keys():
    if zone in zone_visitor_counts:
        poi_counts.append(zone_poi_counts[zone])
        visitor_counts.append(zone_visitor_counts[zone])

# Check if we have enough data for correlation
if len(poi_counts) > 1 and len(visitor_counts) > 1:
    correlation, p_value = pearsonr(poi_counts, visitor_counts)
    print(f"\nPearson correlation coefficient: {correlation}")
    print(f"P-value: {p_value}")
    if p_value < 0.05:
        print("The correlation is statistically significant.")
    else:
        print("The correlation is NOT statistically significant.")
else:
    print("\nNot enough valid data for Pearson correlation analysis.")
