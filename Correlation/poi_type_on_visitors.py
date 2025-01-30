import json
import pandas as pd

# Step 1: Count POIs by Type for Each Zone
with open("poi_data.json", "r", encoding="utf-8") as f:
    poi_data = json.load(f)

zone_poi_counts = {}
zone_poi_types = {}
zone_address_mapping = {
    "bz2454": "Mußstraße",
    "bz2453": "Obere Sandstraße",
    "bz2452": "gabelmann",
    "bz2457": "Domplatz",
    "bz2458": "Geyerswörthstraße",
    "bz2460": "Ob. Brücke",
    "bz2462": "Maximilianspl"
}

# Count POIs by type for each zone
for zone_code, address in zone_address_mapping.items():
    poi_type_counts = {}
    for entry in poi_data:
        if entry["address"] == address:
            for poi in entry["pois"]:
                poi_type = poi.get("type", "unknown")  # Assuming POIs have a "type" key
                poi_type_counts[poi_type] = poi_type_counts.get(poi_type, 0) + 1
    zone_poi_types[zone_code] = poi_type_counts

print("POI Type Counts per Zone:")
for zone, types in zone_poi_types.items():
    print(f"{zone}: {types}")

# Step 2: Read visitor data as a CSV after skipping headers
visitor_data = pd.read_csv("mobithek_data_with_rssi_60min.xls", sep=";", skiprows=3)

# Print columns to identify zone-related data
# print("Visitor Data Columns:", visitor_data.columns)

# Step 3: Identify visitor data columns corresponding to zones (wide, mid, close)
zone_visitor_counts = {}

for zone_code in zone_address_mapping.keys():
    matching_columns = []
    for col in visitor_data.columns[1:]:  # Skip the first column (timestamp)
        if zone_code in col:
            matching_columns.append(col)

    if matching_columns:
        # Sum values for matching columns
        visitor_sum = visitor_data[matching_columns].sum().sum()
        zone_visitor_counts[zone_code] = visitor_sum
    else:
        print(f"No matching columns for {zone_code}. Skipping this zone.")

zone_visitor_counts = {zone: int(visitor_count) for zone, visitor_count in zone_visitor_counts.items()}
print("Visitor Counts per Zone:", zone_visitor_counts)

# Step 4: Analyze Correlations for All POI Types
poi_correlations = {}

# Extract all unique POI types from the data
all_poi_types = set()
for types in zone_poi_types.values():
    all_poi_types.update(types.keys())

# Calculate correlation for each POI type
for poi_type in all_poi_types:
    poi_type_counts = []
    visitor_counts = []

    for zone, poi_types in zone_poi_types.items():
        poi_type_count = poi_types.get(poi_type, 0)
        visitor_count = zone_visitor_counts.get(zone, 0)
        poi_type_counts.append(poi_type_count)
        visitor_counts.append(visitor_count)

    # Remove zones with zero visitors or missing data
    valid_indices = [i for i in range(len(visitor_counts)) if visitor_counts[i] != 0]
    poi_type_counts = [poi_type_counts[i] for i in valid_indices]
    visitor_counts = [visitor_counts[i] for i in valid_indices]

    if poi_type_counts and visitor_counts:
        correlation = pd.Series(poi_type_counts).corr(pd.Series(visitor_counts))
        poi_correlations[poi_type] = correlation
    else:
        poi_correlations[poi_type] = None

# Print results
print("\nCorrelations between POI types and visitor counts:")
for poi_type, correlation in poi_correlations.items():
    print(f"{poi_type}: {correlation}")

# Identify the POI type with the strongest correlation
strongest_poi_type = max(poi_correlations, key=lambda k: abs(poi_correlations[k]) if poi_correlations[k] is not None else 0)
print(f"\nThe POI type with the strongest correlation is '{strongest_poi_type}' with a correlation of {poi_correlations[strongest_poi_type]:.4f}.")
