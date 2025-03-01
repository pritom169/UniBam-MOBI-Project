import json
import pandas as pd


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
    count = 0
    for entry in poi_data:
        if entry["address"] == address:
            count += len(entry["pois"])
    zone_poi_counts[zone_code] = count

print("POI Counts per Zone:", zone_poi_counts)

# Step 2: Read the data as CSV after skipping headers
visitor_data = pd.read_csv("data/mobithek_data_with_rssi_60min.xls", sep=";", skiprows=3)



# Step 3: Identify visitor data columns corresponding to the zones (wide, mid, close)
zone_visitor_counts = {}

# Zone-wise mapping for wide, mid, and close columns
for zone_code in zone_poi_counts.keys():
    matching_columns = []
    
    # Looking for columns corresponding to each zone (wide, mid, close)
    for col in visitor_data.columns[1:]:  # Skip the first column which is the timestamp
        if zone_code in col:
            matching_columns.append(col)
    
    if matching_columns:
        # Sum the values for the matching columns
        visitor_sum = visitor_data[matching_columns].sum().sum()  # Sum across rows and columns
        zone_visitor_counts[zone_code] = visitor_sum
    else:
        print(f"No matching columns for {zone_code}. Skipping this zone.")

# Convert np.int64 to native Python int
zone_visitor_counts = {zone: int(visitor_count) for zone, visitor_count in zone_visitor_counts.items()}

print("Visitor Counts per Zone:", zone_visitor_counts)

# Step 4: Combine Data and Perform Correlation
combined_data = {zone: {"poi_count": zone_poi_counts[zone], "visitor_count": zone_visitor_counts.get(zone, 0)} for zone in zone_poi_counts}

print("\nCombined Data (Zone: POI Count, Visitor Count):")
for zone, data in combined_data.items():
    print(f"{zone}: {data}")

# Optional: Perform a correlation analysis (make sure there are non-NaN values)
poi_counts = [data["poi_count"] for data in combined_data.values()]
visitor_counts = [data["visitor_count"] for data in combined_data.values()]

# Remove NaN values for correlation calculation
valid_indices = [i for i, val in enumerate(visitor_counts) if not pd.isna(val) and visitor_counts[i] != 0]
poi_counts = [poi_counts[i] for i in valid_indices]
visitor_counts = [visitor_counts[i] for i in valid_indices]

if poi_counts and visitor_counts:
    correlation = pd.Series(poi_counts).corr(pd.Series(visitor_counts))
    print("\nCorrelation between POI counts and visitor counts:", correlation)
else:
    print("\nNot enough valid data for correlation analysis.")
