import requests
import json
import os

# Define paths for saving data files
current_folder = os.path.dirname(os.path.abspath(__file__))  # Folder containing this script
parent_folder = os.path.dirname(current_folder)             # Parent folder (Point_of_Interest)
json_output_folder = os.path.join(parent_folder, "json_outputs")  # Sibling 'json_outputs' folder

# Ensure the output folder exists
if not os.path.exists(json_output_folder):
    os.makedirs(json_output_folder)

"""
The 'get_coordinates' function makes a request to the OpenStreetMap Nominatim API
to get the latitude and longitude of a given address. It returns the coordinates if found,
or None if the address cannot be found.
"""
def get_coordinates(address):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": address,
        "format": "json",
        "addressdetails": 1,
        "limit": 1
    }
    headers = {
        "User-Agent": "POIFetcherApp/1.0 (mobi@app.com)"
    }
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
        else:
            print(f"Address not found: {address}")
            return None, None
    except requests.RequestException as e:
        print(f"Error fetching coordinates for address '{address}': {e}")
        return None, None

"""
The 'fetch_pois' function uses the Overpass API to search for Points of Interest (POIs)
within a given radius around the latitude and longitude provided. It returns a list of POIs.
"""
def fetch_pois(lat, lon, poi_type, radius=200):
    url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    node["amenity"="{poi_type}"](around:{radius},{lat},{lon});
    out;
    """
    try:
        response = requests.get(url, params={"data": query})
        response.raise_for_status()
        data = response.json()
        return data.get("elements", [])
    except requests.RequestException as e:
        print(f"Error fetching POIs for coordinates ({lat}, {lon}): {e}")
        return []

"""
The 'save_poi_data_to_json' function collects POI data for a list of addresses and stores the results
in a JSON file. Each address's data includes coordinates and a list of POIs found within the specified radius.
"""
def save_poi_data_to_json(addresses, pois_to_search, filename="poi_data.json"):
    all_data = []
    for address in addresses:
        lat, lon = get_coordinates(address)
        if lat is None or lon is None:
            continue
        address_data = {"address": address, "coordinates": {"lat": lat, "lon": lon}, "pois": []}
        print(f"Fetching POIs for: {address}")
        for poi_type in pois_to_search:
            pois = fetch_pois(lat, lon, poi_type)
            for poi in pois:
                poi_info = {
                    "name": poi.get("tags", {}).get("name", "Unnamed POI"),
                    "type": poi_type,
                    "lat": poi["lat"],
                    "lon": poi["lon"],
                    "opening_hours": poi.get("tags", {}).get("opening_hours", "Not available"),
                }
                address_data["pois"].append(poi_info)
        all_data.append(address_data)
    
    # Save the data in the JSON folder
    json_file_path = os.path.join(json_output_folder, filename)
    with open(json_file_path, "w") as f:
        json.dump(all_data, f, indent=4)
    print(f"POI data saved to {json_file_path}")
    return all_data

def main():
    """
    Main function to define addresses and POI types, then save the collected POI data to a JSON file.
    """
    addresses = ["Ob. Brücke", "Ludwig-Donau-Main-Kanal", "Mühlwörth", "Am Kranen", "Herrenstraße", 
                 "Am Leinritt", "Spinnerei", "Untere Mühlbrücke", "Dompl", "Altenburg Castle", "Maximilianspl", 
                 "Michelsberg", "Ob. Mühlbrücke", "Schiffbauplatz", "Obere Sandstr", "Gertrauden Straße", 
                 "Heiliggrabstraße", "Benediktinerweg", "Geyerswörthstraße", "Michaelsberg", "Judenstraße", 
                 "Schönleinsplatz", "Centurione I", "Obere Sandstraße", "Hans-Schütz-Straße", "Kunigundendamm",
                 "Schillerplatz", "Theuerstadt", "Pfahlplaetzchen", "Gruener Markt", "Hainstraße", "Heumarkt",
                 "St.-Getreu-Strasse"]
    
    pois_to_search = ["cafe", "restaurant", "bakery", "fuel", "parking", "park", "food", "museum", "pharmacy", "atm",
                      "motorcycle_parking", "charging_station", "bicycle_parking", "university", "school",
                      "library", "clinic"]

    # Save POI data to JSON
    save_poi_data_to_json(addresses, pois_to_search)

if __name__ == "__main__":
    main()
