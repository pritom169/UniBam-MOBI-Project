import os
import requests
import folium
from folium.plugins import MarkerCluster

# Define the path for saving the HTML map file
current_folder = os.path.dirname(os.path.abspath(__file__))  # Folder containing this script
parent_folder = os.path.dirname(current_folder)             # Parent folder (Point_of_Interest)
html_output_folder = os.path.join(parent_folder, "html_outputs")  # Sibling 'html_outputs' folder

# Ensure the output folder exists
if not os.path.exists(html_output_folder):
    os.makedirs(html_output_folder)

"""
The 'get_coordinates' function makes a request to the OpenStreetMap Nominatim API
to get the latitude and longitude of a given address.
It returns the coordinates if found, or None if not.
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
            print("Address not found!")
            return None, None
    except requests.RequestException as e:
        print(f"Error fetching coordinates for address: {e}")
        return None, None

"""
The 'fetch_pois' function uses the Overpass API to search for Points of Interest (POIs)
within a given radius around the latitude and longitude provided.
It returns a list of POIs found within the specified radius.
"""
def fetch_pois(lat, lon, poi_type, radius=300):
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
        print(f"Error fetching POIs: {e}")
        return []

"""
The 'get_poi_icon_and_color' function returns the appropriate icon and color 
for a given type of POI (e.g., cafe, restaurant, park).
It uses a predefined dictionary to map POI types to corresponding icon and color values.
"""
def get_poi_icon_and_color(poi_type):
    poi_styles = {
        "cafe": ("coffee", "orange"),
        "restaurant": ("cutlery", "red"),
        "bakery": ("birthday-cake", "brown"),
        "fuel": ("gas-pump", "blue"),
        "parking": ("parking", "darkblue"),
        "park": ("tree", "green"),
        "food": ("apple-alt", "pink"),
        "museum": ("university", "purple"),
        "pharmacy": ("medkit", "lightgreen"),
        "atm": ("money", "darkgreen"),
        "motorcycle_parking": ("motorcycle", "gray"),
        "charging_station": ("bolt", "yellow"),
        "bicycle_parking": ("bicycle", "cyan"),
        "university": ("graduation-cap", "darkred"),
        "school": ("school", "lightblue"),
        "library": ("book", "teal"),
        "clinic": ("hospital", "lightcoral")
    }
    return poi_styles.get(poi_type, ("info-sign", "gray"))

"""
The 'display_pois_on_map' function adds markers to the map for each POI found.
Each marker includes a popup with details like the POI's name and opening hours.
It groups the markers by POI type using a MarkerCluster for better map organization.
"""
def display_pois_on_map(lat, lon, pois, poi_type, marker_cluster):
    icon, color = get_poi_icon_and_color(poi_type)

    for poi in pois:
        poi_lat = poi["lat"]
        poi_lon = poi["lon"]
        poi_name = poi.get("tags", {}).get("name", "Unnamed POI")
        opening_hours = poi.get("tags", {}).get("opening_hours", "Not available")

        popup_content = f"""
        <b>{poi_name}</b><br>
        Type: {poi_type.capitalize()}<br>
        Opening Hours: {opening_hours}<br>
        """

        folium.Marker(
            [poi_lat, poi_lon],
            popup=folium.Popup(popup_content, max_width=300),
            tooltip=poi_name,
            icon=folium.Icon(color=color, icon=icon, prefix="fa")
        ).add_to(marker_cluster)

"""
The 'create_map_with_grouped_pois' function creates a map centered around a given address.
It fetches POIs for different groups (e.g., Food & Drinks, Essential Services)
and adds them to the map in clustered groups. The map is saved to an HTML file.
"""
def create_map_with_grouped_pois(address, grouped_pois):
    lat, lon = get_coordinates(address)
    if lat is None or lon is None:
        return

    print(f"Coordinates of {address}: Latitude={lat}, Longitude={lon}")

    poi_map = folium.Map(
        location=[lat, lon],
        zoom_start=16,
        tiles="OpenStreetMap",
        attr='Map data © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>'
    )

    folium.Marker(
        [lat, lon],
        popup=f"<b>{address}</b>",
        tooltip="Address Location",
        icon=folium.Icon(color="red", icon="home", prefix="fa")
    ).add_to(poi_map)

    for group_name, poi_types in grouped_pois.items():
        print(f"Processing group: {group_name}")
        group_cluster = MarkerCluster(name=group_name).add_to(poi_map)

        for poi_type in poi_types:
            print(f"  Fetching {poi_type}s near the address...")
            pois = fetch_pois(lat, lon, poi_type)
            print(f"    Found {len(pois)} {poi_type}(s).")
            if pois:
                display_pois_on_map(lat, lon, pois, poi_type, group_cluster)

    # Use the html_output_folder path for saving the map
    map_file_path = os.path.join(html_output_folder, "poi_Ob_Brücke.html")
    poi_map.save(map_file_path)
    print(f"Map saved to {map_file_path}. Open it in your browser to view.")

def main():
    address = "Ob. Brücke, Bamberg, Bavaria"
    grouped_pois = {
        "Food & Drinks": ["cafe", "restaurant", "bakery", "food"],
        "Essential Services": ["atm", "pharmacy", "fuel", "clinic"],
        "Leisure & Recreation": ["park", "museum"],
        "Parking & Transport": ["parking", "motorcycle_parking", "bicycle_parking", "charging_station"],
        "Education & Knowledge": ["university", "school", "library"]
    }

    create_map_with_grouped_pois(address, grouped_pois)

if __name__ == "__main__":
    main()
