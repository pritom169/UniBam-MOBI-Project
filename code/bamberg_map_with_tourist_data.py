import folium
import pandas as pd

def display_bamberg_circle_map(locations, visitor_data, filename="bamberg_map_with_tourist_data.html"):
    """
    Display Bamberg zones with visitor data as dynamic circle radii and labeled markers.
    """
    # Create a base map centered on Bamberg
    bamberg_map = folium.Map(location=[49.8915326908064, 10.886880788328682], zoom_start=14, tiles="OpenStreetMap")

    # Add circles and markers for each location
    for location in locations:
        zone = location["zone"]
        name = location["name"]
        lat = location["lat"]
        lon = location["lon"]

        # Get visitor count for this zone
        visitors = visitor_data.get(zone, 0)

        # Define circle color based on visitor count
        if visitors > 1000:
            color = "red"
        elif visitors > 500:
            color = "orange"
        else:
            color = "green"

        # Add a dynamic radius circle based on visitor count
        folium.Circle(
            location=[lat, lon],
            radius=visitors * 0.0001,  # Scale radius (adjust multiplier as needed)
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6
        ).add_to(bamberg_map)

        # Add a marker with zone details
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(f"<b>{name}</b><br>Visitor Count: {visitors}", max_width=300),
            tooltip=f"{name} ({zone})",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(bamberg_map)

    # Save the map to an HTML file
    bamberg_map.save(filename)
    print(f"Map with visitor data saved to {filename}. Open it in your browser to view.")

def main():
    """
    Main function to generate the Bamberg map with visitor data.
    """
    # Bamberg locations with names, coordinates, and sensor IDs
    bamberg_locations = [
        {"zone": "bz2452", "name": "Gabelmann", "lat": 49.8915326908064, "lon": 10.886880788328682},
        {"zone": "bz2453", "name": "Sandstraße", "lat": 49.890902010804375, "lon": 10.882862587313937},
        {"zone": "bz2454", "name": "Mußstraße", "lat": 49.89377824021776, "lon": 10.877295540783985},
        {"zone": "bz2457", "name": "Domkranz", "lat": 49.89352197587767, "lon": 10.882945771405465},
        {"zone": "bz2458", "name": "Touristeninformation", "lat": 49.88863950847202, "lon": 10.885894010763526},
        {"zone": "bz2460", "name": "Old Rathaus", "lat": 49.88081879723072, "lon": 10.868877325685178},
        {"zone": "bz2464", "name": "Maxplatz", "lat": 49.8900000000000, "lon": 10.8870000000000},
        {"zone": "bz2463", "name": "New Rathaus West", "lat": 49.8890000000000, "lon": 10.8860000000000},
        {"zone": "bz2462", "name": "New Rathaus Ost", "lat": 49.8880000000000, "lon": 10.8850000000000},
    ]

    # Load visitor data (replace with your actual file path if needed)
    visitor_data_file = "D:/Bamberg University/winter24_25/project/UniBam-MOBI-Project/data/Visitor_Traffic_Data_by_Zone.csv"
    data = pd.read_csv(visitor_data_file)

    # Sum visitor counts for each zone
    visitor_counts = data.drop(columns=["timestamp"]).sum().to_dict()

    # Display Bamberg zones on an interactive map
    display_bamberg_circle_map(bamberg_locations, visitor_counts)

if __name__ == "__main__":
    main()
