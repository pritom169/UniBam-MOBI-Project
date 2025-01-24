import folium

def display_bamberg_circle_map(locations, filename="bamberg_map_with_markers.html"):
    """
    Display Bamberg locations with 500-meter radius circles and labeled markers on an interactive map.
    """
    # Create a base map centered on Bamberg
    bamberg_map = folium.Map(location=[49.8915326908064, 10.886880788328682], zoom_start=14, tiles="OpenStreetMap")

    # Add circles and markers for each location
    for location in locations:
        name = location["name"]
        lat = location["lat"]
        lon = location["lon"]

        # Add a 200-meter radius circle
        folium.Circle(
            location=[lat, lon],
            radius=200,  # 200 meters
            color="blue",
            fill=True,
            fill_color="#3f93e8",
            fill_opacity=0.5
        ).add_to(bamberg_map)

        # Add a marker with a label for the location
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(f"<b>{name}</b>", max_width=200),
            tooltip=name,
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(bamberg_map)

    bamberg_map.save(filename)
    print(f"Map with markers saved to {filename}. Open it in your browser to view.")

def main():
    """
    Main function to generate the Bamberg map with circles and markers.
    """
    # Bamberg locations with names, coordinates, and sensor IDs
    bamberg_locations = [
        {"name": "Gabelmann (bz2452)", "lat": 49.8915326908064, "lon": 10.886880788328682},
        {"name": "Sandstraße (bz2453)", "lat": 49.890902010804375, "lon": 10.882862587313937},
        {"name": "Mußstraße (bz2454)", "lat": 49.89377824021776, "lon": 10.877295540783985},
        {"name": "Domkranz (bz2457)", "lat": 49.89352197587767, "lon": 10.882945771405465},
        {"name": "Touristeninformation (bz2458)", "lat": 49.88863950847202, "lon": 10.885894010763526},
        {"name": "Old Rathaus (bz2460)", "lat": 49.88081879723072, "lon": 10.868877325685178},
        {"name": "Maxplatz (bz2464)", "lat": 49.8900000000000, "lon": 10.8870000000000},
        {"name": "New Rathaus West (bz2463)", "lat": 49.8890000000000, "lon": 10.8860000000000},
        {"name": "New Rathaus Ost (bz2462)", "lat": 49.8880000000000, "lon": 10.8850000000000},
    ]

    # Display Bamberg locations on an interactive map
    display_bamberg_circle_map(bamberg_locations)

if __name__ == "__main__":
    main()
