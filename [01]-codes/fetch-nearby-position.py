import json
from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth specified in decimal degrees
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    # Radius of earth in meters
    r = 6371000
    return c * r


def is_poi_near_street(poi_lat, poi_lon, street_points, max_distance=250):
    """
    Check if a POI is within max_distance meters of any point on the street
    """
    for link in street_points:
        for point in link['points']:
            # Note: In your JSON, longitude is 'lng' not 'lon'
            distance = haversine_distance(
                poi_lat,
                poi_lon,
                point['lat'],
                point['lng']  # Changed from 'lng' to match your JSON
            )
            if distance <= max_distance:
                return True
    return False


def analyze_proximity(street_data, poi_data):
    """
    Analyze which POIs are near each street and create a new JSON
    """
    results = []

    # Process each street in the results
    for street_info in street_data.get('results', []):
        # Get street information - using .get() to avoid KeyError
        location = street_info.get('location', {})
        street_name = location.get('description', 'Willy-Lessing-Straße + Luitpoldstraße')
        street_points = location.get('shape', {}).get('links', [])

        # Initialize nearby POIs list for this street
        nearby_pois = []

        # Process all POI entries
        for poi_entry in poi_data:
            # Process each POI in the current entry
            for poi in poi_entry.get('pois', []):
                if is_poi_near_street(poi['lat'], poi['lon'], street_points):
                    nearby_pois.append({
                        'name': poi['name'],
                        'type': poi['type'],
                        'lat': poi['lat'],
                        'lon': poi['lon'],
                        'opening_hours': poi['opening_hours']
                    })

        # Add street and its nearby POIs to results if any were found
        if nearby_pois:
            results.append({
                'street_name': street_name,
                'nearby_pois': nearby_pois
            })

    return results


def main():
    try:
        # Read street data
        with open('2024-12-09-09-00-14.json', 'r', encoding='utf-8') as f:
            street_data = json.load(f)

        # Read POI data
        with open('poi_data.json', 'r', encoding='utf-8') as f:
            poi_data = json.load(f)

        # Analyze proximity
        results = analyze_proximity(street_data, poi_data)

        # Write result to file
        with open('nearby_pois_250m.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    except FileNotFoundError as e:
        print(f"Error: Could not find input file - {e}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file - {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()