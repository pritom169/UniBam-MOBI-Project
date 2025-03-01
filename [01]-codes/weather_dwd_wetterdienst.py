import csv
import requests
from collections import defaultdict
import os

# Function to convert temperature from Kelvin to other units
def convert_temperature(value, to_unit="Celsius"):
    """
    Convert temperature from Kelvin to Celsius or Fahrenheit.
    Default is Celsius.

    Args:
        value (float): The temperature in Kelvin.
        to_unit (str): The unit to convert to. Can be "Celsius" or "Fahrenheit".
    
    Returns:
        str: The temperature in the specified unit, formatted to 2 decimal places.
    """
    if to_unit == "Celsius":
        return f"{value - 273.15:.2f}"
    elif to_unit == "Fahrenheit":
        return f"{(value - 273.15) * 9/5 + 32:.2f}"
    else:
        return f"{value:.2f}"

# Function to convert other weather parameters into readable formats
def convert_parameter(parameter, value):
    """
    Convert various weather parameters into human-readable formats.

    Args:
        parameter (str): The name of the weather parameter.
        value (float): The value of the weather parameter.

    Returns:
        str: A human-readable representation of the parameter value.
    """
    if parameter.startswith("temperature_air"):
        return convert_temperature(value, to_unit="Celsius")
    elif parameter == "cloud_cover_total":
        if value <= 20:
            return "Clear"
        elif value <= 50:
            return "Partly Cloudy"
        elif value <= 84:
            return "Mostly Cloudy"
        else:
            return "Overcast"
    elif parameter == "humidity":
        return f"{value:.2f}"
    elif parameter == "precipitation_form":
        form_mapping = {
            0: "No Precipitation", 
            1: "Rain", 
            2: "Drizzle", 
            3: "Snow", 
            6: "Freezing Rain", 
            7: "Hail"
        }
        return form_mapping.get(value, "Unknown")
    elif parameter == "precipitation_height":
        return "0" if value == 0 else f"{value:.2f}"
    elif parameter == "pressure_air_site":
        return f"{value / 100:.1f}"
    else:
        return value

# Grouping of weather parameters for easier processing
groupings = {
    "temperature": [
        "temperature_air_max_2m",
        "temperature_air_mean_2m",
        "temperature_air_min_2m",
        "temperature_air_min_0_05m",
    ],
    "cloud_cover": ["cloud_cover_total"],
    "humidity": ["humidity"],
    "precipitation": ["precipitation_form", "precipitation_height"],
    "pressure": ["pressure_air_site"],
}

# Function to fetch weather data from the API
def fetch_weather_data():
    """
    Fetch weather data from the wetterdienst API.

    Returns:
        dict: JSON response containing weather data, or an empty list if error.
    """
    api_url = "https://wetterdienst.eobs.org/api/values"
    params = {
        "provider": "dwd",
        "network": "observation",
        "parameters": "daily/kl",  
        "period": "recent",
        "station": "00282",               # Station Id for Bamberg
        "date": "2024-12-01/2024-12-31"   # Date range for the data
    }
    response = requests.get(api_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data: {response.status_code} - {response.text}")
        return {"values": []}

def main():
    """
    Main function that fetches weather data, processes it, and saves it as CSV files.
    """
    # Fetching weather data
    data = fetch_weather_data()
    grouped_data = defaultdict(lambda: defaultdict(list))

    # Processing fetched data and grouping by parameter
    for entry in data.get("values", []):
        date = entry["date"].split("T")[0]  # Extracting the date part
        for group_name, parameters in groupings.items():
            if entry["parameter"] in parameters:
                transformed_value = convert_parameter(entry["parameter"], entry["value"])
                grouped_data[group_name]["station_id"].append(entry["station_id"])
                grouped_data[group_name]["date"].append(date)
                grouped_data[group_name][entry["parameter"]].append(transformed_value)
                break

    # Ensure there are no duplicated dates and all rows are aligned correctly
    all_dates = sorted(set(grouped_data["temperature"]["date"]))  # Get unique sorted dates

    for group_name, data_dict in grouped_data.items():
        max_rows = len(all_dates)
        # Align the rows based on dates
        data_dict["date"] = all_dates  # Use the sorted unique dates for all groups

        for param in data_dict.keys():
            if param not in ["station_id", "date"]:
                # Extend with None if data for specific parameter is missing
                while len(data_dict[param]) < max_rows:
                    data_dict[param].append(None)

    # Writing the grouped data to CSV files in the csv_outputs folder
    for group_name, data_dict in grouped_data.items():
        # Creating the path for csv file in the csv_outputs folder
        output_folder = os.path.join(os.path.dirname(__file__), "../[02]-source-files/csv_outputs")
        os.makedirs(output_folder, exist_ok=True)
        file_name = os.path.join(output_folder, f"{group_name}_data.csv")
        
        # Writing to CSV
        with open(file_name, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            headers = ["Date", "Station ID"] + [key for key in data_dict.keys() if key not in ["station_id", "date"]]
            writer.writerow(headers)
            for i in range(max_rows):
                row = [
                    data_dict["date"][i],
                    data_dict["station_id"][i] if i < len(data_dict["station_id"]) else None,
                ]
                row.extend(data_dict.get(parameter, [None])[i] for parameter in headers[2:])
                writer.writerow(row)
        print(f"CSV file '{file_name}' created successfully!")

# Running the main function when script is executed
if __name__ == "__main__":
    main()
