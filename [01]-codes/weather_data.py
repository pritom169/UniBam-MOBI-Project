import requests
import csv
from datetime import datetime, timedelta

# Location: Bamberg, Germany
LATITUDE = 49.8917
LONGITUDE = 10.8858

# Define Date Range
START_DATE = "2023-07-10"
END_DATE = "2023-08-20"

# Generate dates
start_date = datetime.strptime(START_DATE, "%Y-%m-%d")
end_date = datetime.strptime(END_DATE, "%Y-%m-%d")

# Open-Meteo API URL
API_URL = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
           "&start_date={start}&end_date={end}&daily=temperature_2m_max,temperature_2m_min,"
           "precipitation_sum,wind_speed_10m_max&timezone=Europe/Berlin")

api_url = API_URL.format(lat=LATITUDE, lon=LONGITUDE, start=START_DATE, end=END_DATE)

# Fetch Data
response = requests.get(api_url)
if response.status_code == 200:
    data = response.json()
    daily_data = data.get("daily", {})

    if daily_data:
        csv_filename = f"bamberg_weather.csv"
        with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Date", "Max Temperature (°C)", "Min Temperature (°C)", "Average Temperature (°C)",
                             "Max Wind Speed (m/s)"])

            for i, date in enumerate(daily_data["time"]):
                max_temp = round(daily_data["temperature_2m_max"][i], 2)
                min_temp = round(daily_data["temperature_2m_min"][i], 2)
                avg_temp = round((max_temp + min_temp) / 2,
                                 2) if max_temp is not None and min_temp is not None else "N/A"
                wind_speed = round(daily_data["wind_speed_10m_max"][i], 2)

                writer.writerow([
                    date,
                    max_temp,
                    min_temp,
                    avg_temp,
                    wind_speed
                ])

        print(f"Weather data saved to {csv_filename}")
    else:
        print("No historical weather data available.")
else:
    print(f"Failed to fetch data. Status Code: {response.status_code}")
