### Wetterdienst API - Explanation of Parameters

#### 1. DWD Observation API

URL:  
`https://wetterdienst.eobs.org/api/values?provider=dwd&network=observation&parameters=daily/kl&period=recent&station=01420&date=2025-01-01/2025-01-15`

- **provider**: Specifies the data provider. In this case, it's the German Weather Service (DWD).
- **network**: Refers to the type of weather network. `observation` refers to historical observational data.
- **parameters**: Specifies the weather parameter(s) to query. `daily/kl` refers to daily temperature data.
- **period**: Defines the time period for the data. `recent` will fetch the most recent available data.
- **station**: The station ID for the observation station. `01420` refers to a specific station.
- **date**: Specifies the date range for which data is requested. It is in the format `YYYY-MM-DD/YYYY-MM-DD`. For example, `2025-01-01/2025-01-15` represents the period from January 1 to January 15, 2025.

#### 2. DWD Observation API for Specific Parameters

URL:  
`https://wetterdienst.eobs.org/api/values?provider=dwd&network=observation&parameters=daily/kl/temperature_air_mean_2m&period=recent&station=01420&date=2025-01-01/2025-01-15`

- **parameters**: Specifies the specific parameter to query. In this case, `temperature_air_mean_2m` refers to the mean temperature at 2 meters above the ground.

---

### Tomorrow.io API - Historical Weather Data

URL:  
`https://api.tomorrow.io/v4/historical`

This API allows you to retrieve historical weather data for specific locations over a set time range. It provides data such as temperature, precipitation, and other weather metrics.

#### Request Parameters:

- **location**: Specifies the latitude and longitude of the location for which you want weather data. For example, `"42.3478, -71.0466"` represents a location near Boston, MA, USA.
- **fields**: Lists the data fields you want to retrieve. For example, `["temperature"]` retrieves temperature data.
- **timesteps**: Defines the time resolution of the data. For hourly data, use `["1h"]`.
- **startTime**: The start date and time for the data range in ISO 8601 format. For example, `"2019-03-20T14:09:50Z"`.
- **endTime**: The end date and time for the data range, also in ISO 8601 format. For example, `"2019-03-28T14:09:50Z"`.
- **units**: Specifies the units of measurement. Use `"metric"` for metric units (Celsius, meters, etc.).

This API sends a POST request to retrieve the data, and the response is returned in JSON format.


#### Tomorrow.io API Documentation
For more details about the parameters and API usage, refer to the official Tomorrow.io documentation:

URL:  
`https://docs.tomorrow.io/reference/retrieve-historical-timelines`

