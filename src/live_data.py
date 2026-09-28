# Get live data for the air quality forecast
import os
from pathlib import Path

import joblib
import requests
from dotenv import load_dotenv
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env")

OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")
BASE_URL = "https://api.openaq.org/v3"

headers = {
    "X-API-Key": OPENAQ_API_KEY
}


# Get the latest PM2.5 value from OpenAQ
def get_latest_pm25():
    location_id = 8118
    sensor_id = 23534

    response = requests.get(
        f"{BASE_URL}/locations/{location_id}/latest",
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    results = response.json()["results"]

    pm25_data = [
        row for row in results
        if row["sensorsId"] == sensor_id
    ]

    if not pm25_data:
        raise ValueError("No recent PM2.5 data found.")

    latest = pm25_data[0]

    return {
        "pm25": latest["value"],
        "date": latest["datetime"]["local"]
    }

# Get today's weather and tomorrow's weather for Delhi
def get_weather():
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": 28.6139,
        "longitude": 77.2090,
        "daily": [
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_mean"
        ],
        "timezone": "Asia/Kolkata",
        "forecast_days": 2
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    daily = response.json()["daily"]

    return {
        "today": {
            "date": daily["time"][0],
            "temperature": daily["temperature_2m_mean"][0],
            "humidity": daily["relative_humidity_2m_mean"][0],
            "rainfall": daily["precipitation_sum"][0],
            "wind_speed": daily["wind_speed_10m_mean"][0]
        },
        "tomorrow": {
            "date": daily["time"][1],
            "temperature": daily["temperature_2m_mean"][1],
            "humidity": daily["relative_humidity_2m_mean"][1],
            "rainfall": daily["precipitation_sum"][1],
            "wind_speed": daily["wind_speed_10m_mean"][1]
        }
    }

# Get recent regional fire activity from NASA FIRMS
def get_recent_fire_activity():
    import io
    import pandas as pd

    nasa_key = os.getenv("NASA_FIRMS_MAP_KEY")
    fire_area = "73,26,80,32"

    url = (
        "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{nasa_key}/VIIRS_NOAA20_NRT/{fire_area}/1"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    fire_df = pd.read_csv(io.StringIO(response.text))

    if fire_df.empty:
        return {
            "fire_count": 0,
            "total_frp": 0.0,
            "mean_frp": 0.0
        }

    return {
        "fire_count": len(fire_df),
        "total_frp": fire_df["frp"].sum(),
        "mean_frp": fire_df["frp"].mean()
    }

# Build the features needed for tomorrow's prediction
def build_live_features():
    pm25_data = get_latest_pm25()
    weather = get_weather()
    fire = get_recent_fire_activity()

    history_path = project_root / "data" / "processed" / "model_data.csv"
    history = pd.read_csv(history_path)

    history["date"] = pd.to_datetime(history["date"])
    history = history.dropna(subset=["pm25"]).sort_values("date")

    latest_date = pd.to_datetime(pm25_data["date"]).tz_localize(None)
    current_pm25 = pm25_data["pm25"]

    recent_pm25 = history[
        history["date"] < latest_date.normalize()
    ]["pm25"].tail(7).tolist()

    recent_pm25.append(current_pm25)

    forecast_date = pd.to_datetime(weather["tomorrow"]["date"])

    features = {
        "pm25": current_pm25,
        "temperature": weather["tomorrow"]["temperature"],
        "humidity": weather["tomorrow"]["humidity"],
        "rainfall": weather["tomorrow"]["rainfall"],
        "wind_speed": weather["tomorrow"]["wind_speed"],
        "fire_count": fire["fire_count"],
        "total_frp": float(fire["total_frp"]),
        "mean_frp": float(fire["mean_frp"]),
        "month": forecast_date.month,
        "day_of_year": forecast_date.dayofyear,
        "pm25_lag_1": recent_pm25[-2],
        "pm25_lag_2": recent_pm25[-3],
        "pm25_lag_3": recent_pm25[-4],
        "pm25_lag_7": recent_pm25[-8],
        "pm25_roll_3": sum(recent_pm25[-3:]) / 3,
        "pm25_roll_7": sum(recent_pm25[-7:]) / 7
    }

    return pd.DataFrame([features])

# Check how old the latest PM2.5 reading is
def check_pm25_freshness():
    pm25_data = get_latest_pm25()

    latest_date = pd.to_datetime(pm25_data["date"]).tz_localize(None)
    today = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)

    age_hours = (today - latest_date).total_seconds() / 3600

    return {
        "latest_date": latest_date,
        "age_hours": age_hours,
        "is_fresh": age_hours <= 48
    }

# Make tomorrow's prediction when the PM2.5 data is fresh
def get_live_prediction():
    freshness = check_pm25_freshness()

    if not freshness["is_fresh"]:
        return {
            "prediction": None,
            "status": "stale",
            "latest_pm25_date": freshness["latest_date"],
            "age_hours": freshness["age_hours"]
        }

    features = build_live_features()

    model_path = project_root / "models" / "air_quality_model.pkl"
    model = joblib.load(model_path)

    prediction = model.predict(features)[0]

    return {
        "prediction": float(prediction),
        "status": "live",
        "latest_pm25_date": freshness["latest_date"],
        "age_hours": freshness["age_hours"]
    }

if __name__ == "__main__":
    result = get_live_prediction()

    print("Status:", result["status"])
    print("Latest PM2.5 date:", result["latest_pm25_date"])
    print("Data age:", round(result["age_hours"], 1), "hours")

    if result["prediction"] is not None:
        print("Tomorrow's predicted PM2.5:", round(result["prediction"], 1))
    else:
        print("Live forecast not shown because PM2.5 data is too old.")