# Helper functions for the dashboard
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import joblib
import shap
from src.live_data import get_live_prediction, get_latest_pm25

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data_pipeline import load_data
from src.features import create_features
from src.predict import predict_pm25


def get_latest_data():
    df = load_data()
    df = create_features(df)

    return df


# Get the latest row that has all model features
def get_latest_prediction_data():
    df = get_latest_data()

    feature_columns = [
        "temperature", "humidity", "rainfall", "wind_speed",
        "fire_count", "total_frp", "mean_frp",
        "month", "day_of_year",
        "pm25_lag_1", "pm25_lag_2", "pm25_lag_3", "pm25_lag_7",
        "pm25_roll_3", "pm25_roll_7", "pm25"
    ]

    ready_df = df.dropna(subset=feature_columns)

    latest_row = ready_df.iloc[-1]
    input_data = latest_row[feature_columns].to_dict()

    prediction = predict_pm25(input_data)

    return latest_row, prediction


# Explain which features increase or decrease the prediction
def get_prediction_explanation(latest_row):
    model = joblib.load(
        project_root / "models" / "air_quality_model.pkl"
    )

    feature_columns = joblib.load(
        project_root / "models" / "feature_columns.pkl"
    )

    X = pd.DataFrame(
        [latest_row[feature_columns].to_dict()]
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)[0]

    explanation = pd.DataFrame({
        "feature": feature_columns,
        "value": X.iloc[0].values,
        "impact": shap_values
    })

    explanation["strength"] = explanation["impact"].abs()

    return explanation.sort_values(
        "strength",
        ascending=False
    ).reset_index(drop=True)

# Convert PM2.5 into Indian AQI and category
def calculate_aqi(pm25):
    ranges = [
        (0, 30, 0, 50, "Good"),
        (31, 60, 51, 100, "Satisfactory"),
        (61, 90, 101, 200, "Moderate"),
        (91, 120, 201, 300, "Poor"),
        (121, 250, 301, 400, "Very Poor"),
        (251, 500, 401, 500, "Severe")
    ]

    pm25 = max(0, pm25)

    for low_pm, high_pm, low_aqi, high_aqi, category in ranges:
        if pm25 <= high_pm:
            aqi = (
                (high_aqi - low_aqi)
                / (high_pm - low_pm)
                * (pm25 - low_pm)
                + low_aqi
            )

            return round(aqi), category

    return 500, "Severe"

# Get recent fire locations for the map
def get_fire_map_data():
    import io
    import os
    import requests
    from dotenv import load_dotenv

    load_dotenv(project_root / ".env")

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
        return fire_df

    return fire_df[["latitude", "longitude", "frp"]]