# Show the PM2.5 forecast dashboard
import pandas as pd
import streamlit as st

from utils import (
    calculate_aqi,
    get_latest_data,
    get_latest_prediction_data,
    get_prediction_explanation,
    get_fire_map_data,
    get_live_prediction,
    get_latest_pm25
)


st.set_page_config(
    page_title="Delhi Air Quality Forecast",
    page_icon="🌫️",
    layout="wide"
)

st.title("Delhi Air Quality Trend & Forecast")

st.write(
    "Forecast PM2.5 and understand the factors behind the prediction."
)

# Show the type of data used in the dashboard
st.caption(
    "Delhi • PM2.5 forecasting • Weather • Satellite fire activity"
)

# Check if live pollution data can be used
live_result = get_live_prediction()
latest_pm25 = get_latest_pm25()

if live_result["status"] == "live":
    live_prediction = live_result["prediction"]
    live_aqi, live_category = calculate_aqi(live_prediction)

    live_forecast_date = (
        pd.Timestamp.now(tz="Asia/Kolkata").normalize()
        + pd.Timedelta(days=1)
    )

    st.subheader("Live PM2.5 Forecast")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Predicted PM2.5",
        f"{live_prediction:.1f} µg/m³"
    )

    col2.metric(
        "Predicted AQI",
        live_aqi
    )

    col3.metric(
        "Air Quality",
        live_category
    )

    col4.metric(
        "Forecast Date",
        live_forecast_date.strftime("%d %b %Y")
    )

    st.success("Live pollution data is available.")

else:
    latest_time = live_result["latest_pm25_date"]

    st.warning(
        f"Latest available PM2.5 reading is from "
        f"{latest_time.strftime('%d %b %Y, %I:%M %p')}. "
        f"Live forecast is temporarily unavailable because "
        f"the pollution data is too old."
    )

    st.caption(
        f"Latest available PM2.5: "
        f"{latest_pm25['pm25']:.1f} µg/m³"
    )


# Load the latest historical data for model prediction
latest_row, prediction = get_latest_prediction_data()

# Convert predicted PM2.5 into AQI
predicted_aqi, aqi_category = calculate_aqi(prediction)

forecast_date = (
    pd.to_datetime(latest_row["date"]) + pd.Timedelta(days=1)
)

st.subheader("Historical Model Forecast")
st.caption(
    "Model demonstration using the latest date where all historical "
    "pollution, weather, and fire features are available."
)
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Predicted PM2.5",
    f"{prediction:.1f} µg/m³"
)

col2.metric(
    "Predicted AQI",
    predicted_aqi
)

col3.metric(
    "Air Quality",
    aqi_category
)

col4.metric(
    "Forecast Date",
    forecast_date.strftime("%d %b %Y")
)

# Show a simple message for the predicted air quality
if aqi_category in ["Poor", "Very Poor", "Severe"]:
    st.warning(
        f"Predicted air quality is {aqi_category} with an AQI of {predicted_aqi}."
    )
else:
    st.info(
        f"Predicted air quality is {aqi_category} with an AQI of {predicted_aqi}."
    )

# Show the main reasons behind the prediction
explanation = get_prediction_explanation(latest_row)

name_map = {
    "pm25": "Current PM2.5",
    "pm25_lag_1": "Previous PM2.5",
    "pm25_roll_7": "7-day PM2.5 average",
    "pm25_roll_3": "3-day PM2.5 average",
    "wind_speed": "Wind speed",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "rainfall": "Rainfall",
    "fire_count": "Regional fire activity",
    "total_frp": "Total fire intensity",
    "mean_frp": "Average fire intensity",
    "day_of_year": "Time of year"
}

st.subheader("Why is the model predicting this?")

# Show direction and strength of each feature
top_features = explanation.head(5).copy()
max_impact = top_features["impact"].abs().max()

for _, row in top_features.iterrows():
    feature_name = name_map.get(row["feature"], row["feature"])
    impact = row["impact"]

    relative_impact = abs(impact) / max_impact

    if relative_impact >= 0.67:
        strength = "Strong"
    elif relative_impact >= 0.33:
        strength = "Moderate"
    else:
        strength = "Small"

    if impact > 0:
        direction = "↑"
        effect = "upward influence"
    else:
        direction = "↓"
        effect = "downward influence"

    st.write(
        f"**{feature_name}** — {direction} {strength} {effect}"
    )

# Show PM2.5 levels from the last 7 available days
st.subheader("7-Day Pollution Trend")

df = get_latest_data()

forecast_source_date = pd.to_datetime(latest_row["date"])

trend_df = (
    df[
        (df["date"] <= forecast_source_date) &
        (df["pm25"].notna())
    ][["date", "pm25"]]
    .tail(7)
    .set_index("date")
)

st.line_chart(trend_df)

# Compare the prediction with the same month in past years
forecast_month = forecast_date.month

historical_month = df[
    (df["date"].dt.month == forecast_month) &
    (df["date"] < forecast_source_date)
]["pm25"].dropna()

historical_average = historical_month.mean()

st.subheader("Historical Comparison")

if prediction > historical_average:
    difference = prediction - historical_average

    st.write(
        f"The predicted PM2.5 is **{difference:.1f} µg/m³ higher** "
        f"than the historical average for "
        f"{forecast_date.strftime('%B')}."
    )
else:
    difference = historical_average - prediction

    st.write(
        f"The predicted PM2.5 is **{difference:.1f} µg/m³ lower** "
        f"than the historical average for "
        f"{forecast_date.strftime('%B')}."
    )

st.caption(
    f"Historical {forecast_date.strftime('%B')} average: "
    f"{historical_average:.1f} µg/m³"
)

# Show weather conditions used for the prediction
st.subheader("Historical Weather Conditions")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Temperature",
    f"{latest_row['temperature']:.1f} °C"
)

col2.metric(
    "Humidity",
    f"{latest_row['humidity']:.0f}%"
)

col3.metric(
    "Wind Speed",
    f"{latest_row['wind_speed']:.1f} km/h"
)

col4.metric(
    "Rainfall",
    f"{latest_row['rainfall']:.1f} mm"
)

# Show regional fire activity used for the prediction
st.subheader("Historical Regional Fire Activity")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Fire Detections",
    f"{latest_row['fire_count']:.0f}"
)

col2.metric(
    "Total Fire Intensity",
    f"{latest_row['total_frp']:.1f}"
)

col3.metric(
    "Average Fire Intensity",
    f"{latest_row['mean_frp']:.1f}"
)

st.caption(
    "Satellite fire activity is used as a regional signal "
    "in the PM2.5 model."
)

# Show recent satellite fire locations
fire_map_df = get_fire_map_data()

if not fire_map_df.empty:
    st.subheader("Recent Regional Fire Hotspots")

    st.map(
        fire_map_df,
        latitude="latitude",
        longitude="longitude",
        height=400
    )

    st.caption(
        f"{len(fire_map_df)} satellite fire detections "
        f"shown on the map."
    )
else:
    st.info("No recent fire detections found.")