# Load and combine air quality, weather, and fire data
from pathlib import Path

import pandas as pd


project_root = Path(__file__).resolve().parent.parent


def load_data():
    air = pd.read_csv(
        project_root / "data" / "processed" / "air_quality_clean.csv"
    )

    weather = pd.read_csv(
        project_root / "data" / "raw" / "weather.csv"
    )

    fire = pd.read_csv(
        project_root / "data" / "raw" / "fire_data.csv"
    )

    for df in [air, weather, fire]:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    data = (
        air
        .merge(weather, on="date", how="left")
        .merge(fire, on="date", how="left")
        .sort_values("date")
        .reset_index(drop=True)
    )

    return data

# Test the data pipeline when this file is run directly
if __name__ == "__main__":
    df = load_data()

    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())
    print("Date range:", df["date"].min(), "to", df["date"].max())
    print("\nMissing values:")
    print(df.isna().sum())