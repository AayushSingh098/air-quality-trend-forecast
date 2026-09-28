# Create the features needed by the prediction model
import pandas as pd


def create_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Create date features
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    # Create previous PM2.5 features
    df["pm25_lag_1"] = df["pm25"].shift(1)
    df["pm25_lag_2"] = df["pm25"].shift(2)
    df["pm25_lag_3"] = df["pm25"].shift(3)
    df["pm25_lag_7"] = df["pm25"].shift(7)

    # Create recent PM2.5 averages
    previous_pm25 = df["pm25"].shift(1)
    df["pm25_roll_3"] = previous_pm25.rolling(3).mean()
    df["pm25_roll_7"] = previous_pm25.rolling(7).mean()

    # Create tomorrow's PM2.5 target
    df["target_pm25_next_day"] = df["pm25"].shift(-1)

    return df

# Test feature creation when this file is run directly
if __name__ == "__main__":
    from data_pipeline import load_data

    df = load_data()
    df = create_features(df)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print(
        df[
            [
                "date",
                "pm25",
                "pm25_lag_1",
                "pm25_roll_7",
                "target_pm25_next_day"
            ]
        ].tail()
    )