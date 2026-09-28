# Train and save the final Random Forest model
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor

from data_pipeline import load_data
from features import create_features


project_root = Path(__file__).resolve().parent.parent

# Load data and create model features
df = load_data()
df = create_features(df)

feature_columns = [
    "temperature", "humidity", "rainfall", "wind_speed",
    "fire_count", "total_frp", "mean_frp",
    "month", "day_of_year",
    "pm25_lag_1", "pm25_lag_2", "pm25_lag_3", "pm25_lag_7",
    "pm25_roll_3", "pm25_roll_7", "pm25"
]

# Keep only rows that are ready for training
df = df.dropna(
    subset=feature_columns + ["target_pm25_next_day"]
).reset_index(drop=True)

X = df[feature_columns]
y = df["target_pm25_next_day"]

# Train the final model
model = RandomForestRegressor(
    n_estimators=500,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features=0.7,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

# Save the model and feature names
joblib.dump(
    model,
    project_root / "models" / "air_quality_model.pkl"
)

joblib.dump(
    feature_columns,
    project_root / "models" / "feature_columns.pkl"
)

print("Model training completed.")
print("Training rows:", len(df))
print("Model saved successfully.")