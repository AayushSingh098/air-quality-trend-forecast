# Load the trained model and make a PM2.5 prediction
from pathlib import Path

import joblib
import pandas as pd


project_root = Path(__file__).resolve().parent.parent

model = joblib.load(
    project_root / "models" / "air_quality_model.pkl"
)

feature_columns = joblib.load(
    project_root / "models" / "feature_columns.pkl"
)


def predict_pm25(data):
    input_data = pd.DataFrame([data])
    input_data = input_data[feature_columns]

    prediction = model.predict(input_data)[0]

    return round(float(prediction), 2)