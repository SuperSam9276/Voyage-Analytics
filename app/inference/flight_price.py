"""
Flight price prediction.

Loads flight_price_model.pkl + flight_price_preprocessor.pkl (trained in
notebooks/flight_regression.ipynb) and exposes a single predict function.

Expected raw input columns (must match training exactly):
    from, to, flightType, time, distance, agency, year, month, day, day_of_week
"""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

REQUIRED_COLUMNS = [
    "from", "to", "flightType", "time", "distance",
    "agency", "year", "month", "day", "day_of_week",
]


@st.cache_resource
def load_flight_price_artifacts():
    model = joblib.load(MODEL_DIR / "flight_price_model.pkl")
    preprocessor = joblib.load(MODEL_DIR / "flight_price_preprocessor.pkl")
    return model, preprocessor


def predict_flight_price(input_data: dict) -> float:
    """
    input_data must contain all REQUIRED_COLUMNS.
    Raises ValueError if any are missing.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in input_data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    model, preprocessor = load_flight_price_artifacts()

    df = pd.DataFrame([input_data])[REQUIRED_COLUMNS]

    # enforce numeric types the way app.py did
    for col in ["time", "distance"]:
        df[col] = df[col].astype(float)
    for col in ["year", "month", "day", "day_of_week"]:
        df[col] = df[col].astype(int)

    X = preprocessor.transform(df)
    prediction = model.predict(X)
    return float(prediction[0])
