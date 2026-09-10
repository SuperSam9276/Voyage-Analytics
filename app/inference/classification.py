"""
Gender classification from age + company.

Loads classification_model.pkl produced by training/train_classification.py.
NOTE: this .pkl does not exist yet in the original repo -- the original
Classification_pipeline.py never saved a model. Run train_classification.py
first (see that file's docstring) to generate it.
"""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


@st.cache_resource
def load_classification_package():
    package = joblib.load(MODEL_DIR / "classification_model.pkl")
    return package["pipeline"], package["label_encoder"], package["known_companies"]


def predict_gender(age: float, company: str) -> dict:
    pipeline, label_encoder, known_companies = load_classification_package()

    df = pd.DataFrame([{"age": float(age), "company": company}])

    predicted_class = pipeline.predict(df)[0]
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    probabilities = pipeline.predict_proba(df)[0]
    prob_by_label = {
        label_encoder.inverse_transform([i])[0]: float(p)
        for i, p in enumerate(probabilities)
    }

    return {
        "prediction": predicted_label,
        "probabilities": prob_by_label,
        "company_seen_in_training": company in known_companies,
    }
