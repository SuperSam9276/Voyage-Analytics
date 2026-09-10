"""
Smoke tests: confirm each model loads and produces a prediction without
crashing. Run with: pytest tests/test_inference.py
Run from the app/ directory (or add app/ to PYTHONPATH) so `inference` imports resolve.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from inference.flight_price import predict_flight_price
from inference.hotel_recommendation import get_user_ids, recommend_for_user
from inference.classification import predict_gender


def test_flight_price_prediction_runs():
    price = predict_flight_price({
        "from": "Sao Paulo (SP)",
        "to": "Rio de Janeiro (RJ)",
        "flightType": "economic",
        "time": 1.5,
        "distance": 400.0,
        "agency": "Rainbow",
        "year": 2024,
        "month": 6,
        "day": 15,
        "day_of_week": 3,
    })
    assert isinstance(price, float)
    assert price > 0


def test_hotel_recommendation_runs():
    user_ids = get_user_ids()
    assert len(user_ids) > 0
    recs = recommend_for_user(user_ids[0], top_n=3)
    assert recs is not None  # empty DataFrame is a valid "no new hotels" result


def test_classification_runs():
    result = predict_gender(age=30, company="4You")
    assert "prediction" in result
    assert "probabilities" in result
