"""
Hotel recommendation.

IMPORTANT FIX vs the original hotel_recommendation_app.py:
recommendation_model.pkl only contains {recommendation_model, hotel_feature_matrix,
hotel_catalog} (see hotel_recommendation.ipynb, cell that builds
`recommendation_package`). It does NOT contain "hotels_clean" -- that's saved
separately as streamlit_data/hotels_clean.csv. The original app tried to read
package["hotels_clean"] and would KeyError on load. This module loads the CSV
alongside the pkl instead, matching what the notebook actually produces.
"""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@st.cache_resource
def load_recommendation_package():
    package = joblib.load(MODEL_DIR / "recommendation_model.pkl")
    return (
        package["recommendation_model"],
        package["hotel_feature_matrix"],
        package["hotel_catalog"],
    )


@st.cache_data
def load_hotels_clean():
    return pd.read_csv(DATA_DIR / "hotels_clean.csv")


def get_user_ids() -> list:
    hotels_clean = load_hotels_clean()
    return sorted(hotels_clean["usercode"].dropna().unique().tolist())


def get_user_history(user_code) -> pd.DataFrame:
    hotels_clean = load_hotels_clean()
    return hotels_clean[hotels_clean["usercode"] == user_code].copy()


def recommend_similar_hotels(hotel_name: str, top_n: int = 5) -> pd.DataFrame:
    recommendation_model, hotel_feature_matrix, hotel_catalog = load_recommendation_package()

    matching_indices = hotel_catalog.index[hotel_catalog["name"] == hotel_name].tolist()
    if not matching_indices:
        return pd.DataFrame()

    hotel_index = matching_indices[0]
    n_neighbors = min(top_n + 1, len(hotel_catalog))

    distances, indices = recommendation_model.kneighbors(
        hotel_feature_matrix[hotel_index], n_neighbors=n_neighbors
    )

    recommendations = hotel_catalog.iloc[indices[0][1:]].copy()
    recommendations["similarity_score"] = 1 - distances[0][1:]
    return recommendations.reset_index(drop=True)


def recommend_for_user(user_code, top_n: int = 5) -> pd.DataFrame:
    _, _, hotel_catalog = load_recommendation_package()
    hotels_clean = load_hotels_clean()

    user_history = hotels_clean[hotels_clean["usercode"] == user_code]
    if user_history.empty:
        return pd.DataFrame()

    previous_hotels = user_history["name"].dropna().unique().tolist()
    candidate_scores: dict = {}

    for hotel_name in previous_hotels:
        if hotel_name not in set(hotel_catalog["name"]):
            continue

        n_to_search = min(top_n * 3, max(0, len(hotel_catalog) - 1))
        if n_to_search == 0:
            continue

        similar_hotels = recommend_similar_hotels(hotel_name, top_n=n_to_search)

        for _, row in similar_hotels.iterrows():
            candidate = row["name"]
            score = row["similarity_score"]
            if candidate in previous_hotels:
                continue
            candidate_scores[candidate] = candidate_scores.get(candidate, 0) + score

    if not candidate_scores:
        return pd.DataFrame()

    recommendations = pd.DataFrame(
        candidate_scores.items(), columns=["name", "recommendation_score"]
    )
    recommendations = recommendations.sort_values(
        "recommendation_score", ascending=False
    ).head(top_n)
    recommendations = recommendations.merge(hotel_catalog, on="name", how="left")
    return recommendations.reset_index(drop=True)
