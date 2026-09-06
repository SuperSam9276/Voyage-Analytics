
import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Voyage Analytics",
    page_icon="✈️",
    layout="wide"
)


@st.cache_resource
def load_package():

    return joblib.load(
        "recommendation_model.pkl"
    )


package = load_package()

recommendation_model = package[
    "recommendation_model"
]

hotel_feature_matrix = package[
    "hotel_feature_matrix"
]

hotel_catalog = package[
    "hotel_catalog"
]

hotels_clean = package[
    "hotels_clean"
]


def recommend_similar_hotels(
    hotel_name,
    top_n=5
):

    matching_indices = hotel_catalog.index[
        hotel_catalog["name"] == hotel_name
    ].tolist()

    if not matching_indices:
        return pd.DataFrame()

    hotel_index = matching_indices[0]

    n_neighbors = min(
        top_n + 1,
        len(hotel_catalog)
    )

    distances, indices = (
        recommendation_model.kneighbors(
            hotel_feature_matrix[hotel_index],
            n_neighbors=n_neighbors
        )
    )

    recommendations = hotel_catalog.iloc[
        indices[0][1:]
    ].copy()

    recommendations[
        "similarity_score"
    ] = 1 - distances[0][1:]

    return recommendations.reset_index(
        drop=True
    )


def recommend_for_user(
    user_code,
    top_n=5
):

    user_history = hotels_clean[
        hotels_clean["usercode"] == user_code
    ]

    if user_history.empty:
        return pd.DataFrame()

    previous_hotels = (
        user_history["name"]
        .dropna()
        .unique()
        .tolist()
    )

    candidate_scores = {}

    for hotel_name in previous_hotels:

        if hotel_name not in set(
            hotel_catalog["name"]
        ):
            continue

        n_to_search = min(
            top_n * 3,
            max(
                0,
                len(hotel_catalog) - 1
            )
        )

        if n_to_search == 0:
            continue

        similar_hotels = recommend_similar_hotels(
            hotel_name,
            top_n=n_to_search
        )

        for _, row in similar_hotels.iterrows():

            candidate = row["name"]
            score = row["similarity_score"]

            if candidate in previous_hotels:
                continue

            candidate_scores[candidate] = (
                candidate_scores.get(
                    candidate,
                    0
                ) + score
            )

    if not candidate_scores:
        return pd.DataFrame()

    recommendations = pd.DataFrame(
        candidate_scores.items(),
        columns=[
            "name",
            "recommendation_score"
        ]
    )

    recommendations = (
        recommendations
        .sort_values(
            "recommendation_score",
            ascending=False
        )
        .head(top_n)
    )

    recommendations = recommendations.merge(
        hotel_catalog,
        on="name",
        how="left"
    )

    return recommendations.reset_index(
        drop=True
    )


st.title("✈️ Voyage Analytics")

st.subheader(
    "Personalized Hotel Recommendation System"
)

st.write(
    "Select a user to view their travel history "
    "and receive personalized hotel recommendations."
)


user_ids = sorted(
    hotels_clean["usercode"]
    .dropna()
    .unique()
    .tolist()
)

selected_user = st.selectbox(
    "Select User",
    user_ids
)


user_history = hotels_clean[
    hotels_clean["usercode"] == selected_user
].copy()


if not user_history.empty:

    st.subheader("Travel History")

    history_columns = [
        "name",
        "place",
        "price",
        "days",
        "date"
    ]

    available_columns = [
        column
        for column in history_columns
        if column in user_history.columns
    ]

    st.dataframe(
        user_history[
            available_columns
        ],
        use_container_width=True
    )

    st.subheader(
        "Recommended Hotels"
    )

    top_n = st.slider(
        "Number of recommendations",
        min_value=1,
        max_value=5,
        value=5
    )

    recommendations = recommend_for_user(
        selected_user,
        top_n=top_n
    )

    if recommendations.empty:

        st.warning(
            "No new hotels are available for "
            "recommendation for this user."
        )

    else:

        display_columns = [
            "name",
            "place",
            "price",
            "recommendation_score"
        ]

        available_display_columns = [
            column
            for column in display_columns
            if column in recommendations.columns
        ]

        st.dataframe(
            recommendations[
                available_display_columns
            ],
            use_container_width=True
        )

else:

    st.warning(
        "No historical hotel data found for "
        "the selected user."
    )
