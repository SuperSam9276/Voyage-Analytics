import streamlit as st

from inference.flight_price import predict_flight_price
from inference.hotel_recommendation import (
    get_user_ids,
    get_user_history,
    recommend_for_user,
)
from inference.classification import predict_gender

st.set_page_config(page_title="Voyage Analytics", page_icon="✈️", layout="wide")
st.title("✈️ Voyage Analytics")

tab_flight, tab_hotel, tab_classify = st.tabs(
    ["Flight Price Prediction", "Hotel Recommendation", "User Classification"]
)

CITIES = [
    "Aracaju (SE)", "Brasilia (DF)", "Campo Grande (MS)", "Florianopolis (SC)",
    "Natal (RN)", "Recife (PE)", "Rio de Janeiro (RJ)", "Salvador (BH)", "Sao Paulo (SP)",
]
FLIGHT_TYPES = ["economic", "FirstClass", "Premium"]
AGENCIES = ["CloudFy", "FlyingDrops", "Rainbow"]

# ---------------------------------------------------------------- Flight tab
with tab_flight:
    st.subheader("Predict flight price")

    col1, col2 = st.columns(2)
    with col1:
        from_city = st.selectbox("Departure city", CITIES, key="from_city")
        flight_type = st.selectbox("Flight type", FLIGHT_TYPES)
        agency = st.selectbox("Agency", AGENCIES)
        time_val = st.number_input("Time (hours)", min_value=0.0, step=0.1)
        distance = st.number_input("Distance", min_value=0.0, step=1.0)
    with col2:
        to_city = st.selectbox("Destination city", CITIES, key="to_city")
        year = st.number_input("Year", min_value=2000, max_value=2100, value=2024, step=1)
        month = st.number_input("Month", min_value=1, max_value=12, value=1, step=1)
        day = st.number_input("Day", min_value=1, max_value=31, value=1, step=1)
        day_of_week = st.number_input(
            "Day of week (0=Sun ... 6=Sat)", min_value=0, max_value=6, value=0, step=1
        )

    if st.button("Predict price", type="primary"):
        if from_city == to_city:
            st.error("Departure and destination city must be different.")
        else:
            try:
                price = predict_flight_price({
                    "from": from_city,
                    "to": to_city,
                    "flightType": flight_type,
                    "time": time_val,
                    "distance": distance,
                    "agency": agency,
                    "year": year,
                    "month": month,
                    "day": day,
                    "day_of_week": day_of_week,
                })
                st.success(f"Predicted flight price: R$ {price:.2f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ---------------------------------------------------------------- Hotel tab
with tab_hotel:
    st.subheader("Personalized hotel recommendations")

    try:
        user_ids = get_user_ids()
        selected_user = st.selectbox("Select user", user_ids)

        user_history = get_user_history(selected_user)

        if not user_history.empty:
            st.markdown("**Travel history**")
            history_cols = [c for c in ["name", "place", "price", "days", "date"] if c in user_history.columns]
            st.dataframe(user_history[history_cols], use_container_width=True)

            top_n = st.slider("Number of recommendations", 1, 5, 5)
            recommendations = recommend_for_user(selected_user, top_n=top_n)

            st.markdown("**Recommended hotels**")
            if recommendations.empty:
                st.warning("No new hotels available for recommendation for this user.")
            else:
                display_cols = [c for c in ["name", "place", "price", "recommendation_score"] if c in recommendations.columns]
                st.dataframe(recommendations[display_cols], use_container_width=True)
        else:
            st.warning("No historical hotel data found for the selected user.")
    except FileNotFoundError as e:
        st.error(
            f"Model or data file not found: {e}\n\n"
            "Make sure recommendation_model.pkl is in app/models/ and "
            "hotels_clean.csv is in app/data/."
        )

# ------------------------------------------------------------ Classify tab
with tab_classify:
    st.subheader("User classification (predict gender from age + company)")

    age = st.number_input("Age", min_value=0, max_value=100, value=30, step=1)
    company = st.text_input("Company", placeholder="e.g. 4You")

    if st.button("Classify", type="primary"):
        if not company:
            st.error("Please enter a company.")
        else:
            try:
                result = predict_gender(age, company)
                st.success(f"Predicted: {result['prediction']}")
                st.write("Class probabilities:", result["probabilities"])
                if not result["company_seen_in_training"]:
                    st.info(
                        "Note: this company wasn't in the training data — "
                        "prediction relies only on age for this feature."
                    )
            except FileNotFoundError:
                st.error(
                    "classification_model.pkl not found in app/models/. "
                    "Run training/train_classification.py first to generate it."
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")
