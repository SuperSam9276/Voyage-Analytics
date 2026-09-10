# Voyage Analytics

A unified analytics platform for the travel industry, serving three trained
ML models through a single Streamlit application:

- **Flight Price Prediction** — regression model estimating flight prices
  from route, timing, agency, and flight-class inputs.
- **Hotel Recommendation** — KNN-based recommender suggesting hotels similar
  to a user's past bookings.
- **User Classification** — Random Forest classifier predicting gender from
  age and company, used for user segmentation/analytics.

Built as a team project productionizing exploratory notebooks into a single
deployable app, containerized with Docker.

---

## Project structure

```
Voyage-Analytics/
├── app/
│   ├── streamlit_app.py              # main entrypoint — 3 tabs, one per model
│   ├── models/                       # trained model artifacts (tracked via Git LFS)
│   │   ├── flight_price_model.pkl
│   │   ├── flight_price_preprocessor.pkl
│   │   ├── recommendation_model.pkl
│   │   └── classification_model.pkl
│   ├── data/
│   │   └── hotels_clean.csv          # runtime lookup: user booking history
│   └── inference/
│       ├── flight_price.py
│       ├── hotel_recommendation.py
│       └── classification.py
│
├── notebooks/                        # training / EDA notebooks (source of the models)
│   ├── flight_regression.ipynb
│   └── hotel_recommendation.ipynb
│
├── training/
│   ├── data_cleaning.py              # standalone cleaning step (optional)
│   ├── eda.py                        # exploratory only, not used at deploy time
│   └── train_classification.py       # trains + saves classification_model.pkl
│
├── data/
│   └── users.csv                     # raw training data for the classification model
│
├── tests/
│   └── test_inference.py             # smoke tests — each model loads & predicts
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── .gitattributes                    # Git LFS tracking for *.pkl
└── .gitignore
```

---

## Setup

### 1. Clone the repo (with Git LFS)

Model files are tracked via [Git LFS](https://git-lfs.com), so install it
before cloning:

```bash
git lfs install
git clone https://github.com/SuperSam9276/Voyage-Analytics.git
cd Voyage-Analytics
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the classification model (one-time)

`classification_model.pkl` is not pre-trained in the repo — build it from
`data/users.csv`:

```bash
cd training
python train_classification.py
cd ..
```

This saves `app/models/classification_model.pkl`.

The flight price and hotel recommendation models are exported directly from
`notebooks/flight_regression.ipynb` and `notebooks/hotel_recommendation.ipynb`
— run those notebooks (from inside `notebooks/`) if you need to regenerate
`app/models/flight_price_model.pkl`, `app/models/flight_price_preprocessor.pkl`,
`app/models/recommendation_model.pkl`, or `app/data/hotels_clean.csv`.

---

## Running locally

```bash
cd app
streamlit run streamlit_app.py
```

Open `http://localhost:8501`.

---

## Running with Docker

```bash
docker build -t voyage-analytics .
docker run -p 8501:8501 voyage-analytics
```

Open `http://localhost:8501`.

The Docker image bakes the trained `.pkl` files and `hotels_clean.csv`
directly into the image, so the container runs standalone with no external
data dependencies.

---

## Testing

```bash
pip install pytest
pytest tests/test_inference.py -v
```

Runs a smoke test against all three models — confirms each loads and
produces a prediction without crashing. Requires the model artifacts in
`app/models/` and `app/data/hotels_clean.csv` to already exist (see Setup).

---

## Models

| Model | Type | Input | Output |
|---|---|---|---|
| Flight Price | Regression | route, flight type, agency, date/time, distance | predicted price (R$) |
| Hotel Recommendation | KNN similarity | user's booking history | ranked list of similar hotels |
| User Classification | Random Forest | age, company | predicted gender + class probabilities |

All three are wrapped in `app/inference/` modules that load their artifacts
via `joblib` and expose a single predict function each, called from
`streamlit_app.py`.

---

## Deployment

Deployed as a Docker container, suitable for any container-based host
(Hugging Face Spaces, Render, Fly.io, etc.). See `Dockerfile` for the build
definition.

---

## Notes on data

`data/users.csv` and `app/data/hotels_clean.csv` are the datasets used to
train/serve these models. Confirm these do not contain real personal booking
or user data before treating this repo as fully public — anonymize or
restrict access if needed.
