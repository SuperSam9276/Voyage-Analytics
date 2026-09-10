"""
Trains the gender-classification model and SAVES it — the original
Classification_pipeline.py trained and printed metrics but never called
joblib.dump(), so no .pkl existed to deploy.

Also replaces pd.get_dummies(company) with OneHotEncoder(handle_unknown="ignore")
inside a proper sklearn Pipeline. get_dummies breaks silently at inference time
if a company value appears that wasn't in the training data (columns won't line
up with what the model expects) -- OneHotEncoder handles unseen categories safely.

Run this from the training/ folder with users.csv present, e.g.:
    python train_classification.py --input users.csv --output ../app/models/classification_model.pkl
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

    df = df.drop_duplicates()
    df = df.drop_duplicates(subset="code", keep="first")

    df["name"] = df["name"].str.title()
    df["company"] = df["company"].str.strip()

    df["code"] = df["code"].astype(int)
    df["age"] = pd.to_numeric(df["age"], errors="coerce").astype("Int64")
    df = df[(df["age"].isna()) | ((df["age"] >= 0) & (df["age"] <= 100))]

    df["gender"] = df["gender"].replace({"none": np.nan})
    df["gender"] = df["gender"].str.lower()
    df = df.dropna(subset=["gender", "age"])

    df = df.drop(columns=["code", "name"])
    return df


def main(input_path: str, output_path: str, k: int = 5):
    df = load_and_clean(input_path)

    le = LabelEncoder()
    y = le.fit_transform(df["gender"])
    print("Target classes:", dict(zip(le.classes_, le.transform(le.classes_))))

    X = df.drop(columns=["gender"])
    X["age"] = X["age"].astype(float)

    # Encode company via OneHotEncoder inside the pipeline (not pd.get_dummies)
    # so unseen categories at inference time don't break the feature matrix.
    preprocessor = ColumnTransformer(
        transformers=[
            ("company", OneHotEncoder(handle_unknown="ignore"), ["company"]),
        ],
        remainder="passthrough",  # passes "age" through unchanged
    )

    k = min(k, X.shape[1])  # can't select more features than exist pre-encoding safety

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("variance_threshold", VarianceThreshold(threshold=0.01)),
        ("select_k_best", SelectKBest(score_func=mutual_info_classif, k=k)),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))

    package = {
        "pipeline": pipeline,
        "label_encoder": le,
        "known_companies": sorted(df["company"].dropna().unique().tolist()),
    }
    joblib.dump(package, output_path)
    print(f"\nSaved classification package to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../data/users.csv")
    parser.add_argument("--output", default="../app/models/classification_model.pkl")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    main(args.input, args.output, args.k)
