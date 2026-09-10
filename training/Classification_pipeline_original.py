import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    VarianceThreshold,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# load data
df = pd.read_csv("users.csv")

# preprocessing

# strip whitespace from string columns
str_cols = df.select_dtypes(include="object").columns
df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

# drop exact duplicate rows
df = df.drop_duplicates()

# drop duplicate ids, keep first occurrence
df = df.drop_duplicates(subset="code", keep="first")

# standardize text casing
df["name"] = df["name"].str.title()
df["company"] = df["company"].str.strip()

# enforce dtypes
df["code"] = df["code"].astype(int)
df["age"] = pd.to_numeric(df["age"], errors="coerce").astype("Int64")

# drop rows with invalid age
df = df[(df["age"].isna()) | ((df["age"] >= 0) & (df["age"] <= 100))]

# clean, drop rows with missing/unspecified gender
df["gender"] = df["gender"].replace({"none": np.nan})
df["gender"] = df["gender"].str.lower()
df = df.dropna(subset=["gender", "age"])

# drop non-predictive identifier columns
df = df.drop(columns=["code", "name"])


# encode target
le = LabelEncoder()
y = le.fit_transform(df["gender"])
print("Target classes:", dict(zip(le.classes_,le.transform(le.classes_) )))


# encode features
X = pd.get_dummies(df.drop(columns=["gender"]), columns=["company"], drop_first=True)


# remove near-zero-variance features (rare/uninformative company dummies) ---
vt = VarianceThreshold(threshold=0.01)
X_vt = vt.fit_transform(X)
kept_cols = X.columns[vt.get_support()]
X = X[kept_cols]
print(f"\nFeatures after variance threshold: {list(X.columns)}")


# mutual information (handles numeric + categorical dummies) ---
mi_scores = mutual_info_classif(X, y, discrete_features="auto", random_state=42)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
print("\nMutual information scores:\n", mi_series)

# select top K features by mutual information ---
k = min(5, X.shape[1])
selector = SelectKBest(score_func=mutual_info_classif, k=k)
selector.fit(X, y)
top_features = X.columns[selector.get_support()]
print(f"\nTop {k} selected features:\n", list(top_features))

# cross-check with Random Forest Feature importance
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X,y)
rf_importance = pd.Series(rf.feature_importances_,index=X.columns).sort_values(ascending=False)
print("\nRandom Forest feature importance: \n" , rf_importance)


# final selected feature set
X_selected = X[top_features]
print("\nFinal feature matrix shape:", X_selected.shape)


# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)
print("Train target distribution:\n", pd.Series(y_train).value_counts(normalize=True))
print("Test target distribution:\n", pd.Series(y_test).value_counts(normalize=True))


# build and train random forest classifier
clf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
clf.fit(X_train, y_train)

# evaluate on test set
y_pred = clf.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))

# feature importance from the trained classifier
clf_importance = pd.Series(clf.feature_importances_, index=X_selected.columns).sort_values(ascending=False)
print("\nFeature importance (trained classifier):\n", clf_importance)



