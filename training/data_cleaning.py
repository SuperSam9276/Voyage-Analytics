import pandas as pd
import numpy as np

# load data
df=pd.read_csv("users.csv")

# strip white spaces from string columns
str_cols = df.select_dtypes(include="object").columns
df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

# drop exact duplicate rows
before = len(df)
df=df.drop_duplicates()
print(f"Dropped {before - len(df)} duplicate rows")

# duplicate ids
dup_ids = df["code"].duplicated().sum()
print(f"Duplicate 'code' values: {dup_ids}")
if dup_ids:
    df=df.drop_duplicates(subset="code", keep="first")


# normalize gender: "none"
df["gender"] = df["gender"].replace({"none": np.nan})
df["gender"] = df["gender"].str.lower()

# standardize text casing
df["name"] = df["name"].str.title()
df["company"] = df["company"].str.strip()

# enforce dtypes
df["code"] = df["code"].astype(int)
df["age"] = pd.to_numeric(df["age"], errors="coerce").astype("Int64")
df["gender"] = df["gender"].astype("category")
df["company"] = df["company"].astype("category")


# sanity-check age range
invalid_age = df[(df["age"] < 0) | (df["age"] > 100)]
print(f"Invalid age rows: {len(invalid_age)}")
df = df[(df["age"].isna()) | ((df["age"] >= 0) & (df["age"] <= 100))]


# missing value report
print("\nMissing values after cleaning:\n", df.isnull().sum())

# save cleaned data 
df.to_csv("users_cleaned.csv", index=False)
print("\nCleaned shape:", df.shape)
