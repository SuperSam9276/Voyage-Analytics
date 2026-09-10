import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# load data

df=pd.read_csv("users.csv")

# basic structure

print("Shape:", df.shape)
print("\nDtypes: \n", df.dtypes)
print("\nHead:\n", df.head())
print("\nInfo:")
df.info()

# missing values
print("\nMissing values: \n", df.isnull().sum())

# duplicates
print("\nDuplicate rows: \n", df.duplicated().sum())

# summary stats
print("\nNumeric Summary: \n", df.describe())
print("\ncategorical summary: \n", df.describe(include="object"))

# unique value counts for categorical columns
print("\nGender counts: \n", df["gender"].value_counts())
print("\nTop 10 companies:\n", df["company"].value_counts().head(10))

# visualizations

# age distribution
plt.figure(figsize=(8,5))
sns.histplot(df["age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# age boxplot (outlier check)
plt.figure(figsize=(6,4))
sns.boxplot(x=df["age"])
plt.title("Age boxplot")
plt.tight_layout()
plt.show()

# gender count
plt.figure(figsize=(6,4))
sns.countplot(x="gender",data=df)
plt.title("Gender Distribution")
plt.tight_layout()
plt.show()

# age by gender
plt.figure(figsize=(7, 5))
sns.boxplot(x="gender", y="age", data=df)
plt.title("Age by Gender")
plt.tight_layout()
plt.show()

# top companies by user count
top_companies = df["company"].value_counts().head(10)
plt.figure(figsize=(9, 5))
sns.barplot(x=top_companies.values, y=top_companies.index)
plt.title("Top 10 Companies by User Count")
plt.xlabel("Number of Users")
plt.ylabel("Company")
plt.tight_layout()
plt.show()