import pandas as pd
import numpy as np
from pathlib import Path

# Read the included CSV dataset
df = pd.read_csv(Path(__file__).with_name("employee_raw_data.csv"))
print("Raw Dataset:\n", df)
print("\nShape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

df = df.drop_duplicates()
df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Salary"] = df["Salary"].fillna(df["Salary"].median())
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
df["Department"] = df["Department"].fillna(df["Department"].mode()[0])

df["Name"] = df["Name"].str.strip()
df["Gender"] = df["Gender"].str.strip().str.upper().replace(
    {"MALE": "Male", "M": "Male", "FEMALE": "Female", "F": "Female"})
df["Department"] = df["Department"].str.strip().str.upper()
iso = pd.to_datetime(df["JoiningDate"].astype(str), format="%Y-%m-%d", errors="coerce")
dmy = pd.to_datetime(df["JoiningDate"].astype(str), format="%d/%m/%Y", errors="coerce")
df["JoiningDate"] = iso.fillna(dmy)
df["Age"] = df["Age"].astype(int)

q1, q3 = df["Age"].quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
df["Age"] = np.where(df["Age"] > upper, upper,
                     np.where(df["Age"] < lower, lower, df["Age"]))

encoded = pd.get_dummies(df, columns=["Gender", "Department"], drop_first=True)
salary_range = encoded["Salary"].max() - encoded["Salary"].min()
encoded["Salary_Normalized"] = (
    (encoded["Salary"] - encoded["Salary"].min()) / salary_range
    if salary_range else 0
)
print("\nFinal Cleaned Dataset:\n", encoded.to_string(index=False))
