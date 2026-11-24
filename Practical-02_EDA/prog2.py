import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset("titanic")
print("Shape:", df.shape)
print(df.head())
print(df.describe())
for stat, value in {
    "Mean": df["age"].mean(), "Median": df["age"].median(),
    "Mode": df["age"].mode()[0], "Std": df["age"].std(),
    "Variance": df["age"].var(), "Skewness": df["age"].skew(),
    "Kurtosis": df["age"].kurt()
}.items():
    print(stat, "Age:", value)
print("\nClass counts:\n", df["class"].value_counts(dropna=False))
print("\nMissing values:\n", df.isnull().sum())

sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Value Map"); plt.show()
sns.histplot(df["age"].dropna(), kde=True, bins=20)
plt.title("Age Distribution"); plt.show()
sns.countplot(x="class", data=df)
plt.title("Passenger Count by Class"); plt.show()
sns.boxplot(y=df["fare"])
plt.title("Fare Outliers"); plt.show()
sns.scatterplot(x="age", y="fare", hue="survived", data=df)
plt.title("Age vs Fare by Survival"); plt.show()
survival = df.groupby("class", observed=True)["survived"].mean()
print("\nSurvival rate by class:\n", survival)
survival.plot(kind="bar", title="Survival Rate by Class"); plt.show()
numeric = df.select_dtypes(include=[np.number])
sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap"); plt.show()
sns.pairplot(df[["survived", "age", "fare", "sibsp", "parch"]].dropna(), hue="survived")
plt.show()
