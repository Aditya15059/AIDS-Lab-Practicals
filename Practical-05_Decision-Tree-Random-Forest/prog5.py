import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)
dt = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
for name, model in [("Decision Tree", dt), ("Random Forest", rf)]:
    pred = model.predict(X_test)
    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred, average="macro"))
    print("Recall:", recall_score(y_test, pred, average="macro"))
    print("F1:", f1_score(y_test, pred, average="macro"))
pd.Series(rf.feature_importances_, index=data.feature_names).nlargest(10).sort_values().plot(kind="barh")
plt.title("Top 10 Random Forest Features"); plt.show()
plt.figure(figsize=(18, 9))
plot_tree(dt, feature_names=data.feature_names, class_names=data.target_names,
          filled=True, rounded=True, fontsize=7)
plt.show()
