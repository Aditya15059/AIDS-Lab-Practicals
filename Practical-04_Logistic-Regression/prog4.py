import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X = np.arange(1, 11).reshape(-1, 1)
marks = np.array([35, 40, 45, 52, 55, 63, 68, 74, 82, 88])
y = (marks >= 60).astype(int)
print("Class Labels:", y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = LogisticRegression().fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, zero_division=0))
xs = np.sort(X, axis=0)
plt.scatter(X, y, color="red", label="Actual classes")
plt.plot(xs, model.predict_proba(xs)[:, 1], label="Pass probability")
plt.axhline(0.5, color="gray", linestyle="--")
plt.xlabel("Study Hours"); plt.ylabel("Probability of Passing")
plt.legend(); plt.grid(); plt.show()
