import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X = np.arange(1, 11).reshape(-1, 1)
y = np.array([35, 40, 45, 52, 55, 63, 68, 74, 82, 88])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
print("Slope:", model.coef_[0], "Intercept:", model.intercept_)
print("Actual:", y_test, "Predicted:", pred.round(2))
print("MAE:", mean_absolute_error(y_test, pred))
print("MSE:", mean_squared_error(y_test, pred))
print("R2:", r2_score(y_test, pred))
plt.scatter(X, y, label="Actual")
plt.plot(X, model.predict(X), color="orange", label="Regression line")
plt.xlabel("Study Hours"); plt.ylabel("Marks"); plt.legend(); plt.grid(); plt.show()
