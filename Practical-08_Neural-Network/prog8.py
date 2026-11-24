import numpy as np
np.random.seed(42)
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

W1, b1 = np.random.randn(2, 4), np.zeros((1, 4))
W2, b2 = np.random.randn(4, 1), np.zeros((1, 1))
lr = 0.5
for epoch in range(10000):
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    y_pred = sigmoid(a1 @ W2 + b2)
    dz2 = (y_pred - y) * sigmoid_derivative(y_pred)
    dW2 = a1.T @ dz2 / len(X)
    db2 = np.mean(dz2, axis=0, keepdims=True)
    dz1 = (dz2 @ W2.T) * sigmoid_derivative(a1)
    dW1 = X.T @ dz1 / len(X)
    db1 = np.mean(dz1, axis=0, keepdims=True)
    W2 -= lr * dW2; b2 -= lr * db2
    W1 -= lr * dW1; b1 -= lr * db1
    if epoch % 2500 == 0:
        print("Epoch", epoch, "Loss:", np.mean((y_pred-y)**2))
predicted = (y_pred >= 0.5).astype(int)
print("Probabilities:", y_pred.ravel().round(3))
print("Predicted:", predicted.ravel())
print("Actual:", y.astype(int).ravel())
