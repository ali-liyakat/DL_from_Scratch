# scripts/02_mlp_xor.py
import numpy as np

np.random.seed(42)

# XOR dataset
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0], [1], [1], [0]])

# 2 inputs -> 4 hidden neurons -> 1 output
W1 = np.random.randn(4, 2) * 0.5
b1 = np.zeros((4, 1))
W2 = np.random.randn(1, 4) * 0.5
b2 = np.zeros((1, 1))

lr = 0.1

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

for epoch in range(5000):
    total_loss = 0
    dW1 = np.zeros_like(W1); db1 = np.zeros_like(b1)
    dW2 = np.zeros_like(W2); db2 = np.zeros_like(b2)

    for i in range(len(X)):
        x = X[i].reshape(2, 1)
        y = Y[i].reshape(1, 1)

        # forward
        z1 = W1 @ x + b1
        a1 = relu(z1)
        y_pred = W2 @ a1 + b2

        loss = np.sum((y_pred - y) ** 2)
        total_loss += loss

        # backward
        dy_pred = 2 * (y_pred - y)
        dW2 += dy_pred @ a1.T
        db2 += dy_pred
        da1 = W2.T @ dy_pred
        dz1 = da1 * relu_deriv(z1)
        dW1 += dz1 @ x.T
        db1 += dz1

    n = len(X)
    W1 -= lr * dW1 / n; b1 -= lr * db1 / n
    W2 -= lr * dW2 / n; b2 -= lr * db2 / n

    if epoch % 500 == 0:
        print(f"epoch {epoch}: loss={total_loss:.4f}")

print("\nFinal predictions:")
for i in range(len(X)):
    x = X[i].reshape(2, 1)
    a1 = relu(W1 @ x + b1)
    y_pred = W2 @ a1 + b2
    print(f"input={X[i]} -> pred={y_pred[0][0]:.3f} (target={Y[i][0]})")