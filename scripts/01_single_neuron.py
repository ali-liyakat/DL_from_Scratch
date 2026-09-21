# 01_single_neuron.py

# Toy dataset: we want the neuron to learn y = 2x
xs = [1, 2, 3, 4, 5]
ys = [2, 4, 6, 8, 10]

# Start with random-ish (deliberately wrong) values
w = 0.0
b = 0.0

learning_rate = 0.01

for epoch in range(1700):
    total_loss = 0
    dw = 0
    db = 0

    for x, y in zip(xs, ys):
        y_pred = w * x + b
        loss = (y_pred - y) ** 2
        total_loss += loss

        # gradients, from the math above
        dw += 2 * (y_pred - y) * x
        db += 2 * (y_pred - y)

    # average gradient over the dataset
    dw /= len(xs)
    db /= len(xs)

    # gradient descent step: move w, b in the direction that REDUCES loss
    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 20 == 0:
        print(f"epoch {epoch}: loss={total_loss:.4f}, w={w:.3f}, b={b:.3f}")

print(f"\nFinal: w={w:.3f}, b={b:.3f} (target: w=2, b=0)")