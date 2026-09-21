# scripts/03_mlp_xor_pytorch.py
import torch
import torch.nn as nn

torch.manual_seed(42)

X = torch.tensor([[0.,0.], [0.,1.], [1.,0.], [1.,1.]])
Y = torch.tensor([[0.], [1.], [1.], [0.]])

model = nn.Sequential(
    nn.Linear(2, 4),   # same shape as your W1, b1
    nn.ReLU(),          # same as your relu()
    nn.Linear(4, 1)    # same shape as your W2, b2
)

loss_fn = nn.MSELoss()                              # same as your (y_pred - y)**2
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)   # same as your -= lr * dW

for epoch in range(5000):
    y_pred = model(X)              # forward pass — same as your z1, a1, y_pred lines
    loss = loss_fn(y_pred, Y)

    optimizer.zero_grad()          # clear old gradients
    loss.backward()                # <-- THIS replaces every derivative you hand-wrote
    optimizer.step()               # <-- THIS replaces your W1 -= lr * dW1 lines

    if epoch % 500 == 0:
        print(f"epoch {epoch}: loss={loss.item():.4f}")

print("\nFinal predictions:")
with torch.no_grad():
    for i in range(len(X)):
        pred = model(X[i])
        print(f"input={X[i].tolist()} -> pred={pred.item():.3f} (target={Y[i].item()})")