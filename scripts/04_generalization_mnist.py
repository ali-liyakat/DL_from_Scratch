# scripts/04_generalization_mnist.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms

torch.manual_seed(42)

transform = transforms.ToTensor()
full_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Split training data itself into train/val (val = data we check on, but never train on)
train_size = 50000
val_size = len(full_train) - train_size
train_set, val_set = torch.utils.data.random_split(full_train, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=64)

model = nn.Sequential(
    nn.Flatten(),          # 28x28 image -> 784-length vector
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 10)       # 10 output classes: digits 0-9
)

loss_fn = nn.CrossEntropyLoss()   # standard loss for classification (new — MSE was for regression)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

def evaluate(loader):
    model.eval()
    correct, total, total_loss = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x)
            total_loss += loss_fn(pred, y).item() * len(y)
            correct += (pred.argmax(dim=1) == y).sum().item()
            total += len(y)
    model.train()
    return total_loss / total, correct / total

for epoch in range(5):
    for x, y in train_loader:
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss, train_acc = evaluate(train_loader)
    val_loss, val_acc = evaluate(val_loader)
    print(f"epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} | val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

test_loss, test_acc = evaluate(torch.utils.data.DataLoader(test_set, batch_size=64))
print(f"\nFinal test accuracy: {test_acc:.3f}")