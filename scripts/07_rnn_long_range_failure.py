# scripts/07_rnn_long_range_failure.py
import torch
import torch.nn as nn

torch.manual_seed(42)

# The rule: the LAST character must match the FIRST character.
# This forces the model to remember something from step 0 all the way to the end.
def make_example(length):
    import random
    first = random.choice(['a', 'b'])
    middle = ''.join(random.choice('cdefgh') for _ in range(length - 2))
    return first + middle + first   # first char repeated at the end

torch.manual_seed(42)
import random
random.seed(42)

examples_short = [make_example(5) for _ in range(200)]     # short: easy
examples_long = [make_example(40) for _ in range(200)]     # long: hard

chars = sorted(list(set(''.join(examples_short + examples_long))))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s): return [stoi[ch] for ch in s]

embedding_dim = 16
hidden_dim = 32

class CharRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embed(x)
        out, _ = self.rnn(emb)
        logits = self.head(out)
        return logits

def train_and_test(examples, label, epochs=300):
    model = CharRNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    data = [torch.tensor(encode(ex)) for ex in examples]

    for epoch in range(epochs):
        total_loss = 0
        for seq in data:
            X = seq[:-1]
            Y = seq[-1:].repeat(len(X))  # simplify: just predict last-char target at every step isn't quite right
            logits = model(X)
            # We only actually care about predicting the FINAL character correctly
            last_logit = logits[-1].unsqueeze(0)
            target = seq[-1].unsqueeze(0)
            loss = loss_fn(last_logit, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # Test: how often does it correctly predict the last char = first char?
    correct = 0
    with torch.no_grad():
        for seq in data:
            X = seq[:-1]
            logits = model(X)
            pred = logits[-1].argmax().item()
            if pred == seq[-1].item():
                correct += 1
    acc = correct / len(data)
    print(f"{label}: final-char accuracy = {acc:.2%}")
    return acc

print("Training on SHORT sequences (length 5)...")
train_and_test(examples_short, "Short sequences (len=5)")

print("\nTraining on LONG sequences (length 40)...")
train_and_test(examples_long, "Long sequences (len=40)")