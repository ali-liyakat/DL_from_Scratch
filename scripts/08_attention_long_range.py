# scripts/08_attention_long_range.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

torch.manual_seed(42)
random.seed(42)

def make_example(length):
    first = random.choice(['a', 'b'])
    middle = ''.join(random.choice('cdefgh') for _ in range(length - 2))
    return first + middle + first

examples_short = [make_example(5) for _ in range(200)]
examples_long = [make_example(40) for _ in range(200)]

chars = sorted(list(set(''.join(examples_short + examples_long))))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}

def encode(s): return [stoi[ch] for ch in s]

embedding_dim = 16
max_len = 40

class AttentionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed = nn.Embedding(max_len, embedding_dim)   # NEW: position-based, not content-based
        self.W_q = nn.Linear(embedding_dim, embedding_dim)
        self.W_k = nn.Linear(embedding_dim, embedding_dim)
        self.W_v = nn.Linear(embedding_dim, embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        seq_len = x.shape[0]
        positions = torch.arange(seq_len)

        tok_emb = self.token_embed(x)
        pos_emb = self.pos_embed(positions)
        emb = tok_emb + pos_emb              # combine WHAT the token is + WHERE it is

        Q = self.W_q(emb)
        K = self.W_k(emb)
        V = self.W_v(emb)

        scores = Q @ K.T / (embedding_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attended = attn_weights @ V           # every position can now see every other position, directly

        logits = self.head(attended)
        return logits, attn_weights

def train_and_test(examples, label, epochs=300):
    model = AttentionClassifier()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    data = [torch.tensor(encode(ex)) for ex in examples]

    for epoch in range(epochs):
        for seq in data:
            X = seq[:-1]
            target = seq[-1].unsqueeze(0)
            logits, _ = model(X)
            last_logit = logits[-1].unsqueeze(0)
            loss = loss_fn(last_logit, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    correct = 0
    last_attn = None
    with torch.no_grad():
        for seq in data:
            X = seq[:-1]
            logits, attn_weights = model(X)
            pred = logits[-1].argmax().item()
            if pred == seq[-1].item():
                correct += 1
            last_attn = attn_weights

    acc = correct / len(data)
    print(f"{label}: final-char accuracy = {acc:.2%}")
    return acc, last_attn

print("Training attention model on SHORT sequences (length 5)...")
train_and_test(examples_short, "Short sequences (len=5)")

print("\nTraining attention model on LONG sequences (length 40)...")
acc, attn = train_and_test(examples_long, "Long sequences (len=40)")

print("\nRNN comparison (from Step 7): short=100.00%, long=65.50%")

print("\nLast position's attention weights over the full 39-token sequence (last example):")
print("(high weight near position 0 = model learned to look back at the FIRST character)")
last_row = attn[-1]
print(f"Weight on position 0 (the first char): {last_row[0].item():.4f}")
print(f"Average weight on all other positions: {last_row[1:].mean().item():.4f}")