# scripts/10_tiny_gpt.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request
import os

torch.manual_seed(42)

# --- Download a small real text corpus (only if not already present) ---
data_path = "data_shakespeare.txt"
if not os.path.exists(data_path):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, data_path)

with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

text = text[:20000]   # keep it small so CPU training is fast for this exercise

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s): return [stoi[ch] for ch in s]
def decode(ids): return ''.join(itos[i] for i in ids)

data = torch.tensor(encode(text))

block_size = 64     # how many characters of context the model sees at once
d_model = 64
n_heads = 4
n_layers = 3
ff_hidden = 128

def get_batch(batch_size=16):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / (self.d_head ** 0.5)   # (B, n_heads, T, T)

        # --- CAUSAL MASK: block attending to future positions ---
        mask = torch.tril(torch.ones(T, T)).bool()
        scores = scores.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attended = attn_weights @ V
        combined = attended.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_out(combined)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_hidden):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_hidden), nn.ReLU(), nn.Linear(ff_hidden, d_model))
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ff(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, ff_hidden) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T)
        emb = self.token_embed(x) + self.pos_embed(positions)
        for block in self.blocks:
            emb = block(emb)
        emb = self.ln_f(emb)
        logits = self.head(emb)
        return logits


model = TinyGPT()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

for step in range(2000):
    x, y = get_batch()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"step {step}: loss={loss.item():.4f}")

# --- Generate ---
print("\nGenerating 300 characters from a blank start:")
model.eval()
context = torch.zeros((1, 1), dtype=torch.long)   # start with token 0
with torch.no_grad():
    for _ in range(300):
        cond = context[:, -block_size:]
        logits = model(cond)
        probs = F.softmax(logits[0, -1], dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        context = torch.cat([context, next_id.unsqueeze(0)], dim=1)

print(decode(context[0].tolist()))