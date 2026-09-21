# scripts/06_rnn_next_char.py
import torch
import torch.nn as nn

torch.manual_seed(42)

text = "the cat sat on the mat"
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s): return [stoi[ch] for ch in s]
def decode(ids): return ''.join(itos[i] for i in ids)

data = torch.tensor(encode(text))

# Training pairs: input = text[:-1], target = text[1:] (shifted by one -- "predict next char")
X = data[:-1]
Y = data[1:]

embedding_dim = 16
hidden_dim = 32

class CharRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)   # hidden state -> logits over vocab

    def forward(self, x, hidden=None):
        emb = self.embed(x)                        # (seq_len, embed_dim)
        out, hidden = self.rnn(emb.unsqueeze(0), hidden)  # add batch dim
        logits = self.head(out.squeeze(0))          # (seq_len, vocab_size)
        return logits, hidden

model = CharRNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(300):
    logits, _ = model(X)
    loss = loss_fn(logits, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"epoch {epoch}: loss={loss.item():.4f}")

# Generate: start with "t", predict next char, feed it back in, repeat
print("\nGenerating from seed 't':")
model.eval()
context = torch.tensor(encode("t"))
hidden = None
generated = "t"
with torch.no_grad():
    for _ in range(21):
        logits, hidden = model(context, hidden)
        next_id = logits[-1].argmax().item()
        generated += itos[next_id]
        context = torch.tensor([next_id])

print(generated)