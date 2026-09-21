# scripts/09_multihead_transformer_block.py
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model)   # combines heads back together

    def forward(self, x):
        seq_len, d_model = x.shape

        Q = self.W_q(x).view(seq_len, self.n_heads, self.d_head).transpose(0, 1)  # (n_heads, seq_len, d_head)
        K = self.W_k(x).view(seq_len, self.n_heads, self.d_head).transpose(0, 1)
        V = self.W_v(x).view(seq_len, self.n_heads, self.d_head).transpose(0, 1)

        scores = Q @ K.transpose(-2, -1) / (self.d_head ** 0.5)   # (n_heads, seq_len, seq_len)
        attn_weights = F.softmax(scores, dim=-1)
        attended = attn_weights @ V                                # (n_heads, seq_len, d_head)

        # combine all heads back into one d_model-sized vector per position
        combined = attended.transpose(0, 1).contiguous().view(seq_len, d_model)
        return self.W_out(combined), attn_weights


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_hidden):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, attn_weights = self.attn(x)
        x = self.ln1(x + attn_out)      # residual connection #1, then normalize
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)        # residual connection #2, then normalize
        return x, attn_weights


# --- Sanity check: run fake input through one full block ---
seq_len, d_model, n_heads, ff_hidden = 6, 32, 4, 64
x = torch.randn(seq_len, d_model)

block = TransformerBlock(d_model, n_heads, ff_hidden)
out, attn_weights = block(x)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {out.shape}  (same as input — a Transformer block never changes shape)")
print(f"\nAttention weights shape: {attn_weights.shape}  (n_heads={n_heads}, seq_len={seq_len}, seq_len={seq_len})")
print(f"Each head's attention row sums to ~1.0: {attn_weights[0][0].sum().item():.4f}")

print(f"\nParameter count in this one block: {sum(p.numel() for p in block.parameters()):,}")