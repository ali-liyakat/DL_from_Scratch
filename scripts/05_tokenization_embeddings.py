# scripts/05_tokenization_embeddings.py
import torch
import torch.nn as nn

torch.manual_seed(42)

# --- Step A: Tokenization (just a lookup table, nothing learned) ---
text = "the cat sat on the mat"

chars = sorted(list(set(text)))          # every unique character
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}   # string -> integer
itos = {i: ch for ch, i in stoi.items()}       # integer -> string

def encode(s):
    return [stoi[ch] for ch in s]

def decode(ids):
    return ''.join(itos[i] for i in ids)

print("Vocabulary:", chars)
print("Vocab size:", vocab_size)

sample = "cat"
ids = encode(sample)
print(f"\n'{sample}' -> token ids: {ids}")
print(f"decoded back: '{decode(ids)}'")

# --- Step B: Embedding (a LEARNED table: vocab_size x embedding_dim) ---
embedding_dim = 6
embedding_table = nn.Embedding(vocab_size, embedding_dim)

token_ids = torch.tensor(encode(sample))     # e.g. [c, a, t] -> [ids]
vectors = embedding_table(token_ids)

print(f"\nToken ids for '{sample}': {token_ids.tolist()}")
print(f"Embedding vectors shape: {vectors.shape}  (3 chars x {embedding_dim} dims)")
print(vectors)

# --- Step C: prove it's really just indexing into a matrix ---
full_table = embedding_table.weight
print(f"\nFull embedding table shape: {full_table.shape}  (vocab_size x embedding_dim)")

manual_lookup = full_table[token_ids]
print("\nManually indexing the table gives the identical result as calling embedding_table():")
print(torch.equal(manual_lookup, vectors))