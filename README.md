# DL From Scratch: Backprop to GPT

A hands-on journey through deep learning fundamentals — building every core mechanism from scratch, in order, until they assemble into a working GPT. No step skipped, no black boxes: every formula was hand-derived before being handed off to a framework.

## Philosophy

Each step exists to prove *why* the next one is necessary — not just to demo a new API. Several steps deliberately break something (an RNN failing on long sequences, a single neuron failing at XOR) before the fix is introduced, so every architectural choice is motivated by a real, observed limitation rather than taken on faith.

## The Journey

| Step | What | Key Result |
|---|---|---|
| 1 | Single neuron, hand-derived backprop (pure Python) | Learned `y=2x` via gradient descent; explored learning-rate instability |
| 2 | Multi-layer perceptron + ReLU, hand-derived backprop (NumPy) | Solved XOR — provably impossible for a single neuron |
| 3 | Same XOR problem in PyTorch | Confirmed `loss.backward()` reproduces the exact hand-derived gradients from Step 2 |
| 4 | Train/val/test split on MNIST | 95.8% test accuracy; tracked the generalization gap epoch-by-epoch |
| 5 | Character-level tokenization + embeddings | Proved an embedding layer is literally matrix indexing |
| 6 | RNN, next-character prediction | Perfectly memorized and regenerated a 22-character sequence |
| 7 | RNN long-range dependency test | 100% accuracy at length 5, only 65.5% at length 40 — the vanishing gradient problem, made visible |
| 8 | Self-attention on the same long-range task | **100%** accuracy at length 40 — attention weights show the model looking directly at position 0 |
| 9 | Multi-head attention + full Transformer block | Verified shape-preservation (enables stacking) and correct per-head attention |
| 10 | Tiny GPT — causal masking + 3 stacked blocks, trained on Tiny Shakespeare | 112K params, loss 4.22 → 0.83, generated structured Shakespeare-style text (correct character-name formatting, real words, broken grammar — expected at this scale) |

## The core throughline

Steps 1–3 establish *how learning happens* (gradient descent, backprop). Steps 4–6 establish *how to represent and process real data* (generalization, text, sequence). Steps 7–9 establish *why attention replaced RNNs*, proven empirically rather than asserted. Step 10 assembles all of it into the actual architecture behind every modern LLM — including the one used in my [RAG project](../doc-qa-rag), just at a much larger scale.

## What I'd tell someone starting this

- Hand-deriving backprop before touching a framework is what made `.backward()` feel obvious instead of magic
- The RNN-vs-attention comparison (Steps 7 vs 8) is the single most convincing moment in the whole arc — same task, same architecture family otherwise, one architecture structurally fails and the other doesn't
- Reading attention weights directly (Step 8) is a real technique used in interpretability research, not just a teaching device

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install numpy torch torchvision
```

## Running any step

```bash
python scripts\0N_<name>.py > outputs\0N_<name>_output.txt
```

Each script and its corresponding output are committed together, so the git history itself is a step-by-step record of the project.

## What's next

- Read "Attention Is All You Need" (now genuinely legible after building the mechanism by hand)
- Byte-pair encoding tokenization (what real GPT uses instead of character-level)
- Explore attention weights on real trained text, not just the toy long-range task