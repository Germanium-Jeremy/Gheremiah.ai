# GHEREMIAH AI

A hands-on workspace for learning how language models are built — six numbered stages that walk from **NumPy basics** all the way to a **tiny GPT trained on Tiny Shakespeare**. Every stage builds directly on the previous one, so the folders are meant to be read in order.

All `.py` files and notebooks carry **revision comments** (why a variable/function/block exists, what it controls, and how it affects the model or output) so the material is easy to revisit later. Comments never change the code itself.

## The 6-stage path

| Stage | Folder | What it teaches |
|---|---|---|
| 01 | `01_learn_numpy/` | NumPy fundamentals (slicing, broadcasting, boolean masks) applied to a **character-level bigram model**: count pairs → row-normalize → sample next character. `main.py` is the walkthrough, `final.py` refactors it into a `TextModel` class. |
| 02 | `02_learn_pytorch/` | PyTorch tensors + **autograd** (`main.py`), a **trigram** counting model with 2 characters of context (`trigram.py`), and the bridge from counting to **learning**: one-hot → matmul with a trainable `W` → softmax → cross-entropy → `backward()` (`final.py`). |
| 03 | `03_linear_model/` | The first real **training loop**: epochs, SGD, averaged loss, **perplexity**, save/load checkpoints (`training.py`). `training.ipynb` is the Colab version comparing **character vs word vs BPE subword** tokenizations with bigram sampling. |
| 04 | `04_neural_network/` | First neural network: a **2-layer MLP** that predicts the next character from a 5-character one-hot window. `main.ipynb` also trains a byte-level BPE twin; `subword.ipynb` is the dedicated subword (BPE, vocab 512) version — same MLP, wider inputs. |
| 05 | `05_sequence_model_to_attention/` | **Self-attention from scratch**: single-head attention first in NumPy, then as an `nn.Module`; multi-head attention, Transformer blocks (residuals + LayerNorm + FFN), and a **char-level TinyGPT** trained on 64-token chunks with temperature/top-k sampling. |
| 06 | `06_tiny_GPT/` | The payoff: a minimal **decoder-only GPT** with a learned **BPE tokenizer** (vocab 300), fused QKV + `scaled_dot_product_attention`, training loop, generation, and **checkpointing** to `models/06_tiny_GPT.pt` with a save/reload round-trip. |

## The recurring pipeline

Every stage reuses the same skeleton, just with a bigger model:

```
text → tokens (chars / words / BPE) → (context, next-token) pairs
     → model → logits → softmax → cross-entropy → gradients → update
     → autoregressive sampling (temperature, top-k)
```

Key ideas introduced along the way: vocab + char↔index mapping (01), one-hot × W = embedding lookup (02), perplexity and optimizer steps (03), one-hot inputs and hidden capacity (04), causal self-attention and Transformer blocks (05), weight tying, fused attention and checkpoints (06).

## Data & outputs

- `dataset/input.txt` — Tiny Shakespeare (~1.1 MB, the corpus every stage trains on); `dataset/more.txt` — extra text concatenated in stages 01–03. (gitignored)
- `models/` — saved checkpoints (e.g. `06_tiny_GPT.pt`). (gitignored)
- Notebooks 03/04 also download the dataset directly from Karpathy's `char-rnn` repo when running on Colab.

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows; on mac/linux use .venv/bin/pip
```

Core dependencies: `torch`, `numpy`, `tokenizers`, `matplotlib`, `jupyter`. Scripts in 01–03 expect to be run from the workspace root (they load `dataset/...` relative paths); notebooks run top-to-bottom.
