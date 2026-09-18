"""One-shot generator for 07_conversational_lm.ipynb.

Run:  .venv/Scripts/python.exe 07_build_notebook.py
"""
import json

cells = []


def md(src, cid):
    cells.append({"cell_type": "markdown", "id": cid,
                  "metadata": {"language": "markdown"}, "source": src})


def code(src, cid):
    cells.append({"cell_type": "code", "id": cid,
                  "metadata": {"language": "python"},
                  "execution_count": None, "outputs": [], "source": src})


md(r'''# Conversational LM — TinyGPT learns to chat

Notebook 06 trained a small GPT to continue Shakespeare. This notebook keeps the
**same architecture family** but changes the task: a **multi-turn chat model**
trained from scratch on real dialogue data (the 06 checkpoint is not reused).

The pipeline:

1. **Data**: `data/dialogues_train.txt` / `data/dialogues_val.txt` — ~30k
   multi-turn conversations assembled from three dialogue corpora (Cornell
   Movie-Dialogs, EmpatheticDialogues, ConvAI2/PersonaChat) plus a slice of
   Shakespeare as single-speaker assistant turns — §0
2. **Tokenizer**: the same byte-level BPE as 06, extended with the special chat
   tokens `<|system|> <|user|> <|assistant|> <|end|> <|pad|>`, plus a
   `format_chat` helper that defines exactly what the model sees — §1
3. **Loss masking (the key change)**: only tokens inside
   `<|assistant|> ... <|end|>` spans contribute to the cross-entropy loss —
   the user's words are *context*, not *targets* — §2
4. **Model + training**: the same TinyGPT stack (token+position embeddings →
   causal Transformer blocks → tied LM head), AdamW, gradient clipping — §3–4
5. **Checkpoint**: `models/07_conversational_lm.pt`, self-contained (weights +
   config + the tokenizer itself) — §5
6. **Inference**: a multi-turn chat loop that generates one assistant reply,
   stops at `<|end|>`, and trims old turns to fit the context window — §6

| hyperparameter | value |
|---|---|
| BPE vocab size | ~512 (5 of them special chat tokens) |
| embedding dim (`d_model`) | 128 |
| layers | 4 |
| attention heads | 4 |
| context length (`BLOCK_SIZE`) | 128 |
| batch size | 64 |''', "c0")

code(r'''import os
import re
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

torch.manual_seed(42)
# Seed = reproducible init/batches; device flag switches CPU/GPU.

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# On CPU, more threads != faster for a small model: thread sync overhead
# dominates below ~1M parameters, so cap the pool at half the cores.
torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
print(f"torch threads: {torch.get_num_threads()}")

# --- Configuration used across cells ---------------------------------------
BATCH_SIZE = 64     # conversations per step
BLOCK_SIZE = 128    # context length in tokens (>= 128 as required)''', "c1")

md(r'''## Colab setup — download data and choose checkpoint storage

Run this cell once at the start of a fresh runtime. It downloads the raw archives
listed in the repository README files, extracts only the files used by
`07_preprocess.py`, and writes the processed train/validation files. The helper is
idempotent, so rerunning it does not download files that are already present.

Set `USE_GOOGLE_DRIVE = True` when model checkpoints should survive a Colab
runtime reset. The raw data and processed data stay in the project directory;
only checkpoints are redirected to Drive.''', "c1_setup_md")

code(r'''# The notebook works from either the project root or its 07_conversational_lm folder.
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "07_conversational_lm" else Path.cwd()
SETUP_DIR = PROJECT_ROOT / "07_conversational_lm"
if not (SETUP_DIR / "colab_setup.py").exists():
    SETUP_DIR = Path.cwd()
sys.path.insert(0, str(SETUP_DIR))

from colab_setup import checkpoint_dir, download_raw_datasets

USE_GOOGLE_DRIVE = False
DRIVE_FOLDER = "Gheremiah"
download_raw_datasets(PROJECT_ROOT)
MODEL_DIR = checkpoint_dir(PROJECT_ROOT, USE_GOOGLE_DRIVE, DRIVE_FOLDER)

PREPROCESS_SCRIPT = next(
    path for path in (
        PROJECT_ROOT / "07_preprocess.py",
        PROJECT_ROOT / "07_conversational_lm" / "07_preprocess.py",
    )
    if path.exists()
)
subprocess.run([sys.executable, str(PREPROCESS_SCRIPT)], check=True)
print(f"project root: {PROJECT_ROOT}")
print(f"model directory: {MODEL_DIR}")''', "c1_setup")

md(r'''## 0. The dialogue dataset

`07_preprocess.py` converts the three cloned corpora into one simple format and
writes **one full conversation per line**:

    <|system|> optional system text <|end|>
    <|user|> ... <|end|>
    <|assistant|> ... <|end|>
    <|user|> ... <|end|> ...

Mix (≈180k turns total):

| source | role of the first speaker | notes |
|---|---|---|
| Cornell Movie-Dialogs | `user` | film dialogue, speaker pairs |
| EmpatheticDialogues | `user` | emotional support chats, with a situation as system prompt |
| ConvAI2 / PersonaChat | `user` | persona chats, persona text as system prompt |
| Tiny Shakespeare | `assistant` only | single-speaker turns, keeps some literary style |''', "c2")

code(r'''DATA_DIR = str(PROJECT_ROOT / "data")
TRAIN_PATH = os.path.join(DATA_DIR, "dialogues_train.txt")
VAL_PATH = os.path.join(DATA_DIR, "dialogues_val.txt")

for path in (TRAIN_PATH, VAL_PATH):
    assert os.path.exists(path), f"missing {path} — run `python 07_preprocess.py` first"


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


print(f"train conversations: {len(read_lines(TRAIN_PATH)):,}")
print(f"val   conversations: {len(read_lines(VAL_PATH)):,}")

print()
print("one training example, one turn per line:")
sample_line = read_lines(TRAIN_PATH)[0]
print(sample_line.replace("<|", "\n<|").strip())''', "c3")

md(r'''## 1. Task 1 — the tokenizer, extended with chat tokens

Same byte-level BPE as notebook 06 (GPT-2 conventions: ByteLevel pre-tokenizer
and decoder set *before* training, `initial_alphabet` guarantees all 256 byte
tokens exist before any merge). Two changes:

1. **Special tokens added before training**: `<|system|> <|user|> <|assistant|>
   <|end|> <|pad|>`. Added tokens are matched atomically at encode time — they
   can never be split by merges — and each gets a stable single id.
2. **Vocab ~512** instead of 300: 256 bytes + up to 251 merges + 5 specials
   (the trainer may land a few short; the real size is printed below). Dialogue
   is messier than verse (names, slang, punctuation), so a few extra merges are
   worth it.

The `<|pad|>` token exists because conversations have different lengths: every
example is padded to `BLOCK_SIZE + 1` tokens, and the mask (§2) makes sure the
padding never reaches the loss.

The tokenizer is trained on the raw dialogue files themselves (special tokens
included), so — as in 06 — the *corpus defines the vocab*. BPE training is
deterministic for a fixed corpus, and the trained tokenizer is also embedded
into the checkpoint (§5), so a fresh session can rebuild it exactly.''', "c4")

code(r'''SPECIAL_TOKENS = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<|pad|>"]
VOCAB_SIZE = 512   # 256 byte tokens + 251 merges + 5 specials


class ChatBPE:
    """Byte-level BPE tokenizer with special chat tokens added on top."""

    def __init__(self, corpus_texts, vocab_size=VOCAB_SIZE):
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        tok.add_special_tokens(SPECIAL_TOKENS)      # added tokens, ids in order
        tok.train_from_iterator(
            corpus_texts,
            trainers.BpeTrainer(
                # the 5 specials are added on top of the trainer's vocab
                vocab_size=vocab_size - len(SPECIAL_TOKENS),
                # guarantee all 256 byte-tokens exist before merges start
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            ),
        )
        self.tok = tok

    @classmethod
    def from_json(cls, tokenizer_json):
        """Rebuild a saved tokenizer exactly, without retraining (§5)."""
        obj = cls.__new__(cls)
        obj.tok = Tokenizer.from_str(tokenizer_json)
        return obj

    @property
    def vocab_size(self):
        return self.tok.get_vocab_size(with_added_tokens=True)

    def id(self, special):
        return self.tok.token_to_id(special)

    def encode(self, s):
        return self.tok.encode(s).ids

    def decode(self, ids):
        # skip_special_tokens defaults to True in tokenizers >= 0.20 — we WANT
        # the <|...|> tags back so decode(encode(x)) round-trips exactly.
        return self.tok.decode(ids, skip_special_tokens=False)


# Train ON the dialogue files (specials included) — deterministic for this corpus.
tokenizer = ChatBPE(read_lines(TRAIN_PATH) + read_lines(VAL_PATH))

special_tokens = {name.strip("<|>"): tokenizer.id(name) for name in SPECIAL_TOKENS}
assert list(special_tokens.values()) == list(range(len(SPECIAL_TOKENS)))
print("special tokens:", special_tokens)
print(f"vocab size: {tokenizer.vocab_size}")

sample = "<|user|> What do you do for fun? <|end|>"
ids = tokenizer.encode(sample)
print("tokens:", tokenizer.tok.encode(sample).tokens)
print("ids    :", ids)
assert tokenizer.decode(ids) == sample          # round-trip must be exact
print("Round-trip decode(encode(x)) == x: OK")''', "c5")

md(r'''### 1.1 The chat template

The template is the model's "grammar". Every turn is wrapped as
`<|role|> text <|end|>`, an optional system prompt comes first, and at
generation time we append a bare `<|assistant|>` opener so the first thing the
model must produce is the **reply itself**, ending at `<|end|>`.

`format_chat` is the single source of truth for this: training data, batch
building and inference all go through it.''', "c6")

code(r'''def format_chat(messages, system=None, add_assistant_opener=False):
    """Render a message list with the chat template.

    messages: [{"role": "user"|"assistant", "content": str}, ...]
    system:   optional system prompt shown before the first turn
    add_assistant_opener: append a bare "<|assistant|>" so the model's next
        tokens are the reply itself (used at generation time, §6).
    """
    parts = []
    if system:
        parts.append(f"<|system|> {system} <|end|>")
    for msg in messages:
        parts.append(f"<|{msg['role']}|> {msg['content']} <|end|>")
    if add_assistant_opener:
        parts.append("<|assistant|>")
    return " ".join(parts)


def build_ids(messages, system=None, add_assistant_opener=False):
    """Chat template -> the exact token ids the model sees."""
    return tokenizer.encode(format_chat(messages, system, add_assistant_opener))


SYSTEM_ID, USER_ID, ASSISTANT_ID, END_ID, PAD_ID = (
    special_tokens[k] for k in ("system", "user", "assistant", "end", "pad")
)

demo_msgs = [
    {"role": "user", "content": "What do you do for fun?"},
    {"role": "assistant", "content": "I watch old movies and cook."},
    {"role": "user", "content": "What kind of movies?"},
]
demo_system = "You are a friendly chatbot."
demo_text = format_chat(demo_msgs, system=demo_system, add_assistant_opener=True)
demo_ids = build_ids(demo_msgs, system=demo_system, add_assistant_opener=True)
print(demo_text)
print("ids :", demo_ids)
assert tokenizer.decode(demo_ids) == demo_text
assert demo_ids[-1] == ASSISTANT_ID
print("last id is the assistant opener: everything after it is the reply")''', "c7")

md(r'''## 2. Task 2 — next-token targets with **loss masking**

Notebook 06 flattened everything into one stream: `x = ids[:-1]`,
`y = ids[1:]`, and every one of the `B*T` positions trained the model. A chat
model must **not** learn to imitate the user: user tokens are *conditions*, not
predictions.

So each example is one conversation (padded to `BLOCK_SIZE + 1` with
`<|pad|>`), and the shift is unchanged — but the **loss mask** keeps only the
positions whose *target* lies inside an assistant reply:

```
ids    : <|user|>  hi there      <|end|>  <|assistant|>  hello!  <|end|> ...
targets: hi there <|end|>  <|assistant|>  hello!  <|end|>  ...
loss   :   ×       ×         ×           ✓       ✓      ✓
```

Exactly what receives gradient:

- every **reply content** token of an assistant turn,
- the **`<|end|>`** that closes a reply (the model must learn to stop).

Everything else — user text, `<|user|>`/`<|assistant|>` tags, `<|system|>`
blocks, padding, and the speaker tag that *follows* a reply — has zero loss.
The reply content tokens have arbitrary ids, so membership cannot be decided
from a single id: the mask **scans** the input tokens, switching on at each
`<|assistant|>` opener and off at each `<|end|>`. The opener itself is always
given at generation time (never predicted), and it sits in the *input* of the
step that predicts the reply's first token — which is exactly what the scan
exploits.

The scan lives in `assistant_target_mask` below and is applied inside
`TinyGPT.forward` (§3) whenever the model was built with
`assistant_token_ids=(assistant_id, end_id, pad_id)`; with `None` it degrades
to the plain notebook-06 language-model loss.''', "c8")

code(r'''TAG = re.compile(r"<\|(\w+)\|>\s*([^<]*)")


def parse_line(line):
    """One data line -> ([{"role": ..., "content": ...}, ...], system or None)."""
    messages, system = [], None
    for role, content in TAG.findall(line):
        content = content.strip()
        if role == "end" or not content:
            continue
        if role == "system":
            system = content
        else:
            messages.append({"role": role, "content": content})
    return messages, system


def encode_file(path):
    """Every conversation -> a padded id tensor of length BLOCK_SIZE + 1."""
    examples, skipped = [], 0
    for line in read_lines(path):
        messages, system = parse_line(line)
        if len(messages) < 2:
            skipped += 1
            continue
        ids = build_ids(messages, system=system)
        if len(ids) < 4:
            skipped += 1
            continue
        ids = ids[: BLOCK_SIZE + 1]                    # truncate long talks
        ids = ids + [PAD_ID] * (BLOCK_SIZE + 1 - len(ids))   # pad short ones
        examples.append(torch.tensor(ids, dtype=torch.long))
    return examples, skipped


train_examples, n_skipped = encode_file(TRAIN_PATH)
val_examples, _ = encode_file(VAL_PATH)
print(f"train examples: {len(train_examples):,} (skipped {n_skipped})")
print(f"val   examples: {len(val_examples):,}")


def get_batch(source, batch_size=BATCH_SIZE):
    """Random conversations; x = ids[:-1], y = ids[1:] (next-token targets)."""
    ix = torch.randint(len(source), (batch_size,))
    x = torch.stack([source[i][:-1] for i in ix])
    y = torch.stack([source[i][1:] for i in ix])
    return x.to(device), y.to(device)


xb, yb = get_batch(train_examples)
print(f"x: {tuple(xb.shape)}  y: {tuple(yb.shape)}")


def assistant_target_mask(x, y, assistant_id, end_id, pad_id=None):
    """True where the target y[:, t] belongs to an assistant reply.

    A scan over x (the same tokens shifted by one) tracks span membership:
    an <|assistant|> opener switches the span on, an <|end|> switches it off.
    Kept targets = reply content tokens + the closing <|end|> (the model must
    learn to stop) — but not the speaker tag that follows the reply, and never
    a <|pad|> target (conversations truncated by the context window would
    otherwise teach the model to emit padding).
    """
    B, T = x.shape
    keep = torch.empty(B, T, dtype=torch.bool)
    state = torch.zeros(B, dtype=torch.bool)     # inside an assistant span?
    for t in range(T):
        state = (state | (x[:, t] == assistant_id)) & (x[:, t] != end_id)
        keep[:, t] = state
    if pad_id is not None:
        keep &= y != pad_id
    return keep


ASSISTANT_ID = special_tokens["assistant"]
END_ID = special_tokens["end"]
PAD_ID = special_tokens["pad"]
span_ids = (ASSISTANT_ID, END_ID, PAD_ID)   # (opener, stop, pad) mask markers
mask = assistant_target_mask(xb, yb, *span_ids)
print(f"targets that receive loss: {int(mask.sum()):,} / {mask.numel():,} "
      f"({100 * mask.float().mean().item():.1f}%)")''', "c9")

code(r'''# Walk one row and watch the mask sort context from targets.
row = 0
ctx = tokenizer.decode(xb[row, :28].tolist()).replace("\n", " ")
print("first 28 x-tokens of row 0:", repr(ctx))
print()
for t in range(28):
    tgt = tokenizer.decode([int(yb[row, t])])
    keep = "loss" if bool(mask[row, t]) else "----"
    print(f"  pos {t:2d} | target {tgt!r:16} -> {keep}")''', "c10")

md(r'''## 3. Task 3 — the model

Same decoder-only stack as notebook 06, byte for byte:

    token ids
      -> token embedding + positional embedding
      -> N x TransformerBlock   (each causal, so the whole stack stays causal)
      -> final LayerNorm
      -> Linear projection to vocab logits   (weight tied to the embedding)

Position `t` compares its **query** with every key `t' <= t` and mixes the
values proportionally to the match; the `√d_k` scaling keeps dot products from
saturating the softmax, and the causal mask (upper triangle = −∞) hides the
future. Two implementation refinements carry over from 06: **one fused QKV
projection** instead of three separate ones, and
`F.scaled_dot_product_attention`, which applies scaling + causal mask + softmax
in a single hardware-accelerated call (`is_causal=True`).

Each Transformer block pairs communication (attention) with computation (FFN),
each wrapped in a residual and preceded by LayerNorm (pre-norm ordering) —
residuals mean every sub-layer only learns a small *correction*.

The **one functional change** to `TinyGPT`: an optional
`assistant_token_ids` argument. When given, the forward pass applies the
§2 loss mask; when `None` it degrades to the plain 06 language-model loss.''', "c11")

code(r'''class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention: fused QKV projection + one attention call."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # one projection for Q, K and V combined (3x = out projection)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)   # lets heads interact
        self.attn_drop = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)                                            # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)                                # (B, T, C) each
        # (B, T, C) -> (B, n_heads, T, d_head) for all three
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # scaling, causal masking and softmax in one fused call
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,          # the fused kernel builds the causal mask itself
            dropout_p=self.attn_drop if self.training else 0.0,
        )                                            # (B, n_heads, T, d_head)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-merge the heads
        return self.resid_drop(self.out_proj(y))


torch.manual_seed(0)
attn = CausalSelfAttention(d_model=128, n_heads=4)
x_demo = torch.randn(2, 32, 128)
print(f"attention: {tuple(x_demo.shape)} -> {tuple(attn(x_demo).shape)}")
print("QKV weights fused into one matrix:", tuple(attn.qkv.weight.shape))''', "c12")

md(r'''### 3.1 The Transformer block''', "c13")

code(r'''class TransformerBlock(nn.Module):
    """Communication (attention) + computation (FFN), both wrapped in residuals."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),   # wider inner layer (standard 4x)
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual 1
        x = x + self.ffn(self.ln2(x))    # residual 2
        return x


block = TransformerBlock(d_model=128, n_heads=4)
print(f"block: {tuple(x_demo.shape)} -> {tuple(block(x_demo).shape)}")
print("parameters in one block:", sum(p.numel() for p in block.parameters()))''', "c14")

md(r'''### 3.2 The full model: embeddings + stacked blocks + LM head (+ the mask)

Two embedding tables — one learned vector per **token** (what it is), one per
**position** (where it is) — feed N causal blocks; the final linear layer (LM
head) projects each position to vocabulary logits and is **weight-tied** to the
token-embedding matrix.

With `targets` given, the forward pass computes the mean cross-entropy over the
`B*T` next-token predictions — **masked** to assistant spans when
`assistant_token_ids` was provided at construction:

- `assistant_target_mask` (§2) marks the kept positions by scanning for
  `<|assistant|>` ... `<|end|>` spans,
- `F.cross_entropy(..., reduction="none")` scores every position,
- the mean is taken **only over the kept ones** (zero otherwise — exactly the
  `ignore_index`-style behaviour the task asks for).''', "c15")

code(r'''class TinyGPT(nn.Module):
    """A minimal decoder-only transformer (GPT-style) with optional chat masking."""

    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4,
                 block_size=128, dropout=0.1, assistant_token_ids=None):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.lm_head.weight = self.tok_emb.weight   # weight tying

        # (assistant_id, end_id, pad_id) mask markers -> loss masking (§2);
        # None -> plain LM loss (notebook 06)
        self.assistant_ids = (None if assistant_token_ids is None
                              else torch.as_tensor(assistant_token_ids))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        # GPT-2 style init: small normal (std=0.02) weights, zero biases.
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"sequence of {T} exceeds context {self.block_size}"
        pos = torch.arange(T, device=idx.device)

        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))   # (B, T, C)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)                               # (B, T, vocab)
        loss = None
        if targets is not None:
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_targets = targets.reshape(-1)
            per_tok = F.cross_entropy(flat_logits, flat_targets,
                                      reduction="none")
            if self.assistant_ids is None:                     # notebook-06 mode
                loss = per_tok.mean()
            else:                                              # chat mode (§2)
                # uses the assistant_target_mask helper from §2 (shared
                # namespace); never called during generation (no targets there)
                assistant_id, end_id, pad_id = self.assistant_ids.tolist()
                keep = assistant_target_mask(idx, targets, assistant_id,
                                             end_id, pad_id).reshape(-1)
                loss = (per_tok[keep].mean() if keep.any()
                        else torch.zeros((), device=idx.device))
        return logits, loss''', "c16")

code(r'''model = TinyGPT(vocab_size=tokenizer.vocab_size, d_model=128, n_heads=4,
                n_layers=4, block_size=BLOCK_SIZE, dropout=0.1,
                assistant_token_ids=span_ids).to(device)

n_params = sum(p.numel() for p in model.parameters())
n_unique = n_params - model.tok_emb.weight.numel()   # the tied table is counted twice
print(f"Parameters (tied table counted twice): {n_params:,}")
print(f"Parameters (unique):                   {n_unique:,}")

with torch.no_grad():
    logits, loss = model(xb, yb)
print(f"logits: {tuple(logits.shape)}   masked initial loss: {loss.item():.4f}")

# --- verify the mask math against a hand-rolled version ---------------------
with torch.no_grad():
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = yb.reshape(-1)
    keep = assistant_target_mask(xb, yb, *span_ids).reshape(-1)
    per_tok = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    manual = per_tok[keep].mean()
print(f"hand-rolled masked loss: {manual.item():.4f}")
assert torch.allclose(loss, manual, atol=1e-5)
print("masked cross-entropy == manual mean over assistant targets: OK")''', "c17")

md(r'''### 3.3 A quick causality check

Same property worth asserting before training as in 06: the mask must make
**future tokens invisible**. Perturbing positions after `t` must leave every
prediction at position `<= t` unchanged.''', "c18")

code(r'''model.eval()   # disable dropout so the comparison is deterministic
with torch.no_grad():
    seq = xb[:1, :32]
    base = model(seq)[0]
    perturbed = seq.clone()
    perturbed[0, 16:] = torch.roll(perturbed[0, 16:], shifts=1)   # corrupt the future
    changed = model(perturbed)[0]
    past_same = torch.allclose(base[0, :16], changed[0, :16], atol=1e-5)
    future_diff = not torch.allclose(base[0, 16:], changed[0, 16:], atol=1e-5)
print("changing the future leaves past predictions unchanged:", past_same)
print("changing the future does change its own predictions:     ", future_diff)
assert past_same and future_diff
model.train()''', "c19")

md(r'''## 4. Task 4 — training loop

Same recipe as 06: AdamW, gradient clipping, periodic train/val estimates over
fresh batches — no epochs, no shuffling; every step samples 64 random
conversations. Two differences from the Shakespeare run:

- **the loss is masked**, so each step trains on roughly a third of the
  positions (the assistant spans) — the model sees *all* of every conversation
  as context, but only learns to *produce* assistant turns;
- the data is far noisier than verse, so expect a higher val loss floor than
  notebook 06's ~2.6 — for a 4-layer model at this scale anything below ~3.2
  nats per assistant token is a sensible result.

`MAX_STEPS = 4000` is roughly 35–45 minutes on CPU; drop it to ~500 for a
smoke run. Progress is checkpointed at every eval, so a long run can be
resumed by re-running this cell.''', "c20")

code(r'''LEARNING_RATE = 3e-4
MAX_STEPS = 4000       # ~15-25 min on CPU; ~500 for a quick smoke run
EVAL_INTERVAL = 500
EVAL_ITERS = 50        # batches averaged for a more stable loss estimate

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
print(f"Training on conversations of {BLOCK_SIZE} tokens, batch of {BATCH_SIZE}")''', "c21")

code(r'''@torch.no_grad()
def estimate_loss(model, iters=EVAL_ITERS):
    """Average masked loss over fresh batches - a less noisy read on progress."""
    model.eval()
    out = {}
    for name, source in (("train", train_examples), ("val", val_examples)):
        losses = torch.zeros(iters)
        for k in range(iters):
            xb, yb = get_batch(source, BATCH_SIZE)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


# A long run can outlive a single session (or a Colab disconnect), so progress
# is checkpointed at every eval: re-running this cell (or the whole notebook)
# picks up exactly where it left off instead of starting over.
TRAIN_STATE_PATH = str(MODEL_DIR / "07_train_state.pt")
os.makedirs(MODEL_DIR, exist_ok=True)

train_losses, val_losses = [], []
start_step = 1
if os.path.exists(TRAIN_STATE_PATH):
    state = torch.load(TRAIN_STATE_PATH, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    train_losses = state["train_losses"]
    val_losses = state["val_losses"]
    start_step = state["step"] + 1
    print(f"resuming training from step {state['step']}")
else:
    print(f"training fresh for {MAX_STEPS} steps")
model.train()

for step in range(start_step, MAX_STEPS + 1):
    xb, yb = get_batch(train_examples, BATCH_SIZE)

    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # keep updates sane
    optimizer.step()

    train_losses.append(loss.item())

    if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
        losses = estimate_loss(model)
        val_losses.append(losses["val"])
        print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "step": step,
            },
            TRAIN_STATE_PATH,
        )''', "c22")

md(r'''## 5. Task 5 — save (and reload) a self-contained checkpoint

The checkpoint stores the **state dict** (the weights), the **config** (all
hyper-parameters *and* the special-token map), the **tokenizer itself** (as its
serialised JSON — so no retraining or vocab drift on reload), and the final
metrics. Everything is plain tensors/config/strings, so the file loads safely
with `weights_only=True` elsewhere.''', "c23")

code(r'''SAVE_DIR = str(MODEL_DIR)
SAVE_PATH = os.path.join(SAVE_DIR, "07_conversational_lm.pt")
os.makedirs(SAVE_DIR, exist_ok=True)

final_train = train_losses[-100:]     # last-100 average: a calmer "final" figure
final_val = val_losses[-1]
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": tokenizer.vocab_size,
            "d_model": 128,
            "n_heads": 4,
            "n_layers": 4,
            "block_size": BLOCK_SIZE,
            "dropout": 0.1,
            "special_tokens": special_tokens,   # the special-token map itself
        },
        "tokenizer_json": tokenizer.tok.to_str(),
        "train_loss": sum(final_train) / len(final_train),
        "val_loss": final_val,
        "steps": MAX_STEPS,
    },
    SAVE_PATH,
)
print(f"Saved model to: {SAVE_PATH}")
print(f"file size: {os.path.getsize(SAVE_PATH) / 1e6:.1f} MB")

if os.path.exists(TRAIN_STATE_PATH):
    os.remove(TRAIN_STATE_PATH)   # training complete — a future run starts fresh''', "c24")

md(r'''To prove the checkpoint is self-sufficient: rebuild the tokenizer **from the
saved JSON**, rebuild the model **from the saved config**, load the weights,
and check the reloaded model produces identical logits.''', "c25")

code(r'''checkpoint = torch.load(SAVE_PATH, map_location=device, weights_only=True)
print("keys in checkpoint:", list(checkpoint.keys()))
print("checkpoint config:", checkpoint["config"])
print(f"recorded metrics: train {checkpoint['train_loss']:.4f} | "
      f"val {checkpoint['val_loss']:.4f} after {checkpoint['steps']} steps")

cfg = checkpoint["config"]
st = cfg["special_tokens"]
SYSTEM_ID, USER_ID, ASSISTANT_ID, END_ID, PAD_ID = (
    st[k] for k in ("system", "user", "assistant", "end", "pad")
)
span_ids = (st["assistant"], st["end"], st["pad"])

tokenizer = ChatBPE.from_json(checkpoint["tokenizer_json"])
print("reloaded tokenizer vocab:", tokenizer.vocab_size)

reloaded = TinyGPT(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
                   n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
                   block_size=cfg["block_size"], dropout=cfg["dropout"],
                   assistant_token_ids=span_ids).to(device)
reloaded.load_state_dict(checkpoint["model_state_dict"])

model.eval()
reloaded.eval()
with torch.no_grad():
    same = torch.allclose(model(xb[:8])[0], reloaded(xb[:8])[0], atol=1e-5)
print("reloaded model logits match the trained model:", same)
assert same''', "c26")

md(r'''## 6. Task 6 — inference: a multi-turn chat loop

Generation changes character completely. Instead of one prompt → one long
continuation, the loop:

1. keeps a list of `{"role": ..., "content": ...}` messages,
2. formats the **full history** through `format_chat(..., add_assistant_opener=True)`,
3. samples **one token at a time** until it emits `<|end|>` (the stop token it
   was trained to place after every reply) — padding is banned and top-k keeps
   the samples sensible,
4. decodes just the new tokens into the reply and appends it to the history,
5. **truncates the oldest tokens** whenever the formatted history outgrows
   `block_size` (minus room reserved for the reply).

The system prompt, if any, is prepended to every conversation.''', "c27")

code(r'''@torch.no_grad()
def generate_reply(model, tokenizer, messages, system=None,
                   max_new_tokens=64, temperature=0.8, top_k=40):
    """Generate ONE assistant reply for the conversation so far.

    messages: the full history [{"role": ..., "content": ...}, ...]
    Returns the reply text: the tokens sampled between the assistant opener
    and the terminating <|end|>.
    """
    model.eval()
    dev = next(model.parameters()).device
    assistant_id = tokenizer.id("<|assistant|>")
    end_id = tokenizer.id("<|end|>")
    pad_id = tokenizer.id("<|pad|>")

    prompt_ids = build_ids(messages, system=system, add_assistant_opener=True)
    # Truncation: keep the newest tokens, but always leave room for the reply.
    # (The opener is the last id, so it survives the crop by construction.)
    prompt_ids = prompt_ids[-(model.block_size - max_new_tokens):]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=dev)

    new_ids = []
    for _ in range(max_new_tokens):
        logits, _ = model(idx)
        logits = logits[:, -1, :] / temperature      # only the LAST position matters
        logits[:, pad_id] = float("-inf")            # never emit padding

        if top_k is not None:                        # keep only the top-k logits
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

        tok = int(next_id)
        if tok == end_id:                            # the model says "my turn is done"
            break
        new_ids.append(tok)

    return tokenizer.decode(new_ids).strip()''', "c28")

md(r'''### 6.1 Demo — two to three scripted turns

The demo cell shows the loop end-to-end: history grows turn by turn, each
`generate_reply` call re-formats *everything* and generates only the next
assistant message.''', "c29")

code(r'''script = [
    "hello! how are you today?",
    "what do you like to do for fun?",
    "nice! do you have a favorite movie?",
]
history = []
for user_text in script:
    history.append({"role": "user", "content": user_text})
    reply = generate_reply(model, tokenizer, history,
                           system="You are a friendly chatbot.")
    history.append({"role": "assistant", "content": reply})
    print(f"user: {user_text}")
    print(f"bot : {reply}")
    print()''', "c30")

md(r'''### 6.2 The interactive chat loop

`chat()` is the same loop with a REPL skin. The full message history lives in
`history`; truncation to `block_size` happens inside `generate_reply`, so very
long chats simply forget their oldest turns.

It uses `input()`, which notebooks only support while a cell is running —
that is what the `INTERACTIVE` flag guards.''', "c31")

code(r'''def chat(system="You are a friendly chatbot."):
    """Interactive multi-turn chat: type 'quit' to exit, 'reset' to start over."""
    history = []
    print("chat started — type 'quit' to exit, 'reset' to start over")
    while True:
        try:
            user_text = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_text.lower() in {"quit", "exit", "q"}:
            break
        if user_text.lower() == "reset":
            history = []
            print("(conversation reset)")
            continue
        if not user_text:
            continue
        history.append({"role": "user", "content": user_text})
        reply = generate_reply(model, tokenizer, history, system=system)
        history.append({"role": "assistant", "content": reply})
        print(f"bot: {reply}")


INTERACTIVE = False   # set True (or just call chat()) for a live loop
if INTERACTIVE:
    chat()
else:
    print("INTERACTIVE is False — call chat() in a cell to talk to the model live.")''', "c32")

md(r'''## 7. Exercises

1. **Mask on / mask off**: rebuild the model with `assistant_token_ids=None`
   and train the same number of steps. How does the val loss compare (and is
   it even comparable?) — what does the model start imitating?
2. **System prompt dependence**: the ConvAI2 data carries a persona as the
   system prompt. Ask the reloaded model "what are your hobbies?" with and
   without a matching persona and compare the replies.
3. **Vocabulary size**: retrain with `VOCAB_SIZE = 300` vs 1024 — the
   initial loss `ln(vocab)` moves, but what happens to sequence length and
   reply quality?
4. **Context length**: raise `BLOCK_SIZE` to 256 (and `MAX_TURNS_PER_CONV` in
   `07_preprocess.py` accordingly) — do longer histories improve coherence?
5. **Sampling**: compare `top_k=None` vs `top_k=10` at the same temperature.
6. **Greedy decoding**: set `temperature` very low (or take `argmax`) — when
   does the model degenerate into loops?

### What changed vs notebook 06

| | 06 — tiny GPT | 07 — conversational LM |
|---|---|---|
| data | one flat Shakespeare stream | ~30k multi-turn conversations |
| special tokens | none | `<|system|> <|user|> <|assistant|> <|end|> <|pad|>` |
| loss | every position | assistant spans only (masking) |
| target of training | continue any text | produce one reply, then stop |
| generation | free continuation | chat loop, stops at `<|end|>`, trims history |
| checkpoint | weights + config | weights + config + tokenizer (self-contained) |''', "c33")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.14"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("07_conversational_lm.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"wrote 07_conversational_lm.ipynb with {len(cells)} cells")
