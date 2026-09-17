# AGENTS.md — Convert 06_tiny_GPT into a Conversational LM

## Goal
- Take the existing notebook `06_tiny_GPT.ipynb` (TinyGPT + BPE on Shakespeare) and produce a **new** conversational language model.  
- Do **not** fine-tune the old checkpoint. Train a fresh model from scratch on dialogue data.  
- Save the final checkpoint as `07_conversational_lm.pt`.
- You are also not to modify `06_tiny_GPT.ipynb` but write a new notebook `07_conversational_lm.ipynb` that uses most logic from `06_tiny_GPT.ipynb`, modified to create the conversational language model.

## High-level plan
1. Prepare a clean multi-turn dialogue dataset from the three cloned repositories (under repositories folder) + optional Shakespeare.
2. Extend the tokenizer with special chat tokens and define a chat template.
3. Build a new `TinyGPT` (same architecture family) and train it with **loss masking** (only assistant tokens contribute to the loss).
4. Replace the old generation cell with an interactive multi-turn chat loop.
5. Save a self-contained checkpoint.

Keep the educational style of the original notebook: clear markdown, small model, revisional comments, runnable on CPU/Colab, no external heavy libraries beyond what is already used (`tokenizers`, `torch`).

---

## 1. Dataset preparation

Use the already-cloned repositories:
- Cornell Movie Dialogs
- EmpatheticDialogues
- PersonaChat / ConvAI2

**Requirements**
- Convert each corpus into the same simple turn format:
  ```
  <|user|> ... <|end|>
  <|assistant|> ... <|end|>
  ```
- Keep conversations multi-turn when the source provides them.
- Target size: aim for 50k–200k turns total (mix the three sources). Smaller is fine for a first working version.
- Optionally concatenate a portion of the original Shakespeare text (formatted as single-speaker “assistant” turns) so the model retains some literary style.
- Produce two files:
  - `data/dialogues_train.txt`
  - `data/dialogues_val.txt`
  (plain text, one full conversation per line or separated by blank lines).

Write a short preprocessing script (or notebook cells) that:
- Loads the raw files from the three repos,
- Extracts speaker turns,
- Maps them to `user` / `assistant` roles,
- Applies the special-token template,
- Writes the train/val splits.

## 2. Tokenizer & chat template (mandatory)

- Start from the same byte-level BPE approach used in 06_tiny_GPT.
- **Add** these special tokens before training the tokenizer (or add them and resize embeddings later):
  ```
  <|system|>  <|user|>  <|assistant|>  <|end|>
  ```
- Vocab size can stay ~300–500 or be raised modestly (e.g. 512–1024) if the dialogue data benefits.
- Provide a helper:
  ```python
  def format_chat(messages: list[dict]) -> str:
      """messages = [{"role": "user"|"assistant"|"system", "content": str}, ...]"""
  ```
  that produces the exact token sequence the model will see.

## 3. Model & training

- Instantiate a **new** `TinyGPT` (do not load the 06 weights).
- Keep the architecture close to the original (d_model=128, a few layers, causal attention, weight tying) so it stays fast to train.
- **Loss masking (critical)**:
  - When computing cross-entropy, set the loss of every non-assistant token to zero (or ignore_index).
  - Only tokens that belong to `<|assistant|> ... <|end|>` spans contribute to the gradient.
- Training loop:
  - Same style as 06 (random chunks, AdamW, gradient clip, periodic train/val loss).
  - Use the new dialogue files.
  - Context length (`BLOCK_SIZE`) ≥ 128; raise if conversations are longer.
- After training, evaluate a few generated multi-turn examples.

## 4. Checkpoint

Save exactly:
```python
torch.save({
    "model_state_dict": model.state_dict(),
    "config": { ... all hyper-parameters including vocab_size, special token ids ... },
    "train_loss": ...,
    "val_loss": ...,
    "steps": ...,
}, "07_conversational_lm.pt")
```
The checkpoint must be loadable in a fresh session and sufficient to rebuild both the model and the tokenizer/special-token map.

## 5. Inference — chat loop

Replace the old single-prompt `generate` with an interactive loop that:
- Maintains a list of `{"role": ..., "content": ...}` messages,
- Formats the full history with the chat template,
- Generates only the next assistant reply (stop at `<|end|>`),
- Appends the reply to history,
- Truncates oldest turns when the token length exceeds `block_size`.

Support a short system prompt at the start of every conversation.

## 6. Deliverables

- Updated / new notebook (or clean script) that performs all steps above.
- `data/dialogues_train.txt` + `data/dialogues_val.txt`
- Final model file: `07_conversational_lm.pt`
- A short demo cell showing 2–3 turns of conversation.

## Constraints
- Prefer clarity and reproducibility over maximal performance.
- Stay within the spirit and dependencies of the original 06 notebook.
- Do not rely on the old Shakespeare-only checkpoint; train from scratch on the dialogue mix.