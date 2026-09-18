"""07 — Dataset preparation for the conversational language model.

Converts the three cloned dialogue corpora into one simple turn format:

    <|system|> optional system text <|end|>
    <|user|> ... <|end|>
    <|assistant|> ... <|end|>
    ...

and writes one conversation per line to:

    data/dialogues_train.txt
    data/dialogues_val.txt

Sources (raw files must exist under repositories/_raw/ — see README cell in the
notebook for the download commands):
  1. Cornell Movie-Dialogs Corpus   (movie scripts, latin-1 "+++$+++" files)
  2. EmpatheticDialogues            (csv, `_comma_` placeholders)
  3. ConvAI2 / PersonaChat          (ParlAI text format, "your persona: ...")
  4. (optional) Tiny Shakespeare    (single-speaker "assistant" turns only,
                                     so the model keeps a little literary style)

Run from the project root:  python 07_preprocess.py
"""

import csv
import os
import random
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
RAW = os.path.join(ROOT, "repositories", "_raw")
OUT_DIR = os.path.join(ROOT, "data")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
MAX_TURN_CHARS = 280      # skip pathological monologue turns
MAX_TURNS_PER_CONV = 10   # keep conversations inside a 128-token context
VAL_FRACTION = 0.05

# Rough turn budgets per source (target: 50k-200k turns in total).
CAPS = {
    "cornell": 75_000,
    "empathetic": 45_000,
    "convai2": 55_000,
    "shakespeare": 12_000,
}

SPECIALS = ("<|system|>", "<|user|>", "<|assistant|>", "<|end|>")


# --------------------------------------------------------------------------- #
# Cleaning helpers
# --------------------------------------------------------------------------- #
def clean_text(text):
    """Normalize one utterance; return None if it is not worth keeping."""
    s = text.replace("_comma_", ",")            # EmpatheticDialogues quirk
    s = " ".join(s.split())                     # collapse all whitespace
    if len(s) < 2 or not any(c.isalnum() for c in s):
        return None
    if len(s) > MAX_TURN_CHARS:
        return None
    if any(sp in s for sp in SPECIALS):         # never leak the template
        return None
    if "__" in s:                               # __SILENCE__ and friends
        return None
    return s


def build_conversation(turns, min_assistant_turns=2):
    """turns: list of (role, text). Merge same-role neighbours, validate."""
    turns = [(r, t) for r, t in turns if t]
    if not turns:
        return None
    merged = [list(turns[0])]
    for role, text in turns[1:]:
        if role == merged[-1][0]:
            merged[-1][1] += " " + text          # same speaker kept talking
        else:
            merged.append([role, text])
    roles = [r for r, _ in merged]
    if "assistant" not in roles or "user" not in roles:
        return None
    if roles.count("assistant") < min_assistant_turns:
        return None
    return [(r, t) for r, t in merged[:MAX_TURNS_PER_CONV]]


def conversation_to_line(conv, system=None):
    parts = []
    if system:
        parts.append(f"<|system|> {system} <|end|>")
    for role, text in conv:
        parts.append(f"<|{role}|> {text} <|end|>")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Source 1 — Cornell Movie-Dialogs
# --------------------------------------------------------------------------- #
def load_cornell(limit_turns):
    base = os.path.join(RAW, "cornell", "cornell movie-dialogs corpus")
    lines_path = os.path.join(base, "movie_lines.txt")
    convs_path = os.path.join(base, "movie_conversations.txt")

    line_text = {}      # line id -> (character id, cleaned text)
    with open(lines_path, "r", encoding="latin-1") as f:
        for row in f:
            parts = row.split("+++$+++")
            if len(parts) < 5:
                continue
            line_id = parts[0].strip()
            char_id = parts[1].strip()
            text = clean_text(parts[4])
            if text:
                line_text[line_id] = (char_id, text)

    raw_convs = []
    with open(convs_path, "r", encoding="latin-1") as f:
        for row in f:
            parts = row.split("+++$+++")
            if len(parts) < 4:
                continue
            c1, c2 = parts[0].strip(), parts[1].strip()
            try:
                line_ids = eval(parts[3].strip())   # literal list of ids
            except Exception:
                continue
            turns = []
            for lid in line_ids:
                if lid not in line_text:
                    break
                char_id, text = line_text[lid]
                if char_id not in (c1, c2):
                    break
                turns.append((char_id, text))
            if len(turns) < 4:
                continue
            speaker_a = turns[0][0]
            turns = [("user" if cid == speaker_a else "assistant", t)
                     for cid, t in turns]
            conv = build_conversation(turns)
            if conv:
                raw_convs.append(conv)

    rng = random.Random(SEED)
    rng.shuffle(raw_convs)
    out, acc = [], 0
    for conv in raw_convs:
        if acc >= limit_turns:
            break
        out.append(conv)
        acc += len(conv)
    return out


# --------------------------------------------------------------------------- #
# Source 2 — EmpatheticDialogues
# --------------------------------------------------------------------------- #
def load_empathetic(limit_turns):
    folder = os.path.join(RAW, "empatheticdialogues")
    by_conv = {}
    for fname in ("train.csv", "valid.csv"):
        path = os.path.join(folder, fname)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                text = clean_text(row["utterance"])
                if not text:
                    continue
                key = row["conv_id"]
                by_conv.setdefault(key, {"first_speaker": row["speaker_idx"],
                                         "turns": [],
                                         "context": row["context"],
                                         "prompt": row["prompt"]})
                info = by_conv[key]
                role = "user" if row["speaker_idx"] == info["first_speaker"] else "assistant"
                info["turns"].append((role, text))

    system_prefix = ("You are chatting with someone. "
                     "Situation: {context}: {prompt}")
    raw = []
    for info in by_conv.values():
        conv = build_conversation(info["turns"])
        if conv:
            sys_text = clean_text(system_prefix.format(
                context=info["context"], prompt=info["prompt"]))
            raw.append((conv, sys_text))

    rng = random.Random(SEED)
    rng.shuffle(raw)
    out, acc = [], 0
    for conv, sys_text in raw:
        if acc >= limit_turns:
            break
        out.append((conv, sys_text))
        acc += len(conv)
    return out


# --------------------------------------------------------------------------- #
# Source 3 — ConvAI2 / PersonaChat (ParlAI text format)
# --------------------------------------------------------------------------- #
def load_convai2(limit_turns):
    files = [os.path.join(RAW, "train_self_original_no_cands.txt"),
             os.path.join(RAW, "valid_self_original_no_cands.txt")]
    raw = []
    for path in files:
        if not os.path.exists(path):
            continue
        persona, dialog = [], []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "your persona:" in line:
                    if dialog:                       # persona block ended
                        conv = build_conversation(dialog)
                        if conv and persona:
                            sys_text = clean_text(
                                "You are chatting with a stranger. "
                                "Your persona: " + " ".join(persona[:4]))
                            if sys_text:
                                raw.append((conv, sys_text))
                        persona, dialog = [], []
                    persona.append(line.split("your persona:", 1)[1].strip())
                else:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        # ParlAI lines start with the turn number: "12 hi there ..."
                        user_uttr = clean_text(parts[0].split(" ", 1)[-1])
                        asst_uttr = clean_text(parts[1].split(" ", 1)[-1])
                        if user_uttr:
                            dialog.append(("user", user_uttr))
                        if asst_uttr:
                            dialog.append(("assistant", asst_uttr))
        if dialog:
            conv = build_conversation(dialog)
            if conv and persona:
                sys_text = clean_text("You are chatting with a stranger. "
                                      "Your persona: " + " ".join(persona[:4]))
                if sys_text:
                    raw.append((conv, sys_text))

    rng = random.Random(SEED)
    rng.shuffle(raw)
    out, acc = [], 0
    for conv, sys_text in raw:
        if acc >= limit_turns:
            break
        out.append((conv, sys_text))
        acc += len(conv)
    return out


# --------------------------------------------------------------------------- #
# Source 4 — Tiny Shakespeare (single-speaker assistant turns, optional)
# --------------------------------------------------------------------------- #
def load_shakespeare(limit_turns):
    for path in (os.path.join(ROOT, "dataset", "input.txt"),
                 os.path.join(ROOT, "dataset", "input.txt").replace(ROOT, ".")):
        if os.path.exists(path):
            break
    else:
        return []
    with open(os.path.join(ROOT, "dataset", "input.txt"), "r", encoding="utf-8") as f:
        text = f.read()

    blocks = []
    for block in text.split("\n\n"):
        clean = clean_text(block)
        if clean:
            # strip a leading speaker label like "First Citizen:"
            head, sep, rest = clean.partition(": ")
            if sep and len(head) < 40 and ":" not in rest[:40]:
                clean = rest
            clean = clean_text(clean)
            if clean and len(clean) > 40:
                blocks.append(clean)

    # chunk 1-3 consecutive blocks into one "conversation" of assistant turns
    rng = random.Random(SEED)
    rng.shuffle(blocks)
    convs, current, acc = [], [], 0
    for block in blocks:
        current.append(("assistant", block))
        acc += 1
        if len(current) == 3:
            convs.append(list(current))
            current, acc_turn = [], 0
            current = []
        if acc >= limit_turns:
            break
    if current:
        convs.append(current)
    return [(conv, None) for conv in convs]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("Loading Cornell Movie Dialogs ...", flush=True)
    cornell = load_cornell(CAPS["cornell"])
    print(f"  conversations: {len(cornell):,}", flush=True)

    print("Loading EmpatheticDialogues ...", flush=True)
    empathetic = load_empathetic(CAPS["empathetic"])
    print(f"  conversations: {len(empathetic):,}", flush=True)

    print("Loading ConvAI2 / PersonaChat ...", flush=True)
    convai2 = load_convai2(CAPS["convai2"])
    print(f"  conversations: {len(convai2):,}", flush=True)

    print("Loading Shakespeare (optional) ...", flush=True)
    shakespeare = load_shakespeare(CAPS["shakespeare"])
    print(f"  conversations: {len(shakespeare):,}", flush=True)

    all_convs = []
    all_convs += [(c, None) for c in cornell]
    all_convs += empathetic
    all_convs += convai2
    all_convs += shakespeare

    rng = random.Random(SEED)
    rng.shuffle(all_convs)

    n_val = max(1, int(VAL_FRACTION * len(all_convs)))
    val, train = all_convs[:n_val], all_convs[n_val:]

    for name, split in (("dialogues_train.txt", train), ("dialogues_val.txt", val)):
        path = os.path.join(OUT_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            for conv, system in split:
                f.write(conversation_to_line(conv, system) + "\n")
        turns = sum(len(c) for c, _ in split)
        size_mb = os.path.getsize(path) / 1e6
        print(f"wrote {path}  ({len(split):,} conversations, "
              f"{turns:,} turns, {size_mb:.1f} MB)")

    total_turns = sum(len(c) for c, _ in all_convs)
    print(f"TOTAL: {len(all_convs):,} conversations, {total_turns:,} turns")


if __name__ == "__main__":
    sys.exit(main())
