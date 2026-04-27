"""
autoresearch-mythos: data preparation & runtime utilities.
Downloads MathNet olympiad dataset, tokenizes, saves binary shards.
This file is READ-ONLY — the agent modifies only train.py.

One-time setup:  python prepare.py
"""

import os
import math
import numpy as np

# ---------------------------------------------------------------------------
# Constants (fixed — do not modify)
# ---------------------------------------------------------------------------

DATA_DIR = os.path.expanduser("~/.cache/autoresearch-mythos")
TIME_BUDGET = 300          # 5 minutes training wall-clock
MAX_SEQ_LEN = 512          # context length
EVAL_TOKENS = 50_000       # tokens used for validation eval
VAL_RATIO = 0.05           # fraction held out for validation

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class Tokenizer:
    """Wraps tiktoken (gpt2). Falls back to byte-level if unavailable."""

    def __init__(self):
        try:
            import tiktoken
            self.enc = tiktoken.get_encoding("gpt2")
            self._vocab_size = self.enc.n_vocab
            self._kind = "tiktoken"
        except ImportError:
            self.enc = None
            self._vocab_size = 256
            self._kind = "byte"

    def encode(self, text: str):
        if self._kind == "tiktoken":
            return self.enc.encode(text, allowed_special="all")
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        if self._kind == "tiktoken":
            return self.enc.decode(list(tokens))
        return bytes(tokens).decode("utf-8", errors="replace")

    def get_vocab_size(self) -> int:
        return self._vocab_size

    @classmethod
    def from_directory(cls):
        return cls()

# ---------------------------------------------------------------------------
# Data formatting
# ---------------------------------------------------------------------------

def _format_sample(row: dict) -> str:
    """Turn one MathNet row into a training string."""
    prob = row.get("problem_markdown", "") or ""
    sols = row.get("solutions_markdown", None) or []
    sol = sols[0] if sols else ""
    topics = row.get("topics_flat", None) or []
    topic_str = "; ".join(topics) if topics else ""
    parts = [f"<|problem|>\n{prob}\n"]
    if topic_str:
        parts.append(f"<|topics|> {topic_str}\n")
    parts.append(f"<|solution|>\n{sol}\n<|end|>\n")
    return "".join(parts)

# ---------------------------------------------------------------------------
# One-time data preparation
# ---------------------------------------------------------------------------

def prepare():
    os.makedirs(DATA_DIR, exist_ok=True)
    train_path = os.path.join(DATA_DIR, "train.bin")
    val_path   = os.path.join(DATA_DIR, "val.bin")
    meta_path  = os.path.join(DATA_DIR, "meta.txt")

    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"Data already prepared in {DATA_DIR}")
        return

    print("Downloading MathNet dataset (streaming) ...")
    from datasets import load_dataset
    ds = load_dataset("ShadenA/MathNet", split="train", streaming=True)

    tok = Tokenizer()
    print(f"Tokenizer: {tok._kind}  vocab_size={tok._vocab_size}")

    all_tokens = []
    n_samples = 0
    n_bytes = 0

    for row in ds:
        text = _format_sample(row)
        n_bytes += len(text.encode("utf-8"))
        tokens = tok.encode(text)
        all_tokens.extend(tokens)
        n_samples += 1
        if n_samples % 5000 == 0:
            print(f"  {n_samples} samples  |  {len(all_tokens):,} tokens")

    print(f"Done: {n_samples} samples  |  {len(all_tokens):,} tokens  |  {n_bytes:,} bytes")

    bpt = n_bytes / len(all_tokens) if all_tokens else 1.0

    arr = np.array(all_tokens, dtype=np.uint16)
    n_val = int(len(arr) * VAL_RATIO)
    val_arr   = arr[-n_val:]
    train_arr = arr[:-n_val]

    train_arr.tofile(train_path)
    val_arr.tofile(val_path)

    with open(meta_path, "w") as f:
        f.write(f"bytes_per_token={bpt:.6f}\n")
        f.write(f"n_samples={n_samples}\n")
        f.write(f"total_tokens={len(arr)}\n")

    print(f"Saved  train: {len(train_arr):,} tokens → {train_path}")
    print(f"Saved  val:   {len(val_arr):,} tokens → {val_path}")
    print(f"bytes_per_token = {bpt:.4f}")

# ---------------------------------------------------------------------------
# Runtime: dataloader (returns numpy arrays — MLX converts on use)
# ---------------------------------------------------------------------------

def make_dataloader(tokenizer, batch_size, seq_len, split):
    """Infinite generator yielding (x, y, epoch) from memory-mapped tokens."""
    path = os.path.join(DATA_DIR, f"{split}.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    epoch = 0
    while True:
        ix = np.random.randint(0, len(data) - seq_len - 1, size=(batch_size,))
        x = np.stack([data[i   : i + seq_len    ].astype(np.int32) for i in ix])
        y = np.stack([data[i+1 : i + seq_len + 1].astype(np.int32) for i in ix])
        yield x, y, epoch
        epoch += 1

# ---------------------------------------------------------------------------
# Runtime: evaluation (MLX-native)
# ---------------------------------------------------------------------------

def _read_bpt() -> float:
    meta = os.path.join(DATA_DIR, "meta.txt")
    if os.path.exists(meta):
        for line in open(meta):
            if line.startswith("bytes_per_token="):
                return float(line.split("=")[1])
    return 1.0

def evaluate_bpb(model, tokenizer, batch_size):
    """Compute validation bits-per-byte (lower is better, vocab-independent)."""
    import mlx.core as mx
    import mlx.nn as nn

    path = os.path.join(DATA_DIR, "val.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    bpt  = _read_bpt()

    n_batches = max(1, min(EVAL_TOKENS, len(data) - MAX_SEQ_LEN - 1)
                    // (batch_size * MAX_SEQ_LEN))
    total_loss = 0.0
    total_toks = 0

    for _ in range(n_batches):
        ix = np.random.randint(0, len(data) - MAX_SEQ_LEN - 1, size=(batch_size,))
        x = mx.array(np.stack([data[i:i+MAX_SEQ_LEN].astype(np.int32) for i in ix]))
        y = mx.array(np.stack([data[i+1:i+MAX_SEQ_LEN+1].astype(np.int32) for i in ix]))

        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="sum")
        mx.eval(loss)
        total_loss += float(loss.item())
        total_toks += y.size

    avg_nats = total_loss / max(total_toks, 1)
    bpb = avg_nats / math.log(2) * bpt
    return bpb


def evaluate_bpb_torch(model, tokenizer, batch_size, device):
    """PyTorch version of evaluate_bpb for CUDA training."""
    import torch

    path = os.path.join(DATA_DIR, "val.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    bpt  = _read_bpt()

    n_batches = max(1, min(EVAL_TOKENS, len(data) - MAX_SEQ_LEN - 1)
                    // (batch_size * MAX_SEQ_LEN))
    total_loss = 0.0
    total_toks = 0

    with torch.no_grad():
        for _ in range(n_batches):
            ix = np.random.randint(0, len(data) - MAX_SEQ_LEN - 1, size=(batch_size,))
            x = torch.from_numpy(np.stack([data[i:i+MAX_SEQ_LEN].astype(np.int64) for i in ix])).to(device)
            y = torch.from_numpy(np.stack([data[i+1:i+MAX_SEQ_LEN+1].astype(np.int64) for i in ix])).to(device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum"
            )
            total_loss += loss.item()
            total_toks += y.numel()

    avg_nats = total_loss / max(total_toks, 1)
    bpb = avg_nats / math.log(2) * bpt
    return bpb

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    prepare()
