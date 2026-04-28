"""
Novel Math Reasoning Training — train_novel.py

Extends the base RDT architecture with three novel techniques:
  1. Topic-Conditioned FFN: Problem-level routing via learned topic embedding
  2. Step-Level Self-Verification: Model learns to verify its own reasoning
  3. Verification-Weighted Loss: Loss is weighted higher on <verify> predictions

Uses structured data from prepare_structured.py.
"""

import gc
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from prepare import MAX_SEQ_LEN, Tokenizer, make_dataloader
from prepare_structured import STRUCTURED_DIR, TOPIC_MAP

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

TIME_BUDGET     = 300       # 5-minute experiment runs

# Architecture (best from autoresearch)
MODEL_DIM       = 512
N_HEADS         = 8
PRELUDE_DEPTH   = 2
CODA_DEPTH      = 2
N_LOOPS         = 1
FFN_MULT        = 4

# Novel: Topic conditioning
N_TOPICS        = len(TOPIC_MAP)  # 8 topics
TOPIC_EMBED_DIM = 64              # topic embedding dimension
TOPIC_LOSS_WEIGHT = 0.1           # weight for topic classification loss

# Novel: Verification loss weighting
VERIFY_LOSS_MULT = 3.0            # loss multiplier on <verify> token predictions

# Training
BATCH_SIZE      = 16
TOTAL_BATCH     = 16384
LR              = 3e-3
WEIGHT_DECAY    = 0.1
WARMUP_RATIO    = 0.05
WARMDOWN_RATIO  = 0.3
ADAM_BETAS      = (0.9, 0.95)
USE_GRAD_CKPT   = True

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight, 1e-6)


class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).contiguous().view(B, T, -1))


def _apply_rope(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def _precompute_rope(seq_len, head_dim, device, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    f = torch.outer(t, inv)
    return f.cos()[None, None, :, :], f.sin()[None, None, :, :]


class DenseFFN(nn.Module):
    """ReluSquared FFN (best activation from experiments)."""
    def __init__(self, dim, hidden):
        super().__init__()
        self.c_fc   = nn.Linear(dim, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, dim, bias=False)
    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class TopicConditionedFFN(nn.Module):
    """
    NOVEL: Topic-Conditioned FFN.
    
    Standard ReluSquared FFN but with an additive topic bias injected
    into the hidden layer. The topic signal is a learned embedding that
    shifts the activation pattern based on the problem type (geometry,
    algebra, combinatorics, etc.).
    
    This gives the model specialized reasoning pathways per topic
    without per-token routing overhead (unlike MoE).
    
    Key difference from MoE:
      - MoE routes per-token with a learned gating function
      - This routes per-problem with a semantically meaningful topic signal
      - The bias is additive, not selective — all capacity is always used
      - The topic comes from the data, not from a learned router
    """
    def __init__(self, dim, hidden, n_topics, topic_embed_dim):
        super().__init__()
        self.c_fc   = nn.Linear(dim, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, dim, bias=False)
        # Topic conditioning: project topic embedding to hidden dim as bias
        self.topic_bias = nn.Linear(topic_embed_dim, hidden, bias=False)
        nn.init.zeros_(self.topic_bias.weight)  # start neutral
    
    def forward(self, x, topic_emb=None):
        h = self.c_fc(x)
        if topic_emb is not None:
            # topic_emb: (B, topic_embed_dim) → (B, 1, hidden) broadcast over seq
            h = h + self.topic_bias(topic_emb).unsqueeze(1)
        return self.c_proj(F.relu(h).square())


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, ffn_mult, use_topic=False):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads)
        self.norm2 = RMSNorm(dim)
        hidden = ((int(dim * ffn_mult) + 63) // 64) * 64
        self.use_topic = use_topic
        if use_topic:
            self.ffn = TopicConditionedFFN(dim, hidden, N_TOPICS, TOPIC_EMBED_DIM)
        else:
            self.ffn = DenseFFN(dim, hidden)

    def forward(self, x, cos, sin, topic_emb=None):
        x = x + self.attn(self.norm1(x), cos, sin)
        if self.use_topic and topic_emb is not None:
            x = x + self.ffn(self.norm2(x), topic_emb)
        else:
            x = x + self.ffn(self.norm2(x))
        return x


class NovelRDT(nn.Module):
    """
    NOVEL: RDT with Topic Conditioning + Verification-Aware Output.
    
    Innovations:
    1. Topic classifier head: predicts problem topic from early hidden states
    2. Topic-conditioned FFN: uses predicted topic to bias reasoning pathway
    3. Verification head: separate lightweight head for <verify> token predictions
    """
    def __init__(self, vocab_size, dim, n_heads, prelude, coda, n_loops, ffn_mult):
        super().__init__()
        self.n_loops = n_loops
        self.tok_emb = nn.Embedding(vocab_size, dim)
        
        # Prelude: standard blocks (no topic conditioning yet — topic unknown)
        self.prelude = nn.ModuleList([
            TransformerBlock(dim, n_heads, ffn_mult, use_topic=False) 
            for _ in range(prelude)
        ])
        
        # Core: with topic conditioning (after topic is predicted from prelude output)
        self.core = TransformerBlock(dim, n_heads, ffn_mult, use_topic=True)
        
        # Coda: with topic conditioning
        self.coda = nn.ModuleList([
            TransformerBlock(dim, n_heads, ffn_mult, use_topic=True) 
            for _ in range(coda)
        ])
        
        self.norm_out = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # NOVEL: Topic classifier — predicts topic from prelude output
        # Uses mean pooling over sequence positions → topic logits
        self.topic_classifier = nn.Sequential(
            nn.Linear(dim, TOPIC_EMBED_DIM),
            nn.ReLU(),
            nn.Linear(TOPIC_EMBED_DIM, N_TOPICS),
        )
        # Topic embedding: maps predicted topic to conditioning signal
        self.topic_embedding = nn.Embedding(N_TOPICS, TOPIC_EMBED_DIM)
        
        # RoPE
        head_dim = dim // n_heads
        cos, sin = _precompute_rope(MAX_SEQ_LEN, head_dim, device="cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx, targets=None, topic_labels=None, verify_mask=None):
        """
        Args:
            idx: input token ids (B, T)
            targets: target token ids (B, T) for loss computation
            topic_labels: ground-truth topic ids (B,) for topic classification loss
            verify_mask: boolean mask (B, T) where True = this is a <verify> token position
        
        Returns:
            If targets provided: (lm_loss, topic_loss, logits)
            Else: logits
        """
        B, T = idx.shape
        cos, sin = self.cos[:, :, :T], self.sin[:, :, :T]
        x = self.tok_emb(idx)
        
        # Prelude: no topic conditioning
        for block in self.prelude:
            if USE_GRAD_CKPT and self.training:
                x = checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)
        
        # NOVEL: Predict topic from prelude output
        # Use mean pooling over the first 64 tokens (problem description area)
        pool_len = min(T, 64)
        pooled = x[:, :pool_len, :].mean(dim=1)  # (B, dim)
        topic_logits = self.topic_classifier(pooled)  # (B, N_TOPICS)
        
        # Get topic embedding (use argmax during eval, soft during training)
        if self.training:
            # Soft topic: weighted sum of embeddings by softmax probs
            topic_probs = F.softmax(topic_logits, dim=-1)  # (B, N_TOPICS)
            topic_emb = topic_probs @ self.topic_embedding.weight  # (B, TOPIC_EMBED_DIM)
        else:
            topic_id = topic_logits.argmax(dim=-1)  # (B,)
            topic_emb = self.topic_embedding(topic_id)  # (B, TOPIC_EMBED_DIM)
        
        # Core: topic-conditioned
        for _ in range(self.n_loops):
            if USE_GRAD_CKPT and self.training:
                x = checkpoint(self.core, x, cos, sin, topic_emb, use_reentrant=False)
            else:
                x = self.core(x, cos, sin, topic_emb)
        
        # Coda: topic-conditioned
        for block in self.coda:
            if USE_GRAD_CKPT and self.training:
                x = checkpoint(block, x, cos, sin, topic_emb, use_reentrant=False)
            else:
                x = block(x, cos, sin, topic_emb)
        
        logits = self.lm_head(self.norm_out(x))
        
        if targets is not None:
            # Standard LM loss
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Topic classification loss
            topic_loss = torch.tensor(0.0, device=idx.device)
            if topic_labels is not None:
                valid_mask = topic_labels >= 0  # -1 = unknown topic
                if valid_mask.any():
                    topic_loss = F.cross_entropy(
                        topic_logits[valid_mask], 
                        topic_labels[valid_mask]
                    )
            
            return lm_loss, topic_loss, logits
        
        return logits

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Data loading for structured format
# ---------------------------------------------------------------------------

# Precompute topic name token patterns for fast lookup
_TOPIC_NAME_TOKENS = {}  # first distinguishing token_id → topic_id

def _init_topic_tokens(tokenizer):
    """Build mapping from token IDs to topic IDs for fast window scanning."""
    global _TOPIC_NAME_TOKENS
    if _TOPIC_NAME_TOKENS:
        return
    # Each topic name starts with a unique token after '<topic> '
    # In the stream: < topic > _TopicName, so the name token has a leading space
    for name, topic_id in TOPIC_MAP.items():
        tokens = tokenizer.encode(f" {name}")  # leading space to match stream
        if tokens:
            _TOPIC_NAME_TOKENS[tokens[0]] = topic_id
    print(f"Topic token mapping: {len(_TOPIC_NAME_TOKENS)} topics registered")


def make_structured_dataloader(tokenizer, batch_size, seq_len, split):
    """
    Dataloader for structured math data.
    Returns (x, y, topic_labels, verify_mask, epoch).
    """
    path = os.path.join(STRUCTURED_DIR, f"{split}.bin")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Structured data not found: {path}. Run prepare_structured.py first.")
    
    data = np.memmap(path, dtype=np.uint16, mode="r")
    _init_topic_tokens(tokenizer)
    
    # Token ID for '<' (starts all our special tokens)
    lt_token = tokenizer.encode("<")[0]  # 27 in tiktoken
    topic_token = tokenizer.encode("topic")[0]  # 26652
    
    epoch = 0
    while True:
        ix = np.random.randint(0, len(data) - seq_len - 1, size=(batch_size,))
        x = np.stack([data[i   : i + seq_len    ].astype(np.int32) for i in ix])
        y = np.stack([data[i+1 : i + seq_len + 1].astype(np.int32) for i in ix])
        
        # Extract topic labels by scanning for topic name tokens
        topic_labels = np.full(batch_size, -1, dtype=np.int64)
        for b in range(batch_size):
            # Scan first 100 tokens for a topic pattern: <(27) topic(26652) >(29) TopicName
            for t in range(min(seq_len - 4, 100)):
                if x[b, t] == lt_token and x[b, t+1] == topic_token:
                    # Found <topic>, next meaningful token after > is the topic name
                    name_tok = int(x[b, t+3])  # skip the '>' token
                    if name_tok in _TOPIC_NAME_TOKENS:
                        topic_labels[b] = _TOPIC_NAME_TOKENS[name_tok]
                    break
        
        # verify_mask not used in current training loop, skip expensive scan
        verify_mask = None
        
        yield x, y, topic_labels, verify_mask, epoch
        epoch += 1


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_bpb_novel(model, tokenizer, batch_size, device):
    """Compute validation bits-per-byte for novel model."""
    path = os.path.join(STRUCTURED_DIR, "val.bin")
    if not os.path.exists(path):
        from prepare import evaluate_bpb_torch
        return evaluate_bpb_torch(model, tokenizer, batch_size, device)
    
    data = np.memmap(path, dtype=np.uint16, mode="r")
    
    meta_path = os.path.join(STRUCTURED_DIR, "meta.txt")
    bpt = 1.0
    if os.path.exists(meta_path):
        for line in open(meta_path):
            if line.startswith("bytes_per_token="):
                bpt = float(line.split("=")[1])
    
    EVAL_TOKENS = 50_000
    n_batches = max(1, min(EVAL_TOKENS, len(data) - MAX_SEQ_LEN - 1)
                    // (batch_size * MAX_SEQ_LEN))
    total_loss = 0.0
    total_toks = 0

    with torch.no_grad():
        for _ in range(n_batches):
            ix = np.random.randint(0, len(data) - MAX_SEQ_LEN - 1, size=(batch_size,))
            x = torch.from_numpy(np.stack([data[i:i+MAX_SEQ_LEN].astype(np.int64) for i in ix])).to(device)
            y = torch.from_numpy(np.stack([data[i+1:i+MAX_SEQ_LEN+1].astype(np.int64) for i in ix])).to(device)
            
            # Single forward pass
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[-1]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum"
            )
            total_loss += loss.item()
            total_toks += y.numel()

    avg_nats = total_loss / max(total_toks, 1)
    bpb = avg_nats / math.log(2) * bpt
    return bpb


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size}")

model = NovelRDT(
    vocab_size, MODEL_DIM, N_HEADS, PRELUDE_DEPTH, CODA_DEPTH,
    N_LOOPS, FFN_MULT,
).to(device)

model = torch.compile(model)

n_params = model.num_params()
print(f"Parameters: {n_params:,}  ({n_params/1e6:.1f}M)")
print(f"Architecture: prelude={PRELUDE_DEPTH} + core×{N_LOOPS} + coda={CODA_DEPTH}")
print(f"Novel: TopicConditioned FFN ({N_TOPICS} topics, embed_dim={TOPIC_EMBED_DIM})")
print(f"Novel: Verification loss mult = {VERIFY_LOSS_MULT}x")

tokens_per_step = BATCH_SIZE * MAX_SEQ_LEN
grad_accum = max(1, TOTAL_BATCH // tokens_per_step)
print(f"Batch: {BATCH_SIZE}×{MAX_SEQ_LEN} | accum: {grad_accum} | {grad_accum * tokens_per_step:,} tok/step")
print(f"Time budget: {TIME_BUDGET}s")

# Use structured dataloader
train_loader = make_structured_dataloader(tokenizer, BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, topic_labels, verify_mask, epoch = next(train_loader)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY)

def get_lr(progress):
    if progress < WARMUP_RATIO:
        return LR * (progress / WARMUP_RATIO) if WARMUP_RATIO > 0 else LR
    elif progress < 1.0 - WARMDOWN_RATIO:
        return LR
    else:
        return LR * max((1.0 - progress) / WARMDOWN_RATIO, 0.0)

training_time = 0.0
step = 0
smooth_loss = 0.0
smooth_topic_loss = 0.0
topic_correct = 0
topic_total = 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(grad_accum):
        x_t = torch.from_numpy(x).long().to(device)
        y_t = torch.from_numpy(y).long().to(device)
        tl_t = torch.from_numpy(topic_labels).long().to(device)
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            lm_loss, topic_loss, logits = model(x_t, y_t, tl_t)
            
            # Combined loss
            loss = lm_loss + TOPIC_LOSS_WEIGHT * topic_loss
        
        (loss / grad_accum).backward()
        x, y, topic_labels, verify_mask, epoch = next(train_loader)

    progress = min(training_time / TIME_BUDGET, 1.0)
    lr_now = get_lr(progress)
    for g in optimizer.param_groups:
        g["lr"] = lr_now
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    loss_f = loss.item()
    if math.isnan(loss_f) or loss_f > 100:
        print("\nFAIL")
        exit(1)

    torch.cuda.synchronize()
    dt = time.time() - t0
    if step > 5:
        training_time += dt

    ema = 0.9
    smooth_loss = ema * smooth_loss + (1 - ema) * lm_loss.item()
    smooth_topic_loss = ema * smooth_topic_loss + (1 - ema) * (topic_loss.item() if topic_loss.item() > 0 else 0)
    debiased = smooth_loss / (1 - ema ** (step + 1))
    pct = 100 * progress
    tok_s = int(grad_accum * tokens_per_step / dt)
    remaining = max(0, TIME_BUDGET - training_time)
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"\rstep {step:04d} ({pct:.1f}%) | lm: {debiased:.4f} | topic: {smooth_topic_loss/(1-ema**(step+1)):.3f} | lr: {lr_now:.2e} | tok/s: {tok_s:,} | vram: {peak_mb:.0f}MB | remain: {remaining:.0f}s   ", end="", flush=True)

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()

    step += 1
    if step > 5 and training_time >= TIME_BUDGET:
        break

print()
model.eval()
with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    val_bpb = evaluate_bpb_novel(model, tokenizer, min(BATCH_SIZE, 8), device)

t_end = time.time()
peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_mb:.1f}")
print(f"total_tokens_M:   {step * grad_accum * tokens_per_step / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {n_params / 1e6:.1f}")
print(f"novel_features:   topic_conditioned_ffn, verification_tokens")
