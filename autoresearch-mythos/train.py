"""
autoresearch-mythos training script.
Recurrent-Depth Transformer (RDT) on MathNet — Apple Silicon via MLX.

This is the ONLY file the agent modifies.
Everything is fair game: architecture, hyperparameters, optimizer, etc.

Usage: python train.py
"""

import gc
import math
import os
import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, evaluate_bpb, make_dataloader

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# ---------------------------------------------------------------------------
# Hyperparameters (agent can edit all of these)
# ---------------------------------------------------------------------------

# Architecture
MODEL_DIM       = 384       # model embedding dimension
N_HEADS         = 6         # attention heads
PRELUDE_DEPTH   = 2         # transformer layers before recurrent core
CODA_DEPTH      = 2         # transformer layers after recurrent core
N_LOOPS         = 2         # recurrent depth iterations
FFN_MULT        = 4         # FFN hidden dim multiplier

# MoE (set USE_MOE=False for dense FFN)
USE_MOE         = False
N_EXPERTS       = 4         # total routed experts
TOP_K           = 2         # active experts per token
SHARED_EXPERTS  = 1         # always-on shared experts

# Recurrent core
USE_LTI         = True      # LTI stability injection
LORA_RANK       = 16        # LoRA rank for loop adaptation (0=off)

# Training
BATCH_SIZE      = 16         # micro batch size
TOTAL_BATCH     = 2**14     # ~16K tokens per optimizer step
LR              = 3e-4      # peak learning rate
WEIGHT_DECAY    = 0.1       # AdamW weight decay
WARMUP_RATIO    = 0.05      # fraction of time for LR warmup
WARMDOWN_RATIO  = 0.3       # fraction of time for LR cooldown
ADAM_BETAS      = (0.9, 0.95)

# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

def rms_norm(x):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def create_causal_mask(seq_len):
    indices = mx.arange(seq_len)
    blocked = indices[None, :] > indices[:, None]
    return mx.where(blocked, mx.array(float("-inf")), mx.array(0.0))


class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.rope = nn.RoPE(self.head_dim, traditional=True, base=10000)

    def __call__(self, x, mask):
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.rope(q)
        k = self.rope(k)
        scale = 1.0 / math.sqrt(self.head_dim)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.o_proj(y)


class DenseFFN(nn.Module):
    """SwiGLU FFN."""
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up   = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class MoEFFN(nn.Module):
    """Mixture-of-Experts with top-k routing + shared experts."""
    def __init__(self, dim, n_experts, top_k, n_shared, hidden_per_expert):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(dim, n_experts, bias=False)
        self.experts = [DenseFFN(dim, hidden_per_expert) for _ in range(n_experts)]
        self.shared  = [DenseFFN(dim, hidden_per_expert) for _ in range(n_shared)]

    def __call__(self, x):
        B, T, D = x.shape
        # Shared experts always active
        out = sum(e(x) for e in self.shared) if self.shared else mx.zeros_like(x)
        if not self.experts:
            return out
        # Routing
        scores = self.router(x)                         # (B, T, n_experts)
        top_vals = mx.topk(scores, self.top_k, axis=-1)
        weights = mx.softmax(top_vals, axis=-1)         # (B, T, top_k)
        indices = mx.argpartition(-scores, kth=self.top_k - 1, axis=-1)[..., :self.top_k]
        # Compute all experts, mask by routing (simple, works for ≤8 experts)
        for i, expert in enumerate(self.experts):
            mask = mx.any(indices == i, axis=-1, keepdims=True)  # (B, T, 1)
            w = mx.sum(weights * (indices == i).astype(weights.dtype), axis=-1, keepdims=True)
            out = out + w * expert(x) * mask
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, ffn_mult, use_moe):
        super().__init__()
        self.attn = Attention(dim, n_heads)
        hidden = int(dim * ffn_mult * 2 / 3)
        hidden = ((hidden + 63) // 64) * 64
        if use_moe:
            expert_hidden = max(hidden // max(N_EXPERTS, 1), 32)
            self.ffn = MoEFFN(dim, N_EXPERTS, TOP_K, SHARED_EXPERTS, expert_hidden)
        else:
            self.ffn = DenseFFN(dim, hidden)

    def __call__(self, x, mask):
        x = x + self.attn(rms_norm(x), mask)
        x = x + self.ffn(rms_norm(x))
        return x


# ---------------------------------------------------------------------------
# Recurrent-Depth core
# ---------------------------------------------------------------------------

class LTIInjection(nn.Module):
    """LTI state injection with guaranteed stability: ρ(A) < 1 by construction."""
    def __init__(self, dim):
        super().__init__()
        self.log_A = mx.random.normal((dim,)) * 0.1 - 1.0   # A ≈ 0.37
        self.B_proj = nn.Linear(dim, dim, bias=False)

    def __call__(self, h, e):
        A = mx.clip(mx.exp(self.log_A), a_min=0.0, a_max=0.999)
        return A * h + self.B_proj(e)


class RecurrentCore(nn.Module):
    """Single transformer block looped N times with LTI + LoRA."""
    def __init__(self, dim, n_heads, ffn_mult, use_moe, use_lti, lora_rank, max_loops):
        super().__init__()
        self.block = TransformerBlock(dim, n_heads, ffn_mult, use_moe)
        self.lti = LTIInjection(dim) if use_lti else None
        self.loop_embed = nn.Embedding(max_loops, dim) if max_loops > 1 else None
        self.has_lora = lora_rank > 0
        if self.has_lora:
            self.lora_down = nn.Linear(dim, lora_rank, bias=False)
            self.lora_up   = nn.Linear(lora_rank, dim, bias=False)

    def __call__(self, x, e, mask, n_loops):
        h = mx.zeros_like(x)
        for i in range(n_loops):
            if self.lti is not None:
                h = self.lti(h, e)
            x_in = x + h
            if self.loop_embed is not None:
                x_in = x_in + self.loop_embed(mx.array(i))
            x = self.block(x_in, mask)
            if self.has_lora:
                x = x + self.lora_up(self.lora_down(x))
        return x


# ---------------------------------------------------------------------------
# Full RDT model
# ---------------------------------------------------------------------------

class RecurrentDepthTransformer(nn.Module):
    def __init__(self, vocab_size, dim, n_heads, prelude_depth, coda_depth,
                 n_loops, ffn_mult, use_moe, use_lti, lora_rank):
        super().__init__()
        self.n_loops = n_loops
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.prelude = [TransformerBlock(dim, n_heads, ffn_mult, use_moe=False)
                        for _ in range(prelude_depth)]
        self.core = RecurrentCore(dim, n_heads, ffn_mult, use_moe, use_lti,
                                  lora_rank, max_loops=max(n_loops, 32))
        self.coda = [TransformerBlock(dim, n_heads, ffn_mult, use_moe=False)
                     for _ in range(coda_depth)]
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self._mask_cache = {}

    def _get_mask(self, seq_len):
        if seq_len not in self._mask_cache:
            self._mask_cache[seq_len] = create_causal_mask(seq_len)
        return self._mask_cache[seq_len]

    def __call__(self, idx, targets=None):
        B, T = idx.shape
        mask = self._get_mask(T)
        x = self.tok_emb(idx)
        e = x  # save for LTI injection
        for block in self.prelude:
            x = block(x, mask)
        x = self.core(x, e, mask, self.n_loops)
        for block in self.coda:
            x = block(x, mask)
        logits = self.lm_head(rms_norm(x)).astype(mx.float32)

        if targets is None:
            return logits
        return nn.losses.cross_entropy(logits, targets, reduction="mean")


# ---------------------------------------------------------------------------
# AdamW optimizer (MLX-native)
# ---------------------------------------------------------------------------

class AdamW:
    def __init__(self, lr, betas, weight_decay):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.wd = weight_decay
        self.state = {}
        self.t = 0
        self.initial_lr = lr

    def update(self, model, grads):
        self.t += 1
        flat_grads = dict(tree_flatten(grads))
        flat_params = dict(tree_flatten(model.parameters()))

        for path, grad in flat_grads.items():
            param = flat_params[path]
            g = grad.astype(mx.float32)
            p = param.astype(mx.float32)

            if path not in self.state:
                self.state[path] = {
                    "m": mx.zeros_like(g),
                    "v": mx.zeros_like(g),
                }
            s = self.state[path]
            s["m"] = self.beta1 * s["m"] + (1 - self.beta1) * g
            s["v"] = self.beta2 * s["v"] + (1 - self.beta2) * g * g

            m_hat = s["m"] / (1 - self.beta1 ** self.t)
            v_hat = s["v"] / (1 - self.beta2 ** self.t)

            p = p * (1 - self.lr * self.wd) - self.lr * m_hat / (mx.sqrt(v_hat) + 1e-8)
            _set_param(model, path, p.astype(param.dtype))

    def set_lr(self, lr):
        self.lr = lr

    @property
    def state_arrays(self):
        arrays = []
        for s in self.state.values():
            arrays.extend([s["m"], s["v"]])
        return arrays


def _set_param(model, path, value):
    parts = path.split(".")
    obj = model
    for part in parts[:-1]:
        if isinstance(obj, list):
            obj = obj[int(part)]
        elif isinstance(obj, dict):
            obj = obj[part]
        else:
            obj = getattr(obj, part)
    last = parts[-1]
    if isinstance(obj, dict):
        obj[last] = value
    else:
        setattr(obj, last, value)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_global_start = time.time()
mx.random.seed(42)

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

# Build model
model = RecurrentDepthTransformer(
    vocab_size=vocab_size, dim=MODEL_DIM, n_heads=N_HEADS,
    prelude_depth=PRELUDE_DEPTH, coda_depth=CODA_DEPTH,
    n_loops=N_LOOPS, ffn_mult=FFN_MULT, use_moe=USE_MOE,
    use_lti=USE_LTI, lora_rank=LORA_RANK,
)
mx.eval(model.parameters())

n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
print(f"Parameters: {n_params:,}  ({n_params/1e6:.1f}M)")
print(f"Architecture: prelude={PRELUDE_DEPTH} + core×{N_LOOPS} + coda={CODA_DEPTH}")
print(f"MoE: {USE_MOE} ({N_EXPERTS} experts, top-{TOP_K}, {SHARED_EXPERTS} shared)")
print(f"LTI: {USE_LTI}  |  LoRA rank: {LORA_RANK}")

# Dataloader
tokens_per_step = BATCH_SIZE * MAX_SEQ_LEN
grad_accum = max(1, TOTAL_BATCH // tokens_per_step)
train_loader = make_dataloader(tokenizer, BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)

print(f"Batch: {BATCH_SIZE} × {MAX_SEQ_LEN} = {tokens_per_step:,} tok | accum: {grad_accum} | total: {grad_accum * tokens_per_step:,} tok/step")
print(f"Time budget: {TIME_BUDGET}s")

# Optimizer
optimizer = AdamW(lr=LR, betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY)

# Loss + grad function
loss_grad_fn = nn.value_and_grad(
    model, lambda mdl, inp, tgt: mdl(inp, targets=tgt)
)

# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(progress):
    if progress < WARMUP_RATIO:
        return LR * (progress / WARMUP_RATIO) if WARMUP_RATIO > 0 else LR
    elif progress < 1.0 - WARMDOWN_RATIO:
        return LR
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return LR * max(cooldown, 0.0)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

training_time = 0.0
step = 0
smooth_loss = 0.0
t_compiled = None

while True:
    t0 = time.time()
    accum_grads = None

    for _ in range(grad_accum):
        x_mx = mx.array(x.numpy()) if hasattr(x, 'numpy') else mx.array(x)
        y_mx = mx.array(y.numpy()) if hasattr(y, 'numpy') else mx.array(y)
        loss, grads = loss_grad_fn(model, x_mx, y_mx)
        mx.eval(loss, grads)

        if t_compiled is None:
            t_compiled = time.time()
            print(f"Compiled in {t_compiled - t_global_start:.1f}s")

        if accum_grads is None:
            accum_grads = grads
        else:
            accum_grads = tree_map(lambda a, b: a + b, accum_grads, grads)
        x, y, epoch = next(train_loader)

    if grad_accum > 1:
        accum_grads = tree_map(lambda g: g * (1.0 / grad_accum), accum_grads)

    # LR schedule
    progress = min(training_time / TIME_BUDGET, 1.0)
    lr_now = get_lr(progress)
    optimizer.set_lr(lr_now)

    # Update
    optimizer.update(model, accum_grads)
    mx.eval(model.parameters(), *optimizer.state_arrays)

    # Check loss
    loss_f = float(loss.item())
    if math.isnan(loss_f) or loss_f > 100:
        print("\nFAIL — loss exploded")
        exit(1)

    dt = time.time() - t0
    if step > 2:
        training_time += dt

    # Logging
    ema = 0.9
    smooth_loss = ema * smooth_loss + (1 - ema) * loss_f
    debiased = smooth_loss / (1 - ema ** (step + 1))
    pct = 100 * progress
    tok_s = int(grad_accum * tokens_per_step / dt) if dt > 0 else 0
    remaining = max(0, TIME_BUDGET - training_time)

    print(f"\rstep {step:04d} ({pct:.1f}%) | loss: {debiased:.4f} | lr: {lr_now:.2e} | dt: {dt*1000:.0f}ms | tok/s: {tok_s:,} | remain: {remaining:.0f}s   ", end="", flush=True)

    if step == 0:
        gc.collect()

    step += 1
    if step > 2 and training_time >= TIME_BUDGET:
        break

print()

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

print("Running final evaluation...")
val_bpb = evaluate_bpb(model, tokenizer, BATCH_SIZE)

t_end = time.time()
peak_mem = mx.get_peak_memory() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {training_time:.1f}")
print(f"total_seconds:    {t_end - t_global_start:.1f}")
print(f"peak_mem_mb:      {peak_mem:.1f}")
print(f"total_tokens_M:   {step * grad_accum * tokens_per_step / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {n_params / 1e6:.1f}")
print(f"prelude_depth:    {PRELUDE_DEPTH}")
print(f"n_loops:          {N_LOOPS}")
print(f"coda_depth:       {CODA_DEPTH}")
print(f"use_moe:          {USE_MOE}")
print(f"use_lti:          {USE_LTI}")
