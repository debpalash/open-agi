"""
autoresearch-mythos CUDA training script.
Recurrent-Depth Transformer (RDT) on MathNet — NVIDIA GPU (PyTorch).

This is the ONLY file the agent modifies on the CUDA machine.
Usage: uv run train_cuda.py
"""

import gc
import math
import os
import time
from dataclasses import dataclass

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from prepare import MAX_SEQ_LEN, TIME_BUDGET as _TIME_BUDGET, Tokenizer, evaluate_bpb_torch, make_dataloader
TIME_BUDGET     = 1800      # 30-minute full training run

# ---------------------------------------------------------------------------
# Hyperparameters (agent can edit all of these)
# ---------------------------------------------------------------------------

# Architecture
MODEL_DIM       = 512       # model embedding dimension
N_HEADS         = 8         # attention heads
PRELUDE_DEPTH   = 2         # transformer layers before recurrent core
CODA_DEPTH      = 2         # transformer layers after recurrent core
N_LOOPS         = 1         # recurrent depth iterations
FFN_MULT        = 4         # FFN hidden dim multiplier

# MoE
USE_MOE         = False
N_EXPERTS       = 4
TOP_K           = 2
SHARED_EXPERTS  = 1

# Recurrent core
USE_LTI         = False
LORA_RANK       = 0
USE_GRAD_CKPT   = True      # gradient checkpointing (saves VRAM)

# Training
BATCH_SIZE      = 16        # micro batch size
TOTAL_BATCH     = 16384     # ~16K tokens per optimizer step
LR              = 3e-3
WEIGHT_DECAY    = 0.1
WARMUP_RATIO    = 0.05
WARMDOWN_RATIO  = 0.3
ADAM_BETAS      = (0.9, 0.95)

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
    def __init__(self, dim, hidden):
        super().__init__()
        self.c_fc   = nn.Linear(dim, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, dim, bias=False)
    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class MoEFFN(nn.Module):
    def __init__(self, dim, n_experts, top_k, n_shared, hidden):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([DenseFFN(dim, hidden) for _ in range(n_experts)])
        self.shared  = nn.ModuleList([DenseFFN(dim, hidden) for _ in range(n_shared)])

    def forward(self, x):
        out = sum(e(x) for e in self.shared) if self.shared else torch.zeros_like(x)
        if not self.experts:
            return out
        scores = self.router(x)
        weights, indices = scores.topk(self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        for i, expert in enumerate(self.experts):
            mask = (indices == i).any(dim=-1, keepdim=True).float()
            w = (weights * (indices == i).float()).sum(dim=-1, keepdim=True)
            out = out + w * expert(x) * mask
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, ffn_mult, use_moe):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads)
        self.norm2 = RMSNorm(dim)
        hidden = ((int(dim * ffn_mult) + 63) // 64) * 64
        if use_moe:
            eh = max(hidden // max(N_EXPERTS, 1), 32)
            self.ffn = MoEFFN(dim, N_EXPERTS, TOP_K, SHARED_EXPERTS, eh)
        else:
            self.ffn = DenseFFN(dim, hidden)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


class LTIInjection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.log_A = nn.Parameter(torch.randn(dim) * 0.1 - 1.0)
        self.B_proj = nn.Linear(dim, dim, bias=False)
    def forward(self, h, e):
        A = torch.exp(self.log_A).clamp(max=0.999)
        return A * h + self.B_proj(e)


class RecurrentCore(nn.Module):
    def __init__(self, dim, n_heads, ffn_mult, use_moe, use_lti, lora_rank, max_loops):
        super().__init__()
        self.block = TransformerBlock(dim, n_heads, ffn_mult, use_moe)
        self.lti = LTIInjection(dim) if use_lti else None
        self.loop_embed = nn.Embedding(max_loops, dim) if max_loops > 1 else None
        self.has_lora = lora_rank > 0
        if self.has_lora:
            self.lora_down = nn.Linear(dim, lora_rank, bias=False)
            self.lora_up = nn.Linear(lora_rank, dim, bias=False)
            nn.init.zeros_(self.lora_up.weight)

    def forward(self, x, e, cos, sin, n_loops):
        h = torch.zeros_like(x)
        for i in range(n_loops):
            if self.lti is not None:
                h = self.lti(h, e)
            x_in = x + h
            if self.loop_embed is not None:
                x_in = x_in + self.loop_embed(torch.tensor(i, device=x.device))
            if USE_GRAD_CKPT and self.training:
                x = checkpoint(self.block, x_in, cos, sin, use_reentrant=False)
            else:
                x = self.block(x_in, cos, sin)
            if self.has_lora:
                x = x + self.lora_up(self.lora_down(x))
        return x


class RDT(nn.Module):
    def __init__(self, vocab_size, dim, n_heads, prelude, coda, n_loops, ffn_mult, use_moe, use_lti, lora_rank):
        super().__init__()
        self.n_loops = n_loops
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.prelude = nn.ModuleList([TransformerBlock(dim, n_heads, ffn_mult, False) for _ in range(prelude)])
        self.core = RecurrentCore(dim, n_heads, ffn_mult, use_moe, use_lti, lora_rank, max(n_loops, 32))
        self.coda = nn.ModuleList([TransformerBlock(dim, n_heads, ffn_mult, False) for _ in range(coda)])
        self.norm_out = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        head_dim = dim // n_heads
        cos, sin = _precompute_rope(MAX_SEQ_LEN, head_dim, device="cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        cos, sin = self.cos[:, :, :T], self.sin[:, :, :T]
        x = self.tok_emb(idx)
        e = x
        for block in self.prelude:
            if USE_GRAD_CKPT and self.training:
                x = checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)
        x = self.core(x, e, cos, sin, self.n_loops)
        for block in self.coda:
            if USE_GRAD_CKPT and self.training:
                x = checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)
        logits = self.lm_head(self.norm_out(x))
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

model = RDT(
    vocab_size, MODEL_DIM, N_HEADS, PRELUDE_DEPTH, CODA_DEPTH,
    N_LOOPS, FFN_MULT, USE_MOE, USE_LTI, LORA_RANK,
).to(device)

n_params = model.num_params()
print(f"Parameters: {n_params:,}  ({n_params/1e6:.1f}M)")
print(f"Architecture: prelude={PRELUDE_DEPTH} + core×{N_LOOPS} + coda={CODA_DEPTH}")
print(f"MoE: {USE_MOE}  |  LTI: {USE_LTI}  |  LoRA: {LORA_RANK}  |  GradCkpt: {USE_GRAD_CKPT}")

model = torch.compile(model, dynamic=False)

tokens_per_step = BATCH_SIZE * MAX_SEQ_LEN
grad_accum = max(1, TOTAL_BATCH // tokens_per_step)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=ADAM_BETAS, fused=True)

train_loader = make_dataloader(tokenizer, BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)
print(f"Batch: {BATCH_SIZE}×{MAX_SEQ_LEN} | accum: {grad_accum} | {grad_accum*tokens_per_step:,} tok/step")
print(f"Time budget: {TIME_BUDGET}s")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

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

while True:
    torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(grad_accum):
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(torch.from_numpy(x).long().to(device), torch.from_numpy(y).long().to(device))
        (loss / grad_accum).backward()
        x, y, epoch = next(train_loader)

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
    smooth_loss = ema * smooth_loss + (1 - ema) * loss_f
    debiased = smooth_loss / (1 - ema ** (step + 1))
    pct = 100 * progress
    tok_s = int(grad_accum * tokens_per_step / dt)
    remaining = max(0, TIME_BUDGET - training_time)
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"\rstep {step:04d} ({pct:.1f}%) | loss: {debiased:.4f} | lr: {lr_now:.2e} | dt: {dt*1000:.0f}ms | tok/s: {tok_s:,} | vram: {peak_mb:.0f}MB | remain: {remaining:.0f}s   ", end="", flush=True)

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
    val_bpb = evaluate_bpb_torch(model, tokenizer, min(BATCH_SIZE, 8), device)

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
print(f"prelude_depth:    {PRELUDE_DEPTH}")
print(f"n_loops:          {N_LOOPS}")
print(f"coda_depth:       {CODA_DEPTH}")
print(f"use_moe:          {USE_MOE}")
print(f"use_lti:          {USE_LTI}")

# Save trained model checkpoint
import json
ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(ckpt_dir, "rdt_best.pt")
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "step": step,
    "val_bpb": val_bpb,
    "config": {
        "MODEL_DIM": MODEL_DIM, "N_HEADS": N_HEADS,
        "PRELUDE_DEPTH": PRELUDE_DEPTH, "CODA_DEPTH": CODA_DEPTH,
        "N_LOOPS": N_LOOPS, "FFN_MULT": FFN_MULT,
        "USE_MOE": USE_MOE, "USE_LTI": USE_LTI, "LORA_RANK": LORA_RANK,
        "vocab_size": tokenizer.get_vocab_size(),
    },
}, ckpt_path)
print(f"\nCheckpoint saved: {ckpt_path}")
print(f"Model size: {os.path.getsize(ckpt_path) / 1e6:.1f} MB")
