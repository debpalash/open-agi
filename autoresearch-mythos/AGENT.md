# AGENT.md — Instructions for the AI Research Agent

You are an autonomous ML research agent. Your job is to **find the best Recurrent-Depth Transformer (RDT) architecture** by running experiments on the MathNet olympiad dataset. You work alongside another agent on a different machine — you share results via git.

## First: Identify Your Machine

```bash
python -c "import mlx.core; print('MAC')" 2>/dev/null || python -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')"
```

| If you see | You are on | Your train file | Run command |
|---|---|---|---|
| `MAC` | Mac M2 16GB (MLX) | `train.py` | `uv run train.py` |
| `CUDA` | RTX 4070 8GB (PyTorch) | `train_cuda.py` | `uv run train_cuda.py` |

## Setup (one-time)

```bash
# 1. Install dependencies (pick your platform)
uv sync --extra mac       # Mac M2 (installs MLX)
uv sync --extra cuda      # NVIDIA GPU (installs PyTorch)

# 2. Check data exists
ls ~/.cache/autoresearch-mythos/train.bin ~/.cache/autoresearch-mythos/val.bin

# 3. If missing, prepare data
uv run prepare.py

# 4. Read these files for full context (the repo is small)
#    - README.md
#    - prepare.py (READ-ONLY — do not modify)
#    - train.py or train_cuda.py (the file you modify)
#    - best_config.json (shared state with the other machine)

# 5. Check current best
python sync.py status
```

## The Experiment Loop

Run this loop **forever** until the human stops you. Never pause to ask if you should continue.

```
LOOP:
  1. python sync.py pull              ← get latest from other machine
  2. Read best_config.json            ← check if other machine found something better
  3. If new best config from other machine:
       → Apply those hyperparameters to YOUR train file
       → That's your new baseline
  4. Pick ONE experimental idea (see ideas below)
  5. Edit the train file (ONLY your train file, never prepare.py)
  6. git add -A && git commit -m "[mac|cuda] description of change"
  7. Run the experiment:
       Mac:  uv run train.py > run.log 2>&1
       CUDA: uv run train_cuda.py > run.log 2>&1
  8. Read results:
       grep "^val_bpb:\|^peak_mem\|^num_steps:" run.log
  9. If grep is empty → CRASH. Run: tail -n 50 run.log
       → Fix if trivial (typo, import), otherwise discard
  10. DECISION:
       If val_bpb IMPROVED → python sync.py push "what you changed"
       If val_bpb WORSE    → git reset --hard HEAD~1
       If CRASH             → git reset --hard HEAD~1, log it, move on
```

## What You Can and Cannot Do

**CAN modify:**
- `train.py` (if Mac) or `train_cuda.py` (if CUDA)
- Architecture, optimizer, hyperparameters, training loop — everything in that file

**CANNOT modify:**
- `prepare.py` — read-only, contains fixed eval harness
- `sync.py` — coordination infrastructure
- Cannot install new packages beyond `pyproject.toml`

## The Metric

**`val_bpb` (validation bits-per-byte) — LOWER IS BETTER.**

This is vocabulary-independent, so Mac and CUDA results are directly comparable even with different model sizes.

## Experimental Ideas (ordered by expected impact)

### High Priority — RDT-Specific
1. **N_LOOPS sweep**: Try 2, 4, 8, 16, 32. This is THE main knob.
2. **USE_MOE=False**: Dense FFN might beat MoE at small scale (less overhead).
3. **USE_LTI=False**: Does the LTI state injection actually help?
4. **LORA_RANK sweep**: Try 0 (off), 8, 16, 32, 64.
5. **PRELUDE_DEPTH / CODA_DEPTH**: Try 0/0, 1/1, 3/3, 4/4.
6. **Depth extrapolation**: Train with N_LOOPS=4, then manually eval with N_LOOPS=8.

### Medium Priority — Standard Knobs
7. **LR sweep**: Try 1e-4, 3e-4, 1e-3, 3e-3.
8. **BATCH_SIZE increase**: If memory allows, double it.
9. **MODEL_DIM**: Try 256, 384, 512, 768 (watch memory).
10. **WEIGHT_DECAY**: Try 0.0, 0.01, 0.1, 0.3.
11. **WARMUP_RATIO**: Try 0.0, 0.05, 0.1.

### Advanced — Try After Basics
12. **ACT halting**: Let easy tokens use fewer loops.
13. **Different LTI parameterization**: ZOH discretization, diagonal SSM.
14. **Value residual connections** across loops.
15. **Shared vs separate LayerNorm** per loop iteration.
16. **Remove LoRA, add loop-conditioned bias** instead.
17. **Cosine schedule** instead of linear warmdown.

## Decision Criteria

- **Keep** if val_bpb improved (even by 0.001).
- **Discard** if val_bpb equal or worse.
- **Simplicity bonus**: If removing something gives equal val_bpb, KEEP the removal — simpler is better.
- **Complexity tax**: A 0.01 improvement that adds 50 lines of code? Probably not worth it. A 0.01 improvement from a one-line change? Definitely keep.

## Memory Constraints

| Machine | Hard Limit | Safe Budget |
|---|---|---|
| Mac M2 | 16 GB unified | ~13 GB (OS needs ~3 GB) |
| RTX 4070 | 8 GB VRAM | ~7 GB (driver overhead) |

If you OOM: reduce BATCH_SIZE first, then MODEL_DIM, then N_LOOPS.

## When You're Stuck

If you've tried 10+ experiments without improvement:
1. Re-read `prepare.py` for new angles.
2. Combine two near-miss ideas that individually didn't help.
3. Try something radical: remove the recurrent core entirely (pure GPT baseline), then add back components one at a time.
4. Try a completely different optimizer (SGD with momentum, Lion, etc.).
5. Change the data format in the train file (e.g., mask solutions during training so the model only predicts solution tokens).

## Important: NEVER STOP

Once the loop begins, **do not pause to ask the human anything**. They may be sleeping. Run experiments continuously. If each takes ~5 minutes, you can complete ~12/hour, ~100 overnight. The human wakes up to a `best_config.json` full of discoveries.
