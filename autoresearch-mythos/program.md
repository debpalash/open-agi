# autoresearch-mythos

Autonomous AI research on Recurrent-Depth Transformers (RDT) using MathNet olympiad data.

## Dual-Machine Parallel Research

This project runs on **two machines simultaneously**:

| Machine | Script | Branch | Framework | Strengths |
|---|---|---|---|---|
| **Mac M2 16GB** | `uv run train.py` | `autoresearch/mac-<tag>` | MLX | Bigger models (up to ~500M) |
| **RTX 4070 8GB** | `uv run train_cuda.py` | `autoresearch/cuda-<tag>` | PyTorch/CUDA | Faster iteration (~10-20x speed) |

Both machines share `prepare.py` (data/eval) and `pyproject.toml` (deps). Each runs its own experiment branch and logs to its own `results.tsv`. Winning configs from one machine can be ported to the other.

**Cross-pollination**: If the CUDA agent discovers that `N_LOOPS=4` beats `N_LOOPS=8`, the Mac agent should test it too on a bigger model. If the Mac agent finds MoE helps at 400M, the CUDA agent should check if it still helps at 150M.

### Sync protocol

Before and after each experiment, sync with the other machine:

```bash
# Before each experiment — get latest winning config
python sync.py pull

# After a WINNING experiment — share results
python sync.py push "increased N_LOOPS to 16"

# Check current state
python sync.py status
```

The file `best_config.json` tracks the best known hyperparameters across both machines. When you pull and see a new best config from the other machine, **apply those hyperparameters to your train file before your next experiment** — then try to beat it.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr28`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. **Do not modify.**
   - `train.py` — the file you modify. RDT architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch-mythos/` contains `train.bin` and `val.bin`. If not, tell the human to run `python prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Confirm and go**.

## Architecture context

The model in `train.py` is a **Recurrent-Depth Transformer (RDT)** — NOT a standard GPT. Key components:

- **Prelude layers**: Standard transformer blocks that process input before the recurrent core.
- **Recurrent core**: A SINGLE transformer block that is **looped N times**, with:
  - **LTI injection**: Hidden state `h = A*h + B*e` that accumulates across loops (A guaranteed < 1 for stability).
  - **LoRA adaptation**: Low-rank adapters allow per-loop specialization without separate weights.
  - **Loop embeddings**: Positional encoding for which loop iteration the model is in.
- **Coda layers**: Standard transformer blocks after the recurrent core.
- **MoE FFN**: Mixture-of-Experts feed-forward with top-k routing + shared experts.

The hypothesis: **recurrent depth enables deeper reasoning than stacking more layers** because the same weights are reused, forcing the model to learn general transformations rather than memorizing patterns.

## What to explore

The RDT has many unique knobs that standard GPT autoresearch doesn't:

**High priority (RDT-specific):**
- `N_LOOPS` — the main recurrence knob. Try 2, 4, 8, 16, 32.
- `USE_LTI` — does LTI stability injection help? Compare True vs False.
- `USE_MOE` vs dense FFN — which wins at small scale?
- `LORA_RANK` — LoRA for loop adaptation. Try 0, 8, 16, 32.
- `PRELUDE_DEPTH` / `CODA_DEPTH` — how many non-recurrent layers are needed?
- Depth extrapolation: train with N_LOOPS=4, eval with N_LOOPS=8. Does it generalize?

**Standard knobs:**
- `MODEL_DIM` — model width (keep ≤ 512 for memory)
- `N_HEADS` — attention heads
- `FFN_MULT` — FFN hidden dimension multiplier
- `LR`, `WEIGHT_DECAY`, `WARMUP_RATIO`, `WARMDOWN_RATIO`
- `BATCH_SIZE`, `TOTAL_BATCH`

**Advanced experiments:**
- ACT (Adaptive Compute Time) halting — let easy tokens use fewer loops
- Different LTI parameterizations (ZOH discretization, diagonal SSM)
- Value residual connections across loops
- Shared vs separate norm layers per loop

## Experimentation

Each experiment runs on Apple Silicon via MLX. The training script runs for a **fixed time budget of 5 minutes**. Launch: `python train.py`.

**What you CAN do:**
- Modify `train.py` — everything is fair game: architecture, optimizer, hyperparameters, training loop, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only.
- Install new packages beyond what's available.
- Modify the evaluation harness.

**The goal: get the lowest val_bpb on MathNet olympiad data.** Since the time budget is fixed, everything is fair game. The only constraint is the code runs without crashing.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome.

## Output format

```
---
val_bpb:          1.234567
training_seconds: 300.1
total_seconds:    325.9
peak_mem_mb:      4500.0
total_tokens_M:   12.3
num_steps:        150
num_params_M:     45.2
prelude_depth:    2
n_loops:          8
coda_depth:       2
use_moe:          True
use_lti:          True
```

## Logging results

Log to `results.tsv` (tab-separated):

```
commit	val_bpb	memory_mb	status	description
a1b2c3d	1.234567	4500.0	keep	baseline
b2c3d4e	1.198765	4600.0	keep	increase N_LOOPS from 8 to 16
c3d4e5f	1.250000	4400.0	discard	disable LTI injection
d4e5f6g	0.000000	0.0	crash	N_LOOPS=64 OOM
```

## The experiment loop

LOOP FOREVER:

1. **Sync**: `python sync.py pull` — get latest results from the other machine
2. Check `best_config.json` — if the other machine found something better, apply those hyperparameters
3. Tune the train file with an experimental idea
4. git commit
5. Run: `python train.py > run.log 2>&1` (Mac) or `python train_cuda.py > run.log 2>&1` (CUDA)
6. Read results: `grep "^val_bpb:\|^peak_mem" run.log`
7. If empty → crashed. `tail -n 50 run.log` for stack trace.
8. If val_bpb improved → keep the commit, then `python sync.py push "description"`
9. If worse → `git reset --hard` to previous best

**NEVER STOP**: Once begun, do NOT pause to ask the human. Run indefinitely until manually stopped. If you run out of ideas, think harder — try combining previous near-misses, try radical architectural changes, re-read the code for new angles.
