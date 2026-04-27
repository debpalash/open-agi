# open-agi — Autonomous RDT Research Project

## What Is This

This is an autonomous ML research project exploring whether **Recurrent-Depth Transformers (RDTs)** can achieve better reasoning per parameter than standard GPT-style transformers. The research is inspired by Ilya Sutskever's thesis that we're entering an "age of research" — where architectural breakthroughs matter more than raw scaling.

The core hypothesis: **a model that loops its reasoning (recurrent depth) should generalize better than one that stacks unique layers** — especially on mathematical reasoning tasks.

## Directory Layout

```
open-agi/
├── autoresearch-mythos/     ★ THE MAIN PROJECT — start here
│   ├── AGENT.md             ← Read this first. Full agent instructions.
│   ├── train.py             ← MLX model (Mac M2) — agent edits this
│   ├── train_cuda.py        ← PyTorch model (RTX 4070) — agent edits this
│   ├── prepare.py           ← Data prep + eval (READ-ONLY)
│   ├── sync.py              ← Cross-machine sync protocol
│   ├── best_config.json     ← Shared state: best hyperparameters
│   ├── program.md           ← Detailed research protocol
│   └── pyproject.toml       ← uv-managed dependencies
│
├── OpenMythos/              Reference implementation (kyegomez/OpenMythos)
│   └── open_mythos/main.py  ← Original RDT architecture to study
│
├── autoresearch/            Reference framework (karpathy/autoresearch)
│   └── train.py             ← Original autoresearch pattern to follow
│
└── Ilya Sutskever...txt     Interview transcript — theoretical motivation
```

## The Research Strategy

### What we're doing
Training small (50-500M) Recurrent-Depth Transformers on **MathNet** — a dataset of 28K olympiad-level math problems with solutions (18.5M tokens). We use Karpathy's autoresearch pattern: an AI agent autonomously modifies the architecture, trains for 5 minutes, keeps improvements, discards failures, and repeats forever.

### Why math
Math olympiad problems require multi-step reasoning — exactly what recurrent depth should excel at. If a 200M RDT with 16 reasoning loops can solve problems that a standard 200M GPT can't, that's evidence the architecture works. That signal is worth scaling to 4-8B on production GPUs.

### Why RDT (Recurrent-Depth Transformer)
A standard transformer stacks N unique layers. An RDT has:
- **Prelude**: A few standard layers to encode the input
- **Recurrent Core**: ONE transformer block looped N times (same weights, reused)
- **Coda**: A few standard layers to decode the output

The recurrent core also features:
- **LTI Injection**: A stable hidden state (`h = A*h + B*input`) that accumulates across loops — like a working memory
- **LoRA Adaptation**: Per-loop specialization without separate weights
- **MoE FFN**: Sparse expert routing for parameter efficiency

This is like giving the model a "thinking budget" — more loops = more time to reason, without adding parameters.

### Why two machines
We run experiments on two machines simultaneously:

| Machine | Framework | Strength | Train file |
|---|---|---|---|
| **Mac M2 16GB** | MLX | Bigger models (up to ~500M) | `train.py` |
| **RTX 4070 8GB** | PyTorch/CUDA | 10-20x faster iteration | `train_cuda.py` |

They sync via git — when one machine finds a better config, the other pulls it and builds on top.

## Quick Start for an Agent

### Step 1: Go to the project
```bash
cd autoresearch-mythos
```

### Step 2: Read the instructions
```bash
cat AGENT.md
```

### Step 3: Set up (one-time per machine)
```bash
uv sync                # install dependencies
uv run prepare.py      # download MathNet + tokenize (takes ~3 min)
```

### Step 4: Start the experiment loop
```bash
python sync.py pull    # get latest from other machine
# Then follow the loop in AGENT.md — modify train file, run, keep or discard, repeat forever
```

## Current Status

- **Baseline established**: val_bpb = 24.05 (Mac M2, 47.9M params, 45 steps in 5 min)
- **Architecture**: prelude=2, core×8 loops, coda=2, MoE (4 experts top-2), LTI on, LoRA rank 16
- **Dataset**: MathNet v0, 27,817 problems, 18.5M tokens (stored in ~/.cache/autoresearch-mythos/)
- **Next step**: Run the autonomous experiment loop to find the optimal RDT configuration

## Key Research Questions to Answer

1. **Does recurrent depth help?** Compare 8-layer GPT vs 1-block × 8-loops RDT (same compute)
2. **Optimal loop count?** Sweep N_LOOPS = 2, 4, 8, 16, 32
3. **MoE vs Dense at small scale?** Does sparsity help or hurt at 50-200M params?
4. **Does LTI injection matter?** Compare with/without hidden state accumulation
5. **Depth extrapolation?** Train with 4 loops, eval with 8 — does val_bpb improve?

## Theoretical Motivation (from Ilya Sutskever)

> "We're moving from the age of scaling to the age of research."
> "AlexNet was built on two GPUs. The transformer was built on 8 to 64 GPUs. It's far from obvious that you need the largest compute for research."
> "These models generalize dramatically worse than people."

Full transcript: `[English] Ilya Sutskever...txt` in this directory.

The insight: you don't need H100 clusters to discover if recurrent-depth is a better architecture. You need an idea and a way to test it quickly. That's what this project does — 100 architecture experiments overnight while you sleep.

## Repository Links

- **This project**: https://github.com/debpalash/autoresearch-mythos (private)
- **OpenMythos**: https://github.com/kyegomez/OpenMythos (reference)
- **Autoresearch**: https://github.com/karpathy/autoresearch (reference)
- **MathNet dataset**: https://huggingface.co/datasets/ShadenA/MathNet
