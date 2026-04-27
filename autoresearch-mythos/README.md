# autoresearch-mythos

Autonomous AI research on **Recurrent-Depth Transformers (RDT)** trained on [MathNet](https://huggingface.co/datasets/ShadenA/MathNet) olympiad data.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and [OpenMythos](https://github.com/kyegomez/OpenMythos).

## Architecture

The RDT replaces a standard stacked transformer with a **recurrent core** — one transformer block looped N times, with:
- **LTI injection**: Stable hidden state accumulation across loops
- **LoRA adaptation**: Per-loop specialization without separate weights
- **MoE FFN**: Sparse expert routing for parameter efficiency

## Dual-Machine Setup

| Machine | Command | Framework |
|---|---|---|
| Mac M2 16GB | `uv run train.py` | MLX |
| RTX 4070 8GB | `uv run train_cuda.py` | PyTorch/CUDA |

## Quick Start

```bash
# Install deps
uv sync

# One-time data prep (downloads MathNet, tokenizes)
uv run prepare.py

# Train (Mac)
uv run train.py

# Train (CUDA laptop)
uv run train_cuda.py
```

See `program.md` for the full autonomous research protocol.
