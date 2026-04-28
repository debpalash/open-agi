best config
-----

dim=512, heads=8, prelude=2, coda=2, loops=1
Dense FFN (ReluSquared), no MoE/LTI/LoRA
LR=3e-3, WD=0.1, warmup=5%, warmdown=30%
BATCH=16, TOTAL_BATCH=16384
val_bpb = 5.954799


---


Here's the full session summary:

CUDA Experiment Session Summary
19 experiments run on RTX 4070. Best val_bpb: 5.95 (down from Mac's 15.96).

What Worked ✅
Change	val_bpb	Improvement
CUDA baseline (mac config)	7.40	—
dim 256→384	7.13	-0.27
dim 384→512	7.10	-0.03
ReluSquared FFN (from Karpathy)	6.41	-0.69 ⭐
LR 3e-4→1e-3	6.01	-0.40
LR 1e-3→3e-3	5.95	-0.06
What Didn't Work ❌
Change	Result	Why
Logit soft-capping	6.81	Limits confident predictions
More depth (7 layers)	7.11	Slower, fewer steps
dim=1024 batch=64	9.12	Only 276 steps!
dim=768	7.97	Too slow
QK-norm (5 min)	6.63	Helps at 2min but hurts at 5min
Embed norm	7.75	Hurts
LR=1e-2	8.85	Too high
WD=0.0	8.13	Regularization still helps
TOTAL_BATCH 32K	6.20	Fewer steps
Key Insights from Reference Code
Karpathy's ReluSquared was the single biggest improvement (-0.69)
Model size sweet spot is ~67M params (dim=512) — bigger models can't train enough in 5 min
LR sweet spot is 3e-3 — 10x higher than the original Mac config
The CUDA machine is ~16x faster than Mac, so it gets ~10x more training steps in the same budget
Current Best Config
dim=512, heads=8, prelude=2, coda=2, loops=1
Dense FFN (ReluSquared), no MoE/LTI/LoRA
LR=3e-3, WD=0.1, warmup=5%, warmdown=30%
BATCH=16, TOTAL_BATCH=16384
val_bpb = 5.954799
Want me to keep iterating? I could try: cosine schedule, Adam beta tuning, weight tying, prelude=1/coda=1 (reduce depth for speed), or dive back into the RDT recurrence with N_LOOPS=2.

