# Recommended Training Datasets

| Dataset | HuggingFace | Tokens | License | Use |
|---|---|---|---|---|
| FineWeb-Edu | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 1.3T | Apache 2.0 | Primary pretraining |
| OpenHermes 2.5 | [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | ~1M samples | Apache 2.0 | Instruction tuning (~5% mix) |
| OpenWebMath | [open-web-math/open-web-math](https://huggingface.co/datasets/open-web-math/open-web-math) | 14.7B | ODC-By | Math/reasoning boost |

---

## Primary Pretraining

### FineWeb-Edu
- **HuggingFace:** `HuggingFaceFW/fineweb-edu`
- **Size:** 1.3T tokens
- **License:** Apache 2.0
- **Why:** Web text filtered for educational quality. Outperforms The Pile, C4, and RefinedWeb on downstream benchmarks. Already deduplicated and cleaned.
- **Start with:** `sample-10BT` to validate your pipeline, then `sample-100BT` or the full corpus for a serious run.

## Supplementary

### OpenHermes 2.5
- **HuggingFace:** `teknium/OpenHermes-2.5`
- **Size:** ~1M instruction samples
- **License:** Apache 2.0
- **Why:** High-quality instruction-following data. Mix in ~5% by token count on top of FineWeb-Edu to improve instruction following without degrading general capability.

### OpenWebMath
- **HuggingFace:** `open-web-math/open-web-math`
- **Size:** ~14.7B tokens
- **License:** ODC-By
- **Why:** Math-focused web text. Add if you want stronger quantitative and symbolic reasoning. Particularly useful for the 10B+ variants where reasoning depth matters.

## Token Budget Recommendations

| Variant | Chinchilla-optimal | Recommended (looped) |
|---|---|---|
| 1B | ~20B tokens | ~10–15B tokens |
| 3B | ~60B tokens | ~30–40B tokens |
| 10B | ~200B tokens | ~100–150B tokens |
| 50B+ | ~1T+ tokens | ~500B+ tokens |

The looped architecture is more sample-efficient than a standard transformer — same validation loss is reachable with fewer tokens due to faster convergence. The "recommended (looped)" column reflects this and is based on the Tiny Shakespeare result where OpenMythos reached equivalent loss ~2.5× faster than nanoGPT.
