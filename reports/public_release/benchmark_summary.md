# Benchmark Summary

**Cells:** 336
**Failure rate:** 0

## Top leaderboard entries

- `noop` on Llama-3.1-8B — score 1.0 acceptance 1.0
- `int8` on Llama-3.1-8B — score 1.0 acceptance 1.0
- `int4_sim` on Llama-3.1-8B — score 0.6839 acceptance 0.8515
- `spectralquant` on Llama-3.1-8B — score 0.6839 acceptance 0.8515
- `shard` on Llama-3.1-8B — score 0.544 acceptance 0.6323
- `noop` on Mistral-7B — score None acceptance None
- `int8` on Mistral-7B — score None acceptance None
- `int4_sim` on Mistral-7B — score None acceptance None
- `spectralquant` on Mistral-7B — score None acceptance None
- `shard` on Mistral-7B — score None acceptance None

## Claim boundaries

SpectralQuant rows use **fallback/proxy** mode when the real dependency is unavailable. Shard rows are **probe-first** heuristic analysis, not a full Shard integration. Compression ratios in source reports are **stored tensor byte ratios** unless active GPU memory is explicitly measured.
