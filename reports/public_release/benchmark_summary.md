# Benchmark Summary

**Public release cells:** 1500 (Phase H+ scale_7b real GPU benchmark)
**ExactKV failures:** 0
**Historical Phase A cells (internal):** 336

## Top leaderboard entries

- `noop` on Llama-3.1-8B — score 1.0 acceptance 1.0
- `int8` on Llama-3.1-8B — score 1.0 acceptance 1.0
- `noop` on Mistral-7B — score 1.0 acceptance 1.0
- `int8` on Mistral-7B — score 0.9827 acceptance 1.0
- `int4_sim` on Llama-3.1-8B — score 0.8589 acceptance 0.8515
- `spectralquant` on Llama-3.1-8B — score 0.8589 acceptance 0.8515
- `int4_sim` on Mistral-7B — score 0.8512 acceptance 0.8365
- `spectralquant` on Mistral-7B — score 0.8512 acceptance 0.8365
- `shard` on Llama-3.1-8B — score 0.7315 acceptance 0.6323
- `shard` on Mistral-7B — score 0.7268 acceptance 0.6233

## Claim boundaries

SpectralQuant rows use **fallback/proxy** mode when the real dependency is unavailable. Shard rows are **probe-first** heuristic analysis, not a full Shard integration. Compression ratios in source reports are **stored tensor byte ratios** unless active GPU memory is explicitly measured.
