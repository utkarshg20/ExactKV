# ExactKV Evidence Ledger (Release Synthesis — Part 3)

Maps every public claim to its on-disk evidence, strength, required caveat, and exact safe wording. Machine-readable: [`evidence_ledger.json`](evidence_ledger.json).

## Headline facts (from `reports/scale_7b/raw.json`)

- **Total cells:** 1500
- **Models:** meta-llama/Llama-3.1-8B, mistralai/Mistral-7B-Instruct-v0.3
- **Compressors:** noop, int8, int4_sim, spectralquant (MOCK->int4_sim), shard (PROBE_ONLY)
- **ExactKV failures:** 0
- **Deterministic mode:** False
- **Stack:** torch 2.8.0+cu128, transformers 5.12.1
- **Kernel microbenchmark:** int8 1.63x, int4 1.54x, block_sparse 0.98x on kv_shape=[1, 8, 512, 64] — kernel microbenchmark only; NOT end-to-end

### Acceptance summary (scale panel)

| Compressor | Acceptance | Divergence rate | First divergence | Backend |
|------------|-----------:|----------------:|-----------------:|---------|
| `noop` | 1.0 | 0.0 | None | BUILTIN |
| `int8` | 1.0 | 0.0 | None | BUILTIN |
| `int4_sim` | 0.851 | 0.52 | 2.0 | BUILTIN |
| `spectralquant` | 0.851 | 0.52 | 2.0 | MOCK |
| `shard` | 0.632 | — | 1.0 | PROBE_ONLY |

## Claim ledger

| Claim | Evidence | Strength | Caveat req. | Public-safe wording | Forbidden wording |
|-------|----------|----------|:-----------:|---------------------|-------------------|
| Compressor-agnostic token-level drift benchmark | `reports/scale_7b/raw.json; docs/CLAIM_BOUNDARIES.md` | strong | no | ExactKV measures token-level drift across compressors and models. | — |
| 1500-cell real 7B/8B benchmark | `reports/scale_7b/raw.json (total_cells=1500)` | strong | yes | 1500 cells across two real 7B/8B models, deterministic_mode=false. | — |
| Zero exactness failures | `reports/scale_7b/raw.json (exactkv_failures=0)` | strong | yes | Zero exactness failures on the tested panel. | always exact / never fails |
| First-divergence measurement | `docs/METRIC_DEFINITIONS.md; reports/phaseG_unified_truth.json` | strong | no | first_divergence_index is the canonical token-drift metric. | — |
| Acceptance / verifier agreement leaderboard | `reports/public_release/leaderboard_final.json` | strong | yes | Reproducible public leaderboard bundle (repo-local). | hosted SaaS leaderboard |
| Kernel microbenchmark | `reports/phaseF_kernel_benchmark.json` | moderate | yes | Kernel microbenchmark speedups on a fixed kv_shape. | end-to-end speedup / faster inference |
| Compression ratio | `reports/scale_7b/raw.json (compression_ratio); docs/METRIC_DEFINITIONS.md` | moderate | yes | Compression ratios are stored tensor byte ratios. | active GPU memory savings / VRAM savings |
| SpectralQuant support | `reports/scale_7b/raw.json (backend_tier=MOCK)` | moderate | yes | SpectralQuant runs in fallback/proxy mode when the dependency is unavailable. | real SpectralQuant |
| Shard support | `reports/scale_7b/raw.json (backend_tier=PROBE_ONLY)` | moderate | yes | Shard is a probe-first heuristic, not full Shard/ShardCache integration. | real Shard / beats Shard |
| VeriCache relationship | `paper/VeriCache.pdf; docs/VERICACHE_PARITY_CLAIM_GATE.md` | strong | yes | VeriCache owns compressed draft + full-KV verify for lossless **serving**; ExactKV measures drift, does not reproduce VeriCache. | invents draft+verify / reproduces VeriCache / beats VeriCache |
| Production / serving | `docs/CLAIM_BOUNDARIES.md` | strong | yes | Not a production serving system. | production ready / production serving |
| Uniqueness | `docs/NOVELTY_AUDIT.md` | weak | yes | Distinct positioning vs prior art; uniqueness not proven. | first ever / first and only / nothing like this exists |

## Notes

- **Compressor-agnostic token-level drift benchmark:** Core positioning.
- **1500-cell real 7B/8B benchmark:** Sequential model execution (volume constraint).
- **Zero exactness failures:** Panel-scoped hard gate; not a universal guarantee.
- **First-divergence measurement:** Phase G FirstDivergenceAuthority.
- **Acceptance / verifier agreement leaderboard:** Locked composite scoring.
- **Kernel microbenchmark:** block_sparse uses torch backend (0.98x).
- **Compression ratio:** VRAM not measured.
- **SpectralQuant support:** spectralquant_available=False.
- **Shard support:** probe_only=true.
- **VeriCache relationship:** High algorithmic overlap; VeriCache is serving/system prior art. ExactKV novelty is diagnostic measurement only.
- **Production / serving:** vLLM/LMCache are probes only.
- **Uniqueness:** Uniqueness not established.
