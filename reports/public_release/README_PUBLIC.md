# ExactKV Public Benchmark Release

ExactKV is a compressor-agnostic crash-test and leaderboard framework for LLM
KV-cache compression. It measures token-level drift, first divergence, acceptance
rate, verifier agreement, and exactness failures across compressors and models.

**Not a production serving system.** ExactKV does not reproduce VeriCache serving throughput.

## Quick start

```bash
python scripts/exactkv.py run full --deterministic
python scripts/exactkv.py run publish
```

## Current snapshot

- **Total benchmark cells:** 336
- **Models evaluated:** Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-0.5B-Instruct, mistralai/Mistral-7B-Instruct-v0.3, meta-llama/Llama-3.1-8B
- **Deterministic mode:** False
- **Divergence authority:** Phase G `FirstDivergenceAuthority` (canonical)

## Artifacts

| File | Description |
|------|-------------|
| `leaderboard_final.json` | Ranked compressor × model scores |
| `benchmark_summary.md` | Aggregate metrics |
| `methodology.md` | Evaluation methodology |
| `repro_command.sh` | One-command reproduction |

## Claims policy

No end-to-end speedup, latency, or active GPU memory savings claims. Phase F results (when cited) are kernel microbenchmark only. Compression ratios are stored tensor byte ratios unless active GPU memory is explicitly measured. SpectralQuant: fallback/proxy when dependency unavailable. Shard: probe-first analysis only.

Generated: 2026-06-25T14:43:09.307725+00:00
