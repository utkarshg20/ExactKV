# ExactKV Public Benchmark Release

ExactKV is a reproducible KV compression benchmarking platform with plugin-based
compressors, standardized evaluation, and public leaderboard generation.

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

No speedup, latency, or memory savings claims unless directly measured in Phase F.
Token-level acceptance and divergence metrics only.

Generated: 2026-06-25T14:43:09.307725+00:00
