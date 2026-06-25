# ExactKV Unified Benchmark

**Status:** benchmark_complete
**Run ID:** `benchmark_bd20799f820e8d78`
**Config hash:** `bd20799f820e8d78`
**Git commit:** `f2741332d2e1e6454b6a7b3481757e5b753fc47c`
**Total cells:** 336
**ExactKV failure rate:** 0.0000

## Compressors

`noop`, `int8`, `int4_sim`, `k8_v4_sim`, `spectralquant`, `kvquant`, `shard`

## Models

`Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-0.5B-Instruct`, `meta-llama/Llama-3.1-8B`, `mistralai/Mistral-7B-Instruct-v0.3`

## Aggregate acceptance (by compressor)

| Compressor | Mean acceptance | Mean first divergence | Failures |
|------------|----------------:|----------------------:|---------:|
| `int4_sim` | 0.426 | n/a | 0 |
| `int8` | 0.779 | n/a | 0 |
| `k8_v4_sim` | 0.601 | n/a | 0 |
| `kvquant` | 0.534 | n/a | 0 |
| `noop` | 0.775 | n/a | 0 |
| `shard` | 0.686 | n/a | 0 |
| `spectralquant` | 0.585 | n/a | 0 |

## Reproducibility

```bash
python scripts/exactkv.py run benchmark --deterministic
```
