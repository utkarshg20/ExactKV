# External Panel: Mbpp

**Status:** benchmark_in_progress
**Cells:** 20 total, 20 ok
**ExactKV failures:** 0

## Per context bucket

| Bucket | Cells | Divergence rate | Mean acceptance |
|--------|------:|----------------:|----------------:|
| 512 | 12 | 0.333 | 0.772 |
| 1024 | 8 | 0.375 | 0.730 |

## Compressor summary

- `int8`: acceptance=1.000, divergence_rate=0.000, cells=6
- `kvpress_knorm_experimental`: acceptance=0.572, divergence_rate=0.333, cells=6
- `snapkv_experimental`: acceptance=0.445, divergence_rate=1.000, cells=4
- `turboquant_experimental`: acceptance=0.971, divergence_rate=0.250, cells=4

