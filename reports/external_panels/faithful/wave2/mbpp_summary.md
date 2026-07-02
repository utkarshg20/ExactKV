# External Panel: Mbpp

**Status:** benchmark_complete
**Cells:** 64 total, 64 ok
**ExactKV failures:** 0

## Per context bucket

| Bucket | Cells | Divergence rate | Mean acceptance |
|--------|------:|----------------:|----------------:|
| 512 | 32 | 0.375 | 0.785 |
| 1024 | 32 | 0.469 | 0.765 |

## Compressor summary

- `int8`: acceptance=1.000, divergence_rate=0.000, cells=16
- `kvpress_knorm_experimental`: acceptance=0.561, divergence_rate=0.750, cells=16
- `snapkv_experimental`: acceptance=0.547, divergence_rate=0.875, cells=16
- `turboquant_experimental`: acceptance=0.993, divergence_rate=0.062, cells=16

