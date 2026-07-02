# External Panel: Bfcl

**Status:** benchmark_complete
**Cells:** 64 total, 64 ok
**ExactKV failures:** 0

## Per context bucket

| Bucket | Cells | Divergence rate | Mean acceptance |
|--------|------:|----------------:|----------------:|
| 512 | 32 | 0.500 | 0.740 |
| 1024 | 32 | 0.406 | 0.740 |

## Compressor summary

- `int8`: acceptance=1.000, divergence_rate=0.000, cells=16
- `kvpress_knorm_experimental`: acceptance=0.556, divergence_rate=0.812, cells=16
- `snapkv_experimental`: acceptance=0.403, divergence_rate=1.000, cells=16
- `turboquant_experimental`: acceptance=1.000, divergence_rate=0.000, cells=16

