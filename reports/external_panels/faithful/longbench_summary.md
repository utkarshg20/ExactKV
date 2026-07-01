# External Panel: Longbench

**Status:** benchmark_complete
**Cells:** 216 total, 216 ok
**ExactKV failures:** 0

## Per context bucket

| Bucket | Cells | Divergence rate | Mean acceptance |
|--------|------:|----------------:|----------------:|
| 2048 | 72 | 0.750 | 0.440 |
| 4096 | 72 | 0.708 | 0.475 |
| 8192 | 72 | 0.722 | 0.458 |

## Compressor summary

- `int8`: acceptance=0.992, divergence_rate=0.181, cells=72
- `kivi_offline_r32`: acceptance=0.010, divergence_rate=1.000, cells=72
- `snapkv_experimental`: acceptance=0.371, divergence_rate=1.000, cells=72

