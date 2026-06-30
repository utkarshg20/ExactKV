# External Panel: Longbench

**Status:** benchmark_complete
**Cells:** 216 total, 216 ok
**ExactKV failures:** 0

## Per context bucket

| Bucket | Cells | Divergence rate | Mean acceptance |
|--------|------:|----------------:|----------------:|
| 2048 | 72 | 0.681 | 0.464 |
| 4096 | 72 | 0.736 | 0.460 |
| 8192 | 72 | 0.736 | 0.473 |

## Compressor summary

- `int8`: acceptance=0.985, divergence_rate=0.153, cells=72
- `kivi_offline_r32`: acceptance=0.052, divergence_rate=1.000, cells=72
- `snapkv_experimental`: acceptance=0.360, divergence_rate=1.000, cells=72

