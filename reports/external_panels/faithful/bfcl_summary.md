# External Panel: Bfcl

**Status:** benchmark_in_progress
**Cells:** 98 total, 98 ok
**ExactKV failures:** 0

## Per context bucket

| Bucket | Cells | Divergence rate | Mean acceptance |
|--------|------:|----------------:|----------------:|
| 512 | 53 | 0.623 | 0.490 |
| 1024 | 45 | 0.667 | 0.491 |

## Compressor summary

- `int8`: acceptance=1.000, divergence_rate=0.000, cells=35
- `kivi_offline_r32`: acceptance=0.032, divergence_rate=1.000, cells=33
- `snapkv_experimental`: acceptance=0.400, divergence_rate=1.000, cells=30

