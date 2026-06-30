# External Panel: Bfcl

**Status:** benchmark_in_progress
**Cells:** 68 total, 68 ok
**ExactKV failures:** 0

## Per context bucket

| Bucket | Cells | Divergence rate | Mean acceptance |
|--------|------:|----------------:|----------------:|
| 512 | 38 | 0.605 | 0.505 |
| 1024 | 30 | 0.667 | 0.485 |

## Compressor summary

- `int8`: acceptance=1.000, divergence_rate=0.000, cells=25
- `kivi_offline_r32`: acceptance=0.036, divergence_rate=1.000, cells=23
- `snapkv_experimental`: acceptance=0.394, divergence_rate=1.000, cells=20

