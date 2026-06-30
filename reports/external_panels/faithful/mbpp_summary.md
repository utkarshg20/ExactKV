# External Panel: Mbpp

**Status:** benchmark_complete
**Cells:** 48 total, 48 ok
**ExactKV failures:** 0

## Per context bucket

| Bucket | Cells | Divergence rate | Mean acceptance |
|--------|------:|----------------:|----------------:|
| 512 | 24 | 0.583 | 0.533 |
| 1024 | 24 | 0.667 | 0.510 |

## Compressor summary

- `int8`: acceptance=1.000, divergence_rate=0.000, cells=16
- `kivi_offline_r32`: acceptance=0.023, divergence_rate=1.000, cells=16
- `snapkv_experimental`: acceptance=0.541, divergence_rate=0.875, cells=16

