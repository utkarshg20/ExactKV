# External Panel: Bfcl

**Status:** benchmark_complete
**Cells:** 120 total, 120 ok
**ExactKV failures:** 0

## Per context bucket

| Bucket | Cells | Divergence rate | Mean acceptance |
|--------|------:|----------------:|----------------:|
| 512 | 60 | 0.667 | 0.471 |
| 1024 | 60 | 0.667 | 0.493 |

## Compressor summary

- `int8`: acceptance=1.000, divergence_rate=0.000, cells=40
- `kivi_offline_r32`: acceptance=0.002, divergence_rate=1.000, cells=40
- `snapkv_experimental`: acceptance=0.446, divergence_rate=1.000, cells=40

