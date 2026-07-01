# External Panel: Mbpp

**Status:** benchmark_complete
**Cells:** 96 total, 96 ok
**ExactKV failures:** 0

## Per context bucket

| Bucket | Cells | Divergence rate | Mean acceptance |
|--------|------:|----------------:|----------------:|
| 512 | 48 | 0.500 | 0.561 |
| 1024 | 48 | 0.542 | 0.571 |

## Compressor summary

- `int8`: acceptance=1.000, divergence_rate=0.000, cells=32
- `kivi_offline_r32`: acceptance=0.001, divergence_rate=1.000, cells=32
- `snapkv_experimental`: acceptance=0.696, divergence_rate=0.562, cells=32

