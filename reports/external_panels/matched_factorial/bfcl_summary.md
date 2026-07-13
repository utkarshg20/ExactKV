# External Panel: Bfcl

**Status:** benchmark_complete
**Cells:** 270 total, 270 ok
**ExactKV failures:** 0

## Per context bucket

| Bucket | Cells | Divergence rate | Mean acceptance |
|--------|------:|----------------:|----------------:|
| 2048 | 90 | 0.144 | 1.000 |
| 4096 | 90 | 0.056 | 0.998 |
| 8192 | 90 | 0.056 | 0.997 |

## Compressor summary

- `int4_sim`: acceptance=0.995, divergence_rate=0.256, cells=90
- `int8`: acceptance=1.000, divergence_rate=0.000, cells=90
- `noop`: acceptance=1.000, divergence_rate=0.000, cells=90

