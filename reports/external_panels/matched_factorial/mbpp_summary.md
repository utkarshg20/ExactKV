# External Panel: Mbpp

**Status:** benchmark_complete
**Cells:** 162 total, 162 ok
**ExactKV failures:** 0

## Per context bucket

| Bucket | Cells | Divergence rate | Mean acceptance |
|--------|------:|----------------:|----------------:|
| 2048 | 54 | 0.056 | 1.000 |
| 4096 | 54 | 0.056 | 0.999 |
| 8192 | 54 | 0.056 | 0.999 |

## Compressor summary

- `int4_sim`: acceptance=0.997, divergence_rate=0.167, cells=54
- `int8`: acceptance=1.000, divergence_rate=0.000, cells=54
- `noop`: acceptance=1.000, divergence_rate=0.000, cells=54

