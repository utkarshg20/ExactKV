# External Panel: Longbench

**Status:** benchmark_in_progress
**Cells:** 10 total, 10 ok
**ExactKV failures:** 0

## Per context bucket

| Bucket | Cells | Divergence rate | Mean acceptance |
|--------|------:|----------------:|----------------:|
| 2048 | 9 | 0.333 | 0.906 |
| 4096 | 1 | 0.000 | 1.000 |

## Compressor summary

- `int4_sim`: acceptance=0.719, divergence_rate=1.000, cells=3
- `int8`: acceptance=1.000, divergence_rate=0.000, cells=3
- `noop`: acceptance=1.000, divergence_rate=0.000, cells=4

