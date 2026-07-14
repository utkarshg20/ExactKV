# External Panel: Longbench

**Status:** benchmark_complete
**Cells:** 270 total, 270 ok
**ExactKV failures:** 0

## Per context bucket

| Bucket | Cells | Divergence rate | Mean acceptance |
|--------|------:|----------------:|----------------:|
| 2048 | 90 | 0.444 | 0.931 |
| 4096 | 90 | 0.467 | 0.937 |
| 8192 | 90 | 0.444 | 0.945 |

## Compressor summary

- `int4_sim`: acceptance=0.826, divergence_rate=0.911, cells=90
- `int8`: acceptance=0.987, divergence_rate=0.444, cells=90
- `noop`: acceptance=1.000, divergence_rate=0.000, cells=90

