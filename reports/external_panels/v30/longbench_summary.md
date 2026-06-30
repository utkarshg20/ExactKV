# External Panel: Longbench

**Status:** benchmark_complete
**Cells:** 288 total, 288 ok
**ExactKV failures:** 0

## Per context bucket

| Bucket | Cells | Divergence rate | Mean acceptance |
|--------|------:|----------------:|----------------:|
| 2048 | 96 | 0.583 | 0.924 |
| 4096 | 96 | 0.479 | 0.932 |
| 8192 | 96 | 0.510 | 0.923 |

## Compressor summary

- `int4_per_vec_sim`: acceptance=0.935, divergence_rate=0.569, cells=72
- `int4_sim`: acceptance=0.825, divergence_rate=0.847, cells=72
- `int6_sim`: acceptance=0.957, divergence_rate=0.472, cells=72
- `int8`: acceptance=0.988, divergence_rate=0.208, cells=72

