# External Panel: Mbpp

**Status:** benchmark_complete
**Cells:** 96 total, 96 ok
**ExactKV failures:** 0

## Per context bucket

| Bucket | Cells | Divergence rate | Mean acceptance |
|--------|------:|----------------:|----------------:|
| 512 | 48 | 0.000 | 1.000 |
| 1024 | 48 | 0.062 | 0.998 |

## Compressor summary

- `int4_per_vec_sim`: acceptance=1.000, divergence_rate=0.000, cells=24
- `int4_sim`: acceptance=0.996, divergence_rate=0.125, cells=24
- `int6_sim`: acceptance=1.000, divergence_rate=0.000, cells=24
- `int8`: acceptance=1.000, divergence_rate=0.000, cells=24

