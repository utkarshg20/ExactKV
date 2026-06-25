# Phase A Scale Benchmark Summary

**Status:** benchmark_complete
**Mode:** inference
**Models evaluated:** 4
**Total cells:** 336

## Compressor Comparison

| Compressor | Acceptance | Divergence Stability | Failure Rate | Verifier Agreement | Instability (Exp116) |
|------------|------------|----------------------|--------------|--------------------|-----------------------|
| noop | 1.000 | 1.000 | 0.000 | 1.000 | n/a |
| int8 | 0.995 | 0.917 | 0.000 | 0.995 | n/a |
| int4_sim | 0.736 | 0.417 | 0.000 | 0.736 | n/a |
| k8_v4_sim | 0.948 | 0.667 | 0.000 | 0.948 | n/a |
| spectralquant | 0.736 | 0.417 | 0.000 | 0.736 | n/a |
| kvquant | 0.736 | 0.417 | 0.000 | 0.736 | n/a |
| shard | 0.609 | 0.729 | 0.000 | 0.521 | n/a |

## Rankings

### By acceptance rate

1. `noop` — 1.000
2. `int8` — 0.995
3. `k8_v4_sim` — 0.948
4. `int4_sim` — 0.736
5. `kvquant` — 0.736
6. `spectralquant` — 0.736
7. `shard` — 0.609

### By divergence stability

1. `noop` — 1.000
2. `int8` — 0.917
3. `shard` — 0.729
4. `k8_v4_sim` — 0.667
5. `int4_sim` — 0.417
6. `kvquant` — 0.417
7. `spectralquant` — 0.417

### By failure rate (lower is better)

1. `int4_sim` — 0.000
2. `int8` — 0.000
3. `k8_v4_sim` — 0.000
4. `kvquant` — 0.000
5. `noop` — 0.000
6. `shard` — 0.000
7. `spectralquant` — 0.000

## Reproducibility

```bash
python scripts/run_phase_a_scale_benchmark.py --deterministic-mode --device cuda
```

Phase A reports token-level divergence and acceptance only. No speed or memory savings claims unless directly measured. External compressors may use mock/probe fallbacks when adapters are unavailable.
