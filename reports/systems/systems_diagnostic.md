# Systems diagnostic pack

**Cells:** 96 · **exactkv_failures:** 0

**Claim boundary:** Diagnostic peak CUDA allocation and per-path wall-clock on the systems_diagnostic panel (7B/8B). NOT serving throughput, TTFT, RPS, or unqualified production VRAM savings. Peak includes model weights + KV + temporaries.

## Peak CUDA allocation (GiB, mean)

| Model | Compressor | full | lossy | ExactKV |
|-------|------------|-----:|------:|--------:|
| llama | `int4_sim` | 16.099 | 16.673 | 16.716 |
| llama | `int8` | 16.099 | 16.673 | 16.715 |
| llama | `noop` | 16.099 | 16.494 | 16.494 |
| mistral | `int4_sim` | 14.227 | 15.216 | 15.261 |
| mistral | `int8` | 14.227 | 15.216 | 15.261 |
| mistral | `noop` | 14.227 | 14.672 | 14.683 |

## Path wall-clock (ms, mean)

| Model | Compressor | full | lossy | ExactKV |
|-------|------------|-----:|------:|--------:|
| llama | `int4_sim` | 3660.013 | 3674.747 | 8571.778 |
| llama | `int8` | 3691.411 | 3712.755 | 8688.918 |
| llama | `noop` | 3689.776 | 3746.195 | 8472.187 |
| mistral | `int4_sim` | 3534.743 | 3140.91 | 8345.277 |
| mistral | `int8` | 3571.187 | 3584.654 | 8388.943 |
| mistral | `noop` | 3565.501 | 3620.175 | 8275.323 |

## Notes

- Peak is process-level torch.cuda.max_memory_allocated (weights+KV+temps).
- ExactKV arm often peaks higher than lossy-only because full+compressed state coexist.
- Wall-clock is harness path timing, not serving TTFT/RPS.
