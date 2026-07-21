# Serving microbench pack

**Cells:** 76 · **exactkv_failures:** 0

**Claim boundary:** HF multi-request serving microbench (ExactKV harness): TTFT-like latency, completed-requests/sec under serial load, and peak CUDA allocation for full / lossy / ExactKV. NOT vLLM integration, NOT continuous batching, NOT production serving, NOT unqualified VRAM savings. Peak includes weights + KV + temporaries.

## Completed requests/sec (mean, serial load)

| Model | Comp | Load | full | lossy | ExactKV |
|-------|------|------|-----:|------:|--------:|
| llama | `int4_sim` | serial_1 | 0.247 | 0.246 | 0.129 |
| llama | `int4_sim` | serial_16 | 0.347 | 0.345 | 0.177 |
| llama | `int4_sim` | serial_4 | 0.247 | 0.246 | 0.129 |
| llama | `int4_sim` | serial_8 | 0.247 | 0.246 | 0.129 |
| llama | `int8` | serial_1 | 0.247 | 0.244 | 0.129 |
| llama | `int8` | serial_16 | 0.344 | 0.346 | 0.176 |
| llama | `int8` | serial_4 | 0.247 | 0.246 | 0.129 |
| llama | `int8` | serial_8 | 0.247 | 0.246 | 0.129 |
| llama | `noop` | serial_1 | 0.234 | 0.247 | 0.131 |
| llama | `noop` | serial_4 | 0.247 | 0.247 | 0.132 |
| llama | `noop` | serial_8 | 0.247 | 0.247 | 0.132 |
| mistral | `int4_sim` | serial_1 | 0.253 | 0.371 | 0.132 |
| mistral | `int4_sim` | serial_16 | 0.358 | 0.419 | 0.182 |
| mistral | `int4_sim` | serial_4 | 0.253 | 0.37 | 0.132 |
| mistral | `int4_sim` | serial_8 | 0.253 | 0.371 | 0.133 |
| mistral | `int8` | serial_1 | 0.253 | 0.249 | 0.134 |
| mistral | `int8` | serial_16 | 0.353 | 0.356 | 0.183 |
| mistral | `int8` | serial_4 | 0.253 | 0.251 | 0.134 |
| mistral | `int8` | serial_8 | 0.253 | 0.251 | 0.134 |
| mistral | `noop` | serial_1 | 0.238 | 0.253 | 0.135 |
| mistral | `noop` | serial_4 | 0.253 | 0.252 | 0.137 |
| mistral | `noop` | serial_8 | 0.254 | 0.253 | 0.137 |

## Peak CUDA (GiB, mean) and Δ vs full

| Model | Comp | Load | full | lossy | ExactKV | Δlossy | ΔExactKV |
|-------|------|------|-----:|------:|--------:|-------:|---------:|
| llama | `int4_sim` | serial_1 | 16.424 | 17.155 | 16.648 | 0.732 | 0.225 |
| llama | `int4_sim` | serial_16 | 16.196 | 16.43 | 16.089 | 0.234 | -0.107 |
| llama | `int4_sim` | serial_4 | 16.799 | 17.155 | 16.647 | 0.356 | -0.152 |
| llama | `int4_sim` | serial_8 | 16.799 | 17.155 | 16.647 | 0.356 | -0.151 |
| llama | `int8` | serial_1 | 16.424 | 17.155 | 16.648 | 0.732 | 0.225 |
| llama | `int8` | serial_16 | 16.196 | 16.431 | 16.09 | 0.235 | -0.106 |
| llama | `int8` | serial_4 | 16.799 | 17.155 | 16.646 | 0.356 | -0.152 |
| llama | `int8` | serial_8 | 16.799 | 17.155 | 16.647 | 0.356 | -0.152 |
| llama | `noop` | serial_1 | 16.424 | 16.798 | 16.436 | 0.374 | 0.013 |
| llama | `noop` | serial_4 | 16.8 | 16.8 | 16.437 | -0.001 | -0.363 |
| llama | `noop` | serial_8 | 16.8 | 16.799 | 16.436 | -0.002 | -0.364 |
| mistral | `int4_sim` | serial_1 | 14.601 | 15.773 | 15.254 | 1.172 | 0.653 |
| mistral | `int4_sim` | serial_16 | 14.499 | 15.02 | 14.673 | 0.521 | 0.174 |
| mistral | `int4_sim` | serial_4 | 14.99 | 15.774 | 15.255 | 0.784 | 0.264 |
| mistral | `int4_sim` | serial_8 | 14.992 | 15.774 | 15.254 | 0.782 | 0.263 |
| mistral | `int8` | serial_1 | 14.601 | 15.773 | 15.254 | 1.172 | 0.652 |
| mistral | `int8` | serial_16 | 14.5 | 15.02 | 14.672 | 0.52 | 0.172 |
| mistral | `int8` | serial_4 | 14.99 | 15.774 | 15.254 | 0.784 | 0.264 |
| mistral | `int8` | serial_8 | 14.992 | 15.774 | 15.254 | 0.782 | 0.263 |
| mistral | `noop` | serial_1 | 14.601 | 15.04 | 14.677 | 0.439 | 0.075 |
| mistral | `noop` | serial_4 | 14.99 | 15.041 | 14.678 | 0.051 | -0.312 |
| mistral | `noop` | serial_8 | 14.991 | 15.041 | 14.677 | 0.05 | -0.314 |

## TTFT-like (ms, mean)

| Model | Comp | Load | full | lossy | ExactKV |
|-------|------|------|-----:|------:|--------:|
| llama | `int4_sim` | serial_1 | 610.158 | 617.037 | 942.812 |
| llama | `int4_sim` | serial_16 | 392.607 | 398.318 | 719.608 |
| llama | `int4_sim` | serial_4 | 610.778 | 616.98 | 942.778 |
| llama | `int4_sim` | serial_8 | 610.711 | 616.824 | 942.543 |
| llama | `int8` | serial_1 | 609.739 | 634.558 | 942.802 |
| llama | `int8` | serial_16 | 421.266 | 401.043 | 730.4 |
| llama | `int8` | serial_4 | 610.449 | 616.538 | 942.367 |
| llama | `int8` | serial_8 | 610.5 | 616.682 | 942.466 |
| llama | `noop` | serial_1 | 716.202 | 611.005 | 959.821 |
| llama | `noop` | serial_4 | 611.836 | 611.824 | 928.038 |
| llama | `noop` | serial_8 | 610.12 | 610.233 | 929.171 |
| mistral | `int4_sim` | serial_1 | 624.711 | 631.643 | 946.966 |
| mistral | `int4_sim` | serial_16 | 393.727 | 399.692 | 709.771 |
| mistral | `int4_sim` | serial_4 | 624.903 | 631.101 | 946.948 |
| mistral | `int4_sim` | serial_8 | 625.029 | 630.799 | 946.742 |
| mistral | `int8` | serial_1 | 624.918 | 648.372 | 947.526 |
| mistral | `int8` | serial_16 | 426.837 | 401.43 | 717.29 |
| mistral | `int8` | serial_4 | 624.805 | 630.918 | 947.087 |
| mistral | `int8` | serial_8 | 624.858 | 630.808 | 947.193 |
| mistral | `noop` | serial_1 | 748.326 | 625.466 | 980.082 |
| mistral | `noop` | serial_4 | 626.249 | 626.929 | 929.15 |
| mistral | `noop` | serial_8 | 624.431 | 624.567 | 929.703 |

## Notes

- Serial request load on the ExactKV HF harness (one generate after another).
- TTFT-like: prefill→first-token for full/lossy; first verify-commit round for ExactKV.
- peak_delta_vs_full_gib > 0 means the arm peaked higher than full (common for ExactKV).
- NOT vLLM integration / continuous batching / production serving.
