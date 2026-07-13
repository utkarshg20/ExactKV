# Length-opportunity pack

Cell-level divergence rises with `max_new_tokens` partly because each extra token is another greedy argmax trial. Report **both** cell divergence rate and FDI / `max_new` (early vs late).

Generated: `2026-07-13T16:58:18.350951+00:00`

## mbpp (`reports/external_panels/mbpp_gpu_raw.json`)

| Compressor | n | Div. rate | Mean FDI (div) | Mean FDI/`mnt` | Hazard proxy |
|------------|--:|----------:|---------------:|---------------:|-------------:|
| `int4_sim` | 48 | 6.2% | 17.0 | 0.6458 | 0.0027 |
| `int8` | 48 | 0.0% | — | — | 0.0000 |
| `noop` | 48 | 0.0% | — | — | 0.0000 |

## bfcl_short (`reports/external_panels/bfcl_export_50_raw.json`)

| Compressor | n | Div. rate | Mean FDI (div) | Mean FDI/`mnt` | Hazard proxy |
|------------|--:|----------:|---------------:|---------------:|-------------:|
| `int4_sim` | 400 | 11.2% | 10.244 | 0.4042 | 0.0051 |
| `int8` | 400 | 0.0% | — | — | 0.0000 |
| `noop` | 400 | 0.0% | — | — | 0.0000 |

## bfcl_long (`reports/external_panels/bfcl_validity_v27_merged_raw.json`)

| Compressor | n | Div. rate | Mean FDI (div) | Mean FDI/`mnt` | Hazard proxy |
|------------|--:|----------:|---------------:|---------------:|-------------:|
| `int4_sim` | 400 | 50.2% | 84.512 | 0.4102 | 0.0039 |
| `int8` | 400 | 0.8% | 154.333 | 0.7669 | 0.0000 |
| `noop` | 400 | 0.0% | — | — | 0.0000 |

## longbench (`reports/external_panels/hf_longbench_v26_merged_raw.json`)

| Compressor | n | Div. rate | Mean FDI (div) | Mean FDI/`mnt` | Hazard proxy |
|------------|--:|----------:|---------------:|---------------:|-------------:|
| `int4_sim` | 240 | 90.4% | 7.829 | 0.1748 | 0.0797 |
| `int8` | 240 | 24.6% | 23.576 | 0.4677 | 0.0060 |
| `noop` | 240 | 0.0% | — | — | 0.0000 |

Hazard proxy is diagnostic only — not a Kaplan–Meier / formal survival estimate.
