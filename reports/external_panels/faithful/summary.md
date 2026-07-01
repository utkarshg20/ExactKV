# Faithful External Compressor Panel Summary

## bfcl_Llama_3_1_8B_raw

- **Cells:** 120 ok / 120 total
- **ExactKV failures:** 0

| Compressor | n | Div. rate | Mean accept. |
|------------|--:|----------:|-------------:|
| `int8` | 40 | 0.0% | 1.000 |
| `kivi_offline_r32` | 40 | 100.0% | 0.002 |
| `snapkv_experimental` | 40 | 100.0% | 0.446 |

## bfcl_Mistral_7B_Instruct_v0_3_raw

- **Cells:** 120 ok / 120 total
- **ExactKV failures:** 0

| Compressor | n | Div. rate | Mean accept. |
|------------|--:|----------:|-------------:|
| `int8` | 40 | 0.0% | 1.000 |
| `kivi_offline_r32` | 40 | 100.0% | 0.031 |
| `snapkv_experimental` | 40 | 100.0% | 0.405 |

## longbench_Llama_3_1_8B_raw

- **Cells:** 216 ok / 216 total
- **ExactKV failures:** 0

| Compressor | n | Div. rate | Mean accept. |
|------------|--:|----------:|-------------:|
| `int8` | 72 | 18.1% | 0.992 |
| `kivi_offline_r32` | 72 | 100.0% | 0.010 |
| `snapkv_experimental` | 72 | 100.0% | 0.371 |

## longbench_Mistral_7B_Instruct_v0_3_raw

- **Cells:** 216 ok / 216 total
- **ExactKV failures:** 0

| Compressor | n | Div. rate | Mean accept. |
|------------|--:|----------:|-------------:|
| `int8` | 72 | 15.3% | 0.985 |
| `kivi_offline_r32` | 72 | 100.0% | 0.052 |
| `snapkv_experimental` | 72 | 100.0% | 0.360 |

## mbpp_Llama_3_1_8B_raw

- **Cells:** 96 ok / 96 total
- **ExactKV failures:** 0

| Compressor | n | Div. rate | Mean accept. |
|------------|--:|----------:|-------------:|
| `int8` | 32 | 0.0% | 1.000 |
| `kivi_offline_r32` | 32 | 100.0% | 0.001 |
| `snapkv_experimental` | 32 | 56.2% | 0.696 |

## mbpp_Mistral_7B_Instruct_v0_3_raw

- **Cells:** 96 ok / 96 total
- **ExactKV failures:** 0

| Compressor | n | Div. rate | Mean accept. |
|------------|--:|----------:|-------------:|
| `int8` | 32 | 0.0% | 1.000 |
| `kivi_offline_r32` | 32 | 100.0% | 0.026 |
| `snapkv_experimental` | 32 | 87.5% | 0.527 |

## mbpp_mistral_smoke_raw

- **Cells:** 48 ok / 48 total
- **ExactKV failures:** 0

| Compressor | n | Div. rate | Mean accept. |
|------------|--:|----------:|-------------:|
| `int8` | 16 | 0.0% | 1.000 |
| `kivi_offline_r32` | 16 | 100.0% | 0.023 |
| `snapkv_experimental` | 16 | 87.5% | 0.541 |

## merged_raw

- **Cells:** 1230 ok / 1230 total
- **ExactKV failures:** 0

| Compressor | n | Div. rate | Mean accept. |
|------------|--:|----------:|-------------:|
| `int8` | 416 | 7.9% | 0.992 |
| `kivi_offline_r32` | 412 | 100.0% | 0.041 |
| `snapkv_experimental` | 402 | 97.0% | 0.411 |

## smoke2_mbpp_llama_raw

- **Cells:** 0 ok / 0 total
- **ExactKV failures:** 0

| Compressor | n | Div. rate | Mean accept. |
|------------|--:|----------:|-------------:|

## smoke_mbpp_llama_raw

- **Cells:** 2 ok / 2 total
- **ExactKV failures:** 0

| Compressor | n | Div. rate | Mean accept. |
|------------|--:|----------:|-------------:|
| `int8` | 1 | 0.0% | 1.000 |
| `kivi_offline_r32` | 1 | 100.0% | 0.000 |

## Overall (all families)

| Compressor | n | Div. rate | Mean accept. |
|------------|--:|----------:|-------------:|
| `int8` | 721 | 7.9% | 0.993 |
| `kivi_offline_r32` | 717 | 100.0% | 0.033 |
| `snapkv_experimental` | 706 | 95.5% | 0.425 |
