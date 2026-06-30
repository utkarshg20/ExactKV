# Faithful External Compressor Panel Summary

## bfcl_Mistral_7B_Instruct_v0_3_raw

- **Cells:** 98 ok / 98 total
- **ExactKV failures:** 0

| Compressor | n | Div. rate | Mean accept. |
|------------|--:|----------:|-------------:|
| `int8` | 35 | 0.0% | 1.000 |
| `kivi_offline_r32` | 33 | 100.0% | 0.032 |
| `snapkv_experimental` | 30 | 100.0% | 0.400 |

## longbench_Mistral_7B_Instruct_v0_3_raw

- **Cells:** 216 ok / 216 total
- **ExactKV failures:** 0

| Compressor | n | Div. rate | Mean accept. |
|------------|--:|----------:|-------------:|
| `int8` | 72 | 15.3% | 0.985 |
| `kivi_offline_r32` | 72 | 100.0% | 0.052 |
| `snapkv_experimental` | 72 | 100.0% | 0.360 |

## mbpp_mistral_smoke_raw

- **Cells:** 48 ok / 48 total
- **ExactKV failures:** 0

| Compressor | n | Div. rate | Mean accept. |
|------------|--:|----------:|-------------:|
| `int8` | 16 | 0.0% | 1.000 |
| `kivi_offline_r32` | 16 | 100.0% | 0.023 |
| `snapkv_experimental` | 16 | 87.5% | 0.541 |

## merged_raw

- **Cells:** 384 ok / 384 total
- **ExactKV failures:** 0

| Compressor | n | Div. rate | Mean accept. |
|------------|--:|----------:|-------------:|
| `int8` | 131 | 8.4% | 0.992 |
| `kivi_offline_r32` | 129 | 100.0% | 0.041 |
| `snapkv_experimental` | 124 | 96.8% | 0.412 |

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
| `int8` | 255 | 8.6% | 0.991 |
| `kivi_offline_r32` | 251 | 100.0% | 0.042 |
| `snapkv_experimental` | 242 | 97.5% | 0.404 |
