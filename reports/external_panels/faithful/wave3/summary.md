# Faithful External Compressor Panel Summary

## bfcl_Llama_3_1_8B_wave3_raw

- **Cells:** 80 ok / 80 total
- **ExactKV failures:** 0

| Compressor | n | Div. rate | Mean accept. |
|------------|--:|----------:|-------------:|
| `int8` | 40 | 0.0% | 1.000 |
| `turboquant_experimental` | 40 | 10.0% | 1.000 |

## bfcl_Mistral_7B_Instruct_v0_3_wave3_raw

- **Cells:** 80 ok / 80 total
- **ExactKV failures:** 0

| Compressor | n | Div. rate | Mean accept. |
|------------|--:|----------:|-------------:|
| `int8` | 40 | 0.0% | 1.000 |
| `turboquant_experimental` | 40 | 0.0% | 1.000 |

## longbench_Llama_3_1_8B_wave3_raw

- **Cells:** 144 ok / 180 total
- **ExactKV failures:** 0

| Compressor | n | Div. rate | Mean accept. |
|------------|--:|----------:|-------------:|
| `int8` | 72 | 18.1% | 0.992 |
| `turboquant_experimental` | 72 | 62.5% | 0.861 |

## longbench_Mistral_7B_Instruct_v0_3_wave3_raw

- **Cells:** 144 ok / 144 total
- **ExactKV failures:** 0

| Compressor | n | Div. rate | Mean accept. |
|------------|--:|----------:|-------------:|
| `int8` | 72 | 15.3% | 0.985 |
| `turboquant_experimental` | 72 | 66.7% | 0.874 |

## mbpp_Llama_3_1_8B_wave3_raw

- **Cells:** 64 ok / 64 total
- **ExactKV failures:** 0

| Compressor | n | Div. rate | Mean accept. |
|------------|--:|----------:|-------------:|
| `int8` | 32 | 0.0% | 1.000 |
| `turboquant_experimental` | 32 | 0.0% | 1.000 |

## mbpp_Mistral_7B_Instruct_v0_3_wave3_raw

- **Cells:** 64 ok / 64 total
- **ExactKV failures:** 0

| Compressor | n | Div. rate | Mean accept. |
|------------|--:|----------:|-------------:|
| `int8` | 32 | 0.0% | 1.000 |
| `turboquant_experimental` | 32 | 3.1% | 0.996 |

## Overall (all families)

| Compressor | n | Div. rate | Mean accept. |
|------------|--:|----------:|-------------:|
| `int8` | 288 | 8.3% | 0.994 |
| `turboquant_experimental` | 288 | 34.0% | 0.933 |
