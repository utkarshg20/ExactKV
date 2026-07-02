# Faithful External Compressor Panel Summary

## bfcl_Mistral_7B_Instruct_v0_3_wave2_smoke_raw

- **Cells:** 64 ok / 64 total
- **ExactKV failures:** 0

| Compressor | n | Div. rate | Mean accept. |
|------------|--:|----------:|-------------:|
| `int8` | 16 | 0.0% | 1.000 |
| `kvpress_knorm_experimental` | 16 | 81.2% | 0.556 |
| `snapkv_experimental` | 16 | 100.0% | 0.403 |
| `turboquant_experimental` | 16 | 0.0% | 1.000 |

## mbpp_Mistral_7B_Instruct_v0_3_wave2_smoke_raw

- **Cells:** 64 ok / 64 total
- **ExactKV failures:** 0

| Compressor | n | Div. rate | Mean accept. |
|------------|--:|----------:|-------------:|
| `int8` | 16 | 0.0% | 1.000 |
| `kvpress_knorm_experimental` | 16 | 75.0% | 0.561 |
| `snapkv_experimental` | 16 | 87.5% | 0.547 |
| `turboquant_experimental` | 16 | 6.2% | 0.993 |

## Overall (all families)

| Compressor | n | Div. rate | Mean accept. |
|------------|--:|----------:|-------------:|
| `int8` | 32 | 0.0% | 1.000 |
| `kvpress_knorm_experimental` | 32 | 78.1% | 0.558 |
| `snapkv_experimental` | 32 | 93.8% | 0.475 |
| `turboquant_experimental` | 32 | 3.1% | 0.996 |
