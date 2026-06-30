# LongBench Answer-Overlap Pack

Diagnostic max token-F1 vs HF LongBench reference answers — NOT official LongBench scores. Short max_new_tokens limits answer overlap; use for relative compressor comparison only.

## hf_longbench_v26_merged_raw.json

- Cells scored: 720
- Missing reference: 0

| Compressor | n | Mean F1 (full-KV) | Mean F1 (lossy) | Mean F1 (ExactKV) | ExactKV=full |
|------------|--:|------------------:|----------------:|------------------:|-------------:|
| `int4_sim` | 240 | 0.043 | 0.044 | 0.043 | 100% |
| `int8` | 240 | 0.043 | 0.042 | 0.043 | 100% |
| `noop` | 240 | 0.043 | 0.043 | 0.043 | 100% |

## hf_longbench_v26_Llama_3_1_8B_raw.json

- Cells scored: 360
- Missing reference: 0

| Compressor | n | Mean F1 (full-KV) | Mean F1 (lossy) | Mean F1 (ExactKV) | ExactKV=full |
|------------|--:|------------------:|----------------:|------------------:|-------------:|
| `int4_sim` | 120 | 0.037 | 0.040 | 0.037 | 100% |
| `int8` | 120 | 0.037 | 0.037 | 0.037 | 100% |
| `noop` | 120 | 0.037 | 0.037 | 0.037 | 100% |

## hf_longbench_v26_Mistral_7B_raw.json

- Cells scored: 360
- Missing reference: 0

| Compressor | n | Mean F1 (full-KV) | Mean F1 (lossy) | Mean F1 (ExactKV) | ExactKV=full |
|------------|--:|------------------:|----------------:|------------------:|-------------:|
| `int4_sim` | 120 | 0.049 | 0.049 | 0.049 | 100% |
| `int8` | 120 | 0.049 | 0.046 | 0.049 | 100% |
| `noop` | 120 | 0.049 | 0.049 | 0.049 | 100% |

