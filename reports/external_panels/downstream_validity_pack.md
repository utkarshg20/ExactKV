# Downstream Validity Pack

Downstream validity is diagnostic only — not official BFCL/MBPP/HumanEval scores. BFCL: balanced-brace JSON tool-call scan. MBPP/HumanEval: ast.parse on extracted Python.

## bfcl — `bfcl_validity_v27_merged_raw.json`

- Metric: `bfcl_json_tool_call`
- Cells: 1200
- Full-KV valid: 318 (26.5%)
- ExactKV valid: 318 (26.5%)
- Preserved among full-KV valid: 318 / 318 (100.0%)
- Lost among full-KV valid: 0
- Divergence rate: 17.0%

| Compressor | full valid | exactkv valid | preserved | lost | div |
|------------|----------:|-------------:|----------:|-----:|----:|
| `int4_sim` | 106 | 106 | 106 | 0 | 50.2% |
| `int8` | 106 | 106 | 106 | 0 | 0.8% |
| `noop` | 106 | 106 | 106 | 0 | 0.0% |

## bfcl — `bfcl_Llama_3_1_8B_raw.json`

- Metric: `bfcl_json_tool_call`
- Cells: 400
- Full-KV valid: 148 (37.0%)
- ExactKV valid: 148 (37.0%)
- Preserved among full-KV valid: 148 / 148 (100.0%)
- Lost among full-KV valid: 0
- Divergence rate: 15.0%

| Compressor | full valid | exactkv valid | preserved | lost | div |
|------------|----------:|-------------:|----------:|-----:|----:|
| `int4_per_vec_sim` | 37 | 37 | 37 | 0 | 0.0% |
| `int4_sim` | 37 | 37 | 37 | 0 | 60.0% |
| `int6_sim` | 37 | 37 | 37 | 0 | 0.0% |
| `int8` | 37 | 37 | 37 | 0 | 0.0% |

## bfcl — `bfcl_Mistral_7B_Instruct_v0_3_raw.json`

- Metric: `bfcl_json_tool_call`
- Cells: 400
- Full-KV valid: 116 (29.0%)
- ExactKV valid: 116 (29.0%)
- Preserved among full-KV valid: 116 / 116 (100.0%)
- Lost among full-KV valid: 0
- Divergence rate: 11.2%

| Compressor | full valid | exactkv valid | preserved | lost | div |
|------------|----------:|-------------:|----------:|-----:|----:|
| `int4_per_vec_sim` | 29 | 29 | 29 | 0 | 0.0% |
| `int4_sim` | 29 | 29 | 29 | 0 | 45.0% |
| `int6_sim` | 29 | 29 | 29 | 0 | 0.0% |
| `int8` | 29 | 29 | 29 | 0 | 0.0% |

## mbpp — `mbpp_Llama_3_1_8B_raw.json`

- Metric: `python_ast_syntax`
- Cells: 96
- Full-KV valid: 12 (12.5%)
- ExactKV valid: 12 (12.5%)
- Preserved among full-KV valid: 12 / 12 (100.0%)
- Lost among full-KV valid: 0
- Divergence rate: 3.1%

| Compressor | full valid | exactkv valid | preserved | lost | div |
|------------|----------:|-------------:|----------:|-----:|----:|
| `int4_per_vec_sim` | 3 | 3 | 3 | 0 | 0.0% |
| `int4_sim` | 3 | 3 | 3 | 0 | 12.5% |
| `int6_sim` | 3 | 3 | 3 | 0 | 0.0% |
| `int8` | 3 | 3 | 3 | 0 | 0.0% |

## mbpp — `mbpp_Mistral_7B_Instruct_v0_3_raw.json`

- Metric: `python_ast_syntax`
- Cells: 96
- Full-KV valid: 0 (0.0%)
- ExactKV valid: 0 (0.0%)
- Preserved among full-KV valid: 0 / 0
- Lost among full-KV valid: 0
- Divergence rate: 0.0%

| Compressor | full valid | exactkv valid | preserved | lost | div |
|------------|----------:|-------------:|----------:|-----:|----:|
| `int4_per_vec_sim` | 0 | 0 | 0 | 0 | 0.0% |
| `int4_sim` | 0 | 0 | 0 | 0 | 0.0% |
| `int6_sim` | 0 | 0 | 0 | 0 | 0.0% |
| `int8` | 0 | 0 | 0 | 0 | 0.0% |

## mbpp — `mbpp_mistral_smoke_raw.json`

- Metric: `python_ast_syntax`
- Cells: 48
- Full-KV valid: 0 (0.0%)
- ExactKV valid: 0 (0.0%)
- Preserved among full-KV valid: 0 / 0
- Lost among full-KV valid: 0
- Divergence rate: 62.5%

| Compressor | full valid | exactkv valid | preserved | lost | div |
|------------|----------:|-------------:|----------:|-----:|----:|
| `int8` | 0 | 0 | 0 | 0 | 0.0% |
| `kivi_offline_r32` | 0 | 0 | 0 | 0 | 100.0% |
| `snapkv_experimental` | 0 | 0 | 0 | 0 | 87.5% |

