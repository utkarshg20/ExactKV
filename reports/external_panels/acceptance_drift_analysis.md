# Acceptance vs Drift Analysis

Diagnostic only — not end-to-end serving cost. Useful for: when does verify/commit pay off (high acceptance, low drift)?

| Family | Compressor | Model | Cells | Div% | Mean accept | Mean FDI | Mean ms |
|--------|------------|-------|------:|-----:|------------:|---------:|--------:|
| bfcl | `int4_per_vec_sim` | Llama-3.1-8B | 100 | 0.0 | 1.000 | — | 25970.14714 |
| bfcl | `int4_per_vec_sim` | Mistral-7B-Instruct-v0.3 | 100 | 0.0 | 1.000 | — | 24637.83267 |
| bfcl | `int4_sim` | Llama-3.1-8B | 300 | 50.0 | 0.995 | 103.27333333333333 | 26657.707743333332 |
| bfcl | `int4_sim` | Mistral-7B-Instruct-v0.3 | 300 | 52.0 | 0.988 | 73.78205128205128 | 25436.71226 |
| bfcl | `int6_sim` | Llama-3.1-8B | 100 | 0.0 | 1.000 | — | 25985.91709 |
| bfcl | `int6_sim` | Mistral-7B-Instruct-v0.3 | 100 | 0.0 | 1.000 | — | 24681.00527 |
| bfcl | `int8` | Llama-3.1-8B | 300 | 0.0 | 1.000 | — | 26695.108353333333 |
| bfcl | `int8` | Mistral-7B-Instruct-v0.3 | 300 | 1.0 | 1.000 | 154.33333333333334 | 25533.867553333333 |
| bfcl | `noop` | Llama-3.1-8B | 200 | 0.0 | 1.000 | — | 26629.068675 |
| bfcl | `noop` | Mistral-7B-Instruct-v0.3 | 200 | 0.0 | 1.000 | — | 25676.08595 |
| longbench | `int4_per_vec_sim` | Llama-3.1-8B | 72 | 56.9 | 0.935 | 7.195121951219512 | 8060.298277777778 |
| longbench | `int4_per_vec_sim` | Mistral-7B-Instruct-v0.3 | 72 | 55.6 | 0.930 | 9.725 | 7760.601277777778 |
| longbench | `int4_sim` | Llama-3.1-8B | 72 | 84.7 | 0.825 | 5.639344262295082 | 8128.478875 |
| longbench | `int4_sim` | Mistral-7B-Instruct-v0.3 | 72 | 86.1 | 0.846 | 6.241935483870968 | 7808.770527777778 |
| longbench | `int6_sim` | Llama-3.1-8B | 72 | 47.2 | 0.957 | 9.823529411764707 | 8052.630763888889 |
| longbench | `int6_sim` | Mistral-7B-Instruct-v0.3 | 72 | 37.5 | 0.964 | 9.851851851851851 | 7747.3925 |
| longbench | `int8` | Llama-3.1-8B | 72 | 20.8 | 0.988 | 15.266666666666667 | 8079.953597222222 |
| longbench | `int8` | Mistral-7B-Instruct-v0.3 | 72 | 15.3 | 0.985 | 12.545454545454545 | 7781.7635 |
| mbpp | `int4_per_vec_sim` | Llama-3.1-8B | 24 | 0.0 | 1.000 | — | 3889.680083333333 |
| mbpp | `int4_per_vec_sim` | Mistral-7B-Instruct-v0.3 | 24 | 0.0 | 1.000 | — | 3608.458833333333 |
| mbpp | `int4_sim` | Llama-3.1-8B | 24 | 12.5 | 0.996 | 17 | 3851.4595833333333 |
| mbpp | `int4_sim` | Mistral-7B-Instruct-v0.3 | 24 | 0.0 | 1.000 | — | 3620.4688333333334 |
| mbpp | `int6_sim` | Llama-3.1-8B | 24 | 0.0 | 1.000 | — | 3874.2776666666664 |
| mbpp | `int6_sim` | Mistral-7B-Instruct-v0.3 | 24 | 0.0 | 1.000 | — | 3608.5743333333335 |
| mbpp | `int8` | Llama-3.1-8B | 26 | 0.0 | 1.000 | — | 3769.6357307692306 |
| mbpp | `int8` | Mistral-7B-Instruct-v0.3 | 56 | 0.0 | 1.000 | — | 3635.708482142857 |
| mbpp | `kivi_offline_r32` | Llama-3.1-8B | 2 | 100.0 | 0.000 | 0 | 4913.02 |
| mbpp | `kivi_offline_r32` | Mistral-7B-Instruct-v0.3 | 32 | 100.0 | 0.023 | 0 | 6953.5868125 |
| mbpp | `snapkv_experimental` | Mistral-7B-Instruct-v0.3 | 32 | 87.5 | 0.541 | 1.8571428571428572 | 6197.4510625 |

