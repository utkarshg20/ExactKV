# Paper-ready tables: external smoke panels

Copy into technical report. All values from merged GPU artifacts.

## Table 1. External smoke panel summary

| Dataset family | Prompt source | Categories | Context buckets | max_new_tokens | Cells | Compressors | Acceptance | Div rate | Mean 1st div† | exactkv_failure | Mean/p90 ms |
|----------------|---------------|------------|-----------------|----------------|------:|-------------|----------:|---------:|--------------:|----------------:|------------:|
| LongBench | bundled pilot | 6 tasks (6 prompts) | 2048, 4096 | 16, 32 | 72 | noop, int8, int4_sim | 0.994 | 0.083 | 8 | 0 | 6422.296/8781.113 |
| RULER | bundled pilot | 4 task types | 2048, 4096 | 16, 32 | 48 | noop, int8, int4_sim | 0.993 | 0.083 | 14.500 | 0 | 6436.317/8792.586 |
| RULER | bundled pilot | 4 task types | 8192 | 16, 32 | 24 | noop, int8, int4_sim | 0.993 | 0.083 | 10 | 0 | 12231.604/13662.393 |
| BFCL | bundled pilot (4 prompts) | simple, parallel, multi_turn, ast_eval | 1024, 2048 | 16, 32 | 48 | noop, int8, int4_sim | 0.999 | 0.062 | 8 | 0 | 4715.217/6342.236 |
| HumanEval | bundled pilot (4 prompts) | code | 1024, 2048 | 32 | 24 | noop, int8, int4_sim | 1.000 | 0.000 | n/a | 0 | 5790.775/6391.841 |

† Mean first-divergence index over divergent `int4_sim` cells only.

## Table 2. External smoke findings by compressor

| Compressor | Cells | Div rate | Accept mean | Accept median | Accept min | Accept p90 | Divergent cells |
|------------|------:|---------:|------------:|--------------:|-----------:|-----------:|----------------:|
| `noop` | 72 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| `int8` | 72 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| `int4_sim` | 72 | 0.208 | 0.986 | 1.000 | 0.882 | 1.000 | 15 |

## Table 3. Context bucket summary

| Bucket | Cells | Div rate | Mean ms | P50 ms | P90 ms | P95 ms | Max ms |
|-------:|------:|---------:|--------:|-------:|-------:|-------:|-------:|
| 1024 | 36 | 0.000 | 4565.868 | 5195.599 | 5312.189 | 5325.789 | 5570.054 |
| 2048 | 96 | 0.073 | 5384.081 | 6191.934 | 6399.135 | 6406.956 | 6428.245 |
| 4096 | 60 | 0.100 | 7590.241 | 7443.400 | 8812.586 | 8817.886 | 8843.402 |
| 8192 | 24 | 0.083 | 12231.604 | 12025.110 | 13662.393 | 13679.764 | 13691.905 |

## Table 4. Notable divergent case studies

| Family | Category | Model | Compressor | Ctx | mnt | 1st div | Accept | ExactKV | Interpretation | Snippets |
|--------|----------|-------|------------|----:|----:|--------:|-------:|---------|----------------|----------|
| longbench | passage_retrieval_en | Llama-3.1-8B | `int4_sim` | 4096 | 16 | 2 | 0.882 | ok | semantic | yes |
| ruler | niah_single | Llama-3.1-8B | `int4_sim` | 4096 | 16 | 2 | 0.882 | ok | semantic | yes |
| longbench | passage_retrieval_en | Llama-3.1-8B | `int4_sim` | 4096 | 32 | 2 | 0.939 | ok | semantic | yes |
| ruler | niah_single | Llama-3.1-8B | `int4_sim` | 4096 | 32 | 2 | 0.939 | ok | semantic | yes |
| longbench | gov_report | Llama-3.1-8B | `int4_sim` | 2048 | 16 | 4 | 0.882 | ok | semantic | yes |
| longbench | gov_report | Llama-3.1-8B | `int4_sim` | 2048 | 32 | 4 | 0.939 | ok | semantic | yes |
| bfcl | ast_eval | Llama-3.1-8B | `int4_sim` | 2048 | 16 | 4 | 1.000 | ok | tool-risk | yes |
| bfcl | ast_eval | Llama-3.1-8B | `int4_sim` | 2048 | 32 | 4 | 1.000 | ok | tool-risk | yes |
| ruler | niah_single | Llama-3.1-8B | `int4_sim` | 8192 | 16 | 10 | 0.882 | ok | semantic | yes |
| ruler | niah_single | Llama-3.1-8B | `int4_sim` | 8192 | 32 | 10 | 0.939 | ok | semantic | yes |
| bfcl | parallel | Llama-3.1-8B | `int4_sim` | 2048 | 32 | 16 | 1.000 | ok | tool-risk | yes |
| longbench | gov_report | Llama-3.1-8B | `int4_sim` | 4096 | 32 | 18 | 0.939 | ok | semantic | yes |

## Table 5. Limitations and skipped runs

| Item | Status |
|------|--------|
| LongBench HF export | skipped (`datasets` not installed on pod) |
| Mistral-7B external panels | failed (disk quota exceeded) |
| RULER 16K / 32K | not run |
| Official benchmark scores | not computed (drift panels only) |
| Deterministic offline smoke JSON | excluded from GPU totals |
| Real KIVI / KVQuant / SnapKV | not in external smoke runs |

