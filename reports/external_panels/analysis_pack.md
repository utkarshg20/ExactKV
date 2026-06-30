# External Panel Analysis Pack

Generated: 2026-06-27T16:07:35.232497+00:00

ExactKV drift panels only. Not official LongBench/RULER/BFCL/HumanEval scores.

## Executive summary

- **Total GPU cells (ok):** 216
- **Divergent cells:** 15 (0.069 panel-wide rate)
- **exactkv_failures:** 0
- **Model:** meta-llama/Llama-3.1-8B only (all merged GPU artifacts)
- **Prompt source:** bundled pilot JSONL (not HF LongBench, not official scores)

## Totals

| Metric | Value |
|--------|------:|
| Cells ok | 216 |
| Divergence rate | 0.069 |
| Acceptance mean | 0.995 |
| Acceptance median | 1.000 |
| Acceptance p90 | 1.000 |
| Timing mean ms | 6621.371 |
| Timing p50 ms | 6332.587 |
| Timing p90 ms | 10827.967 |
| Timing p95 ms | 12854.340 |
| Timing max ms | 13691.905 |

## By panel

| Panel | Cells | Div rate | Accept mean | Accept p90 | Failures | Mean ms | P90 ms |
|-------|------:|---------:|------------:|-----------:|---------:|--------:|-------:|
| longbench_pilot | 72 | 0.083 | 0.994 | 1.000 | 0 | 6422.296 | 8781.113 |
| ruler_2048_4096 | 48 | 0.083 | 0.993 | 1.000 | 0 | 6436.317 | 8792.586 |
| ruler_8192 | 24 | 0.083 | 0.993 | 1.000 | 0 | 12231.604 | 13662.393 |
| bfcl | 48 | 0.062 | 0.999 | 1.000 | 0 | 4715.217 | 6342.236 |
| humaneval | 24 | 0.000 | 1.000 | 1.000 | 0 | 5790.775 | 6391.841 |

## By compressor (all panels)

| Compressor | Cells | Div rate | Accept mean | Accept min | Accept p90 | Mean 1st div |
|------------|------:|---------:|------------:|-----------:|-----------:|-------------:|
| noop | 72 | 0.000 | 1.000 | 1.000 | 1.000 | n/a |
| int8 | 72 | 0.000 | 1.000 | 1.000 | 1.000 | n/a |
| int4_sim | 72 | 0.208 | 0.986 | 0.882 | 1.000 | 10.120 |

## By context bucket (all panels)

| Bucket | Cells | Div rate | Mean ms | P90 ms |
|-------:|------:|---------:|--------:|-------:|
| 1024 | 36 | 0.000 | 4565.868 | 5312.189 |
| 2048 | 96 | 0.073 | 5384.081 | 6399.135 |
| 4096 | 60 | 0.100 | 7590.241 | 8812.586 |
| 8192 | 24 | 0.083 | 12231.604 | 13662.393 |

## By category

| Category | Cells | Div rate | Accept mean |
|----------|------:|---------:|------------:|
| bfcl/ast_eval | 12 | 0.167 | 0.997 |
| bfcl/multi_turn | 12 | 0.000 | 1.000 |
| bfcl/parallel | 12 | 0.083 | 1.000 |
| bfcl/simple | 12 | 0.000 | 1.000 |
| humaneval/code | 24 | 0.000 | 1.000 |
| longbench/gov_report | 12 | 0.250 | 0.980 |
| longbench/hotpotqa | 12 | 0.000 | 1.000 |
| longbench/lcc | 12 | 0.000 | 1.000 |
| longbench/narrativeqa | 12 | 0.083 | 1.000 |
| longbench/passage_retrieval_en | 12 | 0.167 | 0.985 |
| longbench/trec | 12 | 0.000 | 1.000 |
| ruler/common_words_extraction | 18 | 0.056 | 0.995 |
| ruler/niah_multi | 18 | 0.000 | 1.000 |
| ruler/niah_single | 18 | 0.222 | 0.980 |
| ruler/variable_tracking | 18 | 0.056 | 0.995 |

## First-divergence histogram (divergent cells only)

**longbench_pilot:** idx 2→2, idx 4→2, idx 18→2
**ruler_2048_4096:** idx 2→2, idx 25→1, idx 29→1
**ruler_8192:** idx 10→2
**bfcl:** idx 4→2, idx 16→1

## Divergent cell rankings

### Earliest first divergence
- `lb_passage_retrieval_001_ctx4096`
- `lb_passage_retrieval_001_ctx4096`
- `ruler_niah_single_4k_ctx4096`
- `ruler_niah_single_4k_ctx4096`
- `lb_gov_report_001_ctx2048`
- `lb_gov_report_001_ctx2048`
- `bfcl_ast_001_ctx2048`
- `bfcl_ast_001_ctx2048`
- `ruler_niah_single_4k_ctx8192`
- `ruler_niah_single_4k_ctx8192`

### Lowest acceptance
- `lb_gov_report_001_ctx2048`
- `lb_passage_retrieval_001_ctx4096`
- `ruler_niah_single_4k_ctx4096`
- `ruler_niah_single_4k_ctx8192`
- `ruler_variable_tracking_16k_ctx4096`
- `ruler_common_words_32k_ctx2048`
- `lb_gov_report_001_ctx2048`
- `lb_gov_report_001_ctx4096`
- `lb_passage_retrieval_001_ctx4096`
- `ruler_niah_single_4k_ctx4096`

### Highest context bucket
- `ruler_niah_single_4k_ctx8192`
- `ruler_niah_single_4k_ctx8192`
- `lb_gov_report_001_ctx4096`
- `lb_passage_retrieval_001_ctx4096`
- `lb_passage_retrieval_001_ctx4096`
- `ruler_niah_single_4k_ctx4096`
- `ruler_niah_single_4k_ctx4096`
- `ruler_variable_tracking_16k_ctx4096`
- `lb_narrativeqa_001_ctx2048`
- `lb_gov_report_001_ctx2048`

### Tool/code risk priority
- `bfcl_ast_001_ctx2048`
- `bfcl_ast_001_ctx2048`
- `bfcl_parallel_001_ctx2048`
- `lb_passage_retrieval_001_ctx4096`
- `lb_passage_retrieval_001_ctx4096`
- `ruler_niah_single_4k_ctx4096`
- `ruler_niah_single_4k_ctx4096`
- `lb_gov_report_001_ctx2048`
- `lb_gov_report_001_ctx2048`
- `ruler_niah_single_4k_ctx8192`


## Validation

- **all_merged_deterministic_mode_false:** True
- **exactkv_failures_zero_all_merged_reports:** True
- **exactkv_failures_zero_per_cell:** True
- **divergence_only_int4_sim:** True
- **non_int4_divergent_cells:** 0
- **noop_int8_divergence_count:** 0
- **int4_sim_cells:** 72
- **int4_sim_divergent_cells:** 15
- **int4_sim_divergence_rate:** 0.20833333333333334
- **non_ok_cell_statuses:** []

**Contradictions:**
- summary_all cell count mismatch for longbench_pilot
- summary_all cell count mismatch for ruler_2048_4096
- summary_all cell count mismatch for ruler_8192
- summary_all cell count mismatch for bfcl
- summary_all cell count mismatch for humaneval
