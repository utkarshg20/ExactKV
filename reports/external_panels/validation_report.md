# External panel artifact validation

Generated: 2026-06-27T00:31:51.336628+00:00

**Input:** `/Users/utkarshgupta/Documents/ExactKV/reports/external_panels`

**Overall:** PASS (23/23 files valid, 0 issues, 5 warnings)

Validator checks ExactKV drift panel schema and internal consistency. It does not certify official LongBench/RULER/BFCL/HumanEval/MBPP scores.

## Per-file results

| File | Family | Mode | Source | Cells | Valid | Issues | Warnings |
|------|--------|------|--------|------:|-------|-------:|---------:|
| reports/external_panels/bfcl_Llama_3_1_8B_raw.json | bfcl | gpu | pilot | 48 | yes | 0 | 0 |
| reports/external_panels/bfcl_Mistral_7B_Instruct_v0_3_raw.json | bfcl | gpu | pilot | 48 | yes | 0 | 0 |
| reports/external_panels/bfcl_export_50_raw.json | bfcl | gpu | export | 1200 | yes | 0 | 0 |
| reports/external_panels/bfcl_merged_raw.json | bfcl | gpu | pilot | 48 | yes | 0 | 1 |
| reports/external_panels/bfcl_raw.json | bfcl | deterministic | pilot | 4 | yes | 0 | 0 |
| reports/external_panels/humaneval_Llama_3_1_8B_raw.json | humaneval | gpu | pilot | 24 | yes | 0 | 0 |
| reports/external_panels/humaneval_Mistral_7B_Instruct_v0_3_raw.json | humaneval | gpu | pilot | 24 | yes | 0 | 0 |
| reports/external_panels/humaneval_merged_raw.json | humaneval | gpu | pilot | 24 | yes | 0 | 1 |
| reports/external_panels/humaneval_raw.json | humaneval | deterministic | pilot | 4 | yes | 0 | 0 |
| reports/external_panels/longbench_hf_raw.json | longbench | gpu | hf | 288 | yes | 0 | 0 |
| reports/external_panels/longbench_pilot_Llama_3_1_8B_raw.json | longbench | gpu | pilot | 72 | yes | 0 | 0 |
| reports/external_panels/longbench_pilot_Mistral_7B_Instruct_v0_3_raw.json | longbench | gpu | pilot | 72 | yes | 0 | 0 |
| reports/external_panels/longbench_pilot_merged_raw.json | longbench | gpu | pilot | 72 | yes | 0 | 1 |
| reports/external_panels/longbench_raw.json | longbench | deterministic | pilot | 4 | yes | 0 | 0 |
| reports/external_panels/mbpp_gpu_raw.json | mbpp | gpu | pilot | 144 | yes | 0 | 0 |
| reports/external_panels/mbpp_raw.json | mbpp | deterministic | pilot | 4 | yes | 0 | 0 |
| reports/external_panels/ruler_2048_4096_Llama_3_1_8B_raw.json | ruler | gpu | pilot | 48 | yes | 0 | 0 |
| reports/external_panels/ruler_2048_4096_Mistral_7B_Instruct_v0_3_raw.json | ruler | gpu | pilot | 48 | yes | 0 | 0 |
| reports/external_panels/ruler_2048_4096_merged_raw.json | ruler | gpu | pilot | 48 | yes | 0 | 1 |
| reports/external_panels/ruler_8192_Llama_3_1_8B_raw.json | ruler | gpu | pilot | 24 | yes | 0 | 0 |
| reports/external_panels/ruler_8192_Mistral_7B_Instruct_v0_3_raw.json | ruler | gpu | pilot | 24 | yes | 0 | 0 |
| reports/external_panels/ruler_8192_merged_raw.json | ruler | gpu | pilot | 24 | yes | 0 | 1 |
| reports/external_panels/ruler_raw.json | ruler | deterministic | pilot | 4 | yes | 0 | 0 |
