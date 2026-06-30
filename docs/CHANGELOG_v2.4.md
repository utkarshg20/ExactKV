# Changelog v2.3 → v2.4

**Date:** 2026-06-26

## Paper (ExactKV_Technical_Report.md / .tex)

- Version bump to **v2.4** (MBPP GPU smoke panel).
- Abstract: 144-cell MBPP pilot GPU smoke (`mbpp_gpu_raw.json`), claim boundary (no test execution / not pass@1).
- Table 4: MBPP row (144 cells, Llama + Mistral, 2.1% divergence, 0 failures).
- Notable findings + MBPP note (3 Llama `int4_sim` cells at 1024; Mistral clean).
- Limitations / future work: MBPP loader and GPU smoke marked done.
- Reproducibility: MBPP GPU command + validator.

## Artifacts (validated)

- `reports/external_panels/mbpp_gpu_raw.json` — 144 GPU cells, `exactkv_failures=0`
- `reports/external_panels/validation_report.json` — PASS (23/23 files)

## Claim boundary

ExactKV drift on bundled MBPP-shaped pilot prompts only. Not official MBPP pass@1; generated code not executed against `test_list`.
