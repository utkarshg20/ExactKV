# Changelog: v2.2.1 → v2.3

**Date:** June 2026  
**Primary artifacts:** `reports/external_panels/analysis_pack.json`, `reports/external_panels/*_merged_raw.json`, `reports/external_panels/summary_all.json`

## Summary

v2.3 adds a **216-cell external smoke panel** (Llama-3.1-8B only) on bundled LongBench-shaped, RULER-shaped, BFCL, and HumanEval pilot prompts at 1024–8192 prefill buckets. All values are artifact-backed. No official benchmark scores are claimed.

## Added

- **§5.2** External benchmark smoke panels (experimental setup)
- **§6.4** External Benchmark Smoke Panels with Table 4 (claim-boundary columns)
- **Notable external findings** paragraph (`noop`/`int8` zero divergence, `int4_sim`-only drift)
- **§6.4.1** External case-study table from `case_studies_extracted.json`
- **§8.4** External smoke case studies (Cases J–L)
- **§14.2–14.3** Completed external panels + expanded future work
- Analysis pack under `reports/external_panels/` (`analysis_pack.md/json`, `paper_tables_external.md`, `run_quality_report.md`, `case_studies_extracted.json`)
- `scripts/build_external_analysis_pack.py`

## Updated

- Abstract: required 216-cell external smoke sentence with `int4_sim` concentration claim
- **§15 Limitations:** smoke-panel scope, HF skip, Mistral disk failure, 4-prompt BFCL/HumanEval pilots
- Conclusion cites external smoke supplement
- PDF (`.tex`) external smoke subsection and limitations/future work

## Verified metrics (GPU external, Llama-3.1-8B)

| Panel | Cells | Div rate | Accept | Failures | Mean ms | P90 ms |
|-------|------:|---------:|-------:|---------:|--------:|-------:|
| LongBench pilot | 72 | 0.083 | 0.994 | 0 | 6422 | 8781 |
| RULER 2048/4096 | 48 | 0.083 | 0.993 | 0 | 6436 | 8793 |
| RULER 8192 | 24 | 0.083 | 0.993 | 0 | 12232 | 13640 |
| BFCL | 48 | 0.063 | 0.999 | 0 | 4715 | 6342 |
| HumanEval | 24 | 0.000 | 1.000 | 0 | 5791 | 6392 |

## Not claimed

- Official LongBench/RULER/BFCL/HumanEval scores
- Mistral-7B external panels (disk quota)
- LongBench HF export (`datasets` not installed)
- VeriCache reproduction or compressed-KV draft novelty
- Real KIVI/KVQuant/SnapKV/SpectralQuant/Shard head-to-head results
