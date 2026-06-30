# ExactKV Technical Report v2.3 — Strict Self-Review

**Date:** June 2026  
**Reviewer:** artifact-backed internal audit (`analysis_pack.json`, `run_quality_report.md`)

## Scores

| Dimension | Score | Notes |
|-----------|------:|-------|
| **Research paper** | **8.2 / 10** | Real external smoke data, claim-safe, single-model external GPU |
| **Technical report** | **8.8 / 10** | Traceable tables, analysis pack, reproducibility paths |
| Benchmark strength | 7.5 / 10 | Pilot JSONL smoke, not full LongBench/RULER suites |
| Case studies | 7.8 / 10 | 15 extracted divergent cells + 3 expanded forensics |
| Systems/runtime | 6.9 / 10 | 4.7–12.2 s/cell diagnostic timing documented |
| Claim hygiene | 9.0 / 10 | Drift vs official scores clearly separated |

## Strengths

1. **216 GPU cells** with zero contradictions in `run_quality_report.md` (cross-check vs `summary_all.json`).
2. **Divergence isolated to `int4_sim`** (15/216 cells, 0 `noop`/`int8` divergence) — clean compressor story.
3. **8192-token RULER pass** extends context beyond evidence-plus (512/1024).
4. **Analysis pack** enables paper updates without inventing numbers.
5. Claim boundaries preserved: VeriCache prior art, no serving claims, proxy/probe slots labeled.

## Weaknesses

1. **Single model** on external panels (Mistral failed, disk quota).
2. **Bundled pilot JSONL** only (6 LongBench, 4 RULER, 4 BFCL, 4 HumanEval prompts).
3. **Built-in compressors only** — no real KIVI/KVQuant/SnapKV in external smoke.
4. **No logits at divergence**, no confidence intervals.
5. **HumanEval shows zero drift** on this panel — weak code-risk evidence (benign only).
6. BFCL/HumanEval prompt counts below CLI `--max-prompts` caps (4 available in pilot files).

## What blocks 8.5+ / 10 (research)

1. Second model on external panels (Mistral after disk fix).
2. Real LongBench HF subset or NVIDIA RULER generated prompts (not pilot JSONL).
3. At least one **real external compressor** in headline or external panels.
4. BFCL executability / schema-validity analysis beyond token drift.
5. HumanEval/MBPP pass@1 downstream (sandboxed).
6. Logits and CIs at first divergence.
7. RULER 16K/32K multi-model scaling curves.

## Paper update safety

**Safe to cite v2.3 external numbers** with these boundaries:

- "ExactKV drift smoke panel on benchmark-shaped pilot prompts"
- Not "we evaluate on LongBench" or official leaderboard language
- Llama-3.1-8B only for external GPU tables
- `exactkv_failures=0` is panel-scoped under verifier correction

## Artifact index

| File | Role |
|------|------|
| `reports/external_panels/analysis_pack.json` | Machine-readable aggregates |
| `reports/external_panels/case_studies_extracted.json` | 15 divergent + 1 benign case |
| `reports/external_panels/paper_tables_external.md` | Copy-paste tables |
| `reports/external_panels/run_quality_report.md` | Validation checklist |
| `reports/external_panels/summary_all.json` | Workflow + merge summary |
