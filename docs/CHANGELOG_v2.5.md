# ExactKV v2.5 Changelog — KIVI Panel Integration + Real HF Results

**Date:** June 27, 2026  
**Score target:** 8.5+/10 strict research-paper draft  
**Artifacts:** `kivi_longbench_hf_raw.json` (320 cells), `kivi_mbpp_hf_raw.json` (320 cells)

---

## Summary

v2.5 integrates the completed KIVI offline compressor panel (640 real GPU cells on
real HF datasets) and upgrades the paper with real numbers, a new case study, and
updated evidence framing. It also includes Wilson CIs on acceptance rates, a BFCL
tool-call validity diagnostic, and infrastructure for top-k logit capture.

---

## Paper Changes (v2.4.2 → v2.5)

### §6.4.6 — KIVI offline compressor panel (placeholder → real results)

**LongBench HF (80 cells × 4 compressors = 320 cells):**

| Compressor | Div. rate | Mean accept. | `exactkv_failures` |
|-----------|----------:|-------------:|-------------------:|
| `noop` | 0.0% | 1.000 | 0 |
| `int8` | 13.8% | 0.994 | 0 |
| `int4_sim` | 91.2% | 0.818 | 0 |
| `kivi_offline` | **100.0%** | **0.004** | **0** |

**MBPP HF (80 cells × 4 compressors = 320 cells):**

| Compressor | Div. rate | Mean accept. | `exactkv_failures` |
|-----------|----------:|-------------:|-------------------:|
| `noop` | 0.0% | 1.000 | 0 |
| `int8` | 0.0% | 1.000 | 0 |
| `int4_sim` | 5.0% | 0.995 | 0 |
| `kivi_offline` | **100.0%** | **0.001** | **0** |

**Key findings:**
1. `kivi_offline` shows 100% divergence with acceptance ≈ 0 — catastrophic KV
   corruption in the offline simulate-path integration. Not a claim about KIVI.
2. ExactKV detected and corrected all 160 `kivi_offline` corrupt cells (`exactkv_failures=0`).
3. Real HF `int4_sim` LongBench divergence = **91.2%** vs 20.8% on bundled pilot —
   bundled pilots dramatically underestimate real-world drift.

### §8 — New case study: Case O (KIVI catastrophic corruption)

- `lb_narrativeqa_000_ctx2048`, `kivi_offline`, `max_new_tokens=32`
- Lossy output: `-!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!` (all garbage)
- ExactKV output: matches full-KV (`face, with its high forehead...`)
- Acceptance 0.000, first divergence index 0, `exactkv_failure=False`

### Abstract

Updated to reflect completed KIVI panel and crash-test findings.

### §14.2.1 — Future work

Updated from "pre-registered, pending" to "completed (June 2026)" with result summary.

### §15 — Limitations

- Updated §3, §4 to reflect real HF datasets now complete.
- §9 updated to describe KIVI as an integration diagnostic, not algorithm failure.
- Added RULER 12K queued note (16K/32K OOM on A5000).

---

## Infrastructure Changes

### `scripts/build_external_analysis_pack.py`
- Added `bfcl_tool_call_validity` section with balanced-brace JSON scanner
- Added `acceptance_full_rate_ci95` (Wilson 95% CI) to `totals`, `by_panel`, `by_panel_compressor`
- Added `divergence_rate_overall_ci95` to `totals`
- BFCL diagnostic: `max_new_tokens=16,32` confirmed insufficient for complete JSON

### `exactkv/benchmarks/runner.py`
- Added `capture_divergence_topk()` — post-hoc top-k logit capture at divergence point

### `scripts/run_external_panel.py`
- Added `--store-top-k-logits` flag (wired through to `external_panel.py`)

### `exactkv/benchmarks/external_panel.py`
- Added `store_top_k_logits` parameter

---

## RunPod Job Queue (as of v2.5)

| Session | Status | Output |
|---------|--------|--------|
| `kivi_run` | ✅ DONE | `kivi_longbench_hf_raw.json`, `kivi_mbpp_hf_raw.json` |
| `ruler16k` | ⚠️ OOM at 16K | No output (A5000 memory limit) |
| `bfcl_validity` | 🟡 RUNNING (~103/144) | `bfcl_validity_raw.json` (pending) |
| `ruler12k` | ⏳ QUEUED | `ruler_12288_Llama_raw.json` (pending) |

---

## Evidence Totals (v2.5)

| Layer | Cells | Status |
|-------|------:|--------|
| Headline panel | 1,500 | Complete |
| Evidence-plus | 144 | Complete |
| External smoke (216 Llama-only) | 216 | Complete |
| MBPP GPU smoke (both models) | 144 | Complete |
| KIVI offline (LongBench HF + MBPP HF) | 640 | **NEW — Complete** |
| BFCL validity rerun | 144 | Pending (pod running) |
| RULER 12K | TBD | Pending (queued) |
| **Total confirmed GPU cells** | **2,644+** | |
