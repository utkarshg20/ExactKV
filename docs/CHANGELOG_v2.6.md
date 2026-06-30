# ExactKV Technical Report — v2.6 Changelog

**Date:** 2026-06-28
**Version bump:** v2.5.4 → v2.6
**Type:** New panels: real HF LongBench drift + BFCL validity (Llama partial)

---

## Summary

v2.6 adds the first fully real (non-pilot, non-bundled) HF LongBench drift panel across
**both models** (720 cells) and the first long-generation BFCL tool-call validity panel
(600 Llama cells; Mistral pending). Total evaluated GPU cells grows from 3,844 to **5,164+**.

---

## New Panels

### 1. HF LongBench Drift Panel (v2.6) — 720 cells

| Property | Value |
|----------|-------|
| Panel ID | `hf_longbench_v26` |
| Source | Real THUDM/LongBench via Hugging Face `datasets` |
| Models | Llama-3.1-8B + Mistral-7B-Instruct-v0.3 |
| Compressors | noop, int8, int4_sim |
| Context buckets | 2048, 4096, 8192 tokens |
| max_new_tokens | 32, 64 |
| Subsets | narrativeqa, qasper, multifieldqa_en, hotpotqa, 2wikimqa, gov_report, trec, samsum, lcc, passage_retrieval_en |
| Cells | 2 prompts/subset × 10 subsets × 3 ctx × 2 mnt × 3 comp × 2 models = **720** |
| `exactkv_failures` | **0** |
| Divergent cells (int4_sim) | **217 / 240 (90.4%)** |
| Divergent cells (int8) | **59 / 240 (24.6%)** |
| Divergent cells (noop) | **0 / 240 (0.0%)** |

**Result:** Highest int4_sim divergence observed across all ExactKV panels — 90.4% on
LongBench open-text tasks. `exactkv_failures=0` throughout.

#### Key scientific finding: Task-type sensitivity of int4_sim divergence

LongBench open-text reading/summarization at 2K–8K prefill produces very high int4_sim
divergence (90.4%), with median first-divergence at token 4–6 (very early). Even int8
shows 24.6% divergence, unlike its 0% on BFCL/MBPP/RULER.

| Panel | Context | Task type | int4_sim div. rate | Mean accept. |
|-------|---------|-----------|-------------------|-------------|
| Headline (v1) | ~500 tok | Mixed | 51.3% | — |
| MBPP supplement | 512–1024 tok | Python code | 6.2% | 0.999 |
| BFCL export-50 Llama | 1K–2K tok | Tool-calling | 5.5% | — |
| BFCL export-50 Mistral | 1K–2K tok | Tool-calling | 17.0% | — |
| **HF LongBench (Llama, v2.6)** | **2K–8K tok** | **Open-text reading** | **91.7%** | 0.825 |
| **HF LongBench (Mistral, v2.6)** | **2K–8K tok** | **Open-text reading** | **89.2%** | 0.861 |

**Conclusion:** Task type is the dominant driver of int4_sim divergence, not context length alone.
ExactKV's verifier catches all divergence types with `exactkv_failures=0`.

#### Artifacts
- `reports/external_panels/hf_longbench_v26_Llama_3_1_8B_raw.json` (9.6 MB)
- `reports/external_panels/hf_longbench_v26_Mistral_7B_raw.json` (8.6 MB)
- `reports/external_panels/hf_longbench_v26_merged_raw.json` (combined)

---

### 2. BFCL Tool-Call Validity Panel (v2.7, Llama partial) — 600 cells

| Property | Value |
|----------|-------|
| Panel ID | `bfcl_validity_v27` |
| Source | BFCL export-50 prompts (`benchmarks/prompts/bfcl_export.jsonl`) |
| Model | Llama-3.1-8B (Mistral pending) |
| Compressors | noop, int8, int4_sim |
| Context buckets | 1024, 2048 tokens |
| max_new_tokens | **128, 256** (first full-generation panel) |
| Cells | 50 prompts × 2 ctx × 2 mnt × 3 comp = **600** (Llama) |
| `exactkv_failures` | **0** |
| Divergent cells | **0 / 600** |

**Validity results (post-processed balanced-brace scan):**

| max_new_tokens | valid JSON rate | Notes |
|---------------|-----------------|-------|
| 128 | 13.0% | Too short for most tool-call completions |
| 256 | **50.0%** | Half of prompts produce complete JSON objects |

**Key findings:**
- At `mnt=256`, 50% of Llama BFCL outputs contain complete valid JSON — identical across
  noop, int8, and int4_sim. KV compression does not corrupt tool-call structure.
- The prior BFCL drift panel (`mnt=16/32`) was too short for JSON-completeness analysis;
  this panel confirms that `mnt=256` is the minimum for valid tool-call evaluation.
- `exactkv_failures=0` across all 600 cells.

---

## Bug Fix

**`write_external_panel_outputs` markdown path bug:**

- `exactkv/benchmarks/external_panel.py`: `markdown_path` was derived as a relative path
  even when `json_path` was absolute. This caused `FileNotFoundError` on RunPod where
  the process cwd was `/workspace` instead of `/workspace/ExactKV`.
- Fix: derive `markdown_path` as a sibling of `json_path` (i.e., `json_path.parent / f"{family}_summary.md"`).
- `exactkv/benchmarks/evidence_plus_panel.py`: added `markdown_path.parent.mkdir(parents=True, exist_ok=True)`
  to prevent the same class of error when writing the markdown summary.

---

## In Progress / Queued

| Job | Status | tmux session |
|-----|--------|-------------|
| Mistral v2.7 BFCL validity (600 cells) | **Running** on RunPod | `v27_mistral` |
| H2O v2.8 eviction panel (1,200 cells) | Queued after v2.7 Mistral | `v28_queue` |

---

## Paper changes

- Version bumped to `v2.6` in both `.md` and `.tex` headers.
- §6.5 Table 4d replaced with complete 720-cell results.
- §6.5 status updated; context-length sensitivity finding added.
- §6.7 status updated to reflect Llama completion; partial Table 4f added.
- §6.8 H2O scaffold section added.
- Appendix A updated with v2.6 and v2.7 Llama panel rows; total updated to 5,164+ cells.
- §14.3 future work rows updated.

---

## Total evidence count (v2.6)

| Panel | Cells |
|-------|------:|
| Headline (v1) | 1,500 |
| Evidence-plus (v2) | 144 |
| External smoke pilot | 216 |
| MBPP supplement | 144 |
| BFCL export-50 drift | 1,200 |
| KIVI offline adapter | 640 |
| **HF LongBench v2.6 (new)** | **720** |
| **BFCL validity v2.7 Llama (new)** | **600** |
| **Total** | **5,164** |

`exactkv_failures = 0` across all 5,164 cells.
