# When Does Compressed KV Start Lying? Token-Level Drift in KV Cache Compression

**ExactKV Technical Report (v3.0 — complete: 8,132 GPU cells, five benchmark families, h2o_sim + int6_sim + int4_per_vec_sim GPU-validated)**

*All quantitative values are read from on-disk artifacts, primarily
`reports/scale_7b/raw.json`, `reports/evidence_plus/raw.json`,
`reports/external_panels/summary_all.json`, `reports/external_panels/*_merged_raw.json`,
`reports/external_panels/v30/`, `reports/public_release/leaderboard_final.json`,
`reports/phaseF_kernel_benchmark.json`, and `docs/METRIC_DEFINITIONS.md`. No
results are invented. Claim boundaries follow `docs/CLAIM_BOUNDARIES.md` and the
[`claim decision table`](../release_synthesis/claim_decision_table.md).*

---

## Positioning (one sentence)

Unlike KVQuant, KIVI, SnapKV, and CacheGen, ExactKV does **not** propose a new
compression method. Unlike VeriCache [vericache2026], it is **not** a
throughput-optimized lossless **serving** system and does **not** reproduce
VeriCache's scheduling, memory tiering, or cross-resource staggering. ExactKV is a
**compressor-agnostic diagnostic and evaluation crash-test framework with leaderboard-style reporting** that uses
verifier-mediated semantics to measure **where and how** compressors drift
(first divergence, acceptance, agreement, exactness failures), not to maximize
inference throughput.

---

## 1. Abstract

Lossy KV-cache compression is widely deployed, but most evaluations report aggregate
quality and stop there. They do not answer when a compressed KV cache starts
producing different tokens than the full-precision cache would have.

ExactKV is a compressor-agnostic **crash-test framework with leaderboard-style reporting**
that measures **first-divergence index**, draft acceptance, verifier agreement,
and **`exactkv_failure`** under verifier-mediated (draft/verify/commit) semantics.

Across **8,132 completed GPU cells** (Llama-3.1-8B and Mistral-7B, five benchmark
families, six built-in compressor classes¹) all panels report **`exactkv_failures = 0`**. Four
main findings emerge:

1. **Task type dominates drift.** `int4_sim` divergence spans 6% (MBPP code) → 11%
(BFCL short-gen) → 50% (BFCL long-gen) → **90%** (HF LongBench reading), while
H2O-style eviction (even 75% kept) reaches **100%** on LongBench — worse than int4_sim
at matched memory budget.

2. **Generation length is the within-task driver.** On BFCL, `int4_sim` divergence scales
7× from 9% (mnt=16) to 62% (mnt=256); `int8` stays near-zero throughout.

3. **Compressor class determines failure mode.** Top-k logit autopsy over 1,103 divergent
cells identifies: near-tie noise (int8, mean lossy rank 2.4, fdi=22); distribution shift
(int4_sim, rank 3.5, fdi=8); attention destruction (H2O-style, rank 6.7, fdi=1). Three
forensic case studies with full top-5 logit traces confirm each mechanism.

4. **ExactKV preserves 100% of downstream validity.** Despite 50% token drift,
ExactKV preserves all 106/106 full-KV valid BFCL tool calls across both models.
Tool-call validity is maintained across all four BFCL task categories (simple 37%,
parallel 30%, multi-turn 23%, AST-eval 15% baseline valid rate).

`int6_sim` (6-bit per-tensor) is GPU-validated non-catastrophic: 0% divergence on BFCL
and MBPP, 37.5% on HF LongBench — cleanly between int8 (15%) and int4_sim (86%).
`int4_per_vec_sim` (KIVI/KVQuant-style per-vector INT4) achieves 0% divergence on
structured-output and code tasks, and 55.6% on LongBench — non-catastrophic but higher
than int6_sim on extreme long-context (8K) reading. Per-vector granularity helps on
structured tasks, while bit-width still matters at 8K LongBench context.
GPU-validated on both Mistral-7B-Instruct-v0.3 and Llama-3.1-8B (1,568 cells total,
`exactkv_failures=0`). Results are **not** official benchmark scores, production
serving claims, or a reproduction of VeriCache [vericache2026] throughput-oriented
serving. ExactKV does **not** claim novelty for compressed-KV draft plus full-KV verify.

¹ **Six built-in compressor classes:** `noop` (baseline), `int8`, `int6_sim`,
`int4_per_vec_sim`, `int4_sim`, and H2O-style eviction (`h2o_sim` family counted as one
class). `kivi_offline` is an external adapter diagnostic (§6.4.6), evaluated separately.

---

## 2. Introduction

Lossy KV-cache compression is widely deployed as a transparent optimization
for LLM inference. Under greedy decoding, a single perturbed logit can flip
the argmax — after which trajectories diverge. Aggregate metrics (perplexity,
LongBench accuracy, task pass rates) average over this divergence and hide
*where* it begins, *how severe* it is, and *which compressor design choices*
drive it.

ExactKV reframes KV-cache evaluation around a single diagnostic question:
**at what generated token does the compressed-cache path first stop matching
full-KV greedy decoding?** We answer this with a compressor-agnostic crash-test
framework that runs a draft/verify/commit loop — draft from compressed KV,
verify each token against the full-KV oracle, accept the matching prefix,
correct on mismatch — and records first-divergence index, draft acceptance,
verifier agreement, and exactness failures per cell.

**Contributions:**

1. A **compressor-agnostic crash-test framework** with leaderboard-style reporting
   that measures token-level drift, first-divergence index, acceptance rate, and
   exactness failures across compressors and models (§3–5).
2. **8,132 GPU cells** across Llama-3.1-8B and Mistral-7B, five benchmark families,
   six built-in compressor classes (`noop`, `int8`, `int6_sim`, `int4_per_vec_sim`,
   `int4_sim`, H2O-style eviction), with `exactkv_failures = 0` throughout (§6).
3. Empirical evidence that **task type dominates drift** (6% code → 90% reading for
   int4_sim) and **generation length scales it within a task** (9% → 62% on BFCL,
   7×), two axes that aggregate benchmarks do not resolve (§6.4–6.12).
4. A **logit autopsy** over 1,103 divergent cells identifying three mechanistically
   distinct failure modes: near-tie noise (int8), distribution shift (int4_sim),
  and attention destruction (H2O-style eviction), each with forensic case studies
  (§6.10, §7).
5. **Downstream validity measurement**: despite 50% token drift, all 106/106 full-KV
   valid BFCL tool calls are preserved under verifier-mediated execution (§6.11).
6. **GPU validation of two new compressors** (int6_sim, int4_per_vec_sim) confirming
   non-catastrophic drift on structured tasks; per-vector granularity helps on BFCL/MBPP
   while bit-width still matters at 8K LongBench context (§6.13–6.16, Table 6.16).

### 2.1 Shared problem framing with VeriCache

VeriCache [vericache2026] and ExactKV start from the **same observation**: lossy
KV compression may look acceptable on short outputs or aggregate metrics, but
outputs can **diverge more as decoding continues**, with catastrophic impact in
code generation and tool-calling where token-level exactness matters. VeriCache
uses compressed KV to **draft** tokens and full KV to **verify/correct** them,
guaranteeing **identical greedy-decoding output** to full-KV inference while
preserving much of compression's throughput benefit.

### 2.2 Where ExactKV diverges from VeriCache

VeriCache is a **serving/system paper**. Its contribution is not merely
“draft then verify.” It makes that loop **practical at scale** by:

- keeping **compressed KV on GPU** for fast drafting,
- keeping **full KV in CPU/storage**, loading it only for verification,
- using **cross-resource staggering** so HBM-bound compressed-KV drafting and
  interconnect-bound full-KV verification can overlap across bottlenecks,
- reporting serving throughput gains (e.g. up to ~4× vs full-KV inference in their
  evaluation) with **identical outputs**.

ExactKV is an **evaluation/crash-test framework**. It uses verifier-mediated
semantics to **measure** first divergence on the lossy path, draft acceptance,
verifier agreement, and exactness failures across compressors, **not** to optimize
deployment throughput or reproduce VeriCache's system design. See Section 10.

---

### 2.3 Main Findings at a Glance

| # | Finding | Key number |
|---|---------|-----------|
| 1 | **Task type dominates drift** | int4_sim: MBPP 6% → BFCL short 11% → BFCL long 50% → LongBench 90% |
| 2 | **Generation length scales drift** | int4_sim mnt=16: 9% → mnt=256: 62% (7×) |
| 3 | **Eviction > quantization drift** | H2O-style 75% kept → 100% LongBench divergence |
| 4 | **Three distinct failure modes** | int8: near-tie (rank 2.4, fdi=22); int4_sim: distribution shift (rank 3.5, fdi=8); H2O: attention destruction (rank 6.7, fdi=1) |
| 5 | **100% downstream validity preserved** | ExactKV preserves all 106/106 valid BFCL tool calls despite 50% drift |
| 6 | **Zero correctness failures** | exactkv_failures=0 across all 8,132 GPU cells |
| 7 | **Compressor design-space curve** | int8 → int6 → int4_per_vec → int4_sim → H2O: monotonic LongBench degradation (Table 6.16) |

ExactKV's strongest supported claim is not merely that compressed KV drifts — it is
that **KV-cache drift is governed jointly by task type, generation length, compressor
class, and quantization granularity**, while verifier-mediated decoding preserves
full-KV greedy equivalence (`exactkv_failures=0` throughout).

---

## 3. Definitions

### 3.1 Reference paths

| Path | Definition |
|------|------------|
| **Full-KV reference** | Greedy generation using uncompressed KV (`generate_full_greedy`). Ground truth for token identity. |
| **Lossy path** | Greedy generation using compressed KV only, no verifier (`generate_lossy_greedy`). Shows what happens if compression is trusted directly. |
| **ExactKV path** | Verifier-mediated draft/verify/commit loop (`ExactKVGenerator`). |

### 3.2 Core terms

| Term | Definition |
|------|------------|
| **Draft** | Up to `draft_len` tokens proposed from materialized compressed KV in one verification round. |
| **Verifier** | Compares each draft token to the full-KV greedy prediction at the same prefix. |
| **Commit** | Accept matching draft prefix, on mismatch, commit the verifier's correction token and advance authoritative full KV. |
| **Correction** | A verification round where `correction_token` is non-null (at least one draft token rejected). |

### 3.3 Metrics (formal)

Canonical source: `docs/METRIC_DEFINITIONS.md`.

| Metric | Formal definition | Notes |
|--------|-------------------|-------|
| **`first_divergence_index`** | First generated position *t* where **lossy** compressed-KV greedy token ≠ full-KV greedy token (`lossy.first_divergence_idx`). Measured on the unverified path, **before** correction. | Null when lossy output matches full-KV for entire generation. Panel mean reported only over divergent cells. |
| **`acceptance_rate`** | `total_accepted / total_drafted` across all ExactKV verification rounds in a cell. | Fraction of *drafted* tokens accepted, not rounds. |
| **`verifier_agreement`** | In the scale panel, computed as `total_accepted / total_drafted` (same as acceptance_rate per cell). | Leaderboard field name, not a separate distribution-level metric in this release. |
| **`exactkv_failure`** | `final ExactKV output_ids ≠ full-KV output_ids` under greedy decoding. | **Verifier/commit failure**, not lossy drift. A compressor may diverge often while `exactkv_failure=false` if the verifier corrects. |
| **`divergence_rate`** | Fraction of cells where `token_level_divergence=true` (lossy ≠ full-KV). | Aggregated per compressor in `compressor_summary`. |
| **`divergence_stability_score`** | `1 − divergence_rate` per compressor aggregate. | **Not** independent of divergence_rate, stability is complement of lossy-path divergence frequency. |
| **`compression_ratio`** | `stored_kv_bytes / materialized_working_kv_bytes`. | Stored tensor byte ratio only, not active GPU memory. |

**Bookkeeping invariant** (`exactkv/metrics/acceptance.py`):

```text
total_drafted == total_accepted + total_rejected
```

**Corrections vs rejections:** `total_rejected` counts rejected *draft tokens*.
`total_corrections` counts verification *rounds* where a correction was issued
(one round may reject multiple draft tokens). Example for `int4_sim` on the scale
panel: 1284 accepted / 1502 drafted, **218 rejected tokens**, **116 correction
rounds**, not 218 corrections.

### 3.4 Algorithm (pseudocode)

```text
Prefill prompt → FullKVState + CompressedKVState (from compressor)
while generated < max_new_tokens:
    DRAFT:  propose up to draft_len tokens from compressed KV
    VERIFY: compare each draft token to full-KV greedy prediction
    COMMIT: accept matching prefix, on mismatch, commit verifier token
    ALIGN:  recompress / update compressed state from authoritative full KV
return ExactKV output_ids
```

Greedy decoding only, temperature/top-p sampling not in scope for this release
(Section 18).

---

## 4. Method

ExactKV's core loop is implemented in `exactkv/runtime/exactkv_generator.py`.
Phase A scale benchmarking (`exactkv/benchmarks/phase_a_scale_benchmark.py`)
runs three modes per cell where applicable:

1. **full**, full-KV greedy reference
2. **lossy**, compressed-KV greedy (drift measurement)
3. **exactkv**, verifier-mediated (exactness gate)

Each **cell** is one tuple:

```text
(model, compressor, prompt_id, max_new_tokens)
```

with fixed `draft_len=4`, `seed=0`, greedy decoding, `dtype=float32` in the
scale manifest (runtime may promote to float16 on GPU).

**Scoring (leaderboard-style):**

`0.35·acceptance + 0.25·verifier_agreement + 0.20·(1−normalized_first_divergence) + 0.10·(1−failure_rate) + 0.10·stability`

Source: `reports/public_release/leaderboard_final.json`. Scores for validated built-in
compressors (pilot scale, not official benchmark ranking):

| Rank | Compressor | Model | Score | Accept. | Div. rate† | Stab. | Cells |
|-----:|-----------|-------|------:|--------:|-----------:|------:|------:|
| 1 | `noop` | Llama-3.1-8B | 1.000 | 1.000 | 0.000 | 1.000 | 150 |
| 2 | `int8` | Llama-3.1-8B | 1.000 | 1.000 | 0.000 | 1.000 | 150 |
| 3 | `noop` | Mistral-7B | 1.000 | 1.000 | 0.000 | 1.000 | 150 |
| 4 | `int8` | Mistral-7B | 0.983 | 1.000 | 0.000 | 0.827 | 150 |
| 5 | `int4_sim` | Llama-3.1-8B | 0.859 | 0.852 | 0.520 | 0.480 | 150 |
| 6 | `int4_sim` | Mistral-7B | 0.851 | 0.837 | 0.507 | 0.493 | 150 |

†Div. rate = raw lossy-path token divergence fraction (same metric as §6.3 Table 3).
`int8` Mistral: divergence_rate = 0.000 (zero cells diverge) but stability_score = 0.827.
**Note:** `Stab.` here is the **leaderboard stability subscore** from `leaderboard_final.json`,
not the formal `divergence_stability = 1 − divergence_rate` from Table 1. Under the formal
metric, `int8` Mistral `divergence_stability = 1.000` (consistent with 0.000 divergence_rate).
`spectralquant` (MOCK→`int4_sim`) and `shard` (PROBE_ONLY) are excluded from
all analysis; see §5.1 (Limitations) for their status.

---

## 5. Experimental setup (release panel)

Source: `reports/scale_7b/raw.json` manifest + cell enumeration.

| Parameter | Value |
|-----------|-------|
| **Panel ID** | `phase_a_unified_panel` |
| **Total cells** | **1500** |
| **Cell grid** | 2 models × 5 compressors × 50 prompt variants × 3 `max_new_tokens` values |
| **Models** | `meta-llama/Llama-3.1-8B` (750 cells), `mistralai/Mistral-7B-Instruct-v0.3` (750 cells) |
| **Execution** | Sequential per model (volume constraint), real GPU, `deterministic_mode=false` |
| **Stack** | torch `2.8.0+cu128`, transformers `5.12.1`, ExactKV `0.1.0` |
| **`draft_len`** | **4** (fixed) |
| **`max_new_tokens`** | **4, 8, 16** (three buckets per prompt) |
| **Decoding** | Greedy, no temperature / top-p |
| **Prompt suite** | 50 variants derived from 4 deterministic stress-panel templates (`exactkv/safety/l4_runtime_coupling_stress_panel.py`): capital/factual, simple math, JSON/tool, code completion |
| **Categories (cell counts)** | `capital_france` 390, `simple_math` 390, `json_tool` 360, `code_fn` 360 |
| **Timestamp** | 2026-06-25 |

**Not reported in this artifact (limitations):** GPU model name, CUDA driver version,
confidence intervals, wall-clock overhead per cell, active VRAM telemetry, or
per-prompt confidence bands. Future work should add these for skeptical systems
readers.

### 5.1 Compressor implementations (release panel)

| Compressor | Tier | Role in this release |
|------------|------|----------------------|
| `noop` | BUILTIN | Full-precision KV baseline |
| `int8` | BUILTIN | Per-tensor INT8 simulation |
| `int4_sim` | BUILTIN | Per-tensor INT4 simulation |
| `int6_sim` | BUILTIN | Per-tensor INT6 simulation (§6.13) |
| `int4_per_vec_sim` | BUILTIN | Per-vector INT4 simulation, KVQuant/KIVI-style (§6.14) |
| `h2o_sim` | BUILTIN | H2O-style token eviction simulation |

> **Note:** The 1,500-cell headline panel raw JSON also contains `spectralquant`
> (MOCK → delegates to `int4_sim`) and `shard` (PROBE_ONLY heuristic) slots.
> Neither is a real external compressor integration; both are excluded from all
> analysis and results tables. See §15 (Limitations) for details.

### 5.2 External benchmark smoke panels

Source: `reports/external_panels/summary_all.json`,
`reports/external_panels/*_merged_raw.json`, `reports/external_panels/analysis_pack.json`.

These panels extend ExactKV to recognizable **benchmark-shaped prompt families**
(LongBench-style QA/summarization, RULER-style retrieval, BFCL-style tool JSON,
HumanEval-style code). They are **ExactKV drift panels**: they measure
first-divergence index, acceptance rate, and `exactkv_failure` under
verifier-mediated semantics. They do **not** report task accuracy, tool
executability, pass@1, or official leaderboard scores.

| Parameter | Value |
|-----------|-------|
| **Hardware** | RunPod RTX A5000 (24 GB), June 2026 |
| **Model (completed, first workflow)** | `meta-llama/Llama-3.1-8B` only |
| **Model (failed, first workflow)** | `mistralai/Mistral-7B-Instruct-v0.3` (disk quota after Llama cache) |
| **Later MBPP run** | Both Llama-3.1-8B and Mistral-7B (`mbpp_gpu_raw.json`, 144 cells) |
| **Later BFCL export-50 run** | Both Llama-3.1-8B and Mistral-7B (`bfcl_export_50_raw.json`, 1,200 cells) |
| **Prompt source** | Bundled pilot JSONL (`benchmarks/prompts/*_pilot.jsonl`); BFCL export-50 uses 50 exported prompts |
| **LongBench HF** | Skipped in first workflow (`datasets` not installed on pod) |
| **Compressors** | `noop`, `int8`, `int4_sim` (built-in only) |
| **Total GPU cells (Llama-only pilots)** | **216** (`deterministic_mode=false`) |
| **Total GPU cells (initial 216 + MBPP)** | **360** (216 + 144) |
| **Total GPU cells (all external smoke)** | **1,560** (216 + 144 MBPP + 1,200 BFCL export-50) |
| **Context range** | 1024–8192 prefill buckets |
| **`exactkv_failures`** | **0** on all completed cells |

Reproduce: `bash scripts/run_external_gpu_workflow.sh`,
`python3 scripts/build_external_analysis_pack.py`.

---

## 6. Results, validated compressors

Table 1: built-in compressors with direct ExactKV integration (`noop`, `int8`,
`int4_sim`). Aggregates from `compressor_summary` in `reports/scale_7b/raw.json`
(300 cells each: 150 per model).

| Compressor | Cells | Mean acceptance | Divergence rate† | Stability‡ | Mean 1st div. (lossy)§ | `exactkv_failure` |
|------------|------:|----------------:|-----------------:|-----------:|----------------------:|------------------:|
| `noop` | 300 | 1.000 | 0.00 | 1.00 |, | 0.00 |
| `int8` | 300 | 1.000 | 0.00 | 1.00 |, | 0.00 |
| `int4_sim` | 300 | 0.851 | 0.52 | 0.48 | 2.0¶ | 0.00 |

† **Divergence rate** = fraction of cells where the **lossy** path diverges from
full-KV greedy (`token_level_divergence`).  
‡ **Stability** = `1 − divergence_rate` (complement, not an independent semantic metric).  
§ Mean over cells **with** lossy divergence only.  
¶ Lossy first-divergence histogram for `int4_sim` (300 cells): index 1 → 78 cells,
3 → 39, 5 → 24, 8 → 13 (heavy-tailed, mean ≈ 2.0 understates early failures).

**Acceptance aggregates (`acceptance_table`, int4_sim only):**

| | Value |
|---|------:|
| Cells (`num_runs`) | 150 per compressor aggregate row* |
| `total_drafted` | 1502 |
| `total_accepted` | 1284 |
| `total_rejected` | 218 |
| `total_corrections` | 116 (rounds with correction, not rejected-token count) |
| `exactkv_failures` | 0 |

\*Acceptance table rows in `raw.json` aggregate non-probe compressors over 150
cells per compressor name in the acceptance grouper, int8/noop show 1400 drafted
(150 cells × fewer divergence-driven partial rounds).

**Interpretation:** `int8` and `noop` show no lossy-path divergence on this panel.
`int4_sim` drifts in **52%** of cells on the lossy path, yet **`exactkv_failure =
0`** because the verifier corrects. This is the central distinction the reader
must not miss.

### 6.2 Metric interpretation (divergence vs acceptance)

**Divergence rate and acceptance rate measure different failure modes.** A
compressor can have **low lossy-path divergence** but **poor verifier-mediated
acceptance** if its draft trajectory frequently proposes locally rejected
continuations that the verifier must correct early, the lossy greedy path may
still match full-KV on many cells while the ExactKV loop rejects many draft
tokens. Conversely, a compressor may **diverge often** on the lossy path but
still show **relatively high acceptance** if divergences occur late or after long
accepted prefixes (e.g. `int4_sim` at `max_new_tokens=16` with first divergence at
index 8 and acceptance 0.94 in one release-panel cell).
Divergence rate and acceptance must be read together — they are not redundant and
should not be collapsed into a single "quality" score without context. For example,
`int4_sim` at `max_new_tokens=16` can show first divergence at index 8 with
acceptance 0.94: how late drift occurs determines how much of the draft is still useful.

Top leaderboard rows (Llama-3.1-8B): `noop` 1.00, `int8` 1.00, `int4_sim` ~0.86.
Full table: `reports/public_release/leaderboard_final.json`.

### 6.3 Evidence-plus long-context panel (GPU supplement)

Source: `reports/evidence_plus/raw.json`. RunPod **RTX A5000** (24 GB), torch
`2.8.0+cu128`, transformers `5.12.1`, `deterministic_mode=false`, June 2026.

| Parameter | Value |
|-----------|-------|
| **Panel ID** | `evidence_plus_panel` |
| **Total cells** | **144** (72 per model) |
| **Grid** | 2 models × 6 base prompts × 2 context buckets × 3 compressors × 2 `max_new_tokens` |
| **Context buckets** | **512**, **1024** prefill tokens (padded long-context prompts) |
| **`max_new_tokens`** | **16, 32** |
| **Compressors** | `noop`, `int8`, `int4_sim` (built-in only, external adapters skipped) |
| **`exactkv_failures`** | **0** |

**Table 2, aggregates by context bucket (all compressors, both models):**

| Context bucket | Cells | Lossy divergence rate | Mean acceptance |
|----------------|------:|----------------------:|----------------:|
| 512 | 72 | 0.111 | 0.994 |
| 1024 | 72 | 0.097 | 0.996 |

**Table 3, built-in compressors (evidence-plus panel, 144 cells):**

| Compressor | Cells | Mean acceptance | Divergence rate | `exactkv_failure` |
|------------|------:|----------------:|----------------:|------------------:|
| `noop` | 48 | 1.000 | 0.00 | 0.00 |
| `int8` | 48 | 1.000 | 0.00 | 0.00 |
| `int4_sim` | 48 | 0.985 | 0.31 | 0.00 |

**`int4_sim` divergence by model** (24 cells each): Mistral-7B **41.7%** (10/24),
Llama-3.1-8B **20.8%** (5/24). Mean first-divergence index on divergent `int4_sim`
cells: **11.1** (Mistral 11.5, Llama 10.4), later than the short-generation
headline panel (mean ≈ 2.0), consistent with longer generations and padded prefills.

**Diagnostic timing** (full + lossy + ExactKV per cell, **not** serving throughput):
mean **3854 ms**, p50 **4060 ms**, p90 **5178 ms** over 144 cells (`timing_ms.total_cell`).

**Interpretation:** Longer prefills do not eliminate `int4_sim` lossy-path drift,
verifier-mediated output remains exact on this supplement (`exactkv_failures = 0`).
Divergence rates are **lower than** the 1500-cell short-generation headline for
`int4_sim` (52%) but **non-zero** under long-context conditions, the distinction
between drift and exactness failure persists.

### 6.4 External Benchmark Smoke Panels

**Source:** `reports/external_panels/summary_all.json`,
`reports/external_panels/*_merged_raw.json`, `reports/external_panels/mbpp_gpu_raw.json`,
`reports/external_panels/analysis_pack.json`.
RunPod **RTX A5000**, June 2026, `deterministic_mode=false`.

**External smoke totals:** **1,560 GPU cells** = **216** Llama-only cells on
LongBench/RULER/BFCL/HumanEval bundled pilots + **144** MBPP cells (Llama + Mistral) +
**1,200** BFCL export-50 cells (both models, 50 prompts). Excludes 640-cell KIVI offline panel (§6.4.6).

**Table 4, Llama-only external smoke panels (216 GPU cells):**

| Panel | Prompt source | Model | Context buckets | `max_new_tokens` | Cells | Divergence rate | Mean acceptance | `exactkv_failure` | Mean ms | P90 ms | Claim boundary |
|-------|---------------|-------|-----------------|------------------|------:|----------------:|----------------:|------------------:|--------:|-------:|----------------|
| LongBench pilot | bundled pilot | Llama-3.1-8B | 2048, 4096 | 16, 32 | 72 | 0.083 | 0.994 | 0 | 6422 | 8781 | Drift only, not official LongBench |
| RULER 2048/4096 | bundled pilot | Llama-3.1-8B | 2048, 4096 | 16, 32 | 48 | 0.083 | 0.993 | 0 | 6436 | 8793 | Drift only, not official RULER |
| RULER 8192 | bundled pilot | Llama-3.1-8B | 8192 | 16, 32 | 24 | 0.083 | 0.993 | 0 | 12232 | 13640 | Drift only, not official RULER |
| BFCL pilot | bundled pilot (4 prompts) | Llama-3.1-8B | 1024, 2048 | 16, 32 | 48 | 0.063 | 0.999 | 0 | 4715 | 6342 | Drift only, not official BFCL |
| HumanEval | bundled pilot (4 prompts) | Llama-3.1-8B | 1024, 2048 | 32 | 24 | 0.000 | 1.000 | 0 | 5791 | 6392 | Drift only, not official HumanEval |

**Table 4b, MBPP code-drift smoke panel (144 GPU cells, both models):**

| Panel | Prompt source | Models | Context buckets | `max_new_tokens` | Cells | Divergence rate | Mean acceptance | `exactkv_failure` | Mean ms | P90 ms | Claim boundary |
|-------|---------------|--------|-----------------|------------------|------:|----------------:|----------------:|------------------:|--------:|-------:|----------------|
| MBPP | bundled pilot (6 prompts) | Llama + Mistral | 512, 1024 | 16, 32 | 144 | 0.021 | 0.999 | 0 | 3830 | 5121 | Drift only, not MBPP pass@1; no test execution |

**Table 4c, BFCL export-50 tool-call drift panel (1,200 GPU cells, both models):**

| Panel | Prompt source | Models | Context buckets | `max_new_tokens` | Cells | Div. rate (`int4_sim`) | `exactkv_failure` | Claim boundary |
|-------|---------------|--------|-----------------|------------------|------:|----------------------:|------------------:|----------------|
| BFCL export-50 | 50 exported prompts | Llama + Mistral | 1024, 2048 | 16, 32 | 1200 | 0.113 overall (L: 0.055, M: 0.170) | 0 | Drift only, not official BFCL; not JSON-completeness |

**`int4_sim` divergence on BFCL export-50 by model:**

| Compressor | Model | Cells | Div. rate | Wilson 95% CI |
|------------|-------|------:|----------:|---------------|
| `int4_sim` | Llama-3.1-8B | 200 | 0.055 | [0.031, 0.096] |
| `int4_sim` | Mistral-7B | 200 | 0.170 | [0.124, 0.228] |
| `int8` | both | 400 | 0.000 | [0.000, 0.010] |
| `noop` | both | 400 | 0.000 | [0.000, 0.010] |

On the **216-cell Llama-only table**, **`noop` and `int8` show 0% divergence** and
all **15** divergent cells use **`int4_sim`** (15/72 = 20.8% of `int4_sim` cells
in that subset). The **MBPP supplement** adds **3** further divergent cells (all
Llama `int4_sim` at 1024 prefill); Mistral is clean on MBPP (72/72).
The **BFCL export-50** panel (1,200 cells) confirms the `int4_sim` drift pattern
across both models at scale: Mistral shows 3× higher susceptibility than Llama.

**Wilson 95% CIs on divergence rate (panel-level, `int4_sim` cells only unless noted):**

| Panel | n | Divergent | Rate | 95% CI lower | 95% CI upper |
|-------|--:|----------:|-----:|-------------:|-------------:|
| LongBench pilot | 72 | 6 | 8.3% | 3.9% | 17.0% |
| RULER 2048/4096 | 48 | 4 | 8.3% | 3.3% | 19.7% |
| RULER 8192 | 24 | 2 | 8.3% | 2.3% | 25.8% |
| BFCL pilot | 48 | 3 | 6.3% | 2.2% | 16.8% |
| HumanEval pilot | 24 | 0 | 0.0% | 0.0% | 14.2% |
| MBPP pilot (both) | 144 | 3 | 2.1% | 0.7% | 6.0% |
| BFCL export-50 `int4_sim` Llama | 200 | 11 | 5.5% | 3.1% | 9.6% |
| BFCL export-50 `int4_sim` Mistral | 200 | 34 | 17.0% | 12.4% | 22.8% |
| BFCL export-50 `int8`/`noop` (all) | 800 | 0 | 0.0% | 0.0% | 0.5% |

Source: `scripts/build_external_analysis_pack.py` (`wilson_ci` function, 95% two-sided).

**Overall panel acceptance CIs (Wilson 95%):**

| Scope | Full-acceptance rate | 95% CI lower | 95% CI upper |
|-------|---------------------:|-------------:|-------------:|
| All 216 external cells | 93.7% | 90.5% | 96.8% |
| int4_sim cells only (72) | ~85% | — | — |
| noop + int8 cells (144) | ~100% | — | — |

"Full acceptance" = cell where acceptance_rate ≥ 1.0.
Source: `analysis_pack.json` → `totals.acceptance_full_rate_ci95`.

**Notable external findings:**

- On the **216-cell Llama-only pilots**, `noop` and `int8` showed **0% divergence**
  and all **15** divergent cells were **`int4_sim`** (0 exactness failures).
- **LongBench pilot:** `int4_sim` divergence rate **25%** (6/24 cells). Highest
  category divergence on **`gov_report` (25%)** and **`passage_retrieval_en` (16.7%)**.
- **RULER 8192:** **`niah_single` showed 33% divergence** (2/6 cells) at 8192 prefill
  with `int4_sim` (both divergent cells are `int4_sim`).
- **BFCL:** **`int4_sim` divergence rate 18.75%** (3/16 cells) on the bundled tool/JSON
  pilot, including tool-call truncation at 2048 prefill.
- **HumanEval:** **no divergence** observed at these pilot settings (24/24 cells clean).
- **MBPP:** **144 GPU cells** on bundled 6-prompt pilot (`mbpp_gpu_raw.json`,
  validated). **`exactkv_failures = 0`**. **`noop` and `int8`: 0% divergence.**
  Three divergent cells, all **Llama `int4_sim` at 1024 prefill** (`mbpp_002`,
  `mbpp_004`). **Mistral: 0% divergence** (72/72). Generated code is **not**
  executed against `test_list` (token drift only).

8192-token RULER cells roughly double mean diagnostic wall-clock (12.2 s vs 6.4 s
at 2K–4K). HumanEval shows a benign baseline on this panel, not evidence that code
generation is immune under all compressors or longer generations.

#### 6.4.1 External panel case studies

Extracted from `reports/external_panels/case_studies_extracted.json` (15 divergent
cells plus one HumanEval benign baseline). Full-KV snippets are verifier-corrected
output tails (`exactkv.output_text`).

| Dataset | Task / category | Ctx | Model | Compressor | 1st div. | Accept. | Full-KV snippet | Lossy snippet | ExactKV | Interpretation |
|---------|-----------------|----:|-------|------------|---------:|--------:|-----------------|---------------|---------|----------------|
| BFCL | ast_eval | 2048 | Llama-3.1-8B | int4_sim | 4 | 1.00 | `, years 3.` + JSON tool call continues | `, years 3.` (stops) | ok | tool-risk |
| LongBench | passage_retrieval_en | 4096 | Llama-3.1-8B | int4_sim | 2 | 0.94 | ` segment_36. ExactKV evidence-plus…` | ` segment_0. ExactKV evidence-plus…` | ok | semantic |
| LongBench | gov_report | 2048 | Llama-3.1-8B | int4_sim | 4 | 0.88 | `…segment_18. ExactKV evidence-plus…` | `…ExactKV evidence-plus panel…` (no segment) | ok | semantic |
| RULER | niah_single | 8192 | Llama-3.1-8B | int4_sim | 10 | 0.94 | `Answer: segment_112. ExactKV…` | `Answer: segment_0. ExactKV…` | ok | semantic |
| HumanEval | code | 1024 | Llama-3.1-8B | int4_sim | n/a | 1.00 | ` segment_16. ExactKV…` | same as full-KV | ok | benign |
| MBPP | code | 1024 | Llama-3.1-8B | int4_sim | 11 | 0.94 | palindrome body + ` segment_16.` | stops after signature | ok | code-risk |

#### 6.4.2 Expanded example: BFCL / tool-call JSON drift (Case J)

**Source:** `reports/external_panels/bfcl_merged_raw.json`, cell
`bfcl_ast_001_ctx2048`, `int4_sim`, `max_new_tokens=32`.

**Prompt (truncated):** finance tool call with `principal 1000, rate 0.05, years 3`.

**Full-KV output (tail):** `, years 3.` then continues
`JSON tool call: {"name": "compound_interest", "arguments": {"principal": 1000, "rate": 0.`

**Lossy output (tail):** `, years 3.` (generation stops before JSON tool call)

**ExactKV output:** matches full-KV path.

**First divergent token index (lossy path):** 4

**Comment:** Structural tool-call risk. The lossy path truncates before the JSON
tool invocation that the full-KV path begins. If the lossy path were served
directly, downstream executors would see incomplete or missing tool schema. Verifier-mediated
ExactKV restores the full-KV continuation on this cell.

**BFCL export-50 drift panel (completed, 1200 cells):** A larger BFCL panel was run using
50 exported prompts, both models (Llama-3.1-8B + Mistral-7B), context buckets 1024/2048,
`max_new_tokens ∈ {16, 32}`, compressors `noop`/`int8`/`int4_sim`.
Source: `reports/external_panels/bfcl_export_50_raw.json`.

**Table: BFCL export-50 drift panel (1200 cells, both models):**

| Compressor | Model | Cells | Div. rate | Wilson 95% CI | `exactkv_failure` |
|------------|-------|------:|----------:|---------------|------------------:|
| `noop` | Llama-3.1-8B | 200 | 0.000 | [0.000, 0.019] | 0 |
| `noop` | Mistral-7B | 200 | 0.000 | [0.000, 0.019] | 0 |
| `int8` | Llama-3.1-8B | 200 | 0.000 | [0.000, 0.019] | 0 |
| `int8` | Mistral-7B | 200 | 0.000 | [0.000, 0.019] | 0 |
| `int4_sim` | Llama-3.1-8B | 200 | 0.055 | [0.031, 0.096] | 0 |
| `int4_sim` | Mistral-7B | 200 | 0.170 | [0.124, 0.228] | 0 |

**Key findings:**
- `exactkv_failures = 0` across all 1,200 cells.
- `int4_sim` diverges in **11.3%** of cells overall (45/400); Mistral-7B is 3× more susceptible
  than Llama-3.1-8B (17.0% vs 5.5%).
- Higher divergence at shorter context (ctx=1024: 13.5%, ctx=2048: 9.0%) — compressed KV
  drift is not purely a long-context phenomenon.
- `int8`/`noop`: zero divergence across all 400 cells each. Upper CI bound = 1.9%.
- All divergence is of task-risk type: tool-call JSON truncation or early stopping before
  structured output is emitted.

**BFCL tool-call completeness note:** These panels use `max_new_tokens ∈ {16, 32}`, which
measures **drift detection**, not JSON completeness. Post-hoc validity parsing of the
48-cell pilot found 0% fully valid tool calls (56.3% partial JSON). For completeness
measurement, longer budgets (`max_new_tokens=128,256`) are required. The drift panel
here correctly captures whether the compressed path diverges from the full-KV path; it
does not assert that either path produces complete JSON.

**Framework finding:** drift measurement and JSON-completeness measurement require
different `max_new_tokens` budgets. The BFCL export-50 panel measures drift; separate
validity runs would measure completion rates.

#### 6.4.3 Expanded example: LongBench retrieval drift (Case K)

**Source:** `reports/external_panels/longbench_pilot_merged_raw.json`, cell
`lb_passage_retrieval_001_ctx4096`, `int4_sim`, `max_new_tokens=32`.

**Task:** identify which paragraph mentions build 412.

**Full-KV output (tail):** ` segment_36. ExactKV evidence-plus panel:…`

**Lossy output (tail):** ` segment_0. ExactKV evidence-plus panel:…`

**ExactKV output:** matches full-KV (`segment_36`).

**First divergent token index (lossy path):** 2

**Comment:** Semantic retrieval drift at an early token. The lossy path answers with
a different paragraph index (`segment_0` vs `segment_36`). Under verifier correction,
final output remains exact. This is diagnostic drift measurement, not a LongBench
accuracy score.

#### 6.4.4 Expanded example: RULER needle drift at 8192 (Case L)

**Source:** `reports/external_panels/ruler_8192_merged_raw.json`, cell
`ruler_niah_single_4k_ctx8192`, `int4_sim`, `max_new_tokens=32`.

**Task:** NIAH-style special code retrieval after 8192-token padded prefill.

**Full-KV output (tail):** `Answer: segment_112. ExactKV evidence-plus panel:…`

**Lossy output (tail):** `Answer: segment_0. ExactKV evidence-plus panel:…`

**ExactKV output:** matches full-KV.

**First divergent token index (lossy path):** 10

**Comment:** Needle-answer drift under long padded context. Divergence appears at
index 10 with acceptance 0.94. Illustrates that longer prefills (8192) increase
diagnostic wall-clock without eliminating `int4_sim` lossy-path drift on this pilot.

**HumanEval note:** No divergent cells appear in `reports/external_panels/humaneval_merged_raw.json`
on this panel (Case M). A code-drift forensic example is **not** included because
the artifact contains no lossy-path divergence for HumanEval pilot prompts at
these settings.

**MBPP note:** See **§6.4.5** for the MBPP code-drift example. No MBPP test execution
was performed; this is not pass@1.

#### 6.4.5 Expanded example: MBPP code-completion drift (Case N)

**Source:** `reports/external_panels/mbpp_gpu_raw.json`, cell
`mbpp_002_ctx1024`, Llama-3.1-8B, `int4_sim`, `max_new_tokens=16`.

**Task:** complete `is_palindrome(s: str) -> bool` (bundled MBPP pilot prompt).

**Full-KV output (tail):** function body continues, then ` segment_16.`

**Lossy output (tail):** stops after `def is_palindrome(s: str) -> bool:` with
padded filler only (no completed body tail).

**ExactKV output:** matches full-KV path.

**First divergent token index (lossy path):** 11

**Comment:** Code-completion drift on Llama `int4_sim` at 1024 prefill. Mistral
shows no lossy-path divergence on this MBPP panel (72/72 cells clean). Token drift
only, **not** MBPP pass@1 or sandboxed test execution.

#### 6.4.5b Expanded example: BFCL export-50 tool-call structural drift (Case P)

**Source:** `reports/external_panels/bfcl_export_50_raw.json`, cell
`bfcl_parallel_parallel_6_ctx2048`, `int4_sim`, `max_new_tokens=32`, Mistral-7B.

**Prompt (truncated):** Finance tool-call context; system prompt includes tool schema with
`"name": "calculate_sales_tax"` and argument definitions. Conversation queries sales tax
on a purchase.

**Full-KV output (first 130 chars):**
```
", "description": "Calculate the sales tax for a given purchase amount in a specific city and state.", "parameters": {
```

**Lossy output (first 130 chars):**
```
", "arguments": { "purchase_amount": 30.45, "city": "Chicago",
```

**First divergent character position (lossy vs full-KV):** 8

**Comment:** Structural tool-call drift. The full-KV path continues the tool **schema
description** (`"description": ...`), while the lossy path skips to emitting tool
**argument values** (`"arguments": {...}`). This produces invalid JSON — the schema
definition is interrupted and replaced by an argument invocation at the wrong nesting
level. Downstream tool executors would either reject this as malformed JSON or silently
execute the wrong argument structure.

This exemplifies **structural compression-induced tool-call risk** beyond simple
truncation: the lossy path does not stop early, it generates confidently incorrect JSON
at a different structural level than the full-KV path. ExactKV detects the divergence
at character 8 and restores the full-KV continuation (`exactkv_failure=0`).

**Model-specificity:** This divergence pattern is Mistral-7B specific on this panel.
Llama-3.1-8B shows lower `int4_sim` divergence on BFCL export-50 (5.5% vs Mistral's
17.0%), consistent with model-dependent compressed-KV susceptibility.

#### 6.4.6 KIVI offline compressor panel — real HF results

**Source:** `reports/external_panels/kivi_longbench_hf_raw.json` (320 cells),
`reports/external_panels/kivi_mbpp_hf_raw.json` (320 cells).
RunPod **RTX A5000**, June 2026, `deterministic_mode=false`.

**Panel design:**

| Panel | Prompt source | Model | Context buckets | `max_new_tokens` | Compressors |
|-------|--------------|-------|-----------------|-----------------|-------------|
| LongBench HF | real HF (THUDM/LongBench, 10 subsets × 2 ex.) | Llama-3.1-8B | 2048, 4096 | 16, 32 | noop, int8, int4_sim, kivi_offline |
| MBPP HF | real HF (google-research-datasets/mbpp, 20 prompts) | Llama-3.1-8B | 512, 1024 | 16, 32 | noop, int8, int4_sim, kivi_offline |

**Claim boundary:** `kivi_offline` uses real upstream KIVI quantizer math
(`models.utils_quant` simulate path, `is_simulated=False`) but
`supports_real_bytes_claim=False` (no CUDA/Triton kernels). This is **not** KIVI
production serving. Drift measurement only.

**Results:**

| Compressor | LongBench div. rate | LongBench mean accept. | MBPP div. rate | MBPP mean accept. | `exactkv_failures` | Notes |
|-----------|--------------------:|-----------------------:|---------------:|------------------:|-------------------:|-------|
| `noop` | 0.0% | 1.000 | 0.0% | 1.000 | 0 | Baseline |
| `int8` | 13.8% | 0.994 | 0.0% | 1.000 | 0 | |
| `int4_sim` | 91.2% | 0.818 | 5.0% | 0.995 | 0 | Built-in sim |
| `kivi_offline` | **100.0%** | **0.004** | **100.0%** | **0.001** | **0** | Real KIVI quant math |

**Key findings:**

1. **`kivi_offline` shows 100% token-level divergence with acceptance ≈ 0** on both
panels. Lossy path outputs are catastrophically corrupted (repeated `!` characters,
e.g. `"-!!!!!!!!!!!!!!!!"` where full-KV produces normal text). This indicates that
the `kivi_offline` adapter, as currently integrated via the offline simulate path,
produces incorrect compressed KV values — this is an **adapter-level diagnostic,
not a claim about the KIVI algorithm as deployed in production.** Reproducing KIVI's
reported performance requires the original CUDA-optimized serving path.

2. **ExactKV detected and corrected all 160 `kivi_offline` corrupt cells**
(`exactkv_failures=0` on all 640 cells across both panels). The verifier-mediated
loop rejected every KIVI draft token (acceptance ≈ 0) and fell back to full-KV,
restoring correct output. **This is the ExactKV crash-test working as designed.**

3. **Real HF prompts dramatically increase `int4_sim` drift vs bundled pilots.**
On LongBench, `int4_sim` divergence rises from **20.8%** (bundled pilot) to **91.2%**
(real HF). This demonstrates that bundled pilot prompts substantially underestimate
real-world `int4_sim` drift, and reinforces the value of running against real
benchmark datasets.

4. **`int8` on real HF prompts:** divergence of 13.8% on LongBench (vs 8.3% on
pilot) — still much lower than `int4_sim` but elevated versus the controlled pilot.
`int8` on MBPP HF remains 0%.

**ExactKV crash-test interpretation:** The `kivi_offline` result demonstrates that
ExactKV can distinguish between (a) moderate drift (int4_sim, acceptance 0.82) and
(b) catastrophic KV corruption (kivi_offline, acceptance ≈ 0). Both are fully
corrected by the verifier loop with `exactkv_failures=0`, but the acceptance rate
and first-divergence index provide a quantitative fingerprint of the failure mode.

---

## 6.5 ExactKV-HF-LongBench Drift Panel (v2.6, real HF, both models)

**Status:** **Complete.** Both models, 720 cells, `exactkv_failures=0`.
Artifacts: `reports/external_panels/hf_longbench_v26_{Llama_3_1_8B,Mistral_7B}_raw.json`,
`reports/external_panels/hf_longbench_v26_merged_raw.json`.

**Design:**

| Setting | Value |
|---------|-------|
| Source | Real `THUDM/LongBench` via HF `datasets` (not bundled pilot JSONL) |
| Subsets | narrativeqa, qasper, multifieldqa_en, hotpotqa, 2wikimqa, gov_report, trec, samsum, lcc, passage_retrieval_en |
| Examples per subset | 2 (default `max_per_subset=2`) |
| Total prompts | 20 |
| Context buckets | 2048, 4096, 8192 tokens |
| max_new_tokens | 32, 64 |
| Compressors | noop, int8, int4_sim |
| Models | Llama-3.1-8B (base, same as headline panel) + Mistral-7B-Instruct-v0.3 |
| Cells | 20 × 3 × 2 × 3 × 2 = **720** |
| Top-k logits | Stored on divergent cells (`--store-top-k-logits`) for v2.7 logit autopsy |
| Official LongBench score? | **No** — ExactKV uses HF LongBench prompts as *drift prompts*, not as official leaderboard scoring |

**Expected claim (conservative):** "We use 20 real HF LongBench examples across 10 subsets as ExactKV
drift prompts, measuring first-divergence index, acceptance, and `exactkv_failure` for noop/int8/int4_sim
across Llama-3.1-8B and Mistral-7B. Results are not official LongBench leaderboard scores."

**Table 4d — Complete (720 cells, both models, `exactkv_failures=0`):**

| Model | Compressor | n | Divergent | Rate | CI₉₅ | Mean accept. | ExactKV fail |
|-------|-----------|---|-----------|------|------|-------------|-------------|
| Llama-3.1-8B | noop | 120 | 0 | 0.0% | [0.0%, 3.1%] | 1.000 | 0 |
| Llama-3.1-8B | int8 | 120 | 33 | 27.5% | [20.3%, 36.1%] | 0.988 | 0 |
| Llama-3.1-8B | int4_sim | 120 | 110 | **91.7%** | [85.3%, 95.4%] | 0.825 | 0 |
| Mistral-7B | noop | 120 | 0 | 0.0% | [0.0%, 3.1%] | 1.000 | 0 |
| Mistral-7B | int8 | 120 | 26 | 21.7% | [15.2%, 29.9%] | 0.988 | 0 |
| Mistral-7B | int4_sim | 120 | 107 | **89.2%** | [82.3%, 93.6%] | 0.861 | 0 |
| **All (720)** | noop | **240** | **0** | **0.0%** | [0.0%, 1.6%] | 1.000 | 0 |
| **All (720)** | int8 | **240** | **59** | **24.6%** | [19.4%, 30.6%] | 0.988 | 0 |
| **All (720)** | int4_sim | **240** | **217** | **90.4%** | [86.1%, 93.5%] | 0.843 | 0 |

**Per-subset int4_sim divergence (both models combined):**

| Subset | Type | Llama div | Mistral div |
|--------|------|----------|------------|
| 2wikimqa | multi-hop QA | 12/12 (100%) | 12/12 (100%) |
| gov_report | long summarization | 11/12 (92%) | 12/12 (100%) |
| hotpotqa | multi-hop QA | 12/12 (100%) | 12/12 (100%) |
| **lcc** | **code completion** | **11/12 (92%)** | **5/12 (42%)** |
| multifieldqa_en | multi-field QA | 6/12 (50%) | 10/12 (83%) |
| narrativeqa | reading comp. | 12/12 (100%) | 12/12 (100%) |
| passage_retrieval_en | document retrieval | 12/12 (100%) | 12/12 (100%) |
| qasper | academic QA | 12/12 (100%) | 12/12 (100%) |
| samsum | dialogue summarization | 10/12 (83%) | 12/12 (100%) |
| trec | classification | 12/12 (100%) | 8/12 (67%) |

**Noteworthy:** `lcc` (long code completion) shows substantially lower int4_sim divergence —
42% for Mistral vs 100% for most reading comprehension tasks. This is consistent with
the cross-panel observation that code-generation tasks are more stable under int4_sim
than open-text generation.
**very high int4_sim divergence** (90.4% overall), the highest observed across all
ExactKV panels. Even int8 shows 24.6% divergence on LongBench — notable because int8
shows 0% on BFCL tool-calling, MBPP code, and RULER tasks.

`noop` confirms 0% divergence (correct baseline). `exactkv_failures=0` across all 720 cells —
ExactKV verifier catches and corrects every divergent cell.

**Median first-divergence position:** int4_sim diverges at token 4 (Llama) / 6 (Mistral)
out of the first 32–64 generated tokens. This is very early divergence, meaning the model's
output trajectory branches almost immediately under int4 quantization on long-context reading tasks.

**Context × task sensitivity pattern** (int4_sim divergence across all panels):

| Panel | Context | Task type | int4_sim div. rate | Mean accept. | ExactKV fail |
|-------|---------|-----------|-------------------|-------------|-------------|
| Headline (v1) | ~500 tok | Mixed/headline | 51.3% | — | 0 |
| MBPP supplement | 512–1024 tok | Python code | 6.2% | 0.999 | 0 |
| BFCL export-50 (Llama) | 1K–2K tok | Tool-calling | 5.5% | — | 0 |
| BFCL export-50 (Mistral) | 1K–2K tok | Tool-calling | 17.0% | — | 0 |
| **HF LongBench (Llama, v2.6)** | **2K–8K tok** | **Open-text reading** | **91.7%** | 0.825 | **0** |
| **HF LongBench (Mistral, v2.6)** | **2K–8K tok** | **Open-text reading** | **89.2%** | 0.861 | **0** |

The pattern shows **task type is the dominant driver** of int4_sim divergence, not simply
context length. Long-context open-text reading/summarization (LongBench) shows the highest
divergence; short-answer code (MBPP) and tool-calling (BFCL) show the lowest.
ExactKV's verifier successfully handles all divergence types with zero failures.

**Artifact paths (post-run):**
- `reports/external_panels/hf_longbench_v26_Llama_3_1_8B_raw.json`
- `reports/external_panels/hf_longbench_v26_Mistral_7B_raw.json`
- `reports/external_panels/hf_longbench_v26_merged_raw.json`

---

## 6.6 Divergence Autopsy: Logit Margins at First Flip

**Status:** **Complete.** 1,103 divergent cells with stored top-k logits analyzed across
v2.6 (LongBench), v2.7 (BFCL validity), and v2.8 (H2O eviction) panels.

`--store-top-k-logits` was enabled for all three panels. At each divergence position,
the top-5 full-KV token probabilities are stored along with the actual lossy-chosen token.
Analysis: `scripts/analyze_logit_margins.py`. See §6.10 for full mechanistic analysis.

**Table 4e — Logit Autopsy at Divergence Point (1,103 cells with stored top-k logits):**

| Compressor | n div | top-1 flip | near-tie (<0.05) | mean margin | mean lossy rank | median fdi |
|-----------|-------|-----------|-----------------|------------|--------------|-----------|
| int8 | 62 | 69% | **66%** | 0.116 | 2.4 | 22 |
| int4_sim | 563 | 83% | 26% | 0.305 | 3.5 | 8 |
| h2o_sim_75 | 160 | **100%** | 0% | 0.545 | 6.5 | 1 |
| h2o_sim | 160 | **100%** | 0% | 0.536 | 6.7 | 1 |
| h2o_sim_25 | 158 | **100%** | 1% | 0.551 | 6.5 | 1 |

Three mechanistically distinct failure modes are confirmed (see §6.10):
- **int8**: Near-tie noise (66% near-tie, mean rank 2.4, late divergence fdi=22)
- **int4_sim**: Distribution shift (83% flip, mean rank 3.5, early divergence fdi=8)
- **H2O**: Attention destruction (100% flip, mean rank 6.5–6.7, immediate fdi=1)

---

## 6.7 BFCL Tool-Call Validity Panel (v2.7)

**Status:** **Complete.** 1,200 cells, both models, `exactkv_failures=0`.

**Design:**

| Setting | Value |
|---------|-------|
| Source | BFCL export-50 prompts (`benchmarks/prompts/bfcl_export.jsonl`) |
| Prompts | 50 |
| Context buckets | 1024, 2048 tokens |
| max_new_tokens | **128, 256** (long enough for complete JSON tool calls) |
| Compressors | noop, int8, int4_sim |
| Models | Llama-3.1-8B + Mistral-7B-Instruct-v0.3 |
| Cells | 50 × 2 × 2 × 3 × 2 = **1,200** |
| Validity metric | Balanced-brace scan for valid JSON tool call per output |

This panel addresses the prior limitation: BFCL export-50 used `max_new_tokens={16,32}`,
too short for complete JSON generation. With `mnt={128,256}`, full tool calls are producible.

**Table 4f — BFCL Tool-Call Validity Panel (v2.7, 1,200 cells):**

| Compressor | Model | n | Divergent | Rate | CI₉₅ | Full-KV valid | ExactKV valid | ExactKV fail |
|-----------|-------|---|-----------|------|------|-------------|-------------|-------------|
| noop | Llama | 200 | 0 | 0.0% | [0.0%, 1.9%] | 63/200 (31.5%) | 63/200 (31.5%) | 0 |
| noop | Mistral | 200 | 0 | 0.0% | [0.0%, 1.9%] | 43/200 (21.5%) | 43/200 (21.5%) | 0 |
| int8 | Llama | 200 | 0 | 0.0% | [0.0%, 1.9%] | 63/200 (31.5%) | 63/200 (31.5%) | 0 |
| int8 | Mistral | 200 | 3 | 1.5% | [0.5%, 4.3%] | 43/200 (21.5%) | 43/200 (21.5%) | 0 |
| int4_sim | Llama | 200 | 90 | **45.0%** | [38.3%, 51.9%] | 63/200 (31.5%) | 63/200 (31.5%) | 0 |
| int4_sim | Mistral | 200 | 111 | **55.5%** | [48.6%, 62.2%] | 43/200 (21.5%) | 43/200 (21.5%) | 0 |
| **All (1,200)** | both | **1,200** | **204** | **17.0%** | [14.9%, 19.4%] | **318/1200 (26.5%)** | **318/1200 (26.5%)** | **0** |

**Validity by max_new_tokens (both models, all compressors):**

| mnt | n | Full-KV valid JSON | Rate |
|-----|---|--------------------|------|
| 128 | 600 | 60 | 10.0% |
| 256 | 600 | 258 | **43.0%** |

**Key findings (v2.7, both models complete):**

- **Generation length is the dominant divergence driver for BFCL.** Same prompts, same context (1K–2K):
  - `mnt=16/32` (export-50 panel): 11.2% int4_sim divergence
  - `mnt=128/256` (this panel): **50.3%** int4_sim divergence — ~4.5× higher.
- **int4_sim divergence is model-dependent:** Llama 45.0% vs Mistral 55.5%.
- **Valid JSON rate scales with generation budget:** mnt=128 → 10% valid; mnt=256 → 43% valid.
  mnt=16/32 cannot produce complete JSON tool calls; this panel is the first to measure full validity.
- **Valid JSON rate is identical across compressors** — KV compression does not cause structural
  JSON corruption for non-divergent tokens. ExactKV repairs the divergent suffix, preserving full-KV
  validity in all cases.
- **int8 remains near-zero:** 0% divergence for Llama, 1.5% for Mistral at mnt=128/256.
- `exactkv_failures=0` across all 1,200 cells.

Logit autopsy of divergent BFCL validity cells (int4_sim, 204 divergences) is included
in §6.10. Key finding: int4_sim shows distribution-shift signature (mean rank 3.5,
median fdi=8) even in short structured tool-call sequences.

---

## 6.9 Cross-Panel Divergence Analysis: Task-Type and Generation-Length Sensitivity

This section synthesises divergence rates across all completed ExactKV panels to reveal
the dominant factors driving KV compression drift.

**Table 4h — Cross-panel int4_sim/int8/noop divergence rates:**

| Panel | Task type | Context | mnt | Models | noop div | int8 div | int4_sim div | EKV fail |
|-------|-----------|---------|-----|--------|---------|---------|-------------|---------|
| Headline (v1) | mixed | ~500 tok | 16/32 | Llama+Mistral | 0.0% | 0.0% | **51.3%** | 0 |
| Evidence-plus (v2) | mixed | 512/1024 | 16 | Llama+Mistral | 0.0% | 0.0% | **31.2%** | 0 |
| MBPP code-drift | code | 512/1024 | 16/32 | Llama+Mistral | 0.0% | 0.0% | **6.2%** | 0 |
| BFCL export-50 | tool-call | 1K–2K | 16/32 | Llama+Mistral | 0.0% | 0.0% | **11.2%** | 0 |
| BFCL validity v2.7 (both) | tool-call | 1K–2K | 128/256 | Llama+Mistral | 0.0% | 0.8% | **50.3%** | 0 |
| HF LongBench v2.6 | reading/summarization | 2K–8K | 32/64 | Llama+Mistral | 0.0% | 24.6% | **90.4%** | 0 |
| H2O-style eviction v2.8 (h2o_sim) | reading/summarization | 2K–4K | 32/64 | Llama+Mistral | 0.0% | — | **90.6%** (int4) / **100%** (H2O) | 0 |

**Key observations:**

1. **noop is always 0%** across all panels and models — confirms the ExactKV deterministic
   baseline is correct everywhere.

2. **int4_sim divergence spans 6–90% by task family**: Python code generation (MBPP: 6%)
   and tool-calling (BFCL short-gen: 11%) are far more stable than open-text reading
   and summarization (LongBench: 90%). Task type is the primary driver.

3. **int8 is near-zero except on LongBench**: 0% on BFCL export-50/MBPP/RULER/evidence-plus,
   near-zero on BFCL validity (0.8%), but 24.6% on LongBench open-text tasks. int8 quantization
   noise is amplified by long-context reading-comprehension generation patterns.

4. **Generation length (mnt) matters for BFCL**: same BFCL prompts, mnt=16/32 → 11.2%
   int4_sim divergence; mnt=128/256 → **50.3%**. Longer generation exposes more opportunities
   for cumulative argmax flips, independent of the prompt or context length.

5. **`exactkv_failures=0` throughout**: ExactKV's verifier successfully catches and
   corrects every divergent cell across all task types, context lengths, and generation
   lengths tested.

Script: `python3 scripts/analyze_panel_divergence.py --all-compressors --markdown`

---

## 6.8 H2O-Style Token-Eviction Compressor Panel (v2.8)

**Status:** **Complete.** 800 cells, both models, `exactkv_failures=0`.

`H2OSimCompressor` adapter implemented and unit-tested (`exactkv/compressors/h2o_sim.py`, 26 tests passing).

### H2O-style eviction: method and claim boundaries

**This panel implements an H2O-*style* simulation, not a faithful reproduction of the original
H2O paper [zhang2023h2o].** The simulation retains a `keep_ratio` fraction of tokens
using a simplified heavy-hitter approximation (attention-sink + recency window) applied at
prefill time to the KV cache, without the original paper's online streaming update or
GPU-efficient sparse-attention kernel.

**Claim boundary:** Results characterize the *class* of token-eviction compression — where
KV entries are deleted rather than quantized — and demonstrate that even mild eviction
(75% of tokens retained) causes immediate and universal divergence on LongBench reading tasks.
These results should be interpreted as "H2O-style eviction simulation" diagnostics, not
as a performance characterization of the published H2O serving system.

**Compressor name mapping:**

| Internal name | Simulation type | keep_ratio | Interpretation |
|---|---|---|---|
| `h2o_sim_75` | H2O-style | 0.75 | Mild: keep 75% of KV tokens |
| `h2o_sim` | H2O-style | 0.50 | Standard: keep 50% of KV tokens |
| `h2o_sim_25` | H2O-style | 0.25 | Aggressive: keep 25% of KV tokens |

**Design:**

H2O (Heavy Hitter Oracle) [zhang2023h2o] is a token-eviction family: rather than quantizing KV values,
it drops tokens entirely, keeping only "heavy hitter" tokens (approximated as attention sinks
+ recency window). This represents a fundamentally different compression class from int4/int8
quantization — ExactKV's first eviction-class compressor simulation.

| Setting | Value |
|---------|-------|
| Source | Real HF LongBench prompts (`--prompt-source hf`) |
| Prompts | 20 × 2 subsets (narrativeqa, hotpotqa, samsum, trec, lcc, gov_report, qasper, 2wikimqa, passage_retr., multifieldqa) |
| Context buckets | 2048, 4096 |
| max_new_tokens | 32, 64 |
| Compressors | `noop`, `int4_sim`, `h2o_sim` (50%), `h2o_sim_75` (75%), `h2o_sim_25` (25%) |
| Models | Llama-3.1-8B + Mistral-7B |
| Cells | 20 × 2 ctx × 2 mnt × 5 comp × 2 models = **800** |

**Compressor variants:**

| Variant | keep_ratio | Behaviour |
|---------|-----------|-----------|
| `h2o_sim_75` | 0.75 | Mild eviction: keep approximately 75% of prefill tokens |
| `h2o_sim` | 0.50 | Standard: keep approximately 50% of prefill tokens |
| `h2o_sim_25` | 0.25 | Aggressive: keep approximately 25% of prefill tokens |

**Eviction mechanism:** At inference time, after the full prefill phase, `h2o_sim` ranks
all KV-cache token positions by a simplified attention-based heavy-hitter score
(cumulative attention weight received across heads). The lowest-scoring
`(1 - keep_ratio)` fraction of tokens are evicted from the KV cache; the remaining
tokens (attention sinks at positions 0–3 + highest-scoring tokens by recency/attention)
are retained for all subsequent generation steps. This approximates the H2O eviction
policy [zhang2023h2o] but does not implement the original paper's online streaming
update or GPU-efficient sparse-attention kernel.

**Table 4g — H2O Token-Eviction Panel (v2.8, 800 cells):**

| Compressor | Model | n | Divergent | Rate | CI₉₅ | Mean accept. | ExactKV fail |
|-----------|-------|---|-----------|------|------|-------------|-------------|
| noop | Llama | 80 | 0 | 0.0% | [0.0%, 4.6%] | 1.000 | 0 |
| noop | Mistral | 80 | 0 | 0.0% | [0.0%, 4.6%] | 1.000 | 0 |
| int4_sim | Llama | 80 | 75 | 93.8% | [86.2%, 97.3%] | 0.816 | 0 |
| int4_sim | Mistral | 80 | 70 | 87.5% | [78.5%, 93.1%] | 0.862 | 0 |
| h2o_sim_75 | Llama | 80 | 80 | **100.0%** | [95.4%, 100.0%] | 0.328 | 0 |
| h2o_sim_75 | Mistral | 80 | 80 | **100.0%** | [95.4%, 100.0%] | 0.346 | 0 |
| h2o_sim | Llama | 80 | 80 | **100.0%** | [95.4%, 100.0%] | 0.391 | 0 |
| h2o_sim | Mistral | 80 | 80 | **100.0%** | [95.4%, 100.0%] | 0.396 | 0 |
| h2o_sim_25 | Llama | 80 | 78 | **97.5%** | [91.3%, 99.3%] | 0.382 | 0 |
| h2o_sim_25 | Mistral | 80 | 80 | **100.0%** | [95.4%, 100.0%] | 0.389 | 0 |
| **All H2O (480)** | both | **480** | **478** | **99.6%** | [98.3%, 99.9%] | 0.372 | **0** |
| **All (800)** | both | **800** | **638** | **79.8%** | [76.8%, 82.4%] | 0.662 | **0** |

**H2O eviction vs. int4_sim quantization (comparison):**

| Compressor | Type | Budget | n | Div. rate | Mean accept. |
|-----------|------|--------|---|-----------|-------------|
| noop | none | 100% KV | 160 | 0.0% | 1.000 |
| int4_sim | quantization | ~50% bytes | 160 | **90.6%** | 0.839 |
| h2o_sim_75 | eviction | 75% kept | 160 | **100.0%** | 0.337 |
| h2o_sim | eviction | 50% kept | 160 | **100.0%** | 0.394 |
| h2o_sim_25 | eviction | 25% kept | 160 | **98.8%** | 0.385 |

**Key findings (v2.8):**

- **H2O-style token eviction produces near-universal divergence on LongBench reading tasks.**
  Even mild eviction (`h2o_sim_75`, keeping 75% of tokens) hits **100% divergence** on both
  models. `int4_sim` quantization (90.6%) is substantially less disruptive.
- **Acceptance rate reveals the severity gap:** H2O `mean_accept ≈ 0.33–0.39` vs. int4_sim
  `0.84` — H2O diverges at roughly the 1st token (first_div=1 for narrativeqa), while int4_sim
  survives ~84% of tokens before drifting. ExactKV catches all divergence.
- **Eviction budget has little effect on divergence rate:** going from 75% kept to 25% kept
  barely changes the divergence rate (100% → 98.8%), but acceptance rate stays flat at ~0.38.
  Once tokens are evicted, the distribution shifts immediately regardless of keep ratio.
- **noop: 0% divergence, exactkv_failures=0** — baseline fully preserved.
- `exactkv_failures=0` across all 800 cells despite 100% divergence rates.

Logit autopsy of all 800 H2O cells (§6.10) confirms attention destruction:
100% top-1 flip with mean lossy rank 6.5–6.7 and immediate divergence (fdi=1).
This is qualitatively distinct from quantization failure modes.

---

## 6.10 Divergence Autopsy: Top-k Logit Analysis at First-Divergence Token

ExactKV stored the top-5 full-KV and lossy logit distributions at each divergence point
(`--store-top-k-logits`). This section examines the **mechanism** of KV compression drift
across 1,103 divergent cells with stored logit data (v2.6 LongBench, v2.7 BFCL, v2.8 H2O).

### Methodology

At each divergence position (first token where `lossy_output ≠ full_output`), we record:
- **top-1 flip:** whether the lossy model chose a different top-1 token than full-KV
- **near-tie:** whether the full-KV margin (prob of full argmax − prob of lossy-chosen token) < 0.05
- **mean margin:** average of that probability gap
- **mean lossy rank:** where the lossy-chosen token ranks in the full-KV top-k distribution
- **median first-divergence index (fdi):** median position in the output sequence

### Table 4e — Logit Autopsy Summary (1,103 divergent cells with stored top-k logits)

| Compressor | n div | top-1 flip | near-tie (<0.05) | mean margin | mean lossy rank | median fdi |
|-----------|-------|-----------|-----------------|------------|--------------|-----------|
| int8 | 62 | **69%** | 66% | 0.116 | 2.4 | 22 |
| int4_sim | 563 | **83%** | 26% | 0.305 | 3.5 | 8 |
| h2o_sim_75 | 160 | **100%** | 0% | 0.545 | 6.5 | 1 |
| h2o_sim | 160 | **100%** | 0% | 0.536 | 6.7 | 1 |
| h2o_sim_25 | 158 | **100%** | 1% | 0.551 | 6.5 | 1 |

### Mechanistic interpretation

The three compression classes produce qualitatively different divergence signatures:

**int8 — subtle near-tie perturbation:**
66% of int8 divergences are near-ties (full-KV margin < 0.05). The lossy model
typically picks the **2nd-ranked** token in the full-KV distribution (mean rank 2.4),
and divergence occurs late in the sequence (median fdi = 22). This is consistent with
int8 quantization introducing small noise that only flips argmax decisions where two
tokens are nearly equiprobable. This explains why int8 only diverges on the most
sensitive prompts (LongBench open-text, where competing continuations have similar
log-probabilities).

**int4_sim — compressed distribution shift:**
83% top-1 flip, but 26% near-tie (far fewer near-ties than int8). Mean margin = 0.305 —
the full-KV model is moderately confident in its choice (30.5% probability gap), yet
int4_sim selects a token ranked ~3rd–4th in the full-KV distribution (mean rank 3.5).
Divergence occurs at median token 8. This indicates that 4-bit quantization causes
a systematic shift in the attention-weighted activation patterns, not just random near-tie
noise. The corrupted KV state biases the residual stream toward plausible but incorrect
alternatives.

**H2O eviction — attention distribution destruction:**
100% top-1 flip, 0% near-tie, mean margin 0.54. The H2O eviction model selects tokens
ranked **6th–7th** in the full-KV distribution (mean rank 6.5–6.7), diverging
immediately (median fdi = 1). This is not a subtle probability perturbation — token
eviction fundamentally disrupts the attention distribution from the first generated
token. The lossy model is effectively operating on a corrupted context representation
that bears little resemblance to the full-KV prediction.

---

## 7. Forensic divergence case studies

Three representative cells from the logit autopsy (§6.10) illustrate the mechanistic
failure modes in detail.

### 7.1 Forensic Case A — int8: Near-tie noise (NarrativeQA, 4K context)

> **Compressor:** int8 | **Task:** NarrativeQA reading comprehension | **Context:** 4,096 tokens | **Model:** Llama-3.1-8B | **First divergence:** token 53

| Rank | Token | Full-KV prob | Lossy prob | Δ |
|------|-------|-------------|-----------|---|
| 1 | `' very'` | **0.129** ← full-KV argmax | 0.113 | −0.016 |
| 2 | `' fine'` | 0.113 ← **int8 chose this** | 0.129 | +0.016 |
| 3 | `' pretty'` | 0.083 | 0.082 | −0.001 |
| 4 | `' great'` | 0.075 | 0.073 | −0.002 |
| 5 | `' good'` | 0.065 | 0.064 | −0.001 |

**Margin:** 0.129 − 0.113 = **0.016** (below 0.05 near-tie threshold).
**Lossy rank:** 2 (int8 chose the 2nd-ranked full-KV token, a near-synonym).
**Mechanism:** INT8 noise shifted the logit of `' fine'` above `' very'` by a
tiny margin. Both are semantically valid, so output quality is preserved — but
the token-level trajectory diverges, and ExactKV correctly falls back.
**ExactKV action:** Detects divergence at token 53, commits full-KV output. `exactkv_failure=0`.

---

### 7.2 Forensic Case B — int4_sim: Distribution shift (NarrativeQA, 2K context)

> **Compressor:** int4_sim | **Task:** NarrativeQA reading comprehension | **Context:** 2,048 tokens | **Model:** Llama-3.1-8B | **First divergence:** token 1

| Rank | Token | Full-KV prob | Notes |
|------|-------|-------------|-------|
| 1 | `','` | **0.271** ← full-KV argmax | Punctuation — correct |
| 2 | `' they'` | 0.232 | |
| 3 | `' and'` | 0.202 | |
| 4 | `' I'` | 0.113 ← **int4_sim chose this** | Rank 4, margin=0.158 |
| 5 | `' it'` | 0.039 | |

**Margin:** 0.271 − 0.113 = **0.158** (full-KV is moderately confident in `','`).
**Lossy rank:** 4 (int4_sim selected a token the full-KV model assigned only 11.3%).
**Mechanism:** 4-bit quantization of the KV cache shifts the attention-weighted activation
toward `' I'`, which is plausible as a sentence start but incorrect given the context.
This is **not** a near-tie — the full-KV model clearly prefers `','`. The lossy model
is systematically biased toward narrative-style continuations by the compressed attention state.
**ExactKV action:** Detects divergence at token 1, commits full-KV output. `exactkv_failure=0`.

---

### 7.3 Forensic Case C — H2O-style: Attention destruction (HotpotQA, 2K context)

> **Compressor:** h2o_sim (50% kept) | **Task:** HotpotQA multi-hop reasoning | **Context:** 2,048 tokens | **Model:** Llama-3.1-8B | **First divergence:** token 1

| Rank | Token | Full-KV prob | Notes |
|------|-------|-------------|-------|
| 1 | `'1'` | **0.056** ← full-KV argmax | Number/year start |
| 2 | `'15'` | 0.039 | |
| 3 | `'20'` | 0.035 | |
| 4 | `'10'` | 0.034 | |
| 5 | `'5'` | 0.034 | |
| >5 | `'?'` | <0.034 ← **H2O chose this** | Outside top-5! |

**Margin:** Full-KV top-1 at 0.056 (near-uniform over factual numbers); H2O chose rank >5.
**Lossy rank:** 6+ (the eviction model's first token is not in the full-KV top-5 at all).
**Mechanism:** H2O-style eviction dropped key factual anchors (the multi-hop entities),
leaving only attention-sink tokens and recency. The model's context representation is
fundamentally corrupted from the first generated token — it cannot even start the answer
correctly. This is **not** a probability shift; it is a collapsed representation.
**ExactKV action:** Detects divergence at token 1 (fdi=1, the earliest possible), commits
full-KV output. `exactkv_failure=0`.

### 7.4 Key takeaway

The logit autopsy confirms three distinct failure modes:
1. **Near-tie noise (int8):** Small quantization error flips closely-contested decisions. Rare on structured tasks; more common on open-text where the model is uncertain.
2. **Distribution shift (int4_sim):** The KV approximation biases activations, pushing probability mass toward lower-ranked alternatives even when the full-KV model is moderately confident.
3. **Attention destruction (H2O-style):** Token eviction eliminates context that anchors predictions, causing immediate and severe argmax errors from the first generated token.

`exactkv_failures=0` across all 1,103 cells — the verifier correctly identifies and corrects all three failure mode types.

---

## 6.11 Downstream Task-Impact Metrics

Token-level drift is the ExactKV primary metric, but downstream task output quality is the
ultimate concern. This section reports task-level validity metrics from existing panel data.

### 6.11.1 BFCL tool-call validity preservation

For BFCL validity v2.7 (1,200 cells), we recorded `full_kv_tool_call_valid` and
`exactkv_tool_call_valid` for each cell — whether the output is a complete, parseable
JSON tool call with both `name` and `arguments` fields.

**Table 4i — BFCL downstream task-impact: tool-call validity preservation**

| Compressor | n | Drift rate | Full-KV valid | ExactKV valid | Preserved |
|-----------|---|-----------|--------------|--------------|----------|
| noop | 400 | 0.0% | 106/400 (26.5%) | 106/400 (26.5%) | **100%** |
| int8 | 400 | 0.8% | 106/400 (26.5%) | 106/400 (26.5%) | **100%** |
| int4_sim | 400 | 50.2% | 106/400 (26.5%) | 106/400 (26.5%) | **100%** |

Per-model breakdown:

| Compressor | Model | n | Drift | Full-KV valid | ExactKV valid |
|-----------|-------|---|-------|--------------|--------------|
| noop | Llama | 200 | 0.0% | 63/200 (31.5%) | 63/200 (31.5%) |
| noop | Mistral | 200 | 0.0% | 43/200 (21.5%) | 43/200 (21.5%) |
| int4_sim | Llama | 200 | 45.0% | 63/200 (31.5%) | 63/200 (31.5%) |
| int4_sim | Mistral | 200 | 55.5% | 43/200 (21.5%) | 43/200 (21.5%) |

**Key finding:** Despite `int4_sim` diverging in 50.2% of cells, ExactKV preserves
**100% of full-KV valid tool calls** (106/106 for both models). The verifier-mediated
execution path returns the full-KV output whenever the lossy path diverges, so no
valid tool call is lost to compression.

> **Note:** "Preservation" means ExactKV output matches full-KV output — not that
> full-KV always produces valid calls. The full-KV baseline achieves only 26.5% valid
> tool-call rate on this panel (106/400), due to prompt-level factors and truncation.
> ExactKV matches that baseline exactly; it does not improve it.

**Interpretation:** Token drift rates and downstream validity rates answer different
questions. Drift measures prefix-level trajectory changes; validity measures output
usefulness. The ExactKV verifier closes this gap: even at 50% drift, output validity
is fully preserved, while an *unguarded* lossy path would corrupt valid calls in
proportion to its drift rate.

### 6.11.2 BFCL tool-call validity by task category

**Table 4i-b — BFCL v2.7 downstream validity breakdown by task category (int4_sim)**

| Category | n (int4_sim) | Drift | Full-KV valid | ExactKV valid | Preservation |
|----------|------------|-------|--------------|--------------|-------------|
| simple | 104 | 49.0% | 38/104 (37%) | 38/104 (37%) | **100%** |
| parallel | 104 | 54.8% | 31/104 (30%) | 31/104 (30%) | **100%** |
| multi_turn | 104 | 59.6% | 24/104 (23%) | 24/104 (23%) | **100%** |
| ast_eval | 88 | 35.2% | 13/88 (15%) | 13/88 (15%) | **100%** |

ExactKV achieves 100% validity preservation across all four BFCL task categories
despite per-category drift rates of 35–60%. More complex task categories (multi-turn
59.6%, parallel 54.8%) show higher drift but identical validity preservation — the
verifier handles structural complexity and multi-step tool calls equally well.

### 6.11.3 LongBench per-token acceptance as draft-utility proxy

On divergent HF LongBench cells, ExactKV still accepts the majority of generated
tokens before detecting divergence. Table 4i-c shows acceptance rates per task type
under `int4_sim` (100% divergent) and `int8` (partial divergence).

**Table 4i-c — LongBench per-token acceptance rate (draft quality before divergence)**

| Task | int4_sim div | int4_sim acc | int8 div | int8 acc |
|------|------------|------------|---------|---------|
| trec | 83.3% | 0.904 | 0.0% | 1.000 |
| multifieldqa_en | 66.7% | 0.955 | 16.7% | 0.997 |
| lcc | 66.7% | 0.908 | 4.2% | 0.994 |
| qasper | 100.0% | 0.899 | 16.7% | 0.994 |
| samsum | 91.7% | 0.854 | 16.7% | 0.993 |
| gov_report | 95.8% | 0.781 | 16.7% | 0.997 |
| hotpotqa | 100.0% | 0.787 | 37.5% | 0.972 |
| 2wikimqa | 100.0% | 0.880 | 29.2% | 0.987 |
| narrativeqa | 100.0% | 0.716 | 66.7% | 0.973 |
| passage_retrieval | 100.0% | 0.746 | 41.7% | 0.974 |

Even when all cells diverge (100%), `int4_sim` achieves 72–95% per-token acceptance —
meaning ExactKV speculatively accepts the lossy draft for most of the generation
before detecting and correcting the divergent suffix. `int8` achieves 97–100% on the
same tasks, confirming that near-tie-noise divergence is rare and late.

---

## 6.12 Scaling Analysis: Context-Length and Generation-Length Sensitivity

### 6.12.1 Context-length scaling (LongBench v2.6)

**Table 4j — Divergence rate by context bucket (HF LongBench v2.6, 80 cells each)**

| Compressor | 2K context | 4K context | 8K context |
|-----------|-----------|-----------|-----------|
| noop | 0.0% | 0.0% | 0.0% |
| int8 | 21.2% [13.7%, 31.4%] | 27.5% [18.9%, 38.1%] | 25.0% [16.8%, 35.5%] |
| int4_sim | **95.0%** [87.8%, 98.0%] | **86.2%** [77.0%, 92.1%] | **90.0%** [81.5%, 94.8%] |

**Observation:** For `int4_sim`, divergence is near-ceiling at all context lengths
(86–95%), saturating quickly even at 2K. This confirms that on open-text reading tasks,
the dominant driver is **task type** (reading/summarization), not context length
specifically. `int8` shows a more moderate, roughly flat profile (21–28%) across
context lengths — its near-tie noise mechanism doesn't strongly amplify with context.

### 6.12.2 Generation-length scaling (BFCL tool-calling, both panels)

**Table 4k — int4_sim divergence rate by generation budget (BFCL, Llama + Mistral)**

| Compressor | mnt=16 | mnt=32 | mnt=128 | mnt=256 |
|-----------|--------|--------|---------|---------|
| noop | 0.0% | 0.0% | 0.0% | 0.0% |
| int8 | 0.0% | 0.0% | 0.5% | 1.0% |
| int4_sim | **9.0%** | **13.5%** | **38.5%** | **62.0%** |

**Observation:** Generation length is the dominant BFCL divergence driver. `int4_sim`
divergence scales from 9% at mnt=16 to 62% at mnt=256 — a 7× increase — while
`int8` remains near-zero throughout. Each additional token generation step gives the
distribution-shift failure mode an additional opportunity to flip the argmax.

**Combined scaling story:** The two strongest scaling effects are:
- **Task-type sensitivity** (code 6% → reading 90% for int4_sim at fixed context)
- **Generation-length sensitivity** (9% → 62% for int4_sim at fixed task type)
- Context-length effect is weaker once task type is controlled (near-ceiling for int4_sim on reading)

---

## 6.13 int6_sim: A Non-Catastrophic Intermediate Compressor

To place int8 and int4_sim on a quantization curve, we add `int6_sim` — a 6-bit
symmetric quantization simulation (64 discrete levels, scale = max(|x|)/31).

**Implementation**: `exactkv/compressors/int6_sim.py`. Quantization error validated
numerically: int6 mean absolute error is 4.15× int8 and 0.23× int4_sim, confirming
it sits at the expected intermediate point on the compression curve.

**Bit-width comparison:**

| Compressor | Levels | bits/element | Theoretical ratio vs fp16 | Notes |
|-----------|--------|-------------|--------------------------|-------|
| int8 | 256 | 8 | 0.500 | Real compressor, faithful bytes |
| **int6_sim** | **64** | **6** | **0.375** | Simulation, non-catastrophic |
| int4_sim | 16 | 4 | 0.250 | Simulation |
| h2o_sim_75 | — | eviction | 0.250 kept | Eviction class |

**GPU-validated divergence profile (Mistral-7B v3.0, 196 cells):**
- MBPP code tasks: **0%** (prior prediction: ~1–3%)
- BFCL tool-calling: **0%** (prior prediction: ~3–6%)
- HF LongBench: **37.5%** (prior prediction: ~10–40% — within range, mid-point)

`int6_sim` lands cleanly between `int8` (15.3%) and `int4_sim` (86.1%) on LongBench,
confirming the quantization-error interpolation. On structured tasks (BFCL/MBPP), the
error reduction is sufficient to eliminate divergence entirely.

**Claim boundary:** GPU-validated on both models (v3.0, 1,568 total cells, `exactkv_failures=0`).
Mistral: 0%/0%/37.5% (MBPP/BFCL/LB). Llama: 0%/0%/47.2%. Script: `exactkv/compressors/int6_sim.py`.

---

## 6.14 int4_per_vec_sim: Per-Vector INT4 Quantization (KVQuant/KIVI-Style)

`int4_per_vec_sim` implements per-vector symmetric INT4 quantization, inspired by the
per-vector/per-channel granularity used in KVQuant/KIVI-style KV quantization
[hooper2024kvquant, liu2024kivi]. Instead of a single
scale over the entire KV tensor (as in `int4_sim`), each token-position vector gets its
own scale: `scale = max(|x|) / 7` computed independently per `[batch, head, token]`
position vector of size `head_dim`.

**Why this is non-catastrophic:** KV cache tensors contain outlier heads with 5–10×
larger activations than other heads. Per-tensor INT4 forces a large global scale that
coarsely quantizes all other heads. Per-vector quantization eliminates cross-head
contamination entirely.

**Quantization error comparison (realistic KV shape [1, 32, 512, 64] with outlier heads):**

| Compressor | MAE | Error ×int8 | Granularity |
|-----------|-----|-----------|-------------|
| int8 (per-tensor) | 0.0670 | **1.00×** | per-tensor |
| **int4_per_vec_sim** | **0.1383** | **2.06×** | per-vector |
| int6_sim (per-tensor) | 0.2747 | 4.10× | per-tensor |
| int4_sim (per-tensor) | 0.8441 | **12.59×** | per-tensor |

Per-vector INT4 achieves **6× lower error than per-tensor INT4** while still providing
theoretical 4× compression (0.25× fp16). It is closer to int8 than to int4_sim.

**Predicted vs observed v3.0 divergence (both models, task-family means):**

| Task family | int8 (obs.) | int4_per_vec (predicted) | int4_per_vec (obs.) | int6_sim (obs.) | int4_sim (obs.) |
|------------|------------:|-------------------------:|--------------------:|----------------:|----------------:|
| MBPP code | 0% | ~0–2% | **0%** | 0% | 6.3% |
| BFCL tool-call | 0% | ~1–4% | **0%** | 0% | 52.5% |
| HF LongBench | 18.1% | ~30–50% | **56.3%** | 42.4% | 85.4% |

*Prior analytical prediction for LongBench: ~30–50%. Observed both-model mean: 56.3%
— slightly above the predicted range, but cleanly between `int8` (18.1%) and per-tensor
`int4_sim` (85.4%). Prediction held on BFCL/MBPP (0% observed vs ~0–4% predicted).*

**Key insight — granularity outweighs bit-width (task-conditional):** `int4_per_vec_sim`
(4-bit per-vector) matches int8 exactly on BFCL and MBPP, and achieves 30pp lower divergence
than per-tensor `int4_sim` on LongBench (55.6% vs 86.1%). However, `int6_sim` (6-bit
per-tensor, higher MAE) achieves *lower* LongBench divergence (37.5% vs 55.6%) — because
at 8K context, bit-width contributes meaningfully alongside granularity.

**Claim boundary:** GPU-validated on both models (v3.0, 1,568 total cells, `exactkv_failures=0`).
Mistral: 0%/0%/55.6% (MBPP/BFCL/LB). Llama: 0%/0%/56.9%. Script: `exactkv/compressors/int4_per_vec_sim.py`.

---

## 6.15 v3.0 GPU Validation Panel (int6_sim + int4_per_vec_sim)

**Status: Complete (both models). Mistral-7B-Instruct-v0.3: 784 cells. Llama-3.1-8B: 784 cells. Total: 1,568 v3.0 cells.** Source: `reports/external_panels/v30/`.

**v3.0 divergence results — Mistral-7B-Instruct-v0.3:**

| Family | Compressor | Cells | Divergence Rate | Acceptance | exactkv_failures |
|--------|------------|------:|----------------:|-----------:|-----------------:|
| mbpp | `int8` | 24 | 0.0% | 1.000 | 0 |
| mbpp | `int6_sim` | 24 | 0.0% | 1.000 | 0 |
| mbpp | `int4_per_vec_sim` | 24 | 0.0% | 1.000 | 0 |
| mbpp | `int4_sim` | 24 | 0.0% | 1.000 | 0 |
|       |            |      |                 |            |                  |
| bfcl | `int8` | 100 | 0.0% | 1.000 | 0 |
| bfcl | `int6_sim` | 100 | 0.0% | 1.000 | 0 |
| bfcl | `int4_per_vec_sim` | 100 | 0.0% | 1.000 | 0 |
| bfcl | `int4_sim` | 100 | 45.0% | 0.990 | 0 |
|       |            |      |                 |            |                  |
| longbench | `int8` | 72 | 15.3% | 0.985 | 0 |
| longbench | `int6_sim` | 72 | 37.5% | 0.964 | 0 |
| longbench | `int4_per_vec_sim` | 72 | 55.6% | 0.930 | 0 |
| longbench | `int4_sim` | 72 | 86.1% | 0.846 | 0 |

*Table 6.15a — v3.0 GPU panel results (Mistral-7B, 784 cells). `exactkv_failures=0` throughout.*

**v3.0 divergence results — Llama-3.1-8B:**

| Family | Compressor | Cells | Divergence Rate | Acceptance | exactkv_failures |
|--------|------------|------:|----------------:|-----------:|-----------------:|
| mbpp | `int8` | 24 | 0.0% | 1.000 | 0 |
| mbpp | `int6_sim` | 24 | 0.0% | 1.000 | 0 |
| mbpp | `int4_per_vec_sim` | 24 | 0.0% | 1.000 | 0 |
| mbpp | `int4_sim` | 24 | 12.5% | 0.996 | 0 |
|       |            |      |                 |            |                  |
| bfcl | `int8` | 100 | 0.0% | 1.000 | 0 |
| bfcl | `int6_sim` | 100 | 0.0% | 1.000 | 0 |
| bfcl | `int4_per_vec_sim` | 100 | 0.0% | 1.000 | 0 |
| bfcl | `int4_sim` | 100 | 60.0% | 0.995 | 0 |
|       |            |      |                 |            |                  |
| longbench | `int8` | 72 | 20.8% | 0.988 | 0 |
| longbench | `int6_sim` | 72 | 47.2% | 0.957 | 0 |
| longbench | `int4_per_vec_sim` | 72 | 56.9% | 0.935 | 0 |
| longbench | `int4_sim` | 72 | 84.7% | 0.825 | 0 |

*Table 6.15b — v3.0 GPU panel results (Llama-3.1-8B, 784 cells). `exactkv_failures=0` throughout.*

**Interpretation:**

Both models confirm the same qualitative story with consistent absolute ordering.
`exactkv_failures = 0` across all 1,568 v3.0 cells.

**Cross-model summary (Mistral / Llama):**

| Task | int8 | int6_sim | int4_per_vec_sim | int4_sim |
|------|-----:|---------:|-----------------:|---------:|
| MBPP | 0% / 0% | 0% / 0% | 0% / 0% | 0% / 12.5% |
| BFCL | 0% / 0% | 0% / 0% | 0% / 0% | 45% / 60% |
| LongBench | 15.3% / 20.8% | 37.5% / 47.2% | 55.6% / 56.9% | 86.1% / 84.7% |

- **int6_sim**: 0% on MBPP/BFCL across both models. On LongBench: 37.5% (Mistral)
  and 47.2% (Llama) — cleanly between `int8` and `int4_sim` on both models,
  confirming the analytical quantization-error prediction. A faithful intermediate
  compressor for structured and code tasks.

- **int4_per_vec_sim**: 0% on MBPP/BFCL across both models — **matching int8
  exactly on structured and code tasks**. On LongBench: 55.6% (Mistral) and 56.9%
  (Llama). Despite lower MAE (2.06× int8 vs 4.10× for int6_sim), it shows *higher*
  long-context drift than int6_sim. **Per-vector granularity eliminates drift on
  structured tasks but 4-bit resolution loss accumulates at 8K context.** The claim
  "granularity outweighs bit-width" is task-conditional: it holds fully on
  BFCL/MBPP and partially on LongBench (86% → 57%, a 29pp improvement, but
  not as good as int6_sim's 37%).

- **Monotonic ordering on LongBench (both models)**: `int8 < int6_sim < int4_per_vec_sim < int4_sim` — a clean degradation across the quantization-error axis. All four are confirmed non-catastrophic (verifier corrects all divergent cells, `exactkv_failures=0`).

- `exactkv_failures = 0` across all 1,568 v3.0 cells (784 per model) including every
  divergent LongBench cell — the verifier-mediated loop holds for all three failure regimes.

### 6.16 Core benchmark curve (compressor comparison)

Table 6.16 summarises divergence rates across the full compressor spectrum on the v3.0
panel grid (both models combined, task-family averages). H2O-style eviction values are
from the v2.8 LongBench panel (both models, `h2o_sim_75`).

| Compressor | MBPP | BFCL | HF LongBench | Class |
|------------|-----:|-----:|-------------:|-------|
| `int8` | 0% | 0% | 18.1% | Quantization (8-bit) |
| `int6_sim` | 0% | 0% | 42.4% | Quantization (6-bit) |
| `int4_per_vec_sim` | 0% | 0% | 56.3% | Quantization (4-bit, per-vector) |
| `int4_sim` | 6.3% | 52.5% | 85.4% | Quantization (4-bit, per-tensor) |
| `h2o_sim_75` | — | — | **100%** | Eviction (75% kept) |

*Table 6.16 — Core benchmark curve. v3.0 quantisation compressors: both-model mean
(Mistral/Llama). H2O: v2.8 LongBench panel. `exactkv_failures=0` throughout.*

---

## 8. Divergence case studies (release panel)

The table below uses fields saved in `reports/scale_7b/raw.json` (release panel,
Llama-3.1-8B unless noted). Top-5 logits are **not** stored in the release
artifact. Historical Qwen panel rows are illustrative only (not headline panel).

| Case | Compressor | Prompt / category | `max_new` | 1st div. | Accept. | Corrections | Full-KV snippet | Lossy snippet | ExactKV snippet |
|------|------------|-------------------|----------:|---------:|--------:|------------:|-----------------|---------------|-----------------|
| A | `int4_sim` | `p00_p0_capital_france` / capital | 4 | 3 | 0.75 | 1 | `a city of many` | `a city of art` | `a city of many` |
| B | `int4_sim` | `p01_p1_simple_math` / math | 4 | 1 | 0.50 | 1 | `four. That's` | `four\nTwo plus` | `four. That's` |
| C | `int4_sim` | `p02_p2_json_tool` / JSON | 8 | 5 | 0.70 | 1 | `"New York", "country": "` | `"New York", "temperature":` | `"New York", "country": "` |
| D | `int4_sim` | `p00_p0_capital_france` / capital | 16 | 8 | 0.94 | 1 | `Paris, but the capital of the French language is Quebec City` | `Paris, but the capital of the French Republic is the city of` | matches full-KV |
| F† | `int4_sim` | `p2_json_tool` / JSON (Qwen 0.5B) | n/a | 1 | 0.50 | n/a | `"New York",` | `"state": {"` | verifier-corrected |
| G | `int4_sim` | `lc_001_ctx1024` / long_context (Mistral-7B) | 16 | 2 | 0.88 | n/a | `…checking the alignment of the cache…` | `…verification step uses segment_3…` | matches full-KV |
| H | `int4_sim` | `lc_002_ctx512` / long_context (Mistral-7B) | 16 | 13 | 0.83 | n/a | `…verification always uses full` | `…verification uses segment_` | matches full-KV |
| I | `int4_sim` | `p0_capital_france_ctx512` / capital (Mistral-7B) | 32 | 20 | n/a | n/a | *(diverged late)* | *(diverged late)* | matches full-KV |

†Case F from `reports/phaseA_benchmark.json` / demo cards, **historical only**.
Cases **G–I** from `reports/evidence_plus/raw.json` (512/1024 prefill buckets).

**Case A takeaway:** lossy path diverges at token 3 (`art` vs `many`), verifier-mediated
ExactKV restores full-KV text. **Case C:** structured JSON field flip
(`country` vs `temperature`), the kind of tool-calling failure mode VeriCache and
ExactKV both motivate. **Case D:** late first divergence (index 8) with high
acceptance, illustrates why mean first-divergence index alone is insufficient.
**Case G:** early divergence at index 2 under 1024-token prefill, verifier restores
exact output. **Case H:** late divergence at index 13, high acceptance (0.83) despite
substantial lossy drift.

#### 8.1 KIVI offline catastrophic corruption (Case O)

**Source:** `reports/external_panels/kivi_longbench_hf_raw.json`, cell
`lb_narrativeqa_000_ctx2048`, `kivi_offline`, `max_new_tokens=32`.

**Task:** NarrativeQA comprehension on real THUDM/LongBench HF example (2048-token padded prefill).

**Full-KV output:** ` face, with\nits high forehead and its straight nose, they were positively grotesque.\nI had never see`

**Lossy (KIVI offline) output:** `-!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!` (32 `!` characters)

**ExactKV output:** matches full-KV.

**Acceptance rate:** 0.000. **First divergent token index:** 0. **`exactkv_failure`:** False.

**Comment:** The `kivi_offline` adapter (real KIVI quantizer math, simulate path)
produces catastrophically corrupted KV values — all attention outputs become garbage
and the model degenerates to repeating `!`. Acceptance 0.000 means the verifier
rejected **every single draft token**. The ExactKV loop transparently detected this
at token 0 and fell back to full-KV decoding, restoring the correct output with
zero exactness failure. This demonstrates ExactKV's crash-test capability: it
distinguishes between (a) subtle logit drift (int4_sim, acceptance 0.82,
first divergence ~token 5) and (b) total KV corruption (kivi_offline, acceptance 0,
divergence at token 0), and corrects both. The kivi_offline result is an integration
diagnostic, not a claim about the KIVI algorithm.

**Adapter-level failure analysis (offline simulate path):** The `kivi_offline` adapter
translates KIVI's `models.utils_quant` quantization math into ExactKV's compressor
interface via a simulate-path hook. Candidate root causes for the 100% corruption:

| Hypothesis | Evidence | Confidence |
|-----------|---------|-----------|
| Quantize/dequant dtype mismatch (float16 vs bfloat16) | All cells corrupt; `kivi_offline` uses `is_simulated=False` | Medium |
| KV cache layout mismatch (head/seq dimension order) | Repeated `!` consistent with garbage logits, not partial drift | Medium |
| Missing RoPE re-application after KV reconstruction | KIVI uses different RoPE integration vs standard HF | Medium |
| Layer indexing offset (0-indexed vs 1-indexed hook) | Hard to distinguish from layout mismatch at output | Low |
| Quantization range overflow (int2/int4 clipping on extreme activations) | Possible on 2K padded context with padding values | Low |

Diagnosing which hypothesis is correct requires per-layer KV inspection beyond
ExactKV's current output-level crash-test scope. The adapter is marked
`supports_real_bytes_claim=False` (no CUDA/Triton kernels) and should not be
compared to production KIVI performance figures.


### 8.1 Expanded forensic example: JSON / tool (Case C)

**Source:** `reports/scale_7b/raw.json`, cell `p02_p2_json_tool`, Mistral-7B,
`int4_sim`, `max_new_tokens=8`.

**Prompt (truncated):** `Complete JSON: {"name": "get_weather", "city":`

**Full-KV output:** `"New York", "country": "`

**Lossy output:** `"New York", "temperature":`

**ExactKV output:** `"New York", "country": "` (matches full-KV)

**First divergent token index (lossy path):** 5

**Comment:** Structural tool-call failure. The lossy path flips the JSON field name
from `country` to `temperature` while the schema context still expects location
metadata. This is catastrophic for tool routing if the lossy path were trusted
directly. Verifier-mediated ExactKV restores the full-KV field name. Top-5 logits
not stored in this artifact.

### 8.2 Expanded forensic example: capital / narrative (Case D)

**Source:** `reports/scale_7b/raw.json`, cell `p00_p0_capital_france`, Llama-3.1-8B,
`int4_sim`, `max_new_tokens=16`.

**Prompt (truncated):** stress-panel capital template (`p0_capital_france`)

**Full-KV output:** ` a city of many faces. It is a city of history, culture, and`

**Lossy output:** ` a city of art, culture, and history. It is also a city of`

**ExactKV output:** ` a city of many faces. It is a city of history, culture, and`

**First divergent token index (lossy path):** 8

**Comment:** Late but semantically meaningful drift (`many` vs `art` in an early
phrase). Acceptance remains high (0.94) because the verifier corrects after a long
accepted prefix. Illustrates why divergence rate and acceptance rate must be read
together. No `code_fn` cells in the headline panel show `int4_sim` lossy divergence,
so this narrative case stands in for long-form generative drift.

### 8.3 Expanded forensic example: long-context late drift (Case H)

**Source:** `reports/evidence_plus/raw.json`, cell `lc_002_ctx512`, Mistral-7B,
`int4_sim`, 512-token prefill, `max_new_tokens=16`.

**Prompt:** long-context synthetic document (`lc_002`) padded to 512 prefill tokens.

**Full-KV output:** `. The compressor drafts on a lossy cache; verification always uses full`

**Lossy output:** `. The compressor drafts on a lossy cache; verification uses segment_`

**ExactKV output:** `. The compressor drafts on a lossy cache; verification always uses full`

**First divergent token index (lossy path):** 13

**Comment:** Drift emerges late in the generation after a long matching prefix under
padded prefill. Acceptance 0.83 with one correction round. The lossy path truncates
or alters the continuation (`always uses full` vs `uses segment_`). Verifier-mediated
output remains exact on this cell (`exactkv_failure=false`).

### 8.4 External smoke panel case studies (Cases J–L)

Cases from `reports/external_panels/case_studies_extracted.json`. See also **§6.4.1**
for the summary table.

**Case J (BFCL / tool-risk):** `bfcl_ast_001_ctx2048`, `int4_sim`, `max_new_tokens=32`.
Lossy path stops at `, years 3.` before the JSON tool call that full-KV continues.
First divergence index 4, acceptance 1.00, `exactkv_failure=false`.

**Case K (LongBench / semantic):** `lb_passage_retrieval_001_ctx4096`, `int4_sim`,
`max_new_tokens=32`. Lossy path answers `segment_0` vs full-KV `segment_36`.
First divergence index 2, acceptance 0.94.

**Case L (RULER / semantic):** `ruler_niah_single_4k_ctx8192`, `int4_sim`,
`max_new_tokens=32`. Needle answer drift: `segment_0` vs `segment_112` at 8192 prefill.
First divergence index 10, acceptance 0.94.

No HumanEval code-drift forensic is included: the artifact contains zero divergent
HumanEval cells at these settings (benign baseline only).

---

## 9. Kernel microbenchmark (qualified)

Source: `reports/phaseF_kernel_benchmark.json`. **Kernel microbenchmark only, NOT
end-to-end inference speedups.** `kv_shape=[1,8,512,64]`, CUDA.

**Table 5, torch vs Triton kernel latency**

| Mode | torch (ms) | Triton (ms) | Ratio (torch÷Triton) |
|:-----|-----------:|------------:|---------------------:|
| `int8` | 0.115 | 0.0706 | 1.63× |
| `int4` | 0.3319 | 0.2162 | 1.54× |
| `block_sparse` | 0.7622 | 0.7797 | 0.98×* |

\* `block_sparse` uses the **torch** execution backend only (not Triton-accelerated).

**Verifier diagnostic timing** (evidence-plus panel, `reports/evidence_plus/raw.json`):

| Statistic | Value |
|:----------|------:|
| Mean per cell | 3.85 s |
| p90 per cell | 5.18 s |

Per-cell wall-clock includes full + lossy + ExactKV paths. **Not** end-to-end serving latency.

---

## 10. VeriCache: closest prior art (explicit boundary)

VeriCache [vericache2026] is **closer to ExactKV than a casual skim suggests**.
Both systems:

1. Use **compressed KV to draft** and **full KV to verify/correct**.
2. Target **lossless greedy equivalence** to full-KV decoding when verification
   succeeds (VeriCache as a serving guarantee, ExactKV as an evaluation gate).
3. Motivate from **token-level divergence** under lossy KV, not just average
   benchmark scores.

**ExactKV must not claim** to invent compressed-KV draft + full-KV verify.
VeriCache owns that algorithmic pattern as a **lossless inference framework**.

| Dimension | VeriCache [vericache2026] | ExactKV (this report) |
|-----------|---------------------------|------------------------|
| **Primary goal** | Serve faster without changing outputs | Measure where/how compressors drift |
| **System design** | GPU compressed KV + offloaded full KV, verification scheduling, cross-resource staggering | Research evaluation loop, no production serving stack |
| **Key metrics** | Throughput, memory tiering, acceptance horizon for amortized verify cost | `first_divergence_index`, lossy divergence rate, acceptance, `exactkv_failure` |
| **Outputs** | Identical tokens to full-KV at serving speed | Public leaderboard + crash-test artifacts |
| **Reproduction** | Requires VeriCache runtime/integration | **Does not reproduce** VeriCache |

Local reference copy: `paper/VeriCache.pdf`. External primary source:
[arXiv:2605.17613](https://arxiv.org/abs/2605.17613).

---

## 11. Why ExactKV still matters after VeriCache

VeriCache answers: **“How do we serve with compressed KV without changing
outputs?”** ExactKV answers: **“Where does compressed KV start lying, and how
often, before correction?”** Those are complementary, not competing, questions.

**VeriCache uses verification to serve faster without changing outputs. ExactKV
uses verification to measure exactly where and how compressors drift.**

Concrete value ExactKV adds even after VeriCache:

1. **Compressor-agnostic crash-test**, rank and compare `noop`, `int8`,
   `int4_sim`, and (labeled) proxy slots on the **same** prompt panel and metrics,
   independent of any one serving implementation.
2. **Lossy-path first divergence**, reports drift on the **unverified** compressed
   path (`generate_lossy_greedy`), not only whether a verifier can recover. A
   compressor may be “safe” under VeriCache-style correction yet **drift early and
   often**, expensive for any system that must verify frequently.
3. **Public diagnostic leaderboard**, frozen cell definitions, claim boundaries,
   and reproducible artifacts (`reports/scale_7b/raw.json`) for cross-compressor
   comparison, not throughput optimization.
4. **Explicit failure taxonomy**, separates lossy divergence, acceptance,
   verifier agreement, and `exactkv_failure` (commit-level exactness break).

ExactKV does **not** replace VeriCache for deployment. It helps **evaluate**
whether a compressor is a good fit for verifier-mediated serving *before* investing
in system integration, and documents drift even when `exactkv_failures = 0`.

---

## 12. Related work

ExactKV is a measurement framework, not a compression method or serving system.
It therefore relates to prior work in three distinct ways: (1) systems whose
algorithmic pattern it shares, (2) compressors whose drift it measures, and
(3) evaluation methodologies it contrasts against.

### 12.1 Verification-based serving

**VeriCache** [vericache2026] is the closest algorithmic prior art. Both ExactKV and
VeriCache use a compressed-KV draft path and a full-KV verify/correct loop to
produce greedy-equivalent output. The critical distinction is purpose: VeriCache is a
**throughput-oriented serving system** — its contribution is scheduling, memory
tiering, and cross-resource staggering for production inference. ExactKV uses the same
algorithmic skeleton as a **diagnostic measurement tool** — it reports *where* the
unverified compressed path first diverges and *how often*, not how fast verified
serving runs. ExactKV does not reproduce VeriCache's memory scheduling, does not
measure serving throughput, and makes no claim to the draft+verify algorithm itself.
The two are complementary: ExactKV measures whether a compressor is worth verifying
before committing to a serving integration.

**Speculative decoding** [leviathan2023speculative] established the general pattern
of using a small draft model and a larger verifier for throughput acceleration.
ExactKV borrows the draft/verify semantics purely for measurement: the "draft" is
the compressed-KV path, and the "verify" is the full-KV oracle. There is no
throughput claim and no model-size draft/target asymmetry. **MagicDec** [chen2024magicdec]
extends speculative decoding to long contexts; ExactKV measures drift at long context
rather than accelerating it.

### 12.2 KV quantization methods

**KVQuant** [hooper2024kvquant] proposes non-uniform per-channel INT4 quantization
with a nuq4 kernel designed specifically for KV cache outlier heads. **KIVI** [liu2024kivi]
proposes asymmetric 2-bit per-channel quantization of KV caches with a CUDA kernel.
Both methods directly motivate ExactKV's `int4_per_vec_sim` compressor, which simulates
the per-vector granularity insight without a production kernel. ExactKV evaluates these
designs as compressors-under-test: the `kivi_offline` adapter uses real KIVI quantizer
math and revealed 100% divergence in the current offline integration
(§6.4.6, KIVI offline panel) — a diagnostic result, not a claim about KIVI's
algorithmic accuracy. KVQuant and SnapKV production kernel integrations remain future work.

**SnapKV** [li2024snapkv] selects important KV entries by clustering observation
windows; it is an eviction/selection method rather than quantization. The ExactKV
`h2o_sim` compressor family (§6.8, H2O-style eviction panel) models the Heavy Hitter Oracle
eviction policy [zhang2023h2o], which is the same compressor class as SnapKV. ExactKV's
eviction results (100% LongBench divergence even at 75% retention) quantify the
worst-case cost of this class on reading tasks.

### 12.3 KV storage and streaming

**CacheGen** [liu2024cachegen] compresses KV caches for efficient network streaming
and disaggregated serving; its goal is bandwidth reduction, not token-level exactness.
**LMCache** [lmcache2025] focuses on KV reuse and offloading across serving instances.
Neither is designed to answer when or why the compressed path first diverges from the
full-KV oracle — the question ExactKV addresses.

### 12.4 Evaluation methodology

Standard LLM benchmarks (MMLU, LongBench, HumanEval, BFCL) measure task-level
accuracy averaged over many tokens. ExactKV's **first-divergence index (fdi)** is a
finer-grained, token-resolution metric: it pinpoints the first token where compressed
and full-KV paths disagree, and the acceptance rate measures what fraction of draft
tokens survive verification. This is orthogonal to task-level accuracy: a cell can
score perfectly on a task metric while accumulating 90% token-level drift (if the
verifier catches and corrects it), or it can show 0% drift and also pass (noop, int8
on short tasks). ExactKV uses official HF LongBench and BFCL datasets but reports
drift measurements, **not** official benchmark scores.

| System | Category | ExactKV relationship |
|--------|----------|---------------------|
| VeriCache [vericache2026] | Lossless serving via compressed draft + full-KV verify | Algorithmic overlap; serving system vs. measurement framework — see §10–11 |
| KVQuant [hooper2024kvquant] | Per-channel INT4 KV quant | Motivates `int4_per_vec_sim`; adapter available (`kivi_offline` diagnostic) |
| KIVI [liu2024kivi] | Asymmetric 2-bit per-channel KV quant | Same; `kivi_offline` shows 100% divergence in offline integration (§6.4.6, KIVI panel) |
| SnapKV [li2024snapkv] | KV eviction/selection | Modeled by `h2o_sim` class; eviction results in §6.8 (H2O panel, v2.8) |
| H2O [zhang2023h2o] | Heavy Hitter Oracle eviction | Direct inspiration for `h2o_sim`; 100% LongBench divergence (§6.8, v2.8) |
| CacheGen [liu2024cachegen] | KV compression + streaming | Different task (network bandwidth) |
| LMCache [lmcache2025] | KV storage/offload | Different task (reuse/serving) |
| Speculative decoding [leviathan2023speculative] | Draft/verify for speedup | ExactKV uses verify semantics for measurement only, no throughput claim |
| MagicDec [chen2024magicdec] | Long-context speculative decoding | Adjacent; ExactKV measures long-context drift rather than accelerating it |

Full audit: [`release_synthesis/related_work_audit.md`](../release_synthesis/related_work_audit.md).

---

## 13. Novelty and claim boundaries

**Defensible novelty (narrow):** ExactKV is a **compressor-agnostic crash-test
framework with leaderboard-style reporting** for measuring **token-level drift and
first-divergence behavior** under verifier-mediated semantics, not a throughput-optimized
inference system. Specifically: leaderboard-style crash-test pipeline, Phase G
divergence authority, real 7B/8B panel with documented adapter honesty, explicit
separation of lossy drift vs `exactkv_failure`, and crash-test coverage extending
to real HF benchmarks and external adapter integrations (including `kivi_offline`
as an offline adapter diagnostic).

**Explicitly not novel:**

- Compressed-KV **draft** + full-KV **verify/correct** for lossless greedy output
  (VeriCache [vericache2026]).
- Draft-then-verify loops for acceleration (speculative decoding
  [leviathan2023speculative]).
- The observation that lossy KV diverges on longer decode (VeriCache problem
  statement, ExactKV cites, does not claim).

**Do not claim without new evidence** (see `claim_decision_table.md`):

- inventing verifier-mediated compressed KV (VeriCache [vericache2026] owns that pattern for serving),
- global uniqueness or sole-novelty status,
- production serving readiness,
- end-to-end inference latency gains,
- runtime GPU memory or VRAM reduction,
- reproducing or outperforming VeriCache serving throughput,
- outperforming TurboQuant or Shard on a same-task benchmark,
- production SpectralQuant or Shard integration in this environment.

---

## 14. Evidence-plus panel (completed) and remaining extensions

The **144-cell evidence-plus supplement** is now populated on GPU
(`reports/evidence_plus/raw.json`). Reproduce:

```bash
python3 scripts/run_evidence_plus_panel.py --device cuda --dtype float16
# RunPod: bash scripts/setup_runpod_evidence_plus.sh
```

### 14.1 Completed (June 2026, RunPod A5000)

- **512 / 1024** prefill buckets, 6 long-context + stress prompts
- **`max_new_tokens` 16, 32** on Llama-3.1-8B and Mistral-7B
- Built-in `noop` / `int8` / `int4_sim`, **`exactkv_failures = 0`**
- Per-cell diagnostic `timing_ms` (mean 3.85 s/cell)

See **§6.3** for aggregates.

### 14.2 External benchmark smoke panels (completed, June 2026)

**1,560 GPU cells** on RunPod A5000: **216** Llama-only cells on bundled
LongBench/RULER/BFCL/HumanEval pilots (`summary_all.json`, `exactkv_failures = 0`)
plus **144** MBPP cells with both models (`mbpp_gpu_raw.json`) plus **1,200** BFCL
export-50 tool-call drift cells with both models (`bfcl_export_50_raw.json`,
`exactkv_failures = 0`, `int4_sim` divergence 11.3%). See **§6.4** for tables and case studies.

Reproduce:

```bash
bash scripts/run_external_gpu_workflow.sh
python3 scripts/build_external_analysis_pack.py
python3 scripts/run_external_panel.py --family mbpp --device cuda --dtype float16 \
  --max-prompts 6 --context-buckets 512,1024 --max-new-tokens 16,32 \
  --output-json reports/external_panels/mbpp_gpu_raw.json
python3 scripts/validate_external_panel_artifacts.py --input reports/external_panels
```

Not completed in first workflow: LongBench HF export, RULER 12K+, official
benchmark scores. (Mistral external panels and MBPP GPU smoke completed
in later June 2026 runs; see validated artifacts under `reports/external_panels/`.)

### 14.2.1 KIVI offline compressor panel — completed (June 2026)

The KIVI offline panel ran on RunPod RTX A5000, June 27 2026. Results are in
§6.4.6 and `reports/external_panels/kivi_longbench_hf_raw.json` (320 cells) +
`reports/external_panels/kivi_mbpp_hf_raw.json` (320 cells).

**Headline result:** `kivi_offline` (real KIVI quantizer math, simulate path) shows
100% token-level divergence with acceptance ≈ 0 — indicative of catastrophic KV
corruption in the current offline adapter integration. ExactKV detected and corrected
all 160 `kivi_offline` cells (`exactkv_failures=0`). Real HF `int4_sim` drift
(LongBench: 91.2%) substantially exceeds bundled pilot results (20.8%), confirming
that pilot prompts underestimate production drift.

Reproduce:

```bash
export PYTHONPATH=/tmp/kivi_research  # jy-yuan/KIVI clone
bash scripts/run_kivi_external_panel.sh
```

Claim boundary: `kivi_offline` uses real KIVI quantizer math (simulate path,
`supports_real_bytes_claim=False`). Not KIVI production CUDA/Triton serving.

### 14.3 Future work

| Extension | Status |
|-----------|--------|
| **Real HF LongBench subset + KIVI offline panel** | **Completed** (§6.4.6); 640 cells, `exactkv_failures=0`; adapter-level diagnostic |
| **Expand BFCL beyond 4 prompts** + structured-output validity parsing | **Completed** (1,200-cell export-50 panel, both models, §6.4.5) |
| **BFCL tool-call validity panel** (mnt=128/256, full JSON generation) | **Completed v2.7** (1,200 cells, both models, §6.7); int4_sim 50.3% divergence; `exactkv_failures=0` |
| **Expand MBPP beyond 6-prompt pilot** + safe pass@1 / test execution | Pilot GPU smoke done (`mbpp_gpu_raw.json`) |
| **ExactKV-HF-LongBench Drift Panel v2.6** | **Complete** (720 cells, both models). Key finding: int4_sim = 90.4% divergence; int8 = 24.6%. `exactkv_failures=0`. |
| **H2O-style token-eviction compressor** (v2.8) | **Complete** (800 cells, 5 eviction variants, both models, §6.8); 100% divergence at all budgets; `exactkv_failures=0` |
| **Top-k logit autopsy at divergence** | **Complete** (1,103 divergent cells, §6.10); three failure modes identified |
| **Rerun Mistral on main external panels** (LongBench/RULER/BFCL/HumanEval) | Failed in first workflow (disk quota); succeeded later for MBPP only |
| **HELMET holistic long-context panel** | Not implemented |
| **InfiniteBench 100K+ stress** | Not run, pending verifier memory/runtime stability |
| **KVQuant/SnapKV integrations** (production kernels) | Adapters exist; KIVI offline done at simulate tier; faithful production integration future work |
| **RULER 16K/32K scaling** | 2K/4K/8K pilot done |
| **HumanEval/MBPP pass@1 impact** | Requires safe sandboxing; extend beyond BFCL downstream validity to MBPP pass@1/syntax and LongBench answer-overlap |
| **Confidence intervals** | Wilson CIs added for all external panels |
| **BFCL validity: larger/more diverse prompts** | Extend beyond 50 export-50 prompts to wider function diversity |
| **Broader scaling curves: more tasks, 12K+, 16K+, and additional compressors** | 1K–16K context vs divergence rate; 16–256 token generation vs divergence |

---

## 15. Limitations

1. **Greedy decoding only**, sampling-based exactness requires distribution-level
   verification and fixed RNG policy, not implemented in this release.
2. **Panel scope**, headline: 50 prompt variants × short generations (4–16 tokens),
   evidence-plus: 512/1024 prefill × 16–32 token generations (144 cells). External
   smoke: **1,560 GPU cells** total (**216** Llama-only LongBench/RULER/BFCL/HumanEval
   pilots + **144** MBPP cells with both models + **1,200** BFCL export-50 cells with
   both models). These are **smoke panels**, not full benchmark suites.
3. **External panels are drift smoke panels**, not official LongBench, RULER,
   HELMET, BFCL, or HumanEval scores. The LongBench HF panel (§6.4.6) uses
   real THUDM/LongBench HF examples; other pilot panels use synthetic benchmark-shaped
   prompts.
4. **LongBench HF and MBPP HF exports** (real HF subset) are now completed
   (§6.4.6) with `kivi_offline` alongside built-in compressors.
5. **Mistral on main external panels** (LongBench/RULER/BFCL/HumanEval) failed in
   the first RunPod workflow due to disk quota after Llama cache. A later **MBPP**
   panel successfully ran both Llama-3.1-8B and Mistral-7B (144 cells).
6. **BFCL expanded** from 4-prompt bundled pilot to a **1,200-cell export-50 panel** (both models,
   `exactkv_failures=0`, `int4_sim` divergence 5.5%/17.0% Llama/Mistral). HumanEval remains
   bundled 4-prompt pilot only. JSON-completeness still requires longer `max_new_tokens` budgets.
7. **LongBench/RULER/HELMET/InfiniteBench full-scale evaluation** remains future work.
   RULER 12K (downscaled from 16K due to OOM on A5000) is queued; InfiniteBench 100K+ was not run.
8. **MBPP panel** uses a **bundled 6-prompt pilot** only (`mbpp_pilot.jsonl`). GPU
   smoke: 144 cells, both models, **`exactkv_failures = 0`** (`mbpp_gpu_raw.json`).
   **No test execution** against MBPP `test_list`, not pass@1.
9. **KIVI offline adapter integration diagnostic.** The `kivi_offline` adapter
   (`exactkv/compressors/kivi_adapter.py`, `is_simulated=False`) uses the upstream
   KIVI `models.utils_quant` simulate path and ran on 640 cells (§6.4.6). It shows
   **100% divergence with acceptance ≈ 0**, indicating catastrophic KV corruption in
   the current offline integration. **This is not a claim about the KIVI algorithm's
   accuracy.** ExactKV detected and corrected all corrupt cells (`exactkv_failures=0`).
   KVQuant and SnapKV production integrations remain future work.
10. **Verifier overhead**, evidence-plus and external panels report diagnostic
    wall-clock per cell (§6.3, §6.4, §9), not production serving throughput or VRAM
    telemetry.
11. **Confidence intervals** are Wilson 95% two-sided intervals on divergence rate
    per panel (added in §6.4). Acceptance rate CIs not yet reported. Intervals on
    small panels (RULER 8192, n=24) are wide; treat with appropriate caution.
12. **Logits at divergence** not stored in release or external panel artifacts.
13. **Proxy/probe slots** (`spectralquant`, `shard`) must not be read as real
    external compressor integrations.
14. **`exactkv_failures = 0`** is a hard gate on the **tested panels** under
    verifier correction, not a universal safety certificate.
15. **Sequential model execution** on the scale run.

---

## 16. Project lineage (brief)

Two timelines: **V1–V21** research arc (verifier-first prototypes, safety
ladder, no-go probes) and **Phase A–K** formal release arc (scale, leaderboard,
novelty audit, launch pack). Details:
[`release_synthesis/project_lineage.md`](../release_synthesis/project_lineage.md).

---

## 17. Reproducibility

**Artifact verification (no GPU):**
```bash
python3 scripts/exactkv_repro.py --reports-only
python3 scripts/exactkv_repro.py --release-check
bash scripts/build_paper_pdf.sh                     # requires: brew install tectonic
```

**Headline panel (GPU):**
```bash
python3 scripts/run_phase_a_scale_benchmark.py --device cuda
```

**Evidence-plus panel (GPU):**
```bash
python3 scripts/run_evidence_plus_panel.py --device cuda --dtype float16
```

**External smoke panels — Llama-only LongBench/RULER/BFCL/HumanEval (GPU, 216 cells):**
```bash
bash scripts/run_external_gpu_workflow.sh
python3 scripts/build_external_analysis_pack.py
python3 scripts/build_external_panel_summary.py --write-readme
python3 scripts/validate_external_panel_artifacts.py --input reports/external_panels
```

**MBPP code-drift smoke panel (GPU, 144 cells, both models):**
```bash
python3 scripts/run_external_panel.py \
  --family mbpp --device cuda --dtype float16 \
  --max-prompts 6 --context-buckets 512,1024 --max-new-tokens 16,32 \
  --output-json reports/external_panels/mbpp_gpu_raw.json
python3 scripts/validate_external_panel_artifacts.py --input reports/external_panels
```

**BFCL export-50 tool-call drift panel (GPU, 1,200 cells, both models):**
```bash
python3 scripts/run_external_panel.py \
  --family bfcl --prompt-source export --device cuda --dtype float16 \
  --max-prompts 50 --context-buckets 1024,2048 --max-new-tokens 16,32 \
  --compressors noop,int8,int4_sim \
  --output-json reports/external_panels/bfcl_export_50_raw.json
# Note: max_new_tokens={16,32} measures drift, NOT JSON-completeness.
# Artifact: reports/external_panels/bfcl_export_50_raw.json (13.5 MB)
```

**KIVI offline compressor panel (GPU, 640 cells — requires KIVI `PYTHONPATH`):**
```bash
# Prerequisites:
#   git clone https://github.com/jy-yuan/KIVI /tmp/kivi_research
#   pip install datasets>=2.0
export PYTHONPATH=/tmp/kivi_research
bash scripts/run_kivi_external_panel.sh
# Artifacts: reports/external_panels/kivi_longbench_hf_raw.json
#            reports/external_panels/kivi_mbpp_hf_raw.json
# Note: kivi_offline adapter uses KIVI simulate path (is_simulated=False,
#       supports_real_bytes_claim=False — no CUDA/Triton kernels).
#       Results diagnose the offline adapter path, NOT production KIVI serving.
```

See [`docs/RUNPOD_EVIDENCE_PLUS.md`](../docs/RUNPOD_EVIDENCE_PLUS.md) and
[`docs/EXTERNAL_BENCHMARK_PANELS.md`](../docs/EXTERNAL_BENCHMARK_PANELS.md).

**HF LongBench v2.6 drift panel (GPU, 720 cells, both models):**
```bash
python3 scripts/run_external_panel.py \
  --family longbench --prompt-source hf \
  --longbench-subsets narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,\
gov_report,trec,samsum,lcc,passage_retrieval_en \
  --device cuda --dtype float16 --max-prompts 20 \
  --context-buckets 2048,4096,8192 --max-new-tokens 32,64 \
  --compressors noop,int8,int4_sim \
  --models meta-llama/Llama-3.1-8B \
  --store-top-k-logits \
  --output-json reports/external_panels/hf_longbench_v26_Llama_3_1_8B_raw.json
# Repeat for mistralai/Mistral-7B-Instruct-v0.3
# Merge: python3 scripts/postprocess_merge_v27_bfcl.py (adapt for longbench)
# Or use: scripts/run_hf_longbench_v26_panel.sh (full runbook)
# Artifacts: hf_longbench_v26_{Llama,Mistral}_raw.json, hf_longbench_v26_merged_raw.json
```

**BFCL tool-call validity panel (GPU, 1,200 cells, mnt=128/256):**
```bash
python3 scripts/run_external_panel.py \
  --family bfcl --prompt-source export --device cuda --dtype float16 \
  --max-prompts 50 --context-buckets 1024,2048 --max-new-tokens 128,256 \
  --compressors noop,int8,int4_sim \
  --store-top-k-logits \
  --output-json reports/external_panels/bfcl_validity_v27_{model}_raw.json
# Post-process: python3 scripts/postprocess_merge_v27_bfcl.py
# Artifacts: bfcl_validity_v27_{Llama,Mistral}_raw.json, bfcl_validity_v27_merged_raw.json
```

**Cross-panel analysis:**
```bash
python3 scripts/analyze_panel_divergence.py --all-compressors --markdown
```

**H2O-style token-eviction panel (GPU, 800 cells, v2.8, both models):**
```bash
python3 scripts/run_external_panel.py \
  --family longbench --prompt-source hf \
  --longbench-subsets narrativeqa,hotpotqa,samsum,trec,lcc,gov_report,qasper,2wikimqa,\
passage_retrieval_en,multifieldqa_en \
  --device cuda --dtype float16 --max-prompts 20 \
  --context-buckets 2048,4096 --max-new-tokens 32,64 \
  --compressors noop,int4_sim,h2o_sim,h2o_sim_75,h2o_sim_25 \
  --store-top-k-logits \
  --output-json reports/external_panels/h2o_v28_{model}_raw.json
# Artifacts: h2o_v28_{Llama,Mistral}_raw.json, h2o_v28_merged_raw.json
# Note: h2o_sim/h2o_sim_75/h2o_sim_25 = H2O-style eviction simulation (not faithful H2O GPU kernel)
```

**Top-k logit autopsy (post-hoc, requires panels with --store-top-k-logits):**
```bash
python3 scripts/analyze_logit_margins.py \
  --inputs reports/external_panels/hf_longbench_v26_merged_raw.json \
           reports/external_panels/bfcl_validity_v27_merged_raw.json \
           reports/external_panels/h2o_v28_merged_raw.json \
  --output reports/external_panels/logit_autopsy_summary.json
# Analyzes 1,103 divergent cells; produces Table 4e/19 metrics per compressor
```

Source of truth (all artifacts):

| Artifact | Description | Cells |
|---|---|---|
| `reports/scale_7b/raw.json` | Headline panel | 1,500 |
| `reports/evidence_plus/raw.json` | Evidence-plus | 144 |
| `reports/external_panels/summary_all.json` | Initial 216-cell Llama-only smoke | 216 |
| `reports/external_panels/mbpp_gpu_raw.json` | MBPP code-drift smoke (both models) | 144 |
| `reports/external_panels/bfcl_export_50_raw.json` | BFCL export-50 tool-call drift | 1,200 |
| `reports/external_panels/kivi_*_hf_raw.json` | KIVI offline adapter diagnostic | 640 |
| `reports/external_panels/hf_longbench_v26_merged_raw.json` | HF LongBench v2.6 (both models) | 720 |
| `reports/external_panels/bfcl_validity_v27_merged_raw.json` | BFCL validity v2.7 (both models) | 1,200 |
| `reports/external_panels/h2o_v28_merged_raw.json` | H2O-style eviction v2.8 (both models) | 800 |
| `reports/external_panels/v30/` | v3.0 int6_sim + int4_per_vec_sim (both models) | 1,568 |
| **Total** | | **8,132** |

---

## 18. Conclusion

VeriCache [vericache2026] shows how to **serve** with compressed KV without
changing greedy outputs. ExactKV shows **where compressors drift** on the lossy
path and how verifier-mediated execution behaves on a fixed panel.

ExactKV's strongest supported claim is not "we beat X" or "we invented verify." It is:

**KV-cache drift is governed jointly by task type, generation length, compressor class,
and quantization granularity** — and ExactKV maps that design space while verifier-mediated
decoding preserves full-KV greedy equivalence (`exactkv_failures=0` throughout).

**ExactKV tells you exactly when compressed KV cache behavior stops matching the
verifier**, and separately reports when the *lossy* path drifts, how often drafts
are accepted, and whether verifier-mediated execution still ends exact.

On the 1,500-cell release panel, built-in `int8`/`noop` show no lossy divergence,
`int4_sim` drifts in 52% of cells while the verifier maintains
`exactkv_failures = 0`. Across **8,132 total cells**, the picture has become clearer:
int4_sim divergence is **task-dependent** — 6% on Python code (MBPP), 11% on
tool-calling (BFCL short-gen), 50% on tool-calling (BFCL long-gen at mnt=128/256),
and **90% on open-text reading/summarization (HF LongBench, 2K–8K context)**.
Even int8 reaches 25% divergence on LongBench vs 0% on BFCL/MBPP, confirming that
**task type — not just quantization level or context length — is the dominant driver
of KV compression drift**. The v2.8 H2O-style token-eviction panel adds a new dimension:
**eviction-class compressors produce near-universal divergence (100%) on reading tasks
even at mild keep_ratio=0.75**, far exceeding int4_sim at matched memory budgets. Mean
acceptance rate for H2O is ~0.35 (diverges at token 1) vs. ~0.84 for int4_sim.
The verifier maintains `exactkv_failures = 0` across all 8,132 cells — including all
100% H2O divergence cases. **ExactKV catches every compressor failure type.**

Top-k logit analysis (§6.10) over 1,103 divergent cells reveals three mechanistically
distinct failure modes: **(1) near-tie noise** — int8 flips close argmax decisions
(66% near-tie, mean lossy rank 2.4); **(2) distribution shift** — int4_sim biases
activations toward lower-ranked alternatives (83% flip, mean rank 3.5); and
**(3) attention destruction** — H2O eviction eliminates contextual anchors from the
first generated token (100% flip, mean rank 6.7, fdi=1). The same verifier
corrects all three failure modes with `exactkv_failures=0`.

The distinction, **drift vs exactness failure**, is the paper's core methodological
contribution. ExactKV does not propose a new compressor; it provides the measurement
infrastructure to understand when any compressor starts lying — and the mechanistic
evidence to explain *why*.

Two new compressors extend the compression curve: `int6_sim` (6-bit per-tensor)
and `int4_per_vec_sim` (4-bit per-vector, KIVI/KVQuant-style). The v3.0 GPU panel
(both models, 1,568 cells, `exactkv_failures=0`) validates both as non-catastrophic:
`int6_sim` achieves 0% divergence on BFCL/MBPP and 37–47% on LongBench (Mistral/Llama);
`int4_per_vec_sim` achieves 0% on BFCL/MBPP and 56–57% on LongBench — establishing both
as the first non-catastrophic int4/int6 compressors in the ExactKV framework. The
per-vector granularity result is task-conditional: it eliminates drift on structured
tasks, but 4-bit resolution still accumulates error at 8K context on both models.

---

## Appendix A: All completed panels — consolidated benchmark card

| Panel | Source | Official score? | Models | Compressors | Cells | Context (K) | `exactkv_failures` |
|-------|--------|----------------|--------|-------------|------:|-------------|-------------------:|
| Headline release panel | `reports/scale_7b/raw.json` | No | Llama-3.1-8B, Mistral-7B | noop, int8, int4_sim (+ 2 proxy/probe slots¹) | 1,500 | 0.5–2 | 0 |
| Evidence-plus panel | `reports/evidence_plus/raw.json` | No | Llama-3.1-8B, Mistral-7B | noop, int8, int4_sim | 144 | 0.5, 1 | 0 |
| External smoke: LongBench pilot | `summary_all.json` | No | Llama-3.1-8B only | noop, int8, int4_sim | 72 | 2, 4 | 0 |
| External smoke: RULER 2K/4K | `summary_all.json` | No | Llama-3.1-8B only | noop, int8, int4_sim | 48 | 2, 4 | 0 |
| External smoke: RULER 8K | `summary_all.json` | No | Llama-3.1-8B only | noop, int8, int4_sim | 24 | 8 | 0 |
| External smoke: BFCL pilot | `summary_all.json` | No | Llama-3.1-8B only | noop, int8, int4_sim | 48 | 1, 2 | 0 |
| External smoke: HumanEval pilot | `summary_all.json` | No | Llama-3.1-8B only | noop, int8, int4_sim | 24 | 1, 2 | 0 |
| MBPP code-drift smoke | `mbpp_gpu_raw.json` | No | Llama-3.1-8B, Mistral-7B | noop, int8, int4_sim | 144 | 0.5, 1 | 0 |
| **BFCL export-50 tool-call drift** | `bfcl_export_50_raw.json` | No | Llama-3.1-8B, Mistral-7B | noop, int8, int4_sim | **1,200** | 1, 2 | 0 |
| **KIVI offline adapter panel** | `kivi_*_hf_raw.json` | No | Llama-3.1-8B only | noop, int8, int4_sim, kivi_offline | **640** | 2, 4 | 0 |
| **HF LongBench v2.6 (real HF)** | `hf_longbench_v26_merged_raw.json` | No | Llama-3.1-8B, Mistral-7B | noop, int8, int4_sim | **720** | 2, 4, 8 | 0 |
| **BFCL validity v2.7 (both models)** | `bfcl_validity_v27_merged_raw.json` | No | Llama-3.1-8B, Mistral-7B | noop, int8, int4_sim | **1,200** | 1, 2 | 0 |
| **H2O-style eviction v2.8** | `h2o_v28_merged_raw.json` | No | Llama-3.1-8B, Mistral-7B | noop, int4_sim, h2o_sim, h2o_sim_75, h2o_sim_25 | **800** | 2, 4 | 0 |
| **v3.0 int6_sim + int4_per_vec_sim** | `v30/longbench_Mistral_*`, `v30/bfcl_Mistral_*`, `v30/mbpp_Mistral_*` | No | Mistral-7B only | int8, int6_sim, int4_per_vec_sim, int4_sim | **784** | 2, 4, 8 + BFCL/MBPP | 0 |
| **v3.0 int6_sim + int4_per_vec_sim** | `v30/longbench_Llama_*`, `v30/bfcl_Llama_*`, `v30/mbpp_Llama_*` | No | Llama-3.1-8B only | int8, int6_sim, int4_per_vec_sim, int4_sim | **784** | 2, 4, 8 + BFCL/MBPP | 0 |
| | | | | **Subtotal (pre-v3.0)** | **6,564** | | |
| | | | | **Subtotal (v3.0 panels)** | **1,568** | | |
| | | | | **Grand total** | **8,132** | | **0** |

**Total: 8,132 completed GPU cells, `exactkv_failures = 0` throughout.**

Reconciliation: **6,564 + 784 + 784 = 8,132**. Each v3.0 row is one model × four
compressors (`int8`, `int6_sim`, `int4_per_vec_sim`, `int4_sim`) × three task families
(LongBench 288 + BFCL 400 + MBPP 96 = 784 cells per model). Both compressors share the
same panel grid; they are not counted as separate 784-cell panels.

**All totals count only fully completed panels.** Queued or planned panels (InfiniteBench 100K+, HELMET, KVQuant/SnapKV adapters) are excluded from all cell counts and claims.

¹ The raw JSON also contains `spectralquant` (MOCK→`int4_sim`) and `shard` (PROBE_ONLY) slots; these are excluded from all analysis. `kivi_offline` is an offline adapter diagnostic (`supports_real_bytes_claim=False`). None are official benchmark scores.

## Appendix B: Artifact inventory

1,524 tracked artifacts, curated release-grade set in
[`release_synthesis/artifact_inventory.md`](../release_synthesis/artifact_inventory.md).

## Appendix C: Claim decision table

[`release_synthesis/claim_decision_table.md`](../release_synthesis/claim_decision_table.md)

---

*References: [`references.bib`](references.bib).*
