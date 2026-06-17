# 07_VERSION_ROADMAP.md

# ExactKV Version Roadmap

## Purpose of this document

This document defines the staged plan for ExactKV.

Every version specifies:

- Goal
- Why the version exists
- New features
- Explicit non-goals
- Success criteria
- Exit tests
- Expected deliverables

This roadmap exists to prevent scope creep.

## Version philosophy

ExactKV should be built in controlled stages:

```text
V0: understand
V1: prove correctness
V2: generalize framework
V3: benchmark seriously
V4: asymmetric K/V compression experiments (simulated)
V5: workspace-aware memory accounting
V6: real backend adapter interface + first backend candidate
V7: attention-aware and V-specific backend ideas (Sparse V, layer-aware V)
V8: serving-stack integration
```

Do not skip versions.

The project should not jump to CUDA, vLLM, LMCache, or TurboQuant before the core exactness loop is correct.

> **Note (post-V4 update).** The original plan labelled V4 as "add real
> compressors" and V5 as "optimize runtime." After V4 shipped *simulated*
> asymmetric K/V compressors (Experiment 003), the roadmap was re-grounded in the
> current KV-cache literature — see
> [`RELATED_WORK_KV_CACHE_COMPRESSION.md`](RELATED_WORK_KV_CACHE_COMPRESSION.md).
> Real backends (KIVI/TurboQuant-style) move to **V6** behind an adapter
> interface, and "runtime optimization" is **not** adopted as a goal: ExactKV
> makes no speedup, throughput, or latency claims. V5 is memory-accounting
> honesty, not performance.

---

# V0: Research and specification

## Goal

Understand VeriCache and define the ExactKV project.

## Why this version exists

Before writing code, the project needs a shared foundation:

- What problem are we solving?
- What exists already?
- What is ExactKV's actual contribution?
- What are the risks?
- What is in scope?

## New features

No code features.

Documentation only.

## Deliverables

- `00_VISION.md`
- `01_PROBLEM.md`
- `02_EXISTING_WORK.md`
- `03_VERICACHE_ANALYSIS.md`
- `04_EXACTKV_THESIS.md`
- `05_ARCHITECTURE.md`
- `06_DESIGN_DECISIONS.md`
- `07_VERSION_ROADMAP.md`
- `08_METRICS.md`
- `09_BENCHMARKS.md`
- `10_RISKS.md`
- `11_NON_GOALS.md`
- `12_COMPRESSOR_INTERFACE.md`
- `13_VERIFICATION_ENGINE.md`
- `14_IMPLEMENTATION_PLAN.md`
- `15_FUTURE_RESEARCH.md`

## Non-goals

- No implementation.
- No benchmarks.
- No performance claims.

## Success criteria

A contributor can read the docs and explain:

- What ExactKV is
- What VeriCache contributes
- What ExactKV adds
- Why Phase 1 is scoped narrowly
- What must not be built yet

## Exit test

Ask Cursor:

```text
Read the docs and summarize the project, risks, and Phase 1 plan without writing code.
```

If Cursor gives a coherent summary, V0 is complete.

---

# V1: Correctness prototype

## Goal

Prove the core ExactKV loop works.

## Why this version exists

Before optimizing, ExactKV must prove:

> Compressed KV can draft tokens, full KV can verify them, and the final output can match full-KV decoding exactly.

## New features

- Hugging Face model loading
- Tokenizer wrapper
- Full-KV greedy generation baseline
- Simple INT8 compressor
- Compressed-KV drafting
- Verification engine
- Accept/reject logic
- Output token ID equality test
- Basic trace logging

## Target model

Start with one small model:

- `Qwen/Qwen2.5-0.5B`

Alternative:

- TinyLlama

## Decoding mode

Greedy only.

```text
do_sample = False
temperature = 0
num_beams = 1
```

## Compressor

INT8 only.

The compressor does not need to be fast.

It must be simple and correct enough to produce an approximate cache.

## Non-goals

- No INT4.
- No token dropping.
- No TurboQuant.
- No vLLM.
- No LMCache.
- No CPU offload.
- No async transfer.
- No batching.
- No sampling.
- No CUDA kernels.
- No Triton.
- No production performance claims.

## Success criteria

For a prompt and max token count:

```python
full_output_ids == exactkv_output_ids
```

Additionally:

- Lossy compressed mode runs separately.
- ExactKV produces a trace.
- At least one test shows acceptance and/or rejection behavior.
- Implementation is readable.

## Exit tests

### Test 1: full baseline

Run full-KV generation on a fixed prompt.

### Test 2: lossy baseline

Run compressed-KV generation without verification.

### Test 3: exactkv

Run ExactKV generation.

### Test 4: equality

Assert:

```python
exactkv_output_ids == full_output_ids
```

### Test 5: trace sanity

Assert trace includes:

- drafted tokens
- verified tokens
- accepted count
- rejected tokens
- correction token when applicable

## Deliverables

- Minimal package structure
- `ExactKVGenerator`
- `Int8Compressor`
- `VerificationEngine`
- Unit tests
- One example script
- One README section showing usage

---

# V2: Framework layer

## Goal

Turn the V1 prototype into a modular framework.

## Why this version exists

V1 proves the idea. V2 makes it extensible.

ExactKV should not remain a hard-coded INT8 demo.

## New features

- Formal `KVCompressor` base class
- `FullKVState`
- `CompressedKVState`
- `DraftResult`
- `AcceptanceResult`
- `ExactKVResult`
- Config objects
- Improved error handling
- Basic metric collection
- More unit tests

## Compressor support

- INT8
- INT4 simple implementation

## Non-goals

- No advanced compressor integration.
- No vLLM.
- No LMCache.
- No CPU offload.
- No Triton.
- No multi-GPU.

## Success criteria

A new compressor can be added by implementing the compressor interface without touching the verification engine.

## Exit tests

- INT8 ExactKV output equals full.
- INT4 ExactKV output equals full.
- Lossy INT4 output may diverge.
- Metrics are collected consistently.
- Compressor interface is documented.

## Deliverables

- Modular package
- Compressor abstraction docs
- Type hints
- Test coverage for all result objects
- Example for swapping compressors

---

# V3: Benchmark suite

## Goal

Make ExactKV useful as a compressor evaluation tool.

## Why this version exists

ExactKV's value is not only runtime execution. It is also a way to measure how useful and safe compressors are.

## New features

- Benchmark runner
- Prompt suites
- JSON report generation
- CSV report generation
- Acceptance-rate metrics
- Throughput metrics
- Memory estimates
- Exactness checks
- Basic plotting scripts

## Benchmark modes

Every benchmark must run:

```text
full
lossy
exactkv
```

## Prompt categories

- Short natural language
- Long natural language
- Code generation
- JSON generation
- Tool-call-like structured output
- Synthetic long-context prompts

## Non-goals

- No large-scale leaderboard yet.
- No production benchmarking claims.
- No distributed serving.

## Success criteria

For each prompt and compressor, produce:

- Full output
- Lossy output
- ExactKV output
- ExactKV vs full equality
- Lossy vs full equality
- Acceptance rate
- Average accepted tokens per verification
- First mismatch position
- Time per mode
- Memory estimate

## Exit tests

- Benchmark report is generated from command line.
- Report can be loaded as JSON.
- CSV is suitable for plotting.
- At least one plot is generated.

## Deliverables

- `exactkv bench ...`
- `benchmarks/prompts/*.jsonl`
- `reports/*.json`
- `reports/*.csv`
- `plots/*.png`

---

# V4: Advanced compressor adapters

## Goal

Support real KV compression methods beyond toy quantization.

## Why this version exists

To be credible, ExactKV must eventually test known compressors.

## Candidate integrations

### Priority 1

- kvpress adapter
- KIVI adapter

### Priority 2

- SnapKV-style token dropping
- KVQuant-style quantization
- RotateKV-style quantization

### Priority 3

- TurboQuant-style adapter
- KVzip-style compressor
- KVzap-style compressor

## Non-goals

- Do not reimplement every paper from scratch.
- Do not optimize all adapters equally.
- Do not claim SOTA compression.

## Success criteria

ExactKV can run the same benchmark suite over at least one external compressor backend.

## Exit tests

- External compressor produces compressed state.
- ExactKV can draft from it.
- Verification works.
- Metrics compare external compressor against INT8/INT4.

## Deliverables

- Adapter interface
- At least one external compressor integration
- Updated benchmark report
- Documentation for integration limitations

---

# V5: Workspace-aware memory accounting

> **Re-grounded after V4.** This section replaces the original "Runtime
> performance" V5. ExactKV does **not** adopt runtime optimization or speedup as
> a goal, and makes **no** throughput, latency, or speedup claims. See
> [`RELATED_WORK_KV_CACHE_COMPRESSION.md`](RELATED_WORK_KV_CACHE_COMPRESSION.md)
> and [`V5_SCOPE_STATEMENT.md`](V5_SCOPE_STATEMENT.md).

## Goal

Make ExactKV's memory reporting **honest**, not faster.

## Why this version exists

Stored compressed KV bytes are not the full memory story. Real backends keep
full-precision residuals (KIVI), reconstruct dense working caches (Palu, KVTC),
and carry scales/codebooks as metadata. Decode also needs temporary
dequantization workspace.

## New features

- Workspace-aware `MemorySummary`: `stored_kv_bytes`,
  `materialized_working_kv_bytes`, `metadata_bytes`, `temporary_workspace_bytes`,
  `total_kv_footprint_bytes`.
- Per-compressor stats populate the new fields honestly.
- Reports/CLI/Markdown surface stored vs materialized vs total.

## Non-goals

- No backend implementation.
- No throughput, latency, or speedup measurement or claim.
- No real bit-packing presented as a default.

## Success criteria

- Memory is reported honestly across all current compressors.
- `materialized_working_kv_bytes == full_kv_bytes` is surfaced (it holds for all
  current compressors).
- `_sim` compressors keep `supports_real_bytes_claim=False`.
- Exactness preserved; no forbidden performance fields.

---

# V6: Real backend adapter interface + first backend candidate ✅ complete (`v0.6.0`)

## Goal

Design a `BackendAdapter` so a real quantisation format could plug into the
`KVCompressor` protocol and be evaluated by acceptance behaviour.

## Delivered

- `BackendAdapter` interface (`docs/BACKEND_ADAPTER_INTERFACE.md`) and
  `backend_passthrough` PoC in the default registry.
- Restricted experimental `KVPressKnormAdapter` (KnormPress only; **not** in default
  registry; requires isolated `[kvpress]` environment).
- Phase C validation (`docs/KVPRESS_KNORM_VALIDATION.md`) and Experiment 005
  (`docs/EXPERIMENT_005_KVPRESS_KNORM.md`): 272 cells, `exactkv_failures == 0`.
- Release notes: [`RELEASE_NOTES_V0.6.0.md`](RELEASE_NOTES_V0.6.0.md).

## Non-goals (unchanged)

- No throughput/latency/speedup claims; acceptance and (real) memory only.
- No production-serving claims; kvpress not in default dependencies.

> See [`V6_SCOPE_STATEMENT.md`](V6_SCOPE_STATEMENT.md) for the full phased scope,
> restrictions, and exit criteria (all phases complete).

---

# V7: Attention-aware and V-specific experiments (complete — `v0.7.0`)

## Goal

Evaluate attention-aware and V-specific compression ideas through ExactKV's
verification, acceptance, divergence, and workspace-memory framework.

## Deliverables

- [`RELEASE_NOTES_V0.7.0.md`](RELEASE_NOTES_V0.7.0.md) — full V7 summary, phase
  results, limitations, deferred work.
- [`V7_SCOPE_STATEMENT.md`](V7_SCOPE_STATEMENT.md) — phased scope and exit criteria
  (all phases complete).

## Experiments

| Phase | Report | Runs | `exactkv_failures` |
|---|---|---:|---|
| A (analysis) | [`EXPERIMENT_006A_DIVERGENCE_ANALYSIS.md`](EXPERIMENT_006A_DIVERGENCE_ANALYSIS.md) | — (reuses 003–005) | — |
| D | [`EXPERIMENT_006_LAYER_AWARE_V.md`](EXPERIMENT_006_LAYER_AWARE_V.md) | 374 | 0 |
| C (boundary-depth ablation, not real-backend) | [`EXPERIMENT_006C_BOUNDARY_DEPTH_ABLATION.md`](EXPERIMENT_006C_BOUNDARY_DEPTH_ABLATION.md) | 170 | 0 |

## Compressors added

- `k8_v4_boundary_v8_sim`, `k8_v4_boundary2_v8_sim`, `k8_v4_boundary4_v8_sim` —
  simulated layer-aware V policies (`layer_aware_sim.py`); int8 containers; no
  true attention weights.

## Headline result

`k8_v4_boundary4_v8_sim` acceptance **0.954** on Experiment 006C — +0.063 vs
`k8_v4_sim`, +0.045 vs `k_full_v4_sim`.

## Non-goals (unchanged)

- No speedup/throughput/latency/runtime/production claims.
- No Sparse V, KVQuant, TurboQuant+, KIVI, LMCache, vLLM, or PagedAttention in V7.
- Phase C was boundary-depth ablation, **not** KIVI/KVQuant/TurboQuant adapter work.

---

# V8: Serving-context evaluation (complete — v0.8.0)

## Goal

Evaluate ExactKV in a realistic serving/cache context — compatibility, cache
lifecycle, memory honesty, and verification correctness — without production-serving
claims.

## Phase 0 deliverable

- [`V8_SCOPE_STATEMENT.md`](V8_SCOPE_STATEMENT.md) — full phased scope, Experiment
  007 plan, serving-context questions, restrictions, exit criteria.

## Phase A complete

- [`SERVING_CONTEXT_FEASIBILITY.md`](SERVING_CONTEXT_FEASIBILITY.md) — vLLM/LMCache
  **no-go** for Phase C; PagedAttention concepts via local harness; Phase B primary.

## Phase B complete

- [`SERVING_CACHE_LIFECYCLE_HARNESS.md`](SERVING_CACHE_LIFECYCLE_HARNESS.md) —
  `exactkv/serving/cache_lifecycle.py`; local ownership/block-mapping harness;
  exactness smoke gate; no vLLM/LMCache dependency.

## Phase D complete

- [`EXPERIMENT_007_SERVING_CONTEXT.md`](EXPERIMENT_007_SERVING_CONTEXT.md) —
  Mode B harness evaluation; 238 runs; `exactkv_failures == 0`; all harness
  gates pass. Artifacts gitignored.

## Phase E complete — V8 / v0.8.0

- [`RELEASE_NOTES_V0.8.0.md`](RELEASE_NOTES_V0.8.0.md) — V8 changelog
- [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md) — experiments 001–007
- [`PROJECT_STATUS_V0.8.0.md`](PROJECT_STATUS_V0.8.0.md) — status at v0.8.0
- [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md) — V9–v1.0.0 tracker

**V8 is complete.** Tag `v0.8.0` ready. **Public launch deferred** — project not
final yet.

## Non-goals (V8, unchanged)

- No vLLM, LMCache, or PagedAttention implementation.
- No throughput/latency/speedup/runtime/production claims as ExactKV results.
- Phase C (direct stack integration) **no-go/deferred**, not forgotten.

---

# V9: Real backend integration gauntlet (Phase 0 — scope only)

## Goal

Integrate and **evaluate** real KV-compression backends behind `BackendAdapter` —
TurboQuant / TurboQuant+ first, then KIVI and KVQuant if feasible. V9 is the
**real backend credibility phase**; **not public launch**.

## Phase 0 deliverable

- [`V9_SCOPE_STATEMENT.md`](V9_SCOPE_STATEMENT.md) — full phased scope, Experiments
  008–009 plan, RunPod GPU requirements, exactness/memory contracts, exit criteria.

## Planned phases (A–F)

| Phase | Focus |
|---|---|
| A | TurboQuant / TurboQuant+ deep feasibility ✅ |
| B | TurboQuant adapter prototype + smoke exactness gate ✅ |
| C | Experiment 008 — restricted Python adapter vs baselines ✅ ([`EXPERIMENT_008_TURBOQUANT_PYTHON.md`](EXPERIMENT_008_TURBOQUANT_PYTHON.md)) |
| D | KIVI offline D1–D3 ✅ ([`EXPERIMENT_009_KIVI_OFFLINE.md`](EXPERIMENT_009_KIVI_OFFLINE.md)); KVQuant D4–D6 ✅ ([`EXPERIMENT_010_KVQUANT_SIM.md`](EXPERIMENT_010_KVQUANT_SIM.md)) |
| E | RunPod larger-model validation (≥1.5B) ✅ ([`EXPERIMENT_011_LARGER_MODEL_VALIDATION.md`](EXPERIMENT_011_LARGER_MODEL_VALIDATION.md)) |
| F | `v0.9.0` release notes; V10 draft; v1.0.0 readiness decision ✅ |

## Phase F complete — V9 / v0.9.0

- [`RELEASE_NOTES_V0.9.0.md`](RELEASE_NOTES_V0.9.0.md) — V9 changelog
- [`PROJECT_STATUS_V0.9.0.md`](PROJECT_STATUS_V0.9.0.md) — status at v0.9.0
- [`V10_SCOPE_STATEMENT.md`](V10_SCOPE_STATEMENT.md) — V10 formal scope (Phase 0)
- [`V10_SCOPE_DRAFT.md`](V10_SCOPE_DRAFT.md) — superseded draft
- [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md) — experiments 001–011

**V9 is complete.** Tag `v0.9.0` ready. **Public launch deferred** — V10 required first.

See [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md) and
[`V9_SCOPE_STATEMENT.md`](V9_SCOPE_STATEMENT.md).

## Positioning

- **v0.9.0 is not final public launch** — V10 hardens evaluation before v1.0.0.
- ExactKV evaluated restricted TurboQuant, KIVI, and KVQuant adapters honestly.
- **v1.0.0** follows V10 (and V11 substance), not immediately after v0.9.0.

---

# V10: Evaluation suite hardening and divergence forensics (complete — v0.10.0)

## Goal

Expand benchmark suites (`core_v2`, category panels), run draft/generation sensitivity
sweeps, upgrade divergence forensics beyond 006A proxies, and spot-check restricted
real backends on harder categories. **Not** a performance benchmark.
**Not public launch** — v1.0.0 deferred until V11 substance and launch package (D17–D20).

## Phase 0 deliverable

- [`V10_SCOPE_STATEMENT.md`](V10_SCOPE_STATEMENT.md) — formal phased scope, Experiments
  012–013 plan, suite taxonomy, exit criteria.

## Phase 1 deliverable

- [`V10_PROMPT_SUITES.md`](V10_PROMPT_SUITES.md) — seven versioned suite files (128 prompts);
  `scripts/validate_v10_prompt_suites.py`; `tests/test_v10_prompt_suites.py`.

## Phase 2 deliverable

- [`EXPERIMENT_012_EVAL_SUITE_EXPANSION.md`](EXPERIMENT_012_EVAL_SUITE_EXPANSION.md) —
  128 V10 prompts × 7 compressors; `exactkv_failures == 0`; per-category leaderboards.

## Phase 3 deliverable

- [`EXPERIMENT_013_SENSITIVITY_FORENSICS.md`](EXPERIMENT_013_SENSITIVITY_FORENSICS.md) —
  60 prompts × 4 compressors × 3×3 grid (2,160 cells); `exactkv_failures == 0`;
  draft/generation sensitivity + divergence forensics (RunPod A5000 fp16).

## Phase 4 deliverable

- [`EXPERIMENT_014_REAL_BACKEND_SPOTCHECKS.md`](EXPERIMENT_014_REAL_BACKEND_SPOTCHECKS.md) —
  40 harder-category prompts × 7 compressors (280 cells, cross-panel merge);
  `exactkv_failures == 0`; factory-only KVQuant / TurboQuant / KIVI spot-check.

## Phase 5 deliverable

- [`V10_READINESS_ASSESSMENT.md`](V10_READINESS_ASSESSMENT.md) — v1.0.0 launch-gate decision
  (deferred); v0.10.0 tag readiness.
- [`PROJECT_STATUS_V0.10.0.md`](PROJECT_STATUS_V0.10.0.md), [`RELEASE_NOTES_V0.10.0.md`](RELEASE_NOTES_V0.10.0.md).

**V10 exit:** tag **`v0.10.0`**. See [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md) §V10 (D26–D29 complete).

---

# V11: Final launch hardening (complete — `v0.11.0`)

## Goal

Close scale, serving-context, profiling, and launch-documentation gaps before v1.0.0.
**Not public launch.**

Formal scope: [`V11_SCOPE_STATEMENT.md`](V11_SCOPE_STATEMENT.md) (Phases 0–6; Experiments 015–020).

| Phase | Focus | Status |
|---|---|---|
| 0 | Scope statement | ✅ |
| 1 | 1.5B on V10 suites (Exp 015) | ✅ [`EXPERIMENT_015_QWEN15B_V10_SUITES.md`](EXPERIMENT_015_QWEN15B_V10_SUITES.md) |
| 2 | 3B built-in stretch (Exp 016) | ✅ [`EXPERIMENT_016_QWEN3B_V10_SUITES.md`](EXPERIMENT_016_QWEN3B_V10_SUITES.md) |
| 3 | Serving sidecar/probe refresh (Exp 017) | ✅ [`EXPERIMENT_017_SERVING_SIDECAR_PROBE.md`](EXPERIMENT_017_SERVING_SIDECAR_PROBE.md) |
| 4 | Active GPU memory methodology (Exp 018) | ✅ [`EXPERIMENT_018_GPU_MEMORY_PILOT.md`](EXPERIMENT_018_GPU_MEMORY_PILOT.md) |
| 5 | Divergence autopsy + repair hypotheses (Exp 019) | ✅ [`EXPERIMENT_019_DIVERGENCE_AUTOPSY.md`](EXPERIMENT_019_DIVERGENCE_AUTOPSY.md) |
| 5b | Repair policy pilot (Exp 020) | ✅ [`EXPERIMENT_020_REPAIR_POLICY_PILOT.md`](EXPERIMENT_020_REPAIR_POLICY_PILOT.md) |
| 6 | Launch package readiness | ✅ [`V11_LAUNCH_READINESS.md`](V11_LAUNCH_READINESS.md) |

**V11 exit:** tag **`v0.11.0`**. Readiness decision: **not v1.0.0 public launch yet**.
See [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md) §V11.

---

# V12: Deferred Work Completion Gauntlet (Phase 7 complete)

## Goal

Finish or conclusively close major deferred technical tracks before public launch —
especially production-fidelity backend checks, larger-model real-backend validation,
full repair-policy validation, and performance/memory truth boundaries.
**Not public launch.**

Formal scope: [`V12_SCOPE_STATEMENT.md`](V12_SCOPE_STATEMENT.md) (Phases 0–8; Experiments 021–027).

| Phase | Focus | Status |
|---|---|---|
| 0 | Scope statement | ✅ |
| 1 | TurboQuant production-fidelity feasibility (Exp 021) | ✅ [`TURBOQUANT_PRODUCTION_FIDELITY_FEASIBILITY.md`](TURBOQUANT_PRODUCTION_FIDELITY_FEASIBILITY.md) |
| 1b | TurboQuant toolchain prep | ✅ [`TURBOQUANT_PRODUCTION_TOOLCHAIN_PREP.md`](TURBOQUANT_PRODUCTION_TOOLCHAIN_PREP.md) |
| 2 | TurboQuant llama.cpp / GGUF probe or no-go (Exp 022) | ✅ [`EXPERIMENT_022_TURBOQUANT_LLAMACPP_PROBE.md`](EXPERIMENT_022_TURBOQUANT_LLAMACPP_PROBE.md) |
| 3 | KVQuant larger-model validation (Exp 023) | ✅ [`EXPERIMENT_023_KVQUANT_LARGER_MODEL.md`](EXPERIMENT_023_KVQUANT_LARGER_MODEL.md) |
| 4 | KIVI CUDA/Triton feasibility or no-go (Exp 024) | ✅ [`EXPERIMENT_024_KIVI_CUDA_TRITON_FEASIBILITY.md`](EXPERIMENT_024_KIVI_CUDA_TRITON_FEASIBILITY.md) |
| 5 | Full-suite repair-policy validation (Exp 025) | ✅ [`EXPERIMENT_025_FULL_SUITE_REPAIR_POLICY.md`](EXPERIMENT_025_FULL_SUITE_REPAIR_POLICY.md) |
| 6 | True attention logging feasibility (Exp 026) | ✅ [`EXPERIMENT_026_ATTENTION_LOGGING_FEASIBILITY.md`](EXPERIMENT_026_ATTENTION_LOGGING_FEASIBILITY.md) |
| 7 | Performance/memory truth boundary (Exp 027) | ✅ [`EXPERIMENT_027_PERFORMANCE_MEMORY_TRUTH_BOUNDARY.md`](EXPERIMENT_027_PERFORMANCE_MEMORY_TRUTH_BOUNDARY.md) |
| 8 | V12 release package + public-launch decision | Planned |

See [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md) §V12.

---

# V13: Practicality Proof (Phase 2 complete; Exp 029 grid passed)

## Goal

Build and measure the missing systems pieces (span verification, diagnostic timing,
GPU memory isolation, hot adapter, Llama-3.1-8B, demo, plots) needed to determine
whether ExactKV is practically useful — not only exactness-safe. **Not public launch.**

Formal scope: [`V13_SCOPE_STATEMENT.md`](V13_SCOPE_STATEMENT.md) (Phases 0–9; Experiments 028–035).

| Phase | Focus | Status |
|---|---|---|
| 0 | Scope statement | ✅ |
| 1 | Span verification design (Exp 028) | ✅ [`SPAN_VERIFICATION_DESIGN.md`](SPAN_VERIFICATION_DESIGN.md) |
| 2 | Span verification (Exp 028 smoke + Exp 029 grid) | ✅ [`EXPERIMENT_029_SPAN_VERIFICATION_GRID.md`](EXPERIMENT_029_SPAN_VERIFICATION_GRID.md) |
| 3 | Diagnostic timing harness (Exp 030) | ✅ [`EXPERIMENT_030_DIAGNOSTIC_TIMING.md`](EXPERIMENT_030_DIAGNOSTIC_TIMING.md) |
| 3b | Span GPU/fp16 parity (Exp 030b) | ✅ [`EXPERIMENT_030B_SPAN_PARITY_INVESTIGATION.md`](EXPERIMENT_030B_SPAN_PARITY_INVESTIGATION.md) |
| 4 | Active GPU memory isolation (Exp 031) | ✅ [`EXPERIMENT_031_GPU_MEMORY_ISOLATION.md`](EXPERIMENT_031_GPU_MEMORY_ISOLATION.md) |
| 5 | SnapKV / Shard / SpectralQuant feasibility (Exp 032 + addendum) | ✅ [`EXPERIMENT_032_ADDENDUM_SHARD_SPECTRALQUANT.md`](EXPERIMENT_032_ADDENDUM_SHARD_SPECTRALQUANT.md) |
| 5b | SnapKV experimental adapter MVP | ✅ [`EXPERIMENT_032B_SNAPKV_EXPERIMENTAL_SMOKE.md`](EXPERIMENT_032B_SNAPKV_EXPERIMENTAL_SMOKE.md) |
| 5c | SpectralQuant experimental adapter | ✅ Exp 043–045 — RESTRICTED BACKEND panel ([`EXPERIMENT_045_SPECTRALQUANT_RESTRICTED_PANEL.md`](EXPERIMENT_045_SPECTRALQUANT_RESTRICTED_PANEL.md)) |
| 6 | Llama-3.1-8B small suite (Exp 033) | ✅ [`EXPERIMENT_033_LLAMA31_8B_SMALL_SUITE.md`](EXPERIMENT_033_LLAMA31_8B_SMALL_SUITE.md) |
| 7 | Killer correction demo (Exp 034) | ✅ [`EXPERIMENT_034_KILLER_CORRECTION_DEMO.md`](EXPERIMENT_034_KILLER_CORRECTION_DEMO.md) |
| 7b | Live correction terminal demo | ✅ [`DEMO_EXACTKV_LIVE_CORRECTION.md`](DEMO_EXACTKV_LIVE_CORRECTION.md) |
| 8 | Visual plot package (Exp 035) | ✅ [`EXPERIMENT_035_VISUAL_PLOTS_AND_LEADERBOARD.md`](EXPERIMENT_035_VISUAL_PLOTS_AND_LEADERBOARD.md) |
| 8b | Public visual polish (Exp 036) | ✅ [`PUBLIC_VISUAL_PACKAGE.md`](PUBLIC_VISUAL_PACKAGE.md) |
| 8c | Cinematic crash-test video (optional) | ✅ [`EXACTKV_CRASH_TEST_VIDEO.md`](EXACTKV_CRASH_TEST_VIDEO.md) |
| 8d | Leaderboard tiering cleanup | ✅ [`leaderboard.md`](leaderboard.md) |
| 8e | Terminal-native crash-test demo (Exp 034b) | ✅ [`EXACTKV_TERMINAL_CRASH_TEST.md`](EXACTKV_TERMINAL_CRASH_TEST.md) |
| 9B | Prelaunch hardening infrastructure | ✅ [`PRELAUNCH_HARDENING_REPORT.md`](PRELAUNCH_HARDENING_REPORT.md) |
| 10A | LongBench-style drift demo (Exp 037) | ✅ [`EXPERIMENT_037_LONGBENCH_STYLE_DRIFT_DEMO.md`](EXPERIMENT_037_LONGBENCH_STYLE_DRIFT_DEMO.md) — **secondary** demo |
| 10B | Shard external-drafter probe (Exp 038) | ✅ [`EXPERIMENT_038_SHARD_EXTERNAL_DRAFTER_PROBE.md`](EXPERIMENT_038_SHARD_EXTERNAL_DRAFTER_PROBE.md) |
| 10B2 | Shard stress panel (Exp 039) | ✅ [`EXPERIMENT_039_SHARD_EXTERNAL_STRESS_PANEL.md`](EXPERIMENT_039_SHARD_EXTERNAL_STRESS_PANEL.md) |
| 10B3 | Shard ablation (Exp 040) | ✅ [`EXPERIMENT_040_SHARD_EXTERNAL_ABLATION.md`](EXPERIMENT_040_SHARD_EXTERNAL_ABLATION.md) |
| 10B4 | Shard combined stress (Exp 041) | ✅ [`EXPERIMENT_041_SHARD_COMBINED_STRESS.md`](EXPERIMENT_041_SHARD_COMBINED_STRESS.md) — `stop_shard_bounded_probe_complete` |
| 10C | Shard leaderboard integration | ✅ RESTRICTED BACKEND tier in [`leaderboard.md`](leaderboard.md) |
| 10D | SpectralQuant probe (Exp 042) | ✅ [`EXPERIMENT_042_SPECTRALQUANT_PROBE.md`](EXPERIMENT_042_SPECTRALQUANT_PROBE.md) |
| 10E | SpectralQuant smoke leaderboard (superseded) | ✅ → promoted in 10G |
| 10F | SpectralQuant real KV + adapter smoke (043–044) | ✅ |
| 10G | SpectralQuant restricted panel (045) | ✅ RESTRICTED BACKEND in [`leaderboard.md`](leaderboard.md) |
| 10H | External methods consolidation | ✅ [`PHASE_10_EXTERNAL_METHODS_SUMMARY.md`](PHASE_10_EXTERNAL_METHODS_SUMMARY.md) |
| 9C | Launch validation (research preview) | ✅ [`LAUNCH_VALIDATION_REPORT.md`](LAUNCH_VALIDATION_REPORT.md) |
| 9D | RC blocker fixes | ✅ deps + static anchors + heading fix |
| 11A | VeriCache parity audit + systems roadmap | ✅ [`VERICACHE_PARITY_AUDIT.md`](VERICACHE_PARITY_AUDIT.md) · [`VERICACHE_SYSTEMS_ROADMAP.md`](VERICACHE_SYSTEMS_ROADMAP.md) |
| 11B | Dual-cache abstraction (Stage 1) | ✅ [`DUAL_CACHE_ABSTRACTION.md`](DUAL_CACHE_ABSTRACTION.md) |
| 11C | Full-KV storage manager spike (Stage 2) | ✅ [`FULL_KV_STORAGE_MANAGER.md`](FULL_KV_STORAGE_MANAGER.md) |
| 11D | Materialized compressed-draft backend (Stage 3) | ✅ [`MATERIALIZED_COMPRESSED_DRAFT_BACKEND.md`](MATERIALIZED_COMPRESSED_DRAFT_BACKEND.md) |
| 11E | Extended verification scheduler (Stage 4) | ✅ [`EXTENDED_VERIFICATION_SCHEDULER.md`](EXTENDED_VERIFICATION_SCHEDULER.md) |
| 11F | vLLM prototype path (Stage 5) | ✅ [`VLLM_PROTOTYPE_PATH.md`](VLLM_PROTOTYPE_PATH.md) |
| 11G | LMCache prototype path (Stage 6) | ✅ [`LMCACHE_PROTOTYPE_PATH.md`](LMCACHE_PROTOTYPE_PATH.md) |
| 11H | Remote prefix cache semantics (Stage 7) | ✅ [`REMOTE_PREFIX_CACHE_SEMANTICS.md`](REMOTE_PREFIX_CACHE_SEMANTICS.md) |
| 11I | Throughput benchmark harness (Stage 8) | ✅ [`THROUGHPUT_BENCHMARK_HARNESS.md`](THROUGHPUT_BENCHMARK_HARNESS.md) |
| 11J | Paper-like reproduction panel (Stage 9) | ✅ [`PAPER_LIKE_REPRODUCTION_PANEL.md`](PAPER_LIKE_REPRODUCTION_PANEL.md) |
| 11K | VeriCache parity RC claim gate (Stage 10) | ✅ [`VERICACHE_PARITY_CLAIM_GATE.md`](VERICACHE_PARITY_CLAIM_GATE.md) |
| 12A | Full-KV restore smoke (real HF KV) | ✅ [`EXPERIMENT_046_FULL_KV_RESTORE_SMOKE.md`](EXPERIMENT_046_FULL_KV_RESTORE_SMOKE.md) |
| 12B | Full-KV restore panel hardening | ✅ [`EXPERIMENT_047_FULL_KV_RESTORE_PANEL.md`](EXPERIMENT_047_FULL_KV_RESTORE_PANEL.md) |
| 12C | Offline verifier restore smoke | ✅ [`EXPERIMENT_048_OFFLINE_VERIFIER_RESTORE_SMOKE.md`](EXPERIMENT_048_OFFLINE_VERIFIER_RESTORE_SMOKE.md) |
| 12D | Offline verifier lossy draft | ✅ [`EXPERIMENT_049_OFFLINE_VERIFIER_LOSSY_DRAFT.md`](EXPERIMENT_049_OFFLINE_VERIFIER_LOSSY_DRAFT.md) |
| 12E | Offline restored-verifier drift stress | ✅ [`EXPERIMENT_050_OFFLINE_RESTORED_VERIFIER_DRIFT_STRESS.md`](EXPERIMENT_050_OFFLINE_RESTORED_VERIFIER_DRIFT_STRESS.md) |
| 12F | Offline verifier CUDA drift panel | ✅ [`EXPERIMENT_051_OFFLINE_VERIFIER_CUDA_DRIFT_PANEL.md`](EXPERIMENT_051_OFFLINE_VERIFIER_CUDA_DRIFT_PANEL.md) |
| 12G | Restored-verifier runner consolidation | ✅ [`RESTORED_VERIFIER_RUNNER.md`](RESTORED_VERIFIER_RUNNER.md) · [`EXPERIMENT_052_RESTORED_VERIFIER_RUNNER_SMOKE.md`](EXPERIMENT_052_RESTORED_VERIFIER_RUNNER_SMOKE.md) |
| 12H | Runner-backed drift panel | ✅ [`EXPERIMENT_053_RESTORED_VERIFIER_RUNNER_PANEL.md`](EXPERIMENT_053_RESTORED_VERIFIER_RUNNER_PANEL.md) |
| 13A | Experimental restored-verifier runtime | ✅ [`EXPERIMENTAL_RESTORED_VERIFIER_RUNTIME.md`](EXPERIMENTAL_RESTORED_VERIFIER_RUNTIME.md) · [`EXPERIMENT_054_EXPERIMENTAL_RESTORED_VERIFIER_RUNTIME.md`](EXPERIMENT_054_EXPERIMENTAL_RESTORED_VERIFIER_RUNTIME.md) |
| 13B | Explicit CLI flag for experimental runtime | ✅ [`EXPERIMENT_055_EXPERIMENTAL_RESTORED_VERIFIER_CLI.md`](EXPERIMENT_055_EXPERIMENTAL_RESTORED_VERIFIER_CLI.md) |
| 14A | CUDA restored-verifier runtime gate | ✅ [`EXPERIMENT_056_CUDA_RESTORED_VERIFIER_RUNTIME_GATE.md`](EXPERIMENT_056_CUDA_RESTORED_VERIFIER_RUNTIME_GATE.md) |
| 14B | GPU memory accounting diagnostic | ✅ [`EXPERIMENT_057_GPU_MEMORY_ACCOUNTING.md`](EXPERIMENT_057_GPU_MEMORY_ACCOUNTING.md) |
| 14C | Expanded GPU memory panel | ✅ [`EXPERIMENT_058_EXPANDED_GPU_MEMORY_PANEL.md`](EXPERIMENT_058_EXPANDED_GPU_MEMORY_PANEL.md) |
| 15A | vLLM feasibility probe (install-safe) | ✅ [`EXPERIMENT_059_VLLM_FEASIBILITY_PROBE.md`](EXPERIMENT_059_VLLM_FEASIBILITY_PROBE.md) |
| 15B | Isolated vLLM venv feasibility | ✅ [`EXPERIMENT_060_VLLM_VENV_FEASIBILITY.md`](EXPERIMENT_060_VLLM_VENV_FEASIBILITY.md) |
| 15B-unblock | vLLM version compatibility sweep | ✅ [`EXPERIMENT_061_VLLM_VERSION_SWEEP.md`](EXPERIMENT_061_VLLM_VERSION_SWEEP.md) |
| 15C-env | vLLM container/CUDA-13 feasibility | ✅ [`EXPERIMENT_062_VLLM_CONTAINER_FEASIBILITY.md`](EXPERIMENT_062_VLLM_CONTAINER_FEASIBILITY.md) |
| 15C | vLLM API surface reconnaissance | ✅ [`EXPERIMENT_063_VLLM_API_SURFACE_RECON.md`](EXPERIMENT_063_VLLM_API_SURFACE_RECON.md) |
| 15D | vLLM KV/cache visibility probe | ✅ [`EXPERIMENT_064_VLLM_KV_VISIBILITY_PROBE.md`](EXPERIMENT_064_VLLM_KV_VISIBILITY_PROBE.md) |
| 15E | Idle-GPU vLLM object KV probe | ⏸ deferred — [`EXPERIMENT_065_IDLE_VLLM_OBJECT_KV_PROBE.md`](EXPERIMENT_065_IDLE_VLLM_OBJECT_KV_PROBE.md) (auto-serving template blocks idle pass) |
| 16A | Streaming quantized-KV attention feasibility | ✅ [`EXPERIMENT_066_STREAMING_QUANT_ATTENTION_FEASIBILITY.md`](EXPERIMENT_066_STREAMING_QUANT_ATTENTION_FEASIBILITY.md) |
| 16F | Full-prefix logit drift smoke | ✅ [`EXPERIMENT_071_FULL_PREFIX_LOGIT_DRIFT_SMOKE.md`](EXPERIMENT_071_FULL_PREFIX_LOGIT_DRIFT_SMOKE.md) |
| 16G | Full-depth divergence trace | ✅ [`EXPERIMENT_072_FULL_DEPTH_DIVERGENCE_TRACE.md`](EXPERIMENT_072_FULL_DEPTH_DIVERGENCE_TRACE.md) |
| 16H | Qwen-family divergence panel | ✅ [`EXPERIMENT_073_QWEN_FAMILY_DIVERGENCE_PANEL.md`](EXPERIMENT_073_QWEN_FAMILY_DIVERGENCE_PANEL.md) |
| 16I | Attention tolerance policy panel | ✅ [`EXPERIMENT_074_ATTENTION_TOLERANCE_POLICY_PANEL.md`](EXPERIMENT_074_ATTENTION_TOLERANCE_POLICY_PANEL.md) |
| 16J | Generation-shadow wiring review | ✅ [`EXPERIMENT_075_GENERATION_SHADOW_WIRING_REVIEW.md`](EXPERIMENT_075_GENERATION_SHADOW_WIRING_REVIEW.md) |
| 16K | Generation-shadow observer smoke | ✅ [`EXPERIMENT_076_GENERATION_SHADOW_OBSERVER_SMOKE.md`](EXPERIMENT_076_GENERATION_SHADOW_OBSERVER_SMOKE.md) |
| 16L | Prompt+generated generation-shadow panel | ✅ [`EXPERIMENT_077_GENERATION_SHADOW_PROMPT_PLUS_GENERATED_PANEL.md`](EXPERIMENT_077_GENERATION_SHADOW_PROMPT_PLUS_GENERATED_PANEL.md) |
| 16M | Expanded generation-shadow panel | ✅ [`EXPERIMENT_078_GENERATION_SHADOW_EXPANDED_PANEL.md`](EXPERIMENT_078_GENERATION_SHADOW_EXPANDED_PANEL.md) |
| 16N | Decode-prefix ladder shadow observer | ✅ [`EXPERIMENT_079_DECODE_PREFIX_LADDER_SHADOW_OBSERVER.md`](EXPERIMENT_079_DECODE_PREFIX_LADDER_SHADOW_OBSERVER.md) |
| 16O | ExactKV round-log shadow observer | ✅ [`EXPERIMENT_080_ROUND_LOG_SHADOW_OBSERVER.md`](EXPERIMENT_080_ROUND_LOG_SHADOW_OBSERVER.md) |
| 16P | Live round observer smoke | ✅ [`EXPERIMENT_081_LIVE_ROUND_OBSERVER_SMOKE.md`](EXPERIMENT_081_LIVE_ROUND_OBSERVER_SMOKE.md) |
| 16Q | Live observer + post-hoc shadow panel | ✅ [`EXPERIMENT_082_LIVE_OBSERVER_SHADOW_PANEL.md`](EXPERIMENT_082_LIVE_OBSERVER_SHADOW_PANEL.md) |
| 16R | Guarded decode-time shadow observer dry-run | ✅ [`EXPERIMENT_083_GUARDED_DECODE_TIME_SHADOW_SMOKE.md`](EXPERIMENT_083_GUARDED_DECODE_TIME_SHADOW_SMOKE.md) |
| 16S | Expanded guarded decode-time shadow panel | ✅ [`EXPERIMENT_084_GUARDED_DECODE_TIME_SHADOW_PANEL.md`](EXPERIMENT_084_GUARDED_DECODE_TIME_SHADOW_PANEL.md) |
| 16T | Phase 16 closeout & claim freeze | ✅ [`PHASE_16_CLOSEOUT.md`](PHASE_16_CLOSEOUT.md) |
| 10I | Benchmark gap analysis | ✅ [`BENCHMARK_GAP_ANALYSIS.md`](BENCHMARK_GAP_ANALYSIS.md) |
| 10C | Parallel work integration | ✅ [`PARALLEL_WORK_INTEGRATION_REPORT.md`](PARALLEL_WORK_INTEGRATION_REPORT.md) |
| 9C | Launch validation & should-fix | Planned |
| 9 | V13 completion / launch decision | **Deferred** |

See [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md) §V13.

---

# v1.0.0: Public launch (after V13 — **NOT APPROVED**)

Git tag `v1.0.0`, `PROJECT_STATUS` / `RELEASE_NOTES` v1.0.0, optional curated report
bundle (D17), and reviewed launch narrative (D18) — only after Phase 9B must-fix
blockers and explicit launch approval ([`LAUNCH_READINESS_GAP_AUDIT.md`](LAUNCH_READINESS_GAP_AUDIT.md)).

**Public launch remains deferred.** V13 Phase 9A audit: strong demos and exactness evidence, but install/repro/README/claims hardening incomplete. Runtime speed, active memory savings, and serving are **future work** ([`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md)).

---

# Recommended build order

Do this:

```text
V0 → V1 → V2 → V3 → V4
```

Do not do this:

```text
V0 → vLLM integration → TurboQuant → CUDA
```

The second path is likely to fail because correctness and metrics will not be stable.

## Public release milestones

### Release 0.1

Corresponds to V1.

Claim:

> ExactKV proves verified compressed-KV generation can preserve full-KV outputs in a small Hugging Face prototype.

### Release 0.2

Corresponds to V2.

Claim:

> ExactKV is now compressor-agnostic with INT8 and INT4 support.

### Release 0.3

Corresponds to V3.

Claim:

> ExactKV now benchmarks KV compressors by acceptance rate and exactness.

### Release 0.4

Corresponds to V4.

Claim:

> ExactKV evaluates **simulated** asymmetric K/V compressors by acceptance
> behaviour under full-KV verification (Experiment 003; 0 failures). These are
> not real packed-bit backends, and no performance claim is made.

### Release 1.0

Should not happen until:

- multiple compressors
- reproducible benchmarks
- exactness tests
- honest workspace-aware memory reporting (V5)
- at least one real backend adapter evaluated by acceptance behaviour (V6)

> Note: "performance report" was intentionally removed from the Release 1.0
> criteria. ExactKV makes no speedup/throughput/latency claims; release readiness
> is judged on correctness, acceptance evaluation, and memory honesty.

## Final roadmap rule

Every version must leave the project in a working state.

Do not merge half-working systems code that breaks exactness.
