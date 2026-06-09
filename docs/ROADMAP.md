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

# V8: Serving-context evaluation (Phase 0 — scope only)

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

## Recommended direction

Experiment 007 Mode B harness evaluation (Phase D) → launch package (Phase E).
vLLM/LMCache PoC remains deferred (Phase C no-go).

## Non-goals

- No vLLM, LMCache, or PagedAttention implementation in Phase 0.
- No throughput/latency/speedup/runtime/production claims as ExactKV results.

See [`RESEARCH_BACKLOG.md`](RESEARCH_BACKLOG.md) and
[`FUTURE_ROADMAP_V6_V8.md`](FUTURE_ROADMAP_V6_V8.md) for planning context.

> Phases A–E require separate explicit approval before any code is written.

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
