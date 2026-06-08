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
V4: add real compressors
V5: optimize runtime
V6: integrate with serving systems
```

Do not skip versions.

The project should not jump to CUDA, vLLM, LMCache, or TurboQuant before the core exactness loop is correct.

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

# V5: Runtime performance

## Goal

Make ExactKV faster, not just correct.

## Why this version exists

The full promise of the VeriCache idea depends on verification overhead being amortized.

## New features

- Full KV offload to CPU
- Compressed KV resident on GPU
- Async CPU/GPU transfer experiments
- Pinned memory
- Draft length tuning
- Adaptive draft length prototype
- Better memory measurement
- Optional parallel verification over draft span

## Non-goals

- Still no full production serving scheduler unless V5 is stable.
- No multi-node remote prefix caching yet.

## Success criteria

ExactKV shows measurable speedup over full KV on at least one controlled setting.

Potential target:

```text
> 1.2x throughput with exact outputs
```

Stretch target:

```text
> 1.5x throughput with exact outputs
```

## Exit tests

- Benchmark shows speedup in a reproducible environment.
- Exactness remains preserved.
- Memory reporting is credible.
- Draft length sensitivity is reported.

## Deliverables

- Performance report
- Speedup plots
- Memory plots
- Draft length sweep
- Compressor ratio sweep

---

# V6: vLLM and LMCache integration

## Goal

Move ExactKV toward real serving infrastructure.

## Why this version exists

The paper's systems benefit depends on scheduler-level integration and KV movement across tiers.

## New features

- vLLM prototype integration
- LMCache prototype integration
- Paged cache awareness
- Request-level scheduler experiments
- Bandwidth and HBM resource model
- Cross-resource staggering prototype

## Non-goals

- No claim of production readiness until robustly tested.
- No multi-tenant reliability guarantees.
- No broad model compatibility guarantee.

## Success criteria

ExactKV can run as a prototype inside or alongside a serving stack.

## Exit tests

- vLLM path runs one model.
- LMCache path can store or load KV in a controlled demo.
- Scheduler can separate draft and verify phases.
- Exactness remains preserved.

## Deliverables

- Integration docs
- Prototype branch or module
- Serving demo
- System benchmark report

---

# V7: Research extensions

## Goal

Explore new research directions unlocked by ExactKV.

## Possible directions

- Acceptance-optimized compressors
- Adaptive draft length
- Prompt-aware compression selection
- Learned acceptance predictors
- Hybrid speculative decoding plus ExactKV
- Sampling-compatible verification
- Verification for approximate prefix reuse
- Multi-compressor ensembles
- Safety-focused structured-output verification

## Success criteria

At least one research extension produces a result worth a blog post, workshop paper, or viral launch.

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

> ExactKV supports real compressor adapters.

### Release 1.0

Should not happen until:

- multiple compressors
- reproducible benchmarks
- exactness tests
- performance report
- at least one serious integration path

## Final roadmap rule

Every version must leave the project in a working state.

Do not merge half-working systems code that breaks exactness.
