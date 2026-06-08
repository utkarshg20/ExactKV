# 11_NON_GOALS.md

# ExactKV Non-Goals

## Purpose of this document

This document defines what ExactKV will not do, especially in early versions.

Non-goals are as important as goals because ExactKV is easy to over-scope.

The default rule:

> If a feature is not required to prove verified compressed-KV generation, it is not part of V1.

---

# Project-level non-goals

## Non-goal 1: ExactKV is not a new foundation model

ExactKV does not train or fine-tune LLMs.

It works with existing causal language models.

## Non-goal 2: ExactKV is not a new KV compressor first

ExactKV may include simple compressors and adapters, but its primary contribution is verification and benchmarking.

## Non-goal 3: ExactKV is not a full serving engine

ExactKV does not replace vLLM, SGLang, TensorRT-LLM, or other serving systems.

It may integrate with them later.

## Non-goal 4: ExactKV is not a TurboQuant-only project

TurboQuant is relevant, but ExactKV must remain compressor-agnostic.

## Non-goal 5: ExactKV is not a production system in V1

V1 is a correctness prototype.

Do not present it as production-ready.

---

# V1 non-goals

## Non-goal 1: No vLLM integration

V1 must not modify or depend on vLLM.

Why:

- vLLM scheduler internals add complexity.
- Paged KV cache layout complicates early debugging.
- Correctness should be proven in Hugging Face first.

## Non-goal 2: No LMCache integration

V1 must not depend on LMCache.

Why:

- Full-KV offload and transfer are not needed for basic correctness.
- LMCache should be considered after the core loop works.

## Non-goal 3: No CPU offload

V1 keeps full KV and compressed KV on the same device.

Why:

- CPU offload introduces transfer and synchronization complexity.
- The first goal is token exactness, not memory speedup.

## Non-goal 4: No async transfer

V1 should not use CUDA streams or async CPU/GPU loading.

Why:

- Debugging async correctness is harder.
- This belongs to performance versions.

## Non-goal 5: No cross-resource staggering

The VeriCache paper uses cross-resource staggering.

V1 does not.

Why:

- Staggering requires batching, scheduling, transfer modeling, and HBM accounting.
- V1 is single-request and correctness-first.

## Non-goal 6: No batching

V1 supports one prompt at a time.

Why:

- Batching complicates sequence offsets and cache alignment.
- Single-request testing is enough for exactness.

## Non-goal 7: No sampling

V1 supports greedy decoding only.

Not supported:

- temperature sampling
- top-p
- top-k
- beam search

Why:

- Exactness is easiest to define under deterministic decoding.
- Sampling-compatible verification is future work.

## Non-goal 8: No Triton kernels

V1 should not include Triton.

Why:

- Kernel work distracts from correctness.
- PyTorch is enough for prototype.

## Non-goal 9: No CUDA kernels

V1 should not include custom CUDA.

Why:

- Too much complexity.
- Not required for proof of concept.

## Non-goal 10: No multi-GPU

V1 uses one model instance on one device.

Why:

- Tensor parallelism and distributed KV complicate debugging.

## Non-goal 11: No remote prefix caching

V1 does not implement the remote prefix caching setting from VeriCache.

Why:

- Requires storage, network, and multiple GPU pools.
- Future work.

## Non-goal 12: No advanced compressors

V1 does not implement:

- KIVI
- KVQuant
- TurboQuant
- SnapKV
- KVzip
- KVzap
- RotateKV

V1 only needs a simple compressor.

## Non-goal 13: No leaderboard

V1 does not need a public leaderboard.

Why:

- Benchmark suite must mature first.

## Non-goal 14: No web dashboard

V1 does not need a UI.

Reports can be JSON and text.

## Non-goal 15: No claim of speedup

V1 should not claim runtime speedup unless actually measured.

It is acceptable if V1 is slower than full KV.

---

# V2 non-goals

V2 still should not include:

- vLLM
- LMCache
- CUDA
- Triton
- CPU offload
- batching
- sampling
- production serving

V2 is about modularity and metrics.

---

# V3 non-goals

V3 should not include:

- production serving claims
- custom kernels
- multi-GPU
- remote prefix caching

V3 is about benchmark maturity.

---

# V4 non-goals

V4 adds advanced compressor adapters but should not:

- reimplement every compressor paper
- optimize every adapter equally
- claim SOTA compression
- rewrite external libraries

Adapters should be pragmatic.

---

# V5 non-goals

V5 starts performance work but still should not claim production readiness until proven.

V5 may include:

- CPU offload
- async transfer experiments
- draft length tuning

But not necessarily:

- production scheduler
- multi-tenant serving
- remote prefix caching

---

# V6 non-goals

V6 may integrate with vLLM and LMCache but should still avoid overclaiming.

V6 is not automatically production-ready.

Production readiness requires more work:

- broad model support
- robust scheduling
- stress testing
- failure handling
- concurrency
- monitoring
- packaging
- documentation

---

# Explicitly forbidden early shortcuts

## Shortcut 1: Comparing text only

Do not use decoded text equality as the primary exactness metric.

Use token IDs.

## Shortcut 2: Claiming speedup from lossy mode

Lossy compressed mode speedup is not ExactKV speedup.

ExactKV speedup must include verification cost.

## Shortcut 3: Reporting memory reduction without full context

If full KV is still stored on CPU, do not say total memory is reduced.

Say active GPU KV memory is reduced.

## Shortcut 4: Skipping full baseline validation

Before implementing ExactKV, custom full-KV generation must match standard generation.

## Shortcut 5: Building around one compressor

Do not make the verification engine depend on INT8, TurboQuant, or any specific compressor.

## Shortcut 6: Hiding exactness failures

If ExactKV output differs from full output, the run failed.

Do not average it away.

---

# Allowed shortcuts for V1

Some shortcuts are acceptable if documented.

## Allowed: recompress after every commit

Inefficient but safe.

## Allowed: sequential verification

Slow but correct.

## Allowed: simulated INT4 using int8 storage

Useful for acceptance testing, even if not memory-optimal.

## Allowed: GPU-only full KV

V1 does not need CPU offload.

## Allowed: small model only

Qwen2.5-0.5B is enough for proof of concept.

---

# Public messaging non-goals

Do not write:

> ExactKV beats vLLM.

Do not write:

> ExactKV is production-ready.

Do not write:

> ExactKV invented lossless KV compression.

Do not write:

> ExactKV solves KV cache memory.

Do write:

> ExactKV is a compressor-agnostic implementation and benchmark suite for verified KV-cache compression, inspired by VeriCache.

---

# Final non-goal rule

When in doubt, ask:

> Is this required to prove exact output equality with compressed-KV drafting and full-KV verification?

If no, it is probably not V1.
