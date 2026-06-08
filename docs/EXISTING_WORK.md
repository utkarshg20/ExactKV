# Existing Work and Competitive Landscape

## Purpose of this document

This document maps the existing work around ExactKV.

It answers:

- What already exists?
- What does not exist?
- What should ExactKV avoid duplicating?
- Where is the real gap?
- Which papers and repositories matter?

The goal is not to create an exhaustive survey of every KV-cache paper. The goal is to give Cursor and future contributors enough context to understand the design space.

## High-level landscape

ExactKV sits at the intersection of five areas:

1. LLM serving engines
2. KV cache memory management
3. KV cache compression
4. Speculative decoding
5. Verification-based exact inference

## 1. LLM serving engines

### vLLM

vLLM is one of the most important open-source LLM serving engines. Its core contribution is PagedAttention, which manages KV cache memory using a block-based paging abstraction inspired by virtual memory.

Relevant concepts:

- Paged KV cache
- Block tables
- Continuous batching
- Efficient serving
- OpenAI-compatible APIs
- Integration target for future ExactKV versions

Why it matters:

- VeriCache itself is built on top of vLLM and LMCache.
- ExactKV Version 1 should not start by modifying vLLM.
- ExactKV should eventually understand vLLM's `past_key_values` and paged-cache model.

Status for ExactKV:

- V1: Hugging Face only.
- Later: vLLM integration.
- Do not start with vLLM because it adds scheduler and cache-layout complexity too early.

### SGLang

SGLang is another high-performance LLM serving framework. It supports structured generation and efficient serving.

Why it matters:

- Future users may want ExactKV support in SGLang.
- LMCache supports integrations with modern inference engines, including vLLM and SGLang.

Status for ExactKV:

- Not in scope for early versions.

### TensorRT-LLM

TensorRT-LLM is NVIDIA's optimized inference stack.

Why it matters:

- It is production-relevant.
- It uses heavily optimized kernels and memory management.
- It is not the right starting point for an early open-source prototype.

Status for ExactKV:

- Future research only.

## 2. KV cache management and reuse

### PagedAttention

PagedAttention addresses memory fragmentation and allocation efficiency. It does not primarily compress KV values. It changes how KV cache is laid out and managed.

It answers:

> How do we allocate and share KV memory efficiently?

It does not answer:

> How do we safely use lossy compressed KV without changing outputs?

ExactKV is complementary.

### vAttention

vAttention argues for keeping the KV cache contiguous in virtual memory while relying on CUDA virtual memory management for physical allocation. It is an alternative to PagedAttention-style non-contiguous KV layout.

Why it matters:

- ExactKV's future runtime may need to care whether KV memory is physically paged, virtually contiguous, offloaded, or moved through a cache layer.
- For V1, this is not needed.

### LMCache

LMCache is highly relevant.

It is an open-source KV cache layer designed for enterprise-scale LLM inference. It supports extracting, storing, moving, and reusing KV caches across inference engines and memory tiers.

Why it matters:

- VeriCache uses LMCache for persistent KV cache storage and transfer.
- ExactKV will eventually need similar ideas for full-KV offload and reload.
- LMCache is likely the most natural future integration point for production-grade ExactKV.

Status for ExactKV:

- V1 should not depend on LMCache.
- V3 or later can explore LMCache integration.
- If ExactKV becomes serious, LMCache integration is strategically important.

## 3. KV cache compression

KV compression methods reduce memory and bandwidth cost by making the cache smaller.

They generally fall into two buckets:

1. Token dropping
2. Quantization

### Token dropping methods

Token dropping changes the shape or sparsity pattern of the cache by removing tokens or token-head entries.

Relevant methods:

- H2O
- StreamingLLM
- SnapKV
- PyramidKV
- PyramidInfer
- DuoAttention
- KVzip
- FastKVzip
- KVzap
- ExpectedAttention

Strength:

- Can significantly reduce attention cost.
- Often works well for retrieval-like long-context tasks.

Weakness:

- Information is literally removed.
- Can fail badly on tasks requiring exact long-range dependencies.
- Output may stay fluent while becoming functionally wrong.

ExactKV relevance:

- Token dropping can serve as a compressed-KV drafter.
- ExactKV can measure how often dropped-token caches still agree with full-KV outputs.
- A future ExactKV compressor could optimize for acceptance length rather than direct-serving quality.

### Quantization methods

Quantization keeps the cache shape but reduces precision.

Relevant methods:

- KIVI
- KVQuant
- KVTuner
- TurboQuant
- CacheGen
- KVTC
- QServe
- GEAR
- LLM.265
- RotateKV

Strength:

- Preserves structure better than token dropping.
- Can be hardware-friendly.
- More compatible with existing attention shapes.

Weakness:

- Still changes attention values.
- Small per-step distribution shifts can compound over long generation.
- Very low-bit implementations may require custom kernels to be fast.

ExactKV relevance:

- Quantization is the easiest starting point for V1.
- Start with INT8.
- Then INT4.
- Later use KIVI, KVQuant, TurboQuant-style, or kvpress backends.

### KIVI

KIVI is a tuning-free asymmetric 2-bit KV cache quantization method. It quantizes keys and values differently based on observed distribution properties.

Why it matters:

- It is a strong known baseline.
- It is conceptually simple enough to understand.
- It has public source code.
- VeriCache uses KIVI for some remote prefix caching experiments.

ExactKV status:

- Not V1.
- Candidate for V4 advanced compressor backend.

### KVQuant

KVQuant targets low-bit KV quantization and long-context inference. It includes techniques such as per-channel key quantization, pre-RoPE key quantization, non-uniform datatypes, and outlier handling.

Why it matters:

- Strong quantization baseline.
- More complex implementation.
- Good future benchmark target.

ExactKV status:

- Not V1.
- Candidate for advanced backend or comparison.

### TurboQuant

TurboQuant is a recent online vector quantization method from Google Research. It got attention because it claims strong compression with no training and strong performance for KV cache compression and vector search.

Why it matters:

- It is part of the viral inspiration for this project.
- It demonstrates public appetite for KV compression systems.
- It is a likely compressor backend to test eventually.

ExactKV status:

- Do not make ExactKV a TurboQuant-only project.
- Treat TurboQuant as one possible compressor backend.

### NVIDIA kvpress

NVIDIA kvpress is a relevant repository for KV cache compression methods. It can serve as a source of compression implementations and baseline ideas.

Why it matters:

- It may reduce the effort required to add several compressors.
- It shows that compression tooling already exists.
- ExactKV's gap is not "another compression toolkit"; the gap is verification and exactness.

ExactKV status:

- Review for integration later.
- Do not depend on it in V1.

## 4. Speculative decoding

Speculative decoding accelerates generation by using a cheaper drafter model to propose tokens and a target model to verify them.

Relevant methods:

- EAGLE
- EAGLE-2
- Medusa-style heads
- Multi-token prediction
- n-gram speculative decoding
- small-model draft verification

Classic speculative decoding structure:

1. Drafter proposes tokens.
2. Target model verifies.
3. Accepted prefix is committed.
4. Mismatches are corrected.

ExactKV is similar in structure but different in source of approximation.

Traditional speculative decoding uses:

> Different model, same KV idea.

VeriCache-style verification uses:

> Same model, compressed KV.

This matters because compressed KV often preserves the target model's weights and dominant attention patterns. This can allow longer accepted runs than a separate small draft model.

## 5. Sparse or compressed self-speculation

Some prior works use sparse or compressed attention as part of speculative decoding.

Relevant methods from the VeriCache paper:

- MagicDec
- QuantSpec
- SparseSpec

These are close to VeriCache but differ in important ways.

According to the VeriCache analysis:

- Prior systems often keep full KV resident in GPU memory.
- This limits the throughput and memory benefits of compression.
- Prior systems may hard-code one compressor.
- Prior systems do not address remote prefix caching in the same way.

ExactKV should study these systems carefully later, but V1 should stay focused.

## 6. Verification-based compressed KV inference

### VeriCache

VeriCache is the primary paper behind ExactKV.

Its core idea:

1. Use compressed KV to draft tokens.
2. Use full KV to verify tokens.
3. Accept matching tokens.
4. Correct mismatches.
5. Maintain identical output to full-KV decoding under greedy decoding.

Its system contributions include:

- Cross-resource staggering
- Full KV kept out of GPU memory
- Extended verification periods
- Uniform compressor interface
- Support for long-context decoding and remote prefix caching
- Composition with traditional speculative decoding

ExactKV's relationship:

- ExactKV is inspired by VeriCache.
- ExactKV should not claim the idea as original.
- ExactKV should initially build a simpler educational and practical implementation.
- ExactKV should later generalize into a compressor benchmark suite.

## What exists

| Area | Exists today | ExactKV stance |
|---|---|---|
| Full LLM serving engines | vLLM, SGLang, TensorRT-LLM | Do not duplicate |
| KV memory management | PagedAttention, vAttention | Build on later |
| KV cache storage and reuse | LMCache, CacheGen-style systems | Integrate later |
| KV quantization | KIVI, KVQuant, TurboQuant, others | Use as backends |
| KV token dropping | SnapKV, KVzip, KVzap, others | Use as backends |
| Speculative decoding | EAGLE, MTP, n-gram, others | Compose later |
| Verified compressed KV inference | VeriCache paper | Implement and productize |

## What does not appear fully solved

The likely open-source gap is:

> A clean, educational, compressor-agnostic, Hugging Face-first implementation and benchmark suite for verified KV-cache compression.

Specifically, ExactKV can own:

- Simple API for verified compressed KV generation
- Compressor abstraction for experiments
- Acceptance-rate benchmarking
- Exact-output test harness
- Clear diagrams and docs
- Reproducible small-model demos
- Later vLLM and LMCache integration
- A public leaderboard-style comparison of compressors by acceptance behavior

## Competitive positioning

ExactKV should not compete head-on with:

- vLLM as a serving engine
- LMCache as a KV storage layer
- kvpress as a compression toolkit
- KIVI or TurboQuant as individual compressors

ExactKV should compete in a narrower and clearer category:

> Verification layer and benchmark harness for lossy KV compressors.

## Related repositories to inspect

Cursor should eventually inspect or the developer should manually review:

- vllm-project/vllm
- LMCache/LMCache
- NVIDIA/kvpress
- jy-yuan/KIVI
- KVQuant implementations
- TurboQuant community implementations
- SnapKV repositories
- EAGLE repositories
- SGLang repositories

Do not integrate any of these before Phase 1.

## Related papers to prioritize

### Primary

- VeriCache: Turning Lossy KV Cache into Lossless LLM Inference

### Serving and cache management

- Efficient Memory Management for Large Language Model Serving with PagedAttention
- LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference
- vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention
- Mooncake: KV-cache-centric disaggregated serving
- DistServe
- Splitwise

### Compression

- KIVI
- KVQuant
- TurboQuant
- KVTC
- CacheGen
- RotateKV
- SnapKV
- KVzip
- KVzap
- DuoAttention
- PyramidKV

### Speculative decoding

- EAGLE
- EAGLE-2
- FastMTP
- MagicDec
- QuantSpec
- SparseSpec

## Conclusion

The ecosystem already has many compressors and serving systems.

ExactKV should not try to be another generic compressor.

The wedge is:

> Make lossy KV compression safe and measurable through full-KV verification.

This is narrow enough to build and broad enough to matter.