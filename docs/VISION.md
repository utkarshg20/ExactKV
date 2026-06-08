# ExactKV Vision

## Project name

**ExactKV**

## Tagline

**Lossy KV-cache compression. Exact full-KV outputs.**

## One-sentence description

ExactKV is an open-source inference runtime and benchmark suite that lets lossy KV-cache compressors draft tokens quickly, then verifies those tokens against full-KV decoding so the final output remains identical to normal full-KV inference under deterministic decoding.

## Core idea

Modern LLM serving has a brutal tradeoff:

- **Full KV cache**
  - Correct
  - Expensive
  - High GPU memory pressure
  - Lower batch size
  - Slower long-context decoding

- **Compressed KV cache**
  - Faster
  - Smaller
  - Higher throughput
  - But lossy
  - Can silently change generated outputs

ExactKV introduces a third path:

- **Compressed KV cache drafts**
- **Full KV cache verifies**
- **Only verified tokens are committed**
- **Final output stays identical to full-KV decoding**

The practical promise is simple:

> Use aggressive KV-cache compression without trusting lossy outputs blindly.

## Why this project should exist

KV cache compression is becoming one of the most important bottlenecks in LLM inference. Long-context models, coding agents, multi-document agents, repository-level assistants, persistent chat systems, and tool-using agents all require increasingly large KV caches.

Most KV compression work focuses on average benchmark quality. That is not enough for production systems where small output deviations can break correctness.

A compressed model output can look semantically close but still fail:

- Invalid JSON
- Wrong function name
- Wrong tool argument
- Broken code syntax
- Incorrect shell command
- Invalid diff format
- One-token drift that compounds into a totally different answer

ExactKV is built around a stronger guarantee:

> If a token does not match full-KV decoding, it is rejected.

## What ExactKV is

ExactKV is intended to become a practical implementation and extension of the verified-compressed-KV idea from the VeriCache paper.

It should be:

1. **A runtime**
   - Executes generation using compressed KV draft passes and full-KV verification passes.
   - Commits only tokens that match full-KV decoding.
   - Falls back safely when compression is too lossy.

2. **A benchmark suite**
   - Measures how different KV compressors behave under verification.
   - Reports acceptance rate, accepted length, mismatch rate, throughput, memory, and exactness.
   - Helps answer which compressors are actually safe enough for codegen, tool use, and structured outputs.

3. **A compressor-agnostic framework**
   - Supports multiple compression strategies behind one interface.
   - Starts with simple INT8, INT4, and token-dropping baselines.
   - Later supports KIVI, KVQuant, TurboQuant-style quantization, SnapKV-style pruning, kvpress integrations, and other backends.

4. **A research playground**
   - Allows experimentation with draft length, compression ratio, acceptance predictors, adaptive policies, and verifier scheduling.

## What ExactKV is not

ExactKV is not trying to be:

- A new LLM model
- A new foundation model training method
- A full replacement for vLLM in Version 1
- A complete CUDA kernel project from day one
- A TurboQuant-only implementation
- A benchmark that only reports semantic similarity
- A claim that the verified-compressed-KV idea was invented here

ExactKV is explicitly a paper-to-practice systems project that turns a promising recent idea into a usable, extensible, benchmarkable open-source package.

## Inspiration

The main research inspiration is:

**VeriCache: Turning Lossy KV Cache into Lossless LLM Inference**

The paper proposes using compressed KV cache for token drafting and full KV cache for verification. Its stated goal is to preserve much of the decoding throughput advantage of KV compression while producing the same output as full-KV decoding.

ExactKV should treat VeriCache as the core research base, not as marketing decoration. The architecture, roadmap, and terminology should remain grounded in the paper.

## Project thesis

The central thesis is:

> The next useful layer in KV-cache compression is not another compressor. It is a verification and evaluation layer that makes lossy compressors safe, comparable, and production-usable.

Most current KV compression projects answer:

> How small can we make the KV cache while preserving benchmark quality?

ExactKV answers:

> How aggressively can we compress the KV cache while still producing exact full-KV outputs after verification?

That difference matters.

## Why this is a strong systems project

ExactKV is attractive because it combines four high-signal systems themes:

1. **LLM inference**
   - Directly tied to serving cost, latency, and GPU memory.

2. **KV cache management**
   - One of the key bottlenecks in long-context inference.

3. **Speculative execution**
   - Uses draft and verify logic, but applies it to compressed KV rather than a separate small model.

4. **Correctness guarantees**
   - Stronger than “similar quality” or “minimal degradation.”

A good version of this project should be legible to engineers working on:

- vLLM
- SGLang
- TensorRT-LLM
- LMCache
- inference platforms
- coding agents
- GPU serving infrastructure
- long-context systems

## Target users

### User 1: AI infra engineer

They want to know whether a KV compressor is safe enough for production. ExactKV gives them acceptance metrics and exactness tests.

### User 2: model serving team

They want to reduce KV memory and improve throughput without silent behavior drift. ExactKV gives them a verified runtime path.

### User 3: researcher

They want a harness for testing new KV compression methods under a more realistic correctness criterion. ExactKV gives them a compressor interface and benchmark suite.

### User 4: agent framework developer

They want tool calls and structured outputs to remain reliable. ExactKV gives them a path to use compression without silently corrupting tool-call behavior.

### User 5: open-source systems recruiter or reviewer

They want evidence that the builder understands inference internals, tradeoffs, benchmarks, and production constraints. ExactKV should make that obvious.

## Public positioning

The public positioning should be clear and honest:

> ExactKV is an open-source implementation and extension of verified KV-cache compression. It lets lossy KV compressors draft tokens, verifies them against full KV, and only commits tokens that match the full-KV output.

Avoid overclaiming.

Do not say:

> We invented VeriCache.

Say:

> Inspired by VeriCache, we are building a usable compressor-agnostic implementation and benchmark suite.

## Viral launch angle

The launch angle should not be abstract.

Bad:

> I implemented a paper about KV cache verification.

Good:

> Lossy KV compression is fast, but it can silently break codegen, JSON, and tool calls. I built ExactKV: compressed KV drafts tokens, full KV verifies them, and wrong tokens are rejected. Final outputs match full-KV decoding.

Best if backed by benchmark numbers:

> On Qwen, ExactKV preserved byte-identical outputs while accepting X percent of compressed-draft tokens and achieving Y times throughput over full KV.

## Long-term vision

ExactKV should become the standard open-source harness for answering:

- Which KV compressor is safest?
- Which compressor gives the longest accepted draft runs?
- Which compression ratio gives the best speed and exactness tradeoff?
- When should verification happen?
- When should the system fall back to full KV?
- Can we design compressors optimized for acceptance length rather than semantic similarity?

The long-term vision is not just a working demo.

The long-term vision is:

> A verification-first runtime and leaderboard for KV-cache compression.