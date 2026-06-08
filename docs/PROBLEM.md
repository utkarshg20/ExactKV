# The Problem ExactKV Solves

## Summary

LLM inference is increasingly limited by KV-cache memory and bandwidth.

KV-cache compression methods reduce this bottleneck, but almost all of them are lossy. They can preserve benchmark scores on short or open-ended tasks while still changing exact outputs. For production systems involving code, tool calls, JSON, shell commands, or long outputs, even tiny deviations can become catastrophic.

ExactKV exists to resolve this tradeoff:

> Keep the efficiency benefits of compressed KV, but preserve the exact output behavior of full-KV decoding.

## Background: what is the KV cache?

During autoregressive transformer inference, each generated token produces key and value tensors for every attention layer. These tensors are stored so future tokens can attend to previous context without recomputing the entire prompt.

This stored state is called the **KV cache**.

Inference has two major phases:

1. **Prefill**
   - Process the prompt.
   - Build the initial KV cache.

2. **Decode**
   - Generate tokens one at a time.
   - Reuse the KV cache at every step.
   - Append new keys and values as tokens are generated.

The KV cache grows with:

- Context length
- Number of layers
- Number of KV heads
- Head dimension
- Precision
- Batch size
- Number of active requests

At long context lengths, the KV cache can dominate memory and bandwidth.

## Why the KV cache becomes a bottleneck

For each decode step, the model needs to read from the cache to compute attention over previous tokens. As context length grows, each new token requires more memory traffic.

This causes several problems:

### 1. GPU memory pressure

The KV cache occupies GPU memory that could otherwise support:

- More concurrent requests
- Larger batch sizes
- Longer contexts
- Larger models

When full KV is too large, batch size collapses.

### 2. HBM bandwidth pressure

Even when the KV cache fits in memory, reading it each decode step can be bandwidth-bound. The GPU may not be compute-bound. It may be waiting on memory.

### 3. Prefix caching transfer cost

In systems that reuse KV caches across requests, precomputed KV may live on CPU, disk, object storage, or another node. Loading full KV onto a serving GPU can dominate latency.

### 4. Long-context applications amplify the problem

The problem becomes severe in:

- Repository-level code generation
- Long chat histories
- Multi-document reasoning
- Agentic workflows
- Enterprise search assistants
- Long-form structured generation
- Tool-calling workflows with many prior turns

## Existing response: compress the KV cache

Many methods compress KV cache by:

### Token dropping

Remove less important cached tokens.

Examples include:

- H2O
- StreamingLLM
- SnapKV
- PyramidKV
- DuoAttention
- KVzip
- FastKVzip
- KVzap

### Quantization

Reduce the precision of key and value tensors.

Examples include:

- KIVI
- KVQuant
- KVTC
- RotateKV
- TurboQuant
- CacheGen
- QServe
- GEAR

### Offloading and streaming

Move full or compressed KV between GPU, CPU, storage, and network tiers.

Examples include:

- LMCache
- CacheGen
- Mooncake-style KV-centric serving
- prefix caching systems

These methods can reduce memory or transfer size by multiple factors.

## The core issue: most KV compression is lossy

A compressed KV cache is not the same as the original KV cache.

Even if the compressed cache is close in some tensor-level or benchmark-level sense, it changes the model's next-token distribution.

This matters because LLM generation is autoregressive. A small change at one step can change the next token, which changes the next hidden state, which changes the next step, and so on.

The error is not just a one-time approximation error. It can compound.

## Why “similar output” is not enough

Many tasks tolerate approximate natural language:

- Summarization
- Brainstorming
- Open-ended chat
- Casual Q&A

But many production tasks do not tolerate small deviations:

### Code generation

A single wrong symbol can break syntax.

Examples:

- Missing bracket
- Wrong import
- Invalid diff format
- Wrong variable name
- One misplaced indentation token

### Tool calling

A single wrong token can break execution.

Examples:

- Wrong function name
- Wrong argument
- Invalid JSON
- Wrong enum value
- Malformed schema

### Shell commands

A small difference can be dangerous.

Examples:

- `rm -rf *.logs`
- `rm -rf * .`
- `rm -rf ./tmp/*`
- `rm -rf ./ tmp/*`

### Financial, legal, or operational workflows

Even when the text is fluent, correctness may depend on exact structured content.

## The production failure mode

Lossy KV compression may produce outputs that are:

- Fluent
- Semantically plausible
- High F1
- High ROUGE
- Low perplexity degradation
- Still functionally wrong

This creates a dangerous mismatch between benchmark quality and real reliability.

A benchmark might say:

> Quality is nearly preserved.

But a production system might experience:

> Tool call failed.
> Generated code no longer applies.
> JSON schema invalid.
> Agent took the wrong action.

ExactKV focuses on this production reliability gap.

## Why this becomes worse for long outputs

For short outputs, lossy KV may match full KV often enough.

For long outputs, divergence probability accumulates.

If a compressed-KV model has even a small per-token probability of disagreeing with full-KV decoding, the chance that an entire long sequence remains identical falls quickly.

This is especially problematic for:

- Long code generation
- Long chain-of-thought-like planning
- Multi-step tool execution
- Repository edits
- Multi-file patches
- Long structured reports
- Agents that must maintain exact state over many steps

## Existing tradeoff

Today, users often face a binary choice:

### Option A: Full KV

Pros:

- Correct relative to baseline full-KV decoding
- Reliable
- No compression-induced drift

Cons:

- Expensive
- Memory-heavy
- Lower throughput
- Worse long-context serving economics

### Option B: Compressed KV

Pros:

- Smaller memory footprint
- Higher throughput
- More requests per GPU
- Lower transfer cost

Cons:

- Lossy
- Can silently diverge
- Risky for code, tools, and structured outputs

## ExactKV's proposed third option

ExactKV aims to provide:

### Option C: Verified compressed KV

Pros:

- Uses compressed KV for cheap drafting
- Uses full KV as source of truth
- Rejects incorrect tokens
- Produces exact full-KV output under deterministic decoding
- Provides acceptance and mismatch metrics for compressors

Cons:

- Requires full KV to exist somewhere
- Verification adds overhead
- Speedup depends on acceptance rate and scheduler efficiency
- Harder than naive compression

## The key technical question

ExactKV is useful only if compressed KV produces long enough matching draft runs to amortize verification.

The central question is:

> How many compressed-draft tokens can we accept per full-KV verification pass?

If the answer is high, ExactKV can improve throughput.

If the answer is low, ExactKV becomes mainly a benchmark and safety harness.

## North-star problem statement

ExactKV solves this:

> Given a full-KV baseline and a lossy KV compressor, generate the same output as full-KV decoding while using compressed KV for most of the cheap drafting work.

## What success looks like

At minimum, ExactKV should prove:

- ExactKV output equals full-KV output under greedy decoding.
- Lossy compressed output often diverges.
- ExactKV detects and corrects divergence.
- Acceptance rate varies by compressor, compression ratio, model, prompt, and draft length.
- The framework can measure those tradeoffs.

A strong version additionally proves:

- ExactKV is faster than full-KV inference for selected workloads.
- ExactKV reduces active GPU KV memory.
- ExactKV identifies which compressors are most suitable for verified inference.
