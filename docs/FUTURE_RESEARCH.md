# 15_FUTURE_RESEARCH.md

# ExactKV Future Research

## Purpose of this document

This document captures research ideas that are important but not part of the early implementation.

These ideas should not distract from V1.

The rule:

> Future research starts only after the correctness prototype and benchmark suite are stable.

---

# Research direction 1: Acceptance-optimized compressors

## Idea

Most KV compressors optimize direct lossy inference quality.

ExactKV needs a different objective:

```text
maximize accepted draft length under full-KV verification
```

A compressor that is mediocre for direct serving may be excellent for ExactKV if it preserves argmax tokens over long draft horizons.

## Research question

Can we design a KV compressor specifically optimized for verification acceptance rather than semantic similarity?

## Possible objective

```text
maximize E[accepted_tokens_per_verification]
subject to memory budget
```

## Why interesting

This could become ExactKV's own novel research contribution.

---

# Research direction 2: Adaptive draft length

## Idea

The optimal draft length depends on:

- prompt
- model
- compressor
- compression ratio
- generation stage
- recent acceptance history

Static draft length is simple but suboptimal.

## Research question

Can ExactKV adapt draft length online?

## Simple policy

```text
if last rounds had high acceptance:
    increase draft_len
else:
    decrease draft_len
```

## Advanced policy

Use a predictor trained on trace features.

Features:

- recent acceptance rate
- entropy of compressed logits
- KL estimate
- prompt category
- compression ratio
- layer statistics

## Expected benefit

Less wasted draft work and lower verification overhead.

---

# Research direction 3: Acceptance prediction

## Idea

Before verifying, estimate whether the compressed draft is likely to match full KV.

## Possible signals

- compressed logits confidence
- margin between top-1 and top-2 tokens
- entropy
- compressor metadata
- previous mismatch pattern
- token type
- position in sequence

## Research question

Can we predict mismatch before paying full verification cost?

## Use cases

- early verification
- dynamic draft length
- fallback to full KV
- compression ratio adjustment

---

# Research direction 4: Prompt-aware compression

## Idea

Some prompts are more sensitive to compression than others.

Examples:

- Code generation may be fragile.
- JSON may be fragile.
- Open-ended prose may tolerate more compression.

## Research question

Can ExactKV choose compression settings based on prompt type?

## Possible policy

```text
if prompt requires JSON or code:
    mild compression
else:
    aggressive compression
```

## Required benchmark

Prompt category vs acceptance behavior.

---

# Research direction 5: Hybrid speculative decoding plus ExactKV

## Idea

VeriCache can compose with traditional speculative decoding.

That means:

```text
small model drafts
compressed KV drafts
full KV verifies
```

or:

```text
small model proposes candidate tree
compressed KV filters
full KV verifies final path
```

## Research question

Can small-model speculation and compressed-KV verification stack multiplicatively?

## Risk

Complexity increases quickly.

## Not before

V5 or later.

---

# Research direction 6: Sampling-compatible ExactKV

## Idea

V1 uses greedy decoding. Real serving often uses sampling.

The VeriCache paper notes sampling can be handled with rejection-sampling-style methods.

## Research question

Can ExactKV preserve the exact sampling distribution of full-KV decoding while using compressed-KV drafting?

## Challenges

- random seeds
- probability correction
- rejection sampling
- numerical stability
- distributional equality rather than token equality

## Potential value

Makes ExactKV usable for creative generation and chat settings.

---

# Research direction 7: Parallel verification in Hugging Face

## Idea

Sequential verification is simple but slow.

Parallel verification verifies a draft span in one forward pass.

## Research question

Can we implement correct parallel verification over Hugging Face cache states?

## Challenges

- position IDs
- attention masks
- cache positions
- model-specific behavior
- bonus token correctness

## Value

Major speed improvement.

---

# Research direction 8: CPU offload and async transfer

## Idea

To reproduce VeriCache's systems benefit, full KV should live outside GPU memory and be loaded only for verification.

## Research question

Can ExactKV overlap compressed drafting with full-KV transfer?

## Needed tools

- pinned CPU memory
- CUDA streams
- async transfer
- careful synchronization
- memory accounting

## Value

This is where ExactKV becomes a real runtime optimization.

---

# Research direction 9: vLLM scheduler integration

## Idea

VeriCache's true speedup depends on request scheduling and cross-resource staggering.

## Research question

Can ExactKV integrate with vLLM's scheduler and paged KV layout?

## Challenges

- PagedAttention layout
- block tables
- request batching
- scheduler hooks
- memory allocation
- external cache movement

## Value

Production relevance.

---

# Research direction 10: LMCache integration

## Idea

LMCache provides KV storage and movement.

ExactKV could use LMCache to store full KV outside GPU and retrieve it for verification.

## Research question

Can LMCache serve as ExactKV's full-KV backing store?

## Value

Avoids rebuilding KV movement infrastructure.

---

# Research direction 11: Remote prefix caching

## Idea

VeriCache supports remote prefix caching where compressed KV is sent to remote GPUs and full KV is verified near storage.

## Research question

Can ExactKV prototype this in a small simulated environment?

## Needed setup

- local verifier process
- remote drafter process
- simulated slow link
- compressed KV transfer
- full KV verification

## Value

Very strong systems demo, but too complex for early versions.

---

# Research direction 12: Verification beyond KV compression

## Idea

Other approximate cache techniques produce drift too.

Examples:

- approximate prefix reuse
- CacheBlend-like non-prefix reuse
- retrieval-fused caches
- approximate context merging

## Research question

Can draft-then-verify restore exactness for approximate cache reuse beyond compression?

## Value

Could broaden ExactKV beyond KV compression.

---

# Research direction 13: Compressor leaderboard

## Idea

Create a public leaderboard ranking compressors by ExactKV metrics.

Metrics:

- acceptance rate
- average accepted length
- compression ratio
- speedup after verification
- exactness
- structured-output robustness

## Value

Makes ExactKV a community resource.

## Risk

Requires careful reproducibility.

---

# Research direction 14: Task-specific acceptance analysis

## Idea

Analyze which tasks cause low acceptance.

Categories:

- prose
- code
- JSON
- tool calls
- math
- long-context QA
- adversarial prompts

## Research question

Which workloads are compression-sensitive?

## Value

Useful for production policy decisions.

---

# Research direction 15: Layer-wise and head-wise sensitivity

## Idea

Not all layers and heads matter equally for exact token agreement.

## Research question

Can we identify which layers or heads are most important for acceptance length?

## Use

Design better compressors.

Possible output:

- layer sensitivity heatmaps
- head sensitivity heatmaps
- accepted length impact by layer

---

# Research direction 16: Error-localizing compression

## Idea

When mismatch occurs, identify which compressed components caused the wrong token.

## Research question

Can ExactKV attribute mismatch to specific layers, heads, or token positions?

## Value

Could guide compressor design.

---

# Research direction 17: Fallback policies

## Idea

If acceptance is poor, ExactKV should fall back to full KV.

## Research question

What is the best policy for fallback?

Possible policies:

- fallback after N low-acceptance rounds
- fallback for certain prompt categories
- reduce compression ratio
- reduce draft length
- switch compressor

## Value

Production robustness.

---

# Research direction 18: Multi-compressor routing

## Idea

Different compressors may work better for different workloads.

## Research question

Can ExactKV choose among compressors online?

Example:

```text
code prompt → KIVI mild compression
prose prompt → token dropping
long context → TurboQuant
```

## Value

Better speed and acceptance tradeoffs.

---

# Research direction 19: ExactKV for structured-output agents

## Idea

Agents often need exact JSON, tool calls, and state transitions.

## Research question

Can ExactKV reduce tool-call failures caused by lossy KV compression?

## Benchmark

- synthetic tool calls
- ComplexFuncBench
- OpenAI-style function calling prompts
- JSON schema tasks

## Value

Strong product story.

---

# Research direction 20: ExactKV safety mode

## Idea

Allow users to choose reliability levels.

Modes:

```text
off: lossy compression only
safe: ExactKV with fallback
strict: full exactness required
adaptive: policy-based
```

## Value

Makes ExactKV production configurable.

---

# Research direction 21: Visualization tools

## Idea

Visualize verification traces.

Plots:

- token acceptance timeline
- mismatch positions
- accepted length histogram
- draft vs verifier tokens
- compressor comparison

## Value

Great for debugging and demos.

---

# Research direction 22: ExactKV with long-context coding agents

## Idea

Test ExactKV on repository-level coding contexts.

## Research question

Does verified compression preserve code patch correctness better than lossy compression?

## Potential benchmark

- small synthetic repos
- SWE-bench Lite subset
- RepoBench-style prompts

## Value

Very strong demo if successful.

---

# Research direction 23: ExactKV for command safety

## Idea

The VeriCache paper gives shell-command examples where tiny changes can be dangerous.

## Research question

Can ExactKV prevent command drift under lossy KV compression?

## Benchmark

Prompts requiring exact shell commands.

Metrics:

- exact command match
- safety rule match
- invalid command rate

---

# Research direction 24: Hardware-aware ExactKV policy

## Idea

Optimal settings depend on hardware.

Factors:

- HBM bandwidth
- PCIe bandwidth
- CPU memory bandwidth
- GPU memory size
- model size
- KV size

## Research question

Can ExactKV choose compression and draft length based on hardware profile?

## Value

Production deployment guidance.

---

# Research direction 25: ExactKV simulator

## Idea

Build a simulator for VeriCache-style scheduling before implementing full vLLM integration.

Inputs:

- model size
- KV size
- compression ratio
- acceptance rate
- HBM bandwidth
- interconnect bandwidth
- batch size

Outputs:

- predicted throughput
- bottleneck resource
- optimal draft length
- optimal compression ratio

## Value

Helps reason about performance before low-level systems work.

---

# Research backlog priority

## High priority after V3

1. Adaptive draft length
2. Parallel verification
3. Advanced compressor adapters
4. Acceptance-optimized compressor objective
5. Benchmark leaderboard

## Medium priority

1. CPU offload
2. LMCache integration
3. vLLM prototype
4. structured-output benchmark
5. visualization tools

## Long-term

1. remote prefix caching
2. sampling-compatible ExactKV
3. multi-compressor routing
4. hardware-aware scheduler
5. approximate cache reuse verification

---

# Final research principle

ExactKV's best future research direction is not merely making compression smaller.

It is understanding and optimizing:

```text
what makes compressed KV agree with full KV for as long as possible
```

That is the research lens that should guide future work.
