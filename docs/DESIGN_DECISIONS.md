# 06_DESIGN_DECISIONS.md

# ExactKV Design Decisions

## Purpose of this document

This document records major design decisions for ExactKV.

Every important design choice should state:

- Decision
- Why it was chosen
- Alternatives considered
- Why alternatives were rejected
- Consequences
- Version where it applies

This file exists so the project does not depend on undocumented reasoning.

## Decision 001: Build ExactKV as a verification framework, not a compressor

### Decision

ExactKV will be built primarily as a verification runtime and benchmark suite for KV-cache compressors, not as a new compressor.

### Why

The ecosystem already contains many KV compressors:

- KIVI
- KVQuant
- TurboQuant
- SnapKV
- KVzip
- KVzap
- RotateKV
- kvpress compressors

The more useful gap is a system that measures and restores exactness.

### Alternatives

1. Build a new KV quantization method.
2. Implement TurboQuant only.
3. Implement VeriCache exactly as a production serving system.

### Why rejected

A new compressor is harder to differentiate and likely requires custom kernels.

A TurboQuant-only implementation is too narrow.

A full VeriCache serving system is too large for an initial project.

### Consequence

ExactKV must have a compressor abstraction from early versions.

## Decision 002: Start with Hugging Face Transformers, not vLLM

### Decision

Phase 1 uses Hugging Face Transformers.

### Why

Hugging Face provides the simplest path to:

- Load small models
- Inspect `past_key_values`
- Implement greedy generation
- Validate exactness
- Iterate quickly

### Alternatives

1. Start directly in vLLM.
2. Start in TensorRT-LLM.
3. Start in SGLang.

### Why rejected

vLLM has complex scheduler and paged-cache internals.

TensorRT-LLM is too optimized and low-level for early experimentation.

SGLang is not necessary for proving the core concept.

### Consequence

V1 may not be fast.

That is acceptable.

## Decision 003: Greedy decoding only in early versions

### Decision

V1 and V2 support greedy decoding only.

### Why

The core exactness claim is easiest to define under deterministic decoding:

```text
do_sample = False
temperature = 0
```

The VeriCache paper defines identical outputs under greedy decoding except for hardware nondeterminism.

### Alternatives

1. Support temperature sampling.
2. Support top-p sampling.
3. Support beam search.

### Why rejected

Sampling introduces randomness and rejection-sampling complexity.

Beam search adds multiple hypotheses and cache branching.

Both distract from the first correctness goal.

### Consequence

Initial ExactKV is limited but well-defined.

Sampling support is future work.

## Decision 004: Correctness before performance

### Decision

The first implementation optimizes for correctness, readability, and traceability.

### Why

If ExactKV does not exactly match full-KV output, the project fails regardless of speed.

### Alternatives

1. Start with Triton kernels.
2. Start with async CPU/GPU transfer.
3. Start with vLLM scheduler changes.

### Why rejected

These make debugging harder and obscure correctness errors.

### Consequence

V1 may not show speedup.

The launch should be honest if V1 is a correctness and benchmarking MVP.

## Decision 005: ExactKV output must equal full-KV output

### Decision

Under deterministic decoding, ExactKV output IDs must exactly equal full-KV output IDs.

### Why

This is the core guarantee.

### Alternatives

1. Compare decoded text only.
2. Compare semantic similarity.
3. Compare task score only.

### Why rejected

Decoded text can hide tokenization differences.

Semantic similarity is too weak.

Task score can hide silent divergence.

### Consequence

Tests must compare token IDs, not just strings.

## Decision 006: Full KV is the source of truth

### Decision

When compressed KV and full KV disagree, the full-KV token is committed.

### Why

ExactKV is designed to preserve full-KV behavior.

### Alternatives

1. Trust compressed KV if confidence is high.
2. Use a voting system.
3. Use a learned verifier.

### Why rejected

These break the exactness guarantee.

### Consequence

The verification engine must always commit full-KV corrections on mismatch.

## Decision 007: Use a simple INT8 compressor first

### Decision

The first compressor should be a simple INT8 KV quantizer.

### Why

INT8 is easy to reason about and implement.

The goal is to test the runtime, not win compression benchmarks.

### Alternatives

1. INT4 first.
2. TurboQuant first.
3. KIVI first.
4. Token dropping first.

### Why rejected

INT4 introduces packing and numerical complexity.

TurboQuant is more complex.

KIVI has specialized quantization choices.

Token dropping changes shapes and attention patterns.

### Consequence

V1 compressor may not be impressive. That is fine.

## Decision 008: Add INT4 after INT8

### Decision

INT4 should be the next simple quantization backend.

### Why

INT4 gives a more meaningful compression ratio and will likely create more visible divergence.

It is useful for acceptance-rate experiments.

### Alternatives

1. Jump straight to KIVI.
2. Jump straight to kvpress.
3. Implement token dropping next.

### Why rejected

The project needs a controlled progression.

### Consequence

INT4 should be implemented in a straightforward, readable way before optimizing bit-packing.

## Decision 009: Token dropping is included after basic quantization

### Decision

A simple token-dropping compressor should be added after INT8/INT4.

### Why

Token dropping represents a different compression class.

ExactKV must support both quantization and token dropping eventually.

### Alternatives

1. Focus only on quantization.
2. Integrate SnapKV immediately.

### Why rejected

Quantization-only would weaken the compressor-agnostic thesis.

SnapKV integration may be too complex for early versions.

### Consequence

The compressor interface must support shape-changing compressors eventually.

## Decision 010: No CPU offload in Phase 1

### Decision

Phase 1 keeps full KV and compressed KV on the same device.

### Why

CPU offload introduces transfer, pinned memory, async scheduling, and synchronization complexity.

### Alternatives

1. Full KV on CPU from day one.
2. LMCache integration from day one.

### Why rejected

Too much systems complexity before correctness is proven.

### Consequence

Phase 1 will not reproduce VeriCache's main systems benefit.

It will reproduce the algorithmic behavior.

## Decision 011: No batching in Phase 1

### Decision

Phase 1 supports one request at a time.

### Why

Batching complicates cache offsets, sequence lengths, and verification traces.

### Alternatives

1. Batch multiple prompts from the start.
2. Implement request scheduler early.

### Why rejected

Batching is not needed to prove exactness.

### Consequence

Throughput numbers from Phase 1 are not production-serving numbers.

## Decision 012: Trace every verification round

### Decision

ExactKV should produce a structured trace for every draft and verify round.

### Why

This makes the system debuggable and makes benchmark results interpretable.

### Alternatives

1. Only return final output.
2. Log human-readable strings only.

### Why rejected

Final output hides why the system accepted or rejected tokens.

Unstructured logs are hard to analyze.

### Consequence

The result object should include machine-readable traces.

## Decision 013: Use token ID equality, not text equality

### Decision

Exactness means exact output token ID sequence equality.

### Why

Tokenization can produce the same text from different IDs in some edge cases.

### Alternatives

1. Compare decoded text.
2. Compare normalized strings.

### Why rejected

The model operates over token IDs.

ExactKV should preserve model-level behavior.

### Consequence

Benchmark reports can include text equality, but token equality is primary.

## Decision 014: Treat hardware nondeterminism as a known issue

### Decision

ExactKV should document and control for nondeterminism.

### Why

Even full-KV inference can show small nondeterminism depending on kernels, device, dtype, and backend.

### Mitigations

- Use greedy decoding.
- Use fixed seeds where applicable.
- Use deterministic settings where possible.
- Compare within the same process and backend.
- Start with small models and stable dtypes.

### Consequence

Exactness tests should be carefully designed.

## Decision 015: Do not claim production readiness early

### Decision

V1 and V2 should be described as research/runtime prototypes.

### Why

Production readiness requires:

- optimized cache layout
- efficient verification
- batching
- async transfers
- fallback policies
- integration with serving frameworks
- robust model support

### Consequence

Public messaging should be technically impressive but honest.

## Decision 016: Separate benchmark modes clearly

### Decision

Benchmarks must always distinguish:

```text
full
lossy
exactkv
```

### Why

The value of ExactKV is visible only when compared against both baselines.

### Definitions

- Full: normal full-KV generation.
- Lossy: compressed KV generation without verification.
- ExactKV: compressed KV draft plus full-KV verification.

### Consequence

Benchmark reports must not collapse these into one comparison.

## Decision 017: Start with Qwen2.5-0.5B or TinyLlama

### Decision

The first target model should be small.

### Why

Small models allow fast iteration and low hardware requirements.

### Alternatives

1. Start with Llama 8B.
2. Start with Qwen 7B.
3. Start with 32B models from the paper.

### Why rejected

Large models slow iteration and require better hardware.

### Consequence

V1 claims should avoid extrapolating to large-model serving.

## Decision 018: Version docs must define what is new and why

### Decision

Every version must specify:

- Features added
- Why they are added
- What is intentionally excluded
- Success criteria
- Exit tests

### Why

The project is complex and otherwise prone to scope creep.

### Consequence

`07_VERSION_ROADMAP.md` is a controlling document.

## Decision 019: ExactKV should integrate with existing projects later

### Decision

ExactKV should eventually integrate rather than rewrite:

- vLLM
- LMCache
- kvpress
- KIVI

### Why

The goal is to own the verification layer, not rebuild the entire inference ecosystem.

### Consequence

Interfaces should be designed with adapters in mind.

## Decision 020: The first viral post should be based on evidence, not ambition

### Decision

The first public launch should only claim numbers the repo can reproduce.

### Why

AI systems audiences are skeptical. Overclaiming kills credibility.

### Good claim

> ExactKV produces identical outputs to full KV while showing acceptance-rate and mismatch traces for lossy KV compressors.

### Better claim if proven

> ExactKV achieves X throughput improvement with exact full-KV outputs on Y model and Z benchmark.

### Bad claim

> ExactKV solves KV compression.

## Decision log maintenance

Future contributors should append new decisions in this format:

```markdown
## Decision NNN: Title

### Decision

### Why

### Alternatives

### Why rejected

### Consequence
```
