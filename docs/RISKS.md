# 10_RISKS.md

# ExactKV Risks

## Purpose of this document

This document identifies risks that could cause ExactKV to fail technically, strategically, or reputationally.

The project is ambitious. The biggest risk is not that it cannot be coded. The biggest risk is building something that looks impressive but does not actually preserve exactness or produce useful measurements.

---

# Risk severity scale

```text
Low: annoying but manageable
Medium: could slow the project
High: could invalidate a milestone
Critical: could invalidate the project thesis
```

---

# Risk 1: Verification implementation is off by one

## Severity

Critical

## Description

The verification engine may compare each draft token to the wrong full-KV prediction position.

This is the most likely serious correctness bug.

## Symptoms

- Acceptance rate is unexpectedly near zero.
- ExactKV output diverges from full output.
- No-op compressor fails to produce 100% acceptance.
- Mismatch appears at token 0 for many prompts.

## Mitigation

- Build a no-op compressor that returns full KV unchanged.
- With no-op compressor, acceptance must be 100%.
- Compare custom full greedy loop against `model.generate`.
- Write unit tests for middle mismatch, first-token mismatch, and all-match cases.
- Trace token IDs and decoded strings for every round.

---

# Risk 2: Cache state gets corrupted by rejected draft tokens

## Severity

Critical

## Description

Rejected tokens may accidentally remain in full or compressed cache after a mismatch.

## Symptoms

- ExactKV output matches full for early tokens but diverges later.
- Cache sequence length grows too fast.
- Accepted trace looks correct but final output is wrong.

## Mitigation

- Never mutate authoritative state during speculative draft verification unless rollback is proven safe.
- Use temporary full states in V1.
- Add cache length assertions after every round.
- Recompress from full KV after commit in V1 if needed.

---

# Risk 3: ExactKV is slower than full KV

## Severity

High for runtime claims, low for benchmark claims

## Description

Naive verification may be expensive enough that ExactKV is slower than full-KV inference.

## Why likely

V1 has:

- sequential verification
- no offload
- no async overlap
- no vLLM scheduler
- possible recompression overhead

## Mitigation

- Do not claim speedup in V1.
- Position V1 as correctness and benchmark harness.
- Report speed honestly.
- Optimize only after exactness works.
- Track verification overhead separately.

---

# Risk 4: Compressor acceptance rate is too low

## Severity

High

## Description

If compressed KV frequently disagrees with full KV, verification accepts few tokens and performance is poor.

## Symptoms

- Average accepted length close to 0 or 1.
- Most rounds mismatch at index 0.
- Increasing draft length does not help.

## Mitigation

- Test multiple compressors.
- Sweep compression ratios.
- Start with mild compression.
- Add adaptive draft length.
- Explore acceptance-optimized compression later.

---

# Risk 5: INT8 is too accurate to demonstrate rejection behavior

## Severity

Medium

## Description

Simple INT8 may match full KV so often that rejection paths are not well-tested.

## Mitigation

- Add debug_noise compressor.
- Add INT4 simulation.
- Add forced mismatch unit tests.
- Use longer generations and more sensitive prompts.

---

# Risk 6: INT4 implementation becomes a bit-packing distraction

## Severity

Medium

## Description

Implementing efficient INT4 storage may distract from verification logic.

## Mitigation

- In V2, simulate INT4 numerically using int8 storage.
- Do not bit-pack until performance versions.
- Document that simulated INT4 measures acceptance, not true memory efficiency.

---

# Risk 7: Hugging Face cache internals vary across models

## Severity

High

## Description

Different models represent `past_key_values` differently.

Some use:

- tuples
- dynamic cache classes
- static cache classes
- grouped-query attention
- multi-query attention
- different position ID behavior

## Mitigation

- Start with one known model.
- Write model-specific adapter.
- Avoid claiming broad model support.
- Add cache introspection utilities.
- Use explicit compatibility matrix.

---

# Risk 8: Hardware nondeterminism breaks exactness tests

## Severity

Medium

## Description

GPU kernels may produce nondeterministic outputs in edge cases.

## Symptoms

- Full baseline differs across runs.
- ExactKV sometimes matches and sometimes does not.
- Differences occur only on GPU.

## Mitigation

- Use greedy decoding.
- Use deterministic settings where possible.
- Compare within the same process and same model instance.
- Test on CPU for small cases if necessary.
- Document hardware nondeterminism.

---

# Risk 9: The project overclaims novelty

## Severity

High

## Description

The draft-and-verify compressed KV idea comes from VeriCache. ExactKV should not imply it invented the concept.

## Mitigation

- Always cite VeriCache as the core research basis.
- Position ExactKV as implementation, productization, benchmark suite, and extension.
- Be precise in README and launch posts.

Good wording:

> Inspired by VeriCache, ExactKV implements a compressor-agnostic verified KV-cache runtime and benchmark suite.

Bad wording:

> I invented lossless KV compression.

---

# Risk 10: A mature VeriCache implementation appears

## Severity

Medium to High

## Description

If the VeriCache authors or another team release a polished implementation, ExactKV may lose uniqueness.

## Mitigation

Differentiate ExactKV as:

- educational implementation
- Hugging Face-first prototype
- benchmark harness
- compressor leaderboard
- simple API
- research playground

Even if a production implementation exists, a clean benchmark suite can remain useful.

---

# Risk 11: External compressor integrations are harder than expected

## Severity

Medium

## Description

KIVI, kvpress, SnapKV, or TurboQuant implementations may have incompatible assumptions.

## Mitigation

- Use adapters.
- Keep V1 independent.
- Treat external integrations as V4+.
- Document limitations per adapter.
- Avoid promising integrations before testing.

---

# Risk 12: Memory metrics are misleading

## Severity

High

## Description

PyTorch memory measurements can be misleading due to caching allocator behavior.

Also, storing both full and compressed KV can reduce active GPU memory but increase total memory.

## Mitigation

Report separately:

- estimated full KV bytes
- estimated compressed KV bytes
- active GPU KV memory
- total KV memory
- PyTorch allocated memory
- PyTorch reserved memory

Never claim total memory savings if full KV is also stored elsewhere.

---

# Risk 13: Phase 1 scope creep

## Severity

High

## Description

The project may try to include too much too early:

- vLLM
- LMCache
- TurboQuant
- Triton
- CUDA
- CPU offload
- batching
- sampling
- dashboards

## Mitigation

Strictly follow `11_NON_GOALS.md`.

V1 goal:

> Exact output equality with a simple compressor.

Nothing else.

---

# Risk 14: Cursor implements code before understanding docs

## Severity

Medium

## Description

Cursor may jump into implementation and miss project constraints.

## Mitigation

Always prompt Cursor to summarize the docs first.

Required Cursor instruction:

```text
Do not write code yet. First summarize the architecture, risks, and Phase 1 scope.
```

---

# Risk 15: Verification trace becomes too large

## Severity

Low to Medium

## Description

Detailed traces for long generations may consume memory or clutter reports.

## Mitigation

- Allow trace verbosity levels.
- Store summaries by default.
- Store full traces only in debug mode.
- Use JSONL for large traces.

---

# Risk 16: Benchmark prompts are too easy

## Severity

Medium

## Description

If prompts are short or simple, lossy compression may not diverge, making ExactKV look unnecessary.

## Mitigation

Include:

- long outputs
- structured outputs
- code prompts
- JSON prompts
- tool-call prompts
- longer context tests

---

# Risk 17: Benchmark prompts are too hard for small models

## Severity

Medium

## Description

Small models may fail structured tasks regardless of compression.

## Mitigation

- Separate model capability from exactness.
- Focus on output equality, not absolute task success, in early versions.
- Use syntax validity only where model can reasonably comply.
- Later test larger models.

---

# Risk 18: ExactKV becomes only a benchmark, not a runtime

## Severity

Medium

## Description

If speedups are never achieved, ExactKV may not become a runtime optimization.

## Mitigation

This is acceptable if positioned correctly.

The benchmark suite is still valuable.

Later runtime work should focus on:

- parallel verification
- CPU offload
- async transfer
- vLLM integration
- LMCache integration
- adaptive draft length

---

# Risk 19: Parallel verification is difficult

## Severity

High for performance versions

## Description

Efficiently verifying multiple draft tokens in one forward pass may be difficult with Hugging Face cache APIs.

## Mitigation

- Implement sequential verification first.
- Write extensive tests.
- Add parallel verification later as an optimization.
- Compare sequential and parallel outputs exactly.

---

# Risk 20: Adaptive policies become premature research

## Severity

Medium

## Description

Adaptive draft length, learned acceptance prediction, and compressor selection are exciting but can distract from the core.

## Mitigation

Move them to `15_FUTURE_RESEARCH.md`.

Do not implement before V3.

---

# Risk 21: Public launch before results are reproducible

## Severity

High

## Description

A viral post with unreproducible numbers will hurt credibility.

## Mitigation

Before launch:

- Make scripts reproducible.
- Include exact commands.
- Include model and hardware specs.
- Include limitations.
- Avoid unsupported claims.

---

# Risk 22: Confusing active GPU memory with total system memory

## Severity

High

## Description

ExactKV may reduce active GPU KV memory while storing full KV on CPU. Saying it “reduces memory” without qualification is misleading.

## Mitigation

Use precise language:

Good:

> reduces active GPU KV memory

Bad:

> reduces total memory

---

# Risk 23: Full-KV baseline implementation is wrong

## Severity

Critical

## Description

If the custom full-KV greedy loop does not match Hugging Face `model.generate`, ExactKV's ground truth may be wrong.

## Mitigation

- First implement and test full greedy generation.
- Compare against `model.generate`.
- Do this before any compression work.

---

# Risk 24: Model positional encoding issues

## Severity

High

## Description

RoPE position IDs and cache positions may be mishandled during draft or verification.

## Mitigation

- Use model APIs carefully.
- Inspect generated cache positions.
- Start with short prompts.
- Add tests comparing one-step custom generation to `model.generate`.

---

# Risk 25: Too many docs and no code

## Severity

Medium

## Description

The project could become over-specified and under-built.

## Mitigation

Docs should feed into V1 implementation quickly.

After Batch 3, move to:

```text
implementation plan → repository scaffold → full baseline → verification tests
```

---

# Top 5 risks to watch immediately

1. Off-by-one verification
2. Cache state corruption
3. Full baseline mismatch with `model.generate`
4. Hugging Face cache compatibility
5. Scope creep into performance before correctness

---

# Final risk rule

If exactness fails, stop.

Do not optimize.

Do not benchmark speed.

Fix exactness first.
