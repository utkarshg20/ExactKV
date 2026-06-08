# 14_IMPLEMENTATION_PLAN.md

# ExactKV Implementation Plan

## Purpose of this document

This document gives Cursor and future contributors a concrete implementation plan.

It translates the documentation package into ordered development steps.

The plan prioritizes correctness.

---

# Implementation rule

Do not start by implementing compression.

Start by proving that the full-KV baseline is correct.

The order is:

```text
1. Full greedy baseline
2. Exact output test
3. No-op compressor
4. Verification engine
5. INT8 compressor
6. ExactKV loop
7. Metrics
8. Benchmarks
```

---

# Repository structure

Recommended initial structure:

```text
exactkv/
├── exactkv/
│   ├── __init__.py
│   ├── runtime/
│   │   ├── model_runtime.py
│   │   └── generation.py
│   ├── cache/
│   │   ├── full_state.py
│   │   ├── compressed_state.py
│   │   └── utils.py
│   ├── compressors/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── noop.py
│   │   ├── int8.py
│   │   └── debug_noise.py
│   ├── verification/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   └── acceptance.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── exactness.py
│   │   ├── acceptance.py
│   │   └── memory.py
│   └── benchmarks/
│       ├── __init__.py
│       ├── runner.py
│       └── prompts.py
├── benchmarks/
│   └── prompts/
│       └── smoke.jsonl
├── examples/
│   └── qwen_smoke.py
├── tests/
│   ├── test_full_generation.py
│   ├── test_verification_engine.py
│   ├── test_noop_exactkv.py
│   └── test_int8_exactkv.py
├── docs/
├── pyproject.toml
└── README.md
```

---

# Recommended dependencies

## Core

```text
python >= 3.10
torch
transformers
accelerate
safetensors
numpy
pydantic or dataclasses-json optional
tqdm
```

## Testing

```text
pytest
```

## Benchmarking

```text
pandas
matplotlib
```

## Optional later

```text
vllm
lmcache
triton
```

Do not include optional later dependencies in V1 unless needed.

---

# Phase 1 implementation steps

## Step 1: Create project skeleton

Create:

- package folders
- test folders
- example folders
- benchmark prompt folder
- pyproject
- README

No model logic yet.

## Step 2: Implement ModelRuntime

File:

```text
exactkv/runtime/model_runtime.py
```

Responsibilities:

- Load model
- Load tokenizer
- Set eval mode
- Encode prompt
- Decode token IDs
- Run forward pass

Suggested class:

```python
class ModelRuntime:
    def __init__(self, model_name, device="cuda", dtype="auto"):
        ...

    def encode(self, prompt: str):
        ...

    def decode(self, ids):
        ...

    def forward(self, input_ids, past_key_values=None, use_cache=True, **kwargs):
        ...
```

## Step 3: Implement custom greedy full generation

File:

```text
exactkv/runtime/generation.py
```

Implement:

```python
def generate_full_greedy(runtime, prompt, max_new_tokens):
    ...
```

This function should:

1. Encode prompt.
2. Run prefill with `use_cache=True`.
3. Iteratively generate next token by argmax.
4. Update `past_key_values`.
5. Stop on EOS or max tokens.
6. Return output IDs, text, and full state.

## Step 4: Validate full greedy generation

Test:

```text
tests/test_full_generation.py
```

Compare custom greedy loop to Hugging Face `model.generate` using:

```text
do_sample=False
num_beams=1
max_new_tokens=N
```

This must pass before moving on.

If this does not pass, do not implement ExactKV yet.

## Step 5: Define data structures

Files:

```text
exactkv/cache/full_state.py
exactkv/cache/compressed_state.py
exactkv/verification/acceptance.py
```

Implement:

- `FullKVState`
- `CompressedKVState`
- `DraftResult`
- `AcceptanceResult`
- `ExactKVResult`
- `CompressionStats`

Use dataclasses.

## Step 6: Implement NoOpCompressor

File:

```text
exactkv/compressors/noop.py
```

Purpose:

Return full KV unchanged.

Why:

The no-op compressor is the most important debugging tool.

Expected behavior:

```text
acceptance rate = 100%
ExactKV output = full output
```

If this fails, verification is wrong.

## Step 7: Implement VerificationEngine

File:

```text
exactkv/verification/engine.py
```

Start with sequential verification.

Function:

```python
def verify_sequential(runtime, full_state, draft_tokens):
    ...
```

Implementation should compare draft tokens against full-KV greedy predictions.

Start simple, even if inefficient.

## Step 8: Unit test verification logic with mocked tokens

Before using a real model, test acceptance logic with artificial sequences.

Test cases:

1. all match
2. first mismatch
3. middle mismatch
4. EOS mismatch
5. empty draft

## Step 9: Implement ExactKV loop with NoOpCompressor

File:

```text
exactkv/runtime/exactkv_generator.py
```

Use:

- full state
- no-op compressed state
- drafter
- verifier
- accept/reject logic

For NoOpCompressor, all drafts should match.

Test:

```python
full_ids == exactkv_ids
```

## Step 10: Implement simple INT8 compressor

File:

```text
exactkv/compressors/int8.py
```

Start with per-tensor symmetric quantization.

Pseudo-code:

```python
scale = tensor.abs().max() / 127
q = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
```

Store:

- q tensor
- scale
- original dtype
- shape

For materialization:

```python
dequantized = q.float() * scale
dequantized = dequantized.to(original_dtype)
```

## Step 11: Add compressed lossy generation mode

Implement:

```python
generate_lossy_greedy(runtime, prompt, compressor, max_new_tokens)
```

This directly uses compressed KV without verification.

Why:

Benchmarks require `full`, `lossy`, and `exactkv`.

## Step 12: ExactKV with INT8

Run ExactKV using INT8 compressor.

Expected:

```python
exactkv_output_ids == full_output_ids
```

Acceptance rate may be high.

## Step 13: Add debug_noise compressor

Purpose:

Force mismatches for testing.

File:

```text
exactkv/compressors/debug_noise.py
```

This compressor perturbs KV values.

Do not use for serious benchmark claims.

## Step 14: Add metrics

Files:

```text
exactkv/metrics/exactness.py
exactkv/metrics/acceptance.py
exactkv/metrics/memory.py
```

Implement:

- token exact match
- text exact match
- first divergence index
- acceptance rate
- avg accepted length
- estimated KV bytes
- compression ratio

## Step 15: Add benchmark runner

File:

```text
exactkv/benchmarks/runner.py
```

The runner should:

1. Load prompt suite.
2. Run full mode.
3. Run lossy mode.
4. Run exactkv mode.
5. Compare outputs.
6. Save JSON report.

## Step 16: Add smoke prompt suite

File:

```text
benchmarks/prompts/smoke.jsonl
```

Include 10 to 20 prompts.

Categories:

- natural
- code
- JSON
- command
- long-ish text

## Step 17: Add example script

File:

```text
examples/qwen_smoke.py
```

Should run one prompt and print:

- full output
- lossy output
- exactkv output
- exact match result
- acceptance rate
- trace summary

---

# Phase 1 acceptance checklist

Before Phase 1 is complete, all must pass:

## Full baseline

- Custom greedy full generation matches `model.generate`.

## NoOp ExactKV

- NoOp ExactKV matches full.
- Acceptance rate is 100%.

## INT8 ExactKV

- INT8 ExactKV matches full.
- Acceptance metrics are reported.

## Debug mismatch

- Debug compressor causes mismatch.
- Verification rejects wrong tokens.
- ExactKV still matches full.

## Benchmark runner

- Produces JSON report.
- Includes full, lossy, exactkv modes.

---

# Suggested Cursor implementation prompt

Use this after all docs are loaded:

```text
You are implementing ExactKV Phase 1 only.

Read docs/00_VISION.md through docs/15_FUTURE_RESEARCH.md.

Do not implement vLLM, LMCache, Triton, CUDA, CPU offload, batching, sampling, or advanced compressors.

Implement only:
1. Hugging Face ModelRuntime
2. Custom full greedy generation
3. Full baseline vs model.generate validation
4. Data structures
5. NoOpCompressor
6. Sequential VerificationEngine
7. ExactKVGenerator with NoOp
8. INT8Compressor
9. Lossy generation mode
10. Basic metrics
11. Smoke benchmark runner

Prioritize correctness over performance.

After each step, add tests.
```

---

# Implementation order for Cursor

Cursor should not implement all files at once.

Recommended order:

1. Skeleton
2. ModelRuntime
3. Full greedy baseline
4. Baseline tests
5. Data structures
6. NoOpCompressor
7. Acceptance logic unit tests
8. VerificationEngine
9. ExactKVGenerator with NoOp
10. INT8Compressor
11. Metrics
12. Benchmark runner
13. Example script

---

# Known tricky implementation areas

## Hugging Face cache format

Some models use a `Cache` object rather than tuple-style `past_key_values`.

Mitigation:

- Start with one model.
- Inspect actual output structure.
- Write utility functions.

## Position IDs

Incorrect position IDs can break generation.

Mitigation:

- Rely on model's built-in cache behavior where possible.
- Test full loop against `model.generate`.

## Updating compressed cache

V1 can recompress from full state after each verification round.

This is slow but safe.

## Verifying without corrupting full state

Use temporary state or recomputation.

Do not mutate authoritative full state with rejected tokens.

---

# Minimum viable public demo

The first demo should print:

```text
Prompt:
...

Full KV output:
...

Lossy compressed output:
...

ExactKV output:
...

ExactKV matches full: True
Lossy matches full: False or True
Acceptance rate: ...
Average accepted length: ...
First lossy divergence: ...
```

This is enough to explain the project.

---

# Phase 2 preview

After V1 works:

- Add INT4 simulated compressor.
- Improve compressor abstraction.
- Add more metrics.
- Add CSV reports.
- Add plots.
- Add structured-output validation.

Do not start Phase 2 before V1 exactness is stable.

---

# Final implementation principle

Build the smallest thing that proves:

```text
compressed KV can be wrong
full KV can verify it
ExactKV can correct it
final output equals full KV
```

Everything else comes later.
