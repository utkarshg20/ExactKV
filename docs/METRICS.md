# 08_METRICS.md

# ExactKV Metrics

## Purpose of this document

This document defines the metrics ExactKV must collect, report, and optimize.

ExactKV is both:

1. A verified generation runtime.
2. A benchmark suite for KV-cache compressors.

The metrics must therefore answer two different questions:

```text
Runtime question:
Does ExactKV generate the same output as full-KV decoding, and does it do so faster or with less active KV memory?

Compressor question:
How useful is this compressor as a drafter under full-KV verification?
```

The most important principle:

> Exactness is mandatory. Speedup is valuable only after exactness is proven.

---

# Metric hierarchy

ExactKV metrics are organized into six groups:

1. Correctness metrics
2. Acceptance metrics
3. Performance metrics
4. Memory metrics
5. Compressor metrics
6. Task-specific metrics

Each group serves a different purpose.

---

# 1. Correctness metrics

Correctness metrics are the highest-priority metrics.

If these fail, all performance numbers are irrelevant.

## 1.1 Token exact match

### Definition

Whether ExactKV output token IDs exactly equal full-KV output token IDs.

```python
token_exact_match = exactkv_output_ids == full_output_ids
```

### Type

Boolean.

### Required

Yes.

### Why it matters

The central claim of ExactKV is exact output equivalence under deterministic greedy decoding.

Decoded text equality is not enough. Compare token IDs.

### Expected value

For all successful ExactKV runs:

```python
token_exact_match == True
```

## 1.2 Text exact match

### Definition

Whether decoded ExactKV text equals decoded full-KV text.

```python
text_exact_match = exactkv_text == full_text
```

### Type

Boolean.

### Required

Yes, but secondary.

### Why it matters

Useful for human-readable reports.

### Warning

Text match is secondary because different token sequences can sometimes decode to the same text.

## 1.3 Lossy token exact match

### Definition

Whether direct lossy compressed-KV output equals full-KV output.

```python
lossy_token_exact_match = lossy_output_ids == full_output_ids
```

### Type

Boolean.

### Required

Yes.

### Why it matters

Shows how often the compressor would have silently matched full KV without verification.

This helps quantify the value of ExactKV.

## 1.4 First output divergence position

### Definition

The first token index where two outputs differ.

```python
first_divergence_idx = min(i for i if output_a[i] != output_b[i])
```

If no divergence exists, return `None`.

### Required comparisons

- lossy vs full
- exactkv vs full

### Expected result

```text
lossy vs full: may diverge
exactkv vs full: should never diverge
```

## 1.5 Exactness failure count

### Definition

Number of benchmark prompts where ExactKV output differs from full output.

```python
exactness_failures = count(not token_exact_match)
```

### Expected value

```python
exactness_failures == 0
```

If this is nonzero, do not report speedups as valid.

---

# 2. Acceptance metrics

Acceptance metrics are the core ExactKV-specific metrics.

They answer:

> How good is the compressed KV cache as a drafter?

## 2.1 Drafted tokens

### Definition

Total number of tokens proposed by compressed-KV drafting.

```python
total_drafted_tokens = sum(round.drafted_count for round in trace)
```

## 2.2 Accepted drafted tokens

### Definition

Number of drafted tokens accepted because they matched full-KV verification.

```python
accepted_drafted_tokens = sum(round.accepted_count for round in trace)
```

## 2.3 Rejected drafted tokens

### Definition

Number of drafted tokens rejected after mismatch.

```python
rejected_drafted_tokens = total_drafted_tokens - accepted_drafted_tokens
```

## 2.4 Acceptance rate

### Definition

Fraction of drafted tokens accepted.

```python
acceptance_rate = accepted_drafted_tokens / total_drafted_tokens
```

### Interpretation

Higher is better.

### Why it matters

This is the main indicator of whether a compressor is useful for ExactKV.

A compressor with high semantic quality but low acceptance rate may be poor for ExactKV.

## 2.5 Average accepted length per verification

### Definition

Average number of drafted tokens accepted in each verification round.

```python
avg_accepted_len = mean(round.accepted_count for round in trace)
```

### Why it matters

This is more operationally meaningful than raw acceptance rate.

If the system verifies every 16 tokens but accepts only 1, it will likely be slow.

If it accepts 12, verification is well amortized.

## 2.6 Median accepted length

### Definition

Median accepted tokens per verification round.

### Why it matters

Average may be skewed by a few long accepting runs.

## 2.7 Acceptance length distribution

### Definition

Histogram of accepted counts per verification round.

Example buckets:

```text
0
1
2-3
4-7
8-15
16+
```

### Why it matters

Shows whether the compressor is reliably useful or only occasionally useful.

## 2.8 First mismatch index

### Definition

Within each draft round, the first position where draft token differs from verifier token.

If all tokens match:

```python
first_mismatch_index = None
```

### Why it matters

Allows analysis of whether mismatches happen early or late.

## 2.9 Mismatch rate by round

### Definition

Fraction of verification rounds with at least one mismatch.

```python
round_mismatch_rate = rounds_with_mismatch / total_rounds
```

## 2.10 All-match round rate

### Definition

Fraction of verification rounds where the entire draft matched.

```python
all_match_rate = all_matched_rounds / total_rounds
```

## 2.11 Correction token count

### Definition

Number of tokens generated directly by the verifier due to mismatch.

```python
correction_token_count = count(round.correction_token is not None)
```

## 2.12 Bonus token count

### Definition

Number of bonus verifier tokens committed after all-match rounds.

Only applies after bonus token support is enabled.

---

# 3. Performance metrics

Performance metrics are meaningful only after correctness passes.

## 3.1 Wall-clock generation time

### Definition

End-to-end time for generating `max_new_tokens`.

```python
generation_time_seconds = end_time - start_time
```

### Modes

Report for:

- full
- lossy
- exactkv

## 3.2 Tokens per second

### Definition

```python
tokens_per_second = generated_tokens / generation_time_seconds
```

### Required modes

- full tokens/sec
- lossy tokens/sec
- exactkv tokens/sec

## 3.3 Speedup over full KV

### Definition

```python
speedup_over_full = exactkv_tokens_per_second / full_tokens_per_second
```

### Interpretation

- `1.0x`: no speedup
- `<1.0x`: slower than full KV
- `>1.0x`: faster than full KV

### Important warning

Phase 1 may be slower than full KV. This is acceptable if correctness and metrics work.

## 3.4 Lossy speedup over full KV

### Definition

```python
lossy_speedup = lossy_tokens_per_second / full_tokens_per_second
```

### Why it matters

This measures the upper bound of what compression could provide if we trusted lossy outputs directly.

## 3.5 Verification overhead ratio

### Definition

Fraction of ExactKV time spent in verification.

```python
verification_overhead_ratio = verification_time / total_exactkv_time
```

### Why it matters

ExactKV is useful only if verification overhead is amortized.

## 3.6 Draft time

### Definition

Time spent generating compressed-KV draft tokens.

## 3.7 Verification time

### Definition

Time spent verifying draft tokens against full KV.

## 3.8 Compression time

### Definition

Time spent compressing or recompressing KV.

### Important

In early versions, recompression may dominate. Report it separately.

## 3.9 Cache update time

### Definition

Time spent updating full and compressed KV states after commit.

## 3.10 Time to first token

### Definition

Time from prompt start to first generated token.

### Phase 1

Optional.

### Later

Important for serving.

---

# 4. Memory metrics

Memory metrics must be reported carefully and honestly.

## 4.1 Full KV estimated bytes

### Definition

Estimated memory footprint of full KV cache.

```python
full_kv_bytes = sum(t.numel() * t.element_size() for t in kv_tensors)
```

## 4.2 Compressed KV estimated bytes

### Definition

Estimated memory footprint of compressed KV representation.

Must include:

- quantized tensors
- scales
- zero points
- dropped index metadata
- any compressor metadata

## 4.3 Compression ratio

### Definition

```python
compression_ratio = compressed_kv_bytes / full_kv_bytes
```

Lower is better.

Example:

```text
0.25 means 4x compression
```

## 4.4 Memory reduction factor

### Definition

```python
memory_reduction_factor = full_kv_bytes / compressed_kv_bytes
```

Higher is better.

Example:

```text
4.0 means compressed KV is 4x smaller
```

## 4.5 Peak GPU memory allocated

### Definition

Use PyTorch CUDA memory APIs:

```python
torch.cuda.max_memory_allocated()
```

### Warning

This is not the same as theoretical KV memory.

Report it separately.

## 4.6 Peak GPU memory reserved

### Definition

```python
torch.cuda.max_memory_reserved()
```

### Warning

PyTorch caching allocator can make this larger than actual tensor memory.

## 4.7 Active GPU KV memory

### Definition

Estimated memory of KV cache actively resident on GPU.

This matters more in later versions when full KV may be on CPU.

## 4.8 Total system KV memory

### Definition

Total memory used across GPU, CPU, and storage.

ExactKV may use more total memory because it stores both full and compressed KV.

Report both:

```text
active GPU KV memory
total KV memory
```

This avoids misleading claims.

---

# 5. Compressor metrics

These metrics compare compressors independent of end-to-end runtime.

## 5.1 Compressor name

Example:

```text
int8
int4
debug_noise
kivi
kvpress_snapkv
turboquant
```

## 5.2 Compressor type

One of:

```text
quantization
token_dropping
hybrid
debug
external_adapter
```

## 5.3 Bit width

For quantization compressors.

Examples:

- 8
- 4
- 2

## 5.4 Retained token ratio

For token-dropping compressors.

```python
retained_token_ratio = retained_tokens / original_tokens
```

## 5.5 Metadata overhead

Memory used by scales, indices, masks, lookup tables, etc.

## 5.6 Materialization cost

Time required to prepare compressed KV for drafting.

## 5.7 Update cost

Time required to update compressed cache after committed tokens.

## 5.8 Acceptance per compression ratio

A key plot:

```text
x-axis: compression ratio
y-axis: acceptance rate
```

This shows which compressors remain useful as they become more aggressive.

---

# 6. Task-specific metrics

ExactKV must eventually evaluate tasks where exactness matters.

## 6.1 JSON validity

For structured-output prompts:

```python
json_valid = can_parse_json(output_text)
```

## 6.2 Function-call validity

For tool-call-like prompts:

- Function name valid
- Arguments parse
- Required fields present
- Enum values valid

## 6.3 Code syntax validity

For code-generation prompts:

- Python AST parses
- TypeScript parses if tooling available
- Diff format valid if patch prompt

## 6.4 Code execution pass rate

Later benchmark:

- Unit tests pass
- HumanEval pass@1
- MBPP pass@1

## 6.5 Constraint satisfaction

For long-generation prompts:

- Required sections present
- Required format preserved
- No missing required fields

## 6.6 Safety or security task success

Optional later metric for prompt-injection or command-generation tasks.

---

# Minimum metrics for each version

## V1 required metrics

- token exact match
- text exact match
- acceptance rate
- average accepted length
- first mismatch index
- drafted tokens
- accepted tokens
- rejected tokens
- simple wall-clock time
- estimated full KV bytes
- estimated compressed KV bytes

## V2 required metrics

All V1 metrics plus:

- compression ratio
- memory reduction factor
- per-round traces
- compressor stats
- exactness failure count
- lossy exact match

## V3 required metrics

All V2 metrics plus:

- benchmark suite aggregates
- JSON reports
- CSV reports
- plots
- per-prompt metrics
- per-compressor metrics

## V5 required metrics

All previous metrics plus:

- verification overhead ratio
- draft time
- verification time
- cache update time
- peak GPU memory
- active GPU KV memory
- speedup over full KV

---

# Recommended report schema

Each benchmark result should be serializable.

```json
{
  "prompt_id": "code_001",
  "model": "Qwen/Qwen2.5-0.5B",
  "compressor": "int8",
  "draft_len": 8,
  "max_new_tokens": 128,
  "full": {
    "tokens_per_second": 20.1,
    "output_ids": [],
    "output_text": ""
  },
  "lossy": {
    "tokens_per_second": 24.5,
    "token_exact_match": false,
    "first_divergence_idx": 37
  },
  "exactkv": {
    "tokens_per_second": 18.7,
    "token_exact_match": true,
    "acceptance_rate": 0.82,
    "avg_accepted_len": 6.3,
    "round_mismatch_rate": 0.31
  },
  "memory": {
    "full_kv_bytes": 123456,
    "compressed_kv_bytes": 45678,
    "compression_ratio": 0.37
  }
}
```

---

# Metrics that should not be overused early

Avoid making early claims based on:

- isolated wall-clock time from one run
- PyTorch reserved memory alone
- semantic similarity
- decoded text equality alone
- GPU utilization without profiling
- speedup from a non-equivalent baseline

---

# North-star metrics

The three most important metrics are:

```text
1. token_exact_match
2. average accepted tokens per verification
3. speedup over full KV after correctness is proven
```

For V1, only the first two matter.

For later versions, all three matter.

---

# Final rule

Every benchmark table must include exactness.

A speedup number without exactness is not an ExactKV result.
