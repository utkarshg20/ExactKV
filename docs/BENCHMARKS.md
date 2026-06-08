# 09_BENCHMARKS.md

# ExactKV Benchmark Suite

## Purpose of this document

This document defines the benchmark suite for ExactKV.

The benchmark suite has two goals:

1. Prove ExactKV preserves full-KV output.
2. Evaluate how useful different KV compressors are under verification.

The benchmark suite must never hide correctness failures behind average task scores.

---

# Benchmark philosophy

Most KV compression benchmarks ask:

> Does compressed KV preserve task quality?

ExactKV asks a more specific question:

> Can compressed KV draft tokens that full KV will accept?

This means the benchmark suite should focus on:

- exact token match
- acceptance length
- first mismatch
- structured-output correctness
- throughput after verification
- memory tradeoffs

---

# Required benchmark modes

Every benchmark must support three modes.

## Mode 1: full

Normal full-KV generation.

This is the ground truth.

## Mode 2: lossy

Direct compressed-KV generation without verification.

This shows what would happen if the compressor were trusted directly.

## Mode 3: exactkv

Compressed-KV drafting with full-KV verification.

This should match full mode under greedy decoding.

---

# Required comparisons

For every prompt:

```text
full vs lossy
full vs exactkv
lossy vs exactkv
```

The most important comparison is:

```python
full_output_ids == exactkv_output_ids
```

---

# Benchmark levels

The suite should be developed in levels.

## Level 1: smoke tests

Purpose:

> Ensure the system runs.

Prompts:

- Very short prompts
- 5 to 20 generated tokens
- Small model

Example prompts:

```text
The capital of France is
Write a Python function that adds two numbers.
Return a JSON object with name and age.
```

Required output:

- full output
- lossy output
- exactkv output
- exactness boolean

## Level 2: acceptance tests

Purpose:

> Measure draft acceptance behavior.

Prompts:

- Short and medium prompts
- 50 to 200 generated tokens
- Multiple draft lengths

Variables:

- compressor
- compression ratio
- draft length
- prompt type

Required output:

- acceptance rate
- accepted length distribution
- mismatch positions

## Level 3: structured-output tests

Purpose:

> Show why exactness matters.

Prompts:

- JSON generation
- function-call-like outputs
- shell-command outputs
- strict formatting tasks

Required output:

- JSON validity
- schema validity
- function name match
- argument match
- exact token match

## Level 4: code-generation tests

Purpose:

> Evaluate failure modes where lossy compression can look close but break correctness.

Prompts:

- Python functions
- small patches
- diff-format outputs
- unit-test-based problems

Required output:

- syntax validity
- test pass where possible
- exact output match
- lossy divergence point

## Level 5: long-context tests

Purpose:

> Stress KV cache length and long generation.

Prompts:

- Long synthetic documents
- Needle-like context
- Multi-section generation
- Long code context

Required output:

- acceptance vs context length
- memory vs context length
- throughput vs context length

## Level 6: external benchmark datasets

Purpose:

> Compare against known benchmarks.

Candidate datasets:

- HumanEval
- MBPP
- LongGenBench
- GSM8K-Long
- ComplexFuncBench
- synthetic tool-call tasks
- LMCache agentic traces if accessible

These are not needed in V1.

---

# V1 benchmark scope

V1 should include only smoke and acceptance tests.

## V1 model

Primary:

```text
Qwen/Qwen2.5-0.5B
```

Alternative:

```text
TinyLlama
```

## V1 compressors

- int8
- optional debug_noise if needed for rejection testing

## V1 decoding

Greedy only.

## V1 prompt count

Start with 10 to 20 prompts.

## V1 max_new_tokens

Use small values first:

```text
32
64
128
```

## V1 draft lengths

Test:

```text
1
2
4
8
```

Later:

```text
16
32
```

## V1 required benchmark command

Example:

```bash
python -m exactkv.benchmarks.runner \
  --model Qwen/Qwen2.5-0.5B \
  --compressor int8 \
  --draft-len 8 \
  --max-new-tokens 128 \
  --prompt-suite smoke
```

---

# Benchmark prompt suites

## Suite: smoke

Purpose:

Quick sanity check.

Example file:

```text
benchmarks/prompts/smoke.jsonl
```

Example rows:

```json
{"id": "smoke_001", "category": "natural", "prompt": "The capital of France is"}
{"id": "smoke_002", "category": "code", "prompt": "Write a Python function that returns the factorial of n."}
{"id": "smoke_003", "category": "json", "prompt": "Return a JSON object with fields name, age, and city."}
```

## Suite: structured

Purpose:

Test strict outputs.

Categories:

- JSON
- XML-like
- function calls
- command generation
- markdown table

Example:

```json
{
  "id": "json_001",
  "category": "json",
  "prompt": "Return only valid JSON with fields action, query, and limit."
}
```

## Suite: code

Purpose:

Test code correctness and syntax.

Example:

```json
{
  "id": "code_001",
  "category": "code",
  "prompt": "Write a Python function binary_search(arr, target) that returns the index of target or -1."
}
```

## Suite: long_context_synthetic

Purpose:

Stress long context without external datasets.

Example:

- Generate a long document with repeated facts.
- Ask the model to produce structured output based on it.
- Measure acceptance over longer context.

---

# Benchmark configuration

Use a structured config.

```python
@dataclass
class BenchmarkConfig:
    model_name: str
    compressor_name: str
    draft_len: int
    max_new_tokens: int
    prompt_suite: str
    device: str
    dtype: str
    seed: int
    output_dir: str
```

---

# Benchmark result schema

Each prompt result should include:

```json
{
  "prompt_id": "...",
  "category": "...",
  "model_name": "...",
  "compressor_name": "...",
  "draft_len": 8,
  "max_new_tokens": 128,
  "full": {},
  "lossy": {},
  "exactkv": {},
  "memory": {},
  "trace_summary": {}
}
```

## Full result

```json
"full": {
  "output_ids": [],
  "output_text": "",
  "generated_tokens": 128,
  "elapsed_seconds": 1.23,
  "tokens_per_second": 104.0
}
```

## Lossy result

```json
"lossy": {
  "output_ids": [],
  "output_text": "",
  "token_exact_match_with_full": false,
  "text_exact_match_with_full": false,
  "first_divergence_idx": 42,
  "elapsed_seconds": 1.01,
  "tokens_per_second": 126.7
}
```

## ExactKV result

```json
"exactkv": {
  "output_ids": [],
  "output_text": "",
  "token_exact_match_with_full": true,
  "text_exact_match_with_full": true,
  "elapsed_seconds": 1.40,
  "tokens_per_second": 91.4,
  "acceptance_rate": 0.83,
  "avg_accepted_len": 6.2,
  "round_mismatch_rate": 0.27
}
```

## Memory result

```json
"memory": {
  "full_kv_bytes_estimated": 1000000,
  "compressed_kv_bytes_estimated": 250000,
  "compression_ratio": 0.25,
  "memory_reduction_factor": 4.0,
  "peak_gpu_memory_allocated": 123456789
}
```

---

# Aggregate report

At the suite level, report:

- total prompts
- ExactKV exactness failures
- lossy exact match rate
- mean acceptance rate
- median acceptance rate
- mean accepted length
- mean speedup over full
- mean compression ratio
- mean first lossy divergence index

Example:

```json
{
  "summary": {
    "num_prompts": 20,
    "exactkv_failures": 0,
    "lossy_exact_match_rate": 0.35,
    "mean_acceptance_rate": 0.82,
    "mean_avg_accepted_len": 5.7,
    "mean_exactkv_speedup": 0.91,
    "mean_lossy_speedup": 1.14
  }
}
```

---

# Required plots

## Plot 1: acceptance rate vs draft length

x-axis:

```text
draft length
```

y-axis:

```text
acceptance rate
```

## Plot 2: accepted length distribution

Histogram of accepted tokens per verification round.

## Plot 3: speedup vs draft length

x-axis:

```text
draft length
```

y-axis:

```text
tokens/sec relative to full
```

## Plot 4: compression ratio vs acceptance rate

x-axis:

```text
compression ratio
```

y-axis:

```text
acceptance rate
```

## Plot 5: first mismatch position distribution

Shows where lossy compression tends to drift.

## Plot 6: memory vs throughput

x-axis:

```text
estimated KV memory
```

y-axis:

```text
tokens/sec
```

---

# Benchmark acceptance gates

A benchmark run is valid only if:

1. Full generation completed.
2. ExactKV generation completed.
3. ExactKV token output equals full token output.
4. Metrics were recorded.
5. No cache alignment errors occurred.

If exactness fails, mark the run invalid for performance claims.

---

# Benchmark naming convention

Reports should be stored with clear names.

```text
reports/
  qwen05b_int8_draft8_smoke_YYYYMMDD_HHMMSS.json
  qwen05b_int8_draft8_smoke_YYYYMMDD_HHMMSS.csv
```

Plots:

```text
plots/
  qwen05b_int8_acceptance_vs_draft_len.png
  qwen05b_int8_speedup_vs_draft_len.png
```

---

# Reproducibility requirements

Each report must record:

- model name
- model revision if known
- tokenizer name
- transformer version
- PyTorch version
- CUDA version if applicable
- GPU name if applicable
- dtype
- device
- compressor config
- draft length
- max new tokens
- seed
- date and time
- git commit hash if available

---

# Benchmark warnings

## Do not overclaim Phase 1

Phase 1 benchmarks may be slower than full KV because:

- no parallel verification
- no offload
- no optimized kernels
- possible recompression every round
- Hugging Face overhead

This is fine.

The Phase 1 benchmark goal is:

> Does the verified logic work, and what are acceptance patterns?

## Do not report speedup without exactness

A method that is fast but not exact is not an ExactKV success.

## Do not compare against vLLM early

Hugging Face prototype numbers should not be compared to production vLLM throughput.

## Do not hide lossy failures

Lossy outputs should be reported directly. They are part of the motivation.

---

# Future benchmark extensions

## HumanEval

Use for code generation pass rate.

## MBPP

Use for simpler code generation.

## ComplexFuncBench

Use for tool-call accuracy.

## LongGenBench

Use for long-form generation constraints.

## GSM8K-Long

Use for long chained reasoning.

## LMCache traces

Use if accessible for agentic long-context traces.

## Synthetic repository benchmark

Create a mini codebase and ask for modifications.

---

# Final benchmark principle

Every ExactKV benchmark must answer:

```text
Did it remain exact?
How many draft tokens were accepted?
How much memory did compression save?
How much speed survived verification?
```

If a benchmark does not answer those four questions, it is incomplete.
