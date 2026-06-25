# What Breaks When You Compress the KV Cache

ExactKV is a compressor-agnostic crash-test and leaderboard framework for LLM KV-cache compression. It measures token-level drift, first divergence, acceptance rate, verifier agreement, and exactness failures across compressors and models.

Everyone wants smaller KV caches. Few teams measure what happens to the *tokens* when compression kicks in.

ExactKV is an evaluation framework that asks a simple question: **does the compressed cache still agree with full-precision generation, token by token?**

## The short answer

**Phase H+ public release:** **1500 real-GPU cells** on Llama-3.1-8B and Mistral-7B-Instruct-v0.3 with **zero ExactKV failures**. On Llama-3.1-8B, `noop` and `int8` score **1.0** with full acceptance.

**Historical Phase A panel (336 cells, 4 small/mid models):** INT8 mean leaderboard score **0.916** with zero failures — supporting cross-model evidence, not the final public release headline.

## Three cases that illustrate the problem

### Structured Output Drift

**Prompt:** `Complete JSON: {"name": "get_weather", "city":`

**Compressor / model:** `int4_sim` on Qwen 0.5B

First divergence at token **1**, acceptance **0.50**. Pharmacy-style intent-flip prompt is not in the Phase A panel; p2_json_tool is the closest in-panel structured-output drift case.

### Qa Partial Drift

**Prompt:** `The capital of France is`

**Compressor / model:** `shard` on Qwen 0.5B

First divergence at token **None**, acceptance **0.66**. Partial prefix acceptance under probe-only shard compression.

### Worst Case Compression

**Prompt:** `The capital of France is`

**Compressor / model:** `int4_sim` on Qwen 0.5B

First divergence at token **1**, acceptance **0.33**. Lowest acceptance int4_sim cell in Phase A benchmark.


## Leaderboard snapshot (Phase H+ scale_7b — Llama-3.1-8B)

| Rank | Compressor | Score | Acceptance |
|-----:|------------|------:|-----------:|
| 1 | `noop` | 1.0 | 1.0 |
| 2 | `int8` | 1.0 | 1.0 |
| 3 | `int4_sim` | 0.684 | 0.852 |
| 4 | `spectralquant` (fallback/proxy) | 0.684 | 0.852 |
| 5 | `shard` (probe-first) | 0.544 | 0.632 |

## Why this matters

Compression is not a single number. A method can look fine on average yet fail on structured outputs, factual QA, or larger models. ExactKV makes that visible before you ship.

## What we are not claiming

ExactKV is **not a production serving system**. It does **not reproduce VeriCache** serving throughput. Phase F kernel results are **kernel microbenchmark** numbers only — **not end-to-end** inference speedups. Reported compression ratios are **stored tensor byte ratios** unless active GPU memory is explicitly measured (we do **not** claim active GPU memory savings). **SpectralQuant** cells use **fallback/proxy** mode when the real dependency is unavailable. **Shard** is **probe-first** heuristic analysis, not a full Shard integration.

No speedups in end-to-end inference. No memory savings unless measured. No production serving integration — this is an evaluation layer, not a deployment stack.

---

*Data: Phase H+ scale_7b (1500 cells) + Phase A (336 cells, historical). Reproduce: `python3 scripts/exactkv_repro.py --reports-only`*
