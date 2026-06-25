# What Breaks When You Compress the KV Cache

Everyone wants smaller KV caches. Few teams measure what happens to the *tokens* when compression kicks in.

ExactKV is an evaluation framework that asks a simple question: **does the compressed cache still agree with full-precision generation, token by token?**

## The short answer

In our latest cross-model panel, **INT8 is the near-optimal baseline** — mean leaderboard score **0.916** with **zero ExactKV failures** across four models. Aggressive simulators and external probes diverge earlier and accept fewer draft tokens.

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


## Leaderboard snapshot

| Rank | Compressor | Mean score |
|-----:|------------|----------:|
| 1 | `noop` | 0.995 |
| 2 | `int8` | 0.916 |
| 3 | `k8_v4_sim` | 0.801 |

## Why this matters

Compression is not a single number. A method can look fine on average yet fail on structured outputs, factual QA, or larger models. ExactKV makes that visible before you ship.

## What we are not claiming

No speedups. No memory savings unless measured. No production serving integration — this is an evaluation layer, not a deployment stack.

---

*Data: Phase A benchmark + Phase B leaderboard. Reproduce: `python scripts/run_leaderboard.py --all`*
