# Experiment 002: Core Suite Sweep (v0.3.0-dev)
_Generated 2026-06-08 by ExactKV. See disclaimers below._

## Experiment Summary
* **Total results:** 204
* **Compressors:** `int4_sim`, `int8`, `noop`
* **Draft lengths:** 4, 8
* **Prompts:** 34
* **Report type:** sweep

## Manifest
_Manifest not available._

## Correctness Summary
| Metric | Value |
|--------|-------|
| Total results | 204 |
| ExactKV failures | **0** (✓ PASS) |
| Lossy divergences | 86 _(expected for lossy compressors)_ |

## Acceptance Leaderboard — by Compressor
| compressor | simulated | real-bytes | accept_rate | avg_accept_len | drafted | accepted | rejected | corrections | runs | exactkv_fail |
| ---------- | --------- | ---------- | ----------- | -------------- | ------- | -------- | -------- | ----------- | ---- | ------------ |
| int4_sim   | yes       | no         | 0.553       | 2.66           | 2369    | 1097     | 1272     | 331         | 68   | 0            |
| int8       | no        | yes        | 0.951       | 5.20           | 1492    | 1400     | 92       | 28          | 68   | 0            |
| noop       | no        | no         | 1.000       | 5.66           | 1428    | 1428     | 0        | 0           | 68   | 0            |

## Acceptance Leaderboard — by Draft Length
| draft_len | accept_rate | avg_accept_len | drafted | accepted | rejected | corrections | runs | exactkv_fail |
| --------- | ----------- | -------------- | ------- | -------- | -------- | ----------- | ---- | ------------ |
| 4         | 0.865       | 3.25           | 2404    | 1960     | 444      | 182         | 102  | 0            |
| 8         | 0.805       | 5.76           | 2885    | 1965     | 920      | 177         | 102  | 0            |

## Acceptance Grid — Compressor × Draft Length
| compressor | draft_len | accept_rate | avg_accept_len | drafted | accepted | rejected | corrections | runs | exactkv_fail |
| ---------- | --------- | ----------- | -------------- | ------- | -------- | -------- | ----------- | ---- | ------------ |
| int4_sim   | 4         | 0.626       | 2.28           | 962     | 546      | 416      | 168         | 34   | 0            |
| int4_sim   | 8         | 0.480       | 3.03           | 1407    | 551      | 856      | 163         | 34   | 0            |
| int8       | 4         | 0.968       | 3.64           | 728     | 700      | 28       | 14          | 34   | 0            |
| int8       | 8         | 0.934       | 6.76           | 764     | 700      | 64       | 14          | 34   | 0            |
| noop       | 4         | 1.000       | 3.84           | 714     | 714      | 0        | 0           | 34   | 0            |
| noop       | 8         | 1.000       | 7.49           | 714     | 714      | 0        | 0           | 34   | 0            |

## Histogram Tables
### Accepted-Length Distribution
_Bucket: avg accepted tokens per verification round (floored to int)._

| avg_accept_len_bucket | count   | share |
| --------------------- | ------- | ----- |
| 0                     | 0       | 0.0%  |
| 1                     | 14      | 6.9%  |
| 2-3                   | 75      | 36.8% |
| 4-7                   | 70      | 34.3% |
| 8-15                  | 45      | 22.1% |
| 16+                   | 0       | 0.0%  |
| **Total**             | **204** |       |

### First-Divergence Position Distribution
_Bucket: first token index where lossy output diverged from full output. `no_divergence` = lossy matched full._

| first_div_idx_bucket | count   | share |
| -------------------- | ------- | ----- |
| no_divergence        | 118     | 57.8% |
| 0                    | 0       | 0.0%  |
| 1-4                  | 60      | 29.4% |
| 5-16                 | 20      | 9.8%  |
| 17-32                | 6       | 2.9%  |
| 33+                  | 0       | 0.0%  |
| **Total**            | **204** |       |

### Rejection-Count Distribution
_Bucket: total tokens rejected (overridden) by the ExactKV verifier. Non-zero is expected for lossy compressors._

| total_rejected_bucket | count   | share |
| --------------------- | ------- | ----- |
| 0                     | 120     | 58.8% |
| 1-2                   | 7       | 3.4%  |
| 3-5                   | 13      | 6.4%  |
| 6-10                  | 15      | 7.4%  |
| 11+                   | 49      | 24.0% |
| **Total**             | **204** |       |

## Lossy Divergence Examples
_These examples show prompts where the unverified lossy output differed from full-KV greedy output.  Lossy divergence is **expected** and is **not** an ExactKV failure._

### Example 1 — `int8` | draft_len=4 | category=natural_language

**Prompt ID:** `core_nat_001`  
**First divergence token index:** 5  
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> The capital of France is

**Full-KV output:**
> Paris. It is the largest city in Europe and the second largest in the world. It is also the capital of France

**Lossy output** _(diverges from full — expected)_**:**
> Paris. It is the seat of the government of France. It is the capital of the country of France. It is

**ExactKV output** _(must match full)_**:**
> Paris. It is the largest city in Europe and the second largest in the world. It is also the capital of France

> _Note: Lossy divergence is expected. The compressor altered the KV cache, causing the unverified lossy output to differ from full-KV greedy. ExactKV corrects this via verification. A non-zero `exactkv_matches_full=False` would be a correctness bug, not a lossy divergence._

### Example 2 — `int8` | draft_len=8 | category=natural_language

**Prompt ID:** `core_nat_001`  
**First divergence token index:** 5  
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> The capital of France is

**Full-KV output:**
> Paris. It is the largest city in Europe and the second largest in the world. It is also the capital of France

**Lossy output** _(diverges from full — expected)_**:**
> Paris. It is the seat of the government of France. It is the capital of the country of France. It is

**ExactKV output** _(must match full)_**:**
> Paris. It is the largest city in Europe and the second largest in the world. It is also the capital of France

> _Note: Lossy divergence is expected. The compressor altered the KV cache, causing the unverified lossy output to differ from full-KV greedy. ExactKV corrects this via verification. A non-zero `exactkv_matches_full=False` would be a correctness bug, not a lossy divergence._

### Example 3 — `int4_sim` | draft_len=4 | category=natural_language

**Prompt ID:** `core_nat_001`  
**First divergence token index:** 1  
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> The capital of France is

**Full-KV output:**
> Paris. It is the largest city in Europe and the second largest in the world. It is also the capital of France

**Lossy output** _(diverges from full — expected)_**:**
> Paris, the capital of France is Paris. Paris is the capital of France. Paris is the capital of France. Paris

**ExactKV output** _(must match full)_**:**
> Paris. It is the largest city in Europe and the second largest in the world. It is also the capital of France

> _Note: Lossy divergence is expected. The compressor altered the KV cache, causing the unverified lossy output to differ from full-KV greedy. ExactKV corrects this via verification. A non-zero `exactkv_matches_full=False` would be a correctness bug, not a lossy divergence._

### Example 4 — `int4_sim` | draft_len=8 | category=natural_language

**Prompt ID:** `core_nat_001`  
**First divergence token index:** 1  
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> The capital of France is

**Full-KV output:**
> Paris. It is the largest city in Europe and the second largest in the world. It is also the capital of France

**Lossy output** _(diverges from full — expected)_**:**
> Paris, the capital of France is Paris. Paris is the capital of France. Paris is the capital of France. Paris

**ExactKV output** _(must match full)_**:**
> Paris. It is the largest city in Europe and the second largest in the world. It is also the capital of France

> _Note: Lossy divergence is expected. The compressor altered the KV cache, causing the unverified lossy output to differ from full-KV greedy. ExactKV corrects this via verification. A non-zero `exactkv_matches_full=False` would be a correctness bug, not a lossy divergence._

### Example 5 — `int8` | draft_len=4 | category=natural_language

**Prompt ID:** `core_nat_002`  
**First divergence token index:** 8  
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> In machine learning, a neural network is

**Full-KV output:**
> a type of artificial neural network that is inspired by the structure and function of the human brain. It is a model that

**Lossy output** _(diverges from full — expected)_**:**
> a type of artificial neural network that is used to model complex relationships between input and output data. It is a type of

**ExactKV output** _(must match full)_**:**
> a type of artificial neural network that is inspired by the structure and function of the human brain. It is a model that

> _Note: Lossy divergence is expected. The compressor altered the KV cache, causing the unverified lossy output to differ from full-KV greedy. ExactKV corrects this via verification. A non-zero `exactkv_matches_full=False` would be a correctness bug, not a lossy divergence._

## ExactKV Failure Examples
_ExactKV failure means the verified output did NOT match `generate_full_greedy`. This is a correctness bug. Should always be empty._

> ✓ **ExactKV failure count: 0.** The ExactKV loop produced output matching full-KV greedy for every prompt in this report.

## Top Rejection Examples
_Sorted by total rejected tokens descending. High rejection is expected for aggressively lossy compressors and does NOT mean the output is wrong._

### Top-rejection 1 — `int4_sim` | draft_len=8 | category=long_prompt

**Prompt ID:** `core_long_004`  
**Acceptance rate:** 0.227  
**Drafted / Accepted / Rejected / Corrections:** 66 / 15 / 51 / 9
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> In Greek mythology, Prometheus was a Titan who stole fire from the gods and gave it to humanity. As punishment, Zeus chained him to a rock where an eagle ate his liver each day, only for it to regener…

> _High rejection count is expected for aggressively lossy compressors. ExactKV corrects all rejections; exactkv_matches_full must be True._

### Top-rejection 2 — `int4_sim` | draft_len=8 | category=natural_language

**Prompt ID:** `core_nat_006`  
**Acceptance rate:** 0.242  
**Drafted / Accepted / Rejected / Corrections:** 66 / 16 / 50 / 8
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> Climate change refers to long-term shifts in global temperatures and weather patterns. The main cause since the 1800s has been

> _High rejection count is expected for aggressively lossy compressors. ExactKV corrects all rejections; exactkv_matches_full must be True._

### Top-rejection 3 — `int4_sim` | draft_len=8 | category=natural_language

**Prompt ID:** `core_nat_005`  
**Acceptance rate:** 0.234  
**Drafted / Accepted / Rejected / Corrections:** 64 / 15 / 49 / 9
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> The process of photosynthesis converts

> _High rejection count is expected for aggressively lossy compressors. ExactKV corrects all rejections; exactkv_matches_full must be True._

### Top-rejection 4 — `int4_sim` | draft_len=8 | category=long_prompt

**Prompt ID:** `core_long_003`  
**Acceptance rate:** 0.234  
**Drafted / Accepted / Rejected / Corrections:** 64 / 15 / 49 / 9
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> The following is a description of a software architecture: The system uses a microservices pattern with three main services: an authentication service, a data processing service, and an API gateway. E…

> _High rejection count is expected for aggressively lossy compressors. ExactKV corrects all rejections; exactkv_matches_full must be True._

### Top-rejection 5 — `int4_sim` | draft_len=8 | category=natural_language

**Prompt ID:** `core_nat_001`  
**Acceptance rate:** 0.242  
**Drafted / Accepted / Rejected / Corrections:** 62 / 15 / 47 / 9
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> The capital of France is

> _High rejection count is expected for aggressively lossy compressors. ExactKV corrects all rejections; exactkv_matches_full must be True._

## Memory Honesty Notes
> **`int4_sim` memory note:** `int4_sim` is simulated and does **not** claim real packed INT4 memory savings. Values are quantized to the INT4 numeric range but stored in `int8` containers. Memory figures for `int4_sim` reflect `int8` storage only.

* **`int4_sim`:** int4_sim uses int8 container storage in V2; do not interpret this as real packed INT4 memory savings.
* **`noop`:** 'noop' is simulated; compressed_kv_bytes reflects storage format, not real compression savings.

## What This Report Proves
* ExactKV produced token IDs that **exactly match** `generate_full_greedy` for every prompt in this report (ExactKV failure count = 0).
* Lossy divergence was detected and corrected by the verification engine.
* Acceptance rates, accepted lengths, and rejection counts reflect the draft-verify-commit loop behaviour for each compressor and draft length.
* The compressor registry and analysis pipeline function correctly.

## What This Report Does Not Prove
* **No speedup claim.** ExactKV does not measure tokens/second, latency, or wall-clock time.
* **No throughput claim.** Sequential verification is used in V1/V2; this is not a production-serving benchmark.
* **No production readiness.** ExactKV runs with locally cached model weights under a research/experimental framework.
* **`int4_sim` is simulated.** No real packed 4-bit storage is used; memory figures are conservative `int8` estimates.
* **No real compressor backends.** All compressors in V2/V3 are implemented in PyTorch for research purposes.
* **VeriCache attribution.** ExactKV is inspired by the VeriCache paper (Yao et al., arXiv:2605.17613, 2026) and does not claim to have invented the draft-then-verify algorithm. This report evaluates the current Hugging Face correctness and analysis framework, not the paper's system.

## Disclaimers
> **Interpretation notes**
>
> * **Lossy divergence is expected and is not an ExactKV failure.** The compressor alters the KV cache, so the unverified lossy output may differ from full-KV greedy output. ExactKV corrects this via verification.
>
> * **ExactKV failure means ExactKV output differs from full-KV output.** This is a correctness bug. ExactKV failure count must be 0 in a correct implementation.
>
> * **`int4_sim` is simulated and does not claim real packed INT4 memory savings.** Values are quantized to the INT4 numeric range but stored in `int8` containers. Memory figures for `int4_sim` reflect `int8` storage.
>
> * **This report does not claim speedup, throughput, latency, or production readiness.** It documents exactness and acceptance behaviour only.
