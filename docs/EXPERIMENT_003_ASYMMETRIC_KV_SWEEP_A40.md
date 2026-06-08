# Experiment 003: Asymmetric K/V Sweep on A40
_Generated 2026-06-08 by ExactKV. See disclaimers below._

## Experiment Summary
* **Total results:** 612
* **Compressors:** `int4_sim`, `int8`, `k4_v8_sim`, `k4_v_full_sim`, `k8_v2_sim`, `k8_v4_sim`, `k8_v_full`, `k_full_v4_sim`, `k_full_v8`
* **Draft lengths:** 4, 8
* **Prompts:** 34
* **Report type:** sweep

## Manifest
_Manifest not available._

## Correctness Summary
| Metric | Value |
|--------|-------|
| Total results | 612 |
| ExactKV failures | **0** (✓ PASS) |
| Lossy divergences | 388 _(expected for lossy compressors)_ |

## Acceptance Leaderboard — by Compressor
| compressor    | simulated | real-bytes | K bits | V bits | avg eff bits | accept_rate | avg_accept_len | drafted | accepted | rejected | corrections | runs | exactkv_fail |
| ------------- | --------- | ---------- | ------ | ------ | ------------ | ----------- | -------------- | ------- | -------- | -------- | ----------- | ---- | ------------ |
| int4_sim      | yes       | no         | 4      | 4      | 4.0          | 0.553       | 2.66           | 2369    | 1097     | 1272     | 331         | 68   | 0            |
| int8          | no        | yes        | 8      | 8      | 8.0          | 0.951       | 5.20           | 1492    | 1400     | 92       | 28          | 68   | 0            |
| k4_v8_sim     | yes       | no         | 4      | 8      | 6.0          | 0.562       | 2.75           | 2359    | 1106     | 1253     | 322         | 68   | 0            |
| k4_v_full_sim | yes       | no         | 4      | full   | 18.0         | 0.561       | 2.73           | 2365    | 1110     | 1255     | 318         | 68   | 0            |
| k8_v2_sim     | yes       | no         | 8      | 2      | 5.0          | 0.330       | 1.58           | 3167    | 865      | 2302     | 563         | 68   | 0            |
| k8_v4_sim     | yes       | no         | 8      | 4      | 6.0          | 0.858       | 4.54           | 1600    | 1360     | 240      | 68          | 68   | 0            |
| k8_v_full     | no        | yes        | 8      | full   | 20.0         | 0.953       | 5.22           | 1488    | 1402     | 86       | 26          | 68   | 0            |
| k_full_v4_sim | yes       | no         | full   | 4      | 18.0         | 0.890       | 4.79           | 1548    | 1374     | 174      | 54          | 68   | 0            |
| k_full_v8     | no        | yes        | full   | 8      | 20.0         | 0.988       | 5.53           | 1444    | 1422     | 22       | 6           | 68   | 0            |

## Acceptance Leaderboard — by Draft Length
| draft_len | accept_rate | avg_accept_len | drafted | accepted | rejected | corrections | runs | exactkv_fail |
| --------- | ----------- | -------------- | ------- | -------- | -------- | ----------- | ---- | ------------ |
| 4         | 0.782       | 2.92           | 7703    | 5569     | 2134     | 857         | 306  | 0            |
| 8         | 0.695       | 4.86           | 10129   | 5567     | 4562     | 859         | 306  | 0            |

## Acceptance Grid — Compressor × Draft Length
| compressor    | draft_len | K bits | V bits | avg eff bits | accept_rate | avg_accept_len | drafted | accepted | rejected | corrections | runs | exactkv_fail |
| ------------- | --------- | ------ | ------ | ------------ | ----------- | -------------- | ------- | -------- | -------- | ----------- | ---- | ------------ |
| int4_sim      | 4         | 4      | 4      | 4.0          | 0.626       | 2.28           | 962     | 546      | 416      | 168         | 34   | 0            |
| int4_sim      | 8         | 4      | 4      | 4.0          | 0.480       | 3.03           | 1407    | 551      | 856      | 163         | 34   | 0            |
| int8          | 4         | 8      | 8      | 8.0          | 0.968       | 3.64           | 728     | 700      | 28       | 14          | 34   | 0            |
| int8          | 8         | 8      | 8      | 8.0          | 0.934       | 6.76           | 764     | 700      | 64       | 14          | 34   | 0            |
| k4_v8_sim     | 4         | 4      | 8      | 6.0          | 0.633       | 2.33           | 953     | 551      | 402      | 163         | 34   | 0            |
| k4_v8_sim     | 8         | 4      | 8      | 6.0          | 0.491       | 3.18           | 1406    | 555      | 851      | 159         | 34   | 0            |
| k4_v_full_sim | 4         | 4      | full   | 18.0         | 0.635       | 2.34           | 955     | 554      | 401      | 160         | 34   | 0            |
| k4_v_full_sim | 8         | 4      | full   | 18.0         | 0.488       | 3.11           | 1410    | 556      | 854      | 158         | 34   | 0            |
| k8_v2_sim     | 4         | 8      | 2      | 5.0          | 0.405       | 1.49           | 1153    | 433      | 720      | 281         | 34   | 0            |
| k8_v2_sim     | 8         | 8      | 2      | 5.0          | 0.254       | 1.67           | 2014    | 432      | 1582     | 282         | 34   | 0            |
| k8_v4_sim     | 4         | 8      | 4      | 6.0          | 0.899       | 3.37           | 756     | 684      | 72       | 30          | 34   | 0            |
| k8_v4_sim     | 8         | 8      | 4      | 6.0          | 0.818       | 5.70           | 844     | 676      | 168      | 38          | 34   | 0            |
| k8_v_full     | 4         | 8      | full   | 20.0         | 0.969       | 3.64           | 728     | 701      | 27       | 13          | 34   | 0            |
| k8_v_full     | 8         | 8      | full   | 20.0         | 0.938       | 6.80           | 760     | 701      | 59       | 13          | 34   | 0            |
| k_full_v4_sim | 4         | full   | 4      | 18.0         | 0.913       | 3.42           | 748     | 689      | 59       | 25          | 34   | 0            |
| k_full_v4_sim | 8         | full   | 4      | 18.0         | 0.866       | 6.17           | 800     | 685      | 115      | 29          | 34   | 0            |
| k_full_v8     | 4         | full   | 8      | 20.0         | 0.990       | 3.77           | 720     | 711      | 9        | 3           | 34   | 0            |
| k_full_v8     | 8         | full   | 8      | 20.0         | 0.986       | 7.29           | 724     | 711      | 13       | 3           | 34   | 0            |

## Histogram Tables
### Accepted-Length Distribution
_Bucket: avg accepted tokens per verification round (floored to int)._

| avg_accept_len_bucket | count   | share |
| --------------------- | ------- | ----- |
| 0                     | 0       | 0.0%  |
| 1                     | 99      | 16.2% |
| 2-3                   | 257     | 42.0% |
| 4-7                   | 182     | 29.7% |
| 8-15                  | 74      | 12.1% |
| 16+                   | 0       | 0.0%  |
| **Total**             | **612** |       |

### First-Divergence Position Distribution
_Bucket: first token index where lossy output diverged from full output. `no_divergence` = lossy matched full._

| first_div_idx_bucket | count   | share |
| -------------------- | ------- | ----- |
| no_divergence        | 224     | 36.6% |
| 0                    | 0       | 0.0%  |
| 1-4                  | 266     | 43.5% |
| 5-16                 | 86      | 14.1% |
| 17-32                | 36      | 5.9%  |
| 33+                  | 0       | 0.0%  |
| **Total**            | **612** |       |

### Rejection-Count Distribution
_Bucket: total tokens rejected (overridden) by the ExactKV verifier. Non-zero is expected for lossy compressors._

| total_rejected_bucket | count   | share |
| --------------------- | ------- | ----- |
| 0                     | 225     | 36.8% |
| 1-2                   | 41      | 6.7%  |
| 3-5                   | 71      | 11.6% |
| 6-10                  | 58      | 9.5%  |
| 11+                   | 217     | 35.5% |
| **Total**             | **612** |       |

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

### Example 5 — `k8_v4_sim` | draft_len=4 | category=natural_language

**Prompt ID:** `core_nat_001`  
**First divergence token index:** 5  
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> The capital of France is

**Full-KV output:**
> Paris. It is the largest city in Europe and the second largest in the world. It is also the capital of France

**Lossy output** _(diverges from full — expected)_**:**
> Paris. It is the seat of the French government and of the French people. It is also the capital of the department

**ExactKV output** _(must match full)_**:**
> Paris. It is the largest city in Europe and the second largest in the world. It is also the capital of France

> _Note: Lossy divergence is expected. The compressor altered the KV cache, causing the unverified lossy output to differ from full-KV greedy. ExactKV corrects this via verification. A non-zero `exactkv_matches_full=False` would be a correctness bug, not a lossy divergence._

## ExactKV Failure Examples
_ExactKV failure means the verified output did NOT match `generate_full_greedy`. This is a correctness bug. Should always be empty._

> ✓ **ExactKV failure count: 0.** The ExactKV loop produced output matching full-KV greedy for every prompt in this report.

## Top Rejection Examples
_Sorted by total rejected tokens descending. High rejection is expected for aggressively lossy compressors and does NOT mean the output is wrong._

### Top-rejection 1 — `k8_v2_sim` | draft_len=8 | category=code

**Prompt ID:** `core_code_001`  
**Acceptance rate:** 0.143  
**Drafted / Accepted / Rejected / Corrections:** 84 / 12 / 72 / 12
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> Write a Python function that computes the factorial of n:

> _High rejection count is expected for aggressively lossy compressors. ExactKV corrects all rejections; exactkv_matches_full must be True._

### Top-rejection 2 — `k8_v2_sim` | draft_len=8 | category=json

**Prompt ID:** `core_json_002`  
**Acceptance rate:** 0.143  
**Drafted / Accepted / Rejected / Corrections:** 84 / 12 / 72 / 12
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> {"name": "Alice", "age": 30, "city":

> _High rejection count is expected for aggressively lossy compressors. ExactKV corrects all rejections; exactkv_matches_full must be True._

### Top-rejection 3 — `k8_v2_sim` | draft_len=8 | category=command

**Prompt ID:** `core_cmd_001`  
**Acceptance rate:** 0.171  
**Drafted / Accepted / Rejected / Corrections:** 82 / 14 / 68 / 10
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> List all files in the current directory sorted by modification time:

> _High rejection count is expected for aggressively lossy compressors. ExactKV corrects all rejections; exactkv_matches_full must be True._

### Top-rejection 4 — `k8_v2_sim` | draft_len=8 | category=json

**Prompt ID:** `core_json_004`  
**Acceptance rate:** 0.165  
**Drafted / Accepted / Rejected / Corrections:** 79 / 13 / 66 / 11
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> {"status": "success", "data": {"user_id": 42, "username":

> _High rejection count is expected for aggressively lossy compressors. ExactKV corrects all rejections; exactkv_matches_full must be True._

### Top-rejection 5 — `k8_v2_sim` | draft_len=8 | category=code

**Prompt ID:** `core_code_003`  
**Acceptance rate:** 0.171  
**Drafted / Accepted / Rejected / Corrections:** 76 / 13 / 63 / 11
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> # Sort a list of integers using merge sort
def merge_sort(arr):

> _High rejection count is expected for aggressively lossy compressors. ExactKV corrects all rejections; exactkv_matches_full must be True._

## Memory Honesty Notes
> **`int4_sim` memory note:** `int4_sim` is simulated and does **not** claim real packed INT4 memory savings. Values are quantized to the INT4 numeric range but stored in `int8` containers. Memory figures for `int4_sim` reflect `int8` storage only.

* **`int4_sim`:** int4_sim uses int8 container storage in V2; do not interpret this as real packed INT4 memory savings.
* **`k4_v8_sim`:** 'k4_v8_sim' is simulated; compressed_kv_bytes reflects storage format, not real compression savings.
* **`k4_v_full_sim`:** 'k4_v_full_sim' is simulated; compressed_kv_bytes reflects storage format, not real compression savings.
* **`k8_v2_sim`:** 'k8_v2_sim' is simulated; compressed_kv_bytes reflects storage format, not real compression savings.
* **`k8_v4_sim`:** 'k8_v4_sim' is simulated; compressed_kv_bytes reflects storage format, not real compression savings.
* **`k_full_v4_sim`:** 'k_full_v4_sim' is simulated; compressed_kv_bytes reflects storage format, not real compression savings.

## K/V Compression Metadata
_This report includes asymmetric compressors that compress keys and values at different bit-widths.  K bits and V bits are declared by each compressor's capabilities; they are not derived from measured memory. Average effective bits is a comparison aid only._

| Compressor | K bits | V bits | Avg eff bits | Simulated | Real bytes |
|------------|--------|--------|--------------|-----------|------------|
| `int4_sim` | 4 | 4 | 4.0 | yes ⚠️ | no |
| `int8` | 8 | 8 | 8.0 | no | yes |
| `k4_v8_sim` | 4 | 8 | 6.0 | yes ⚠️ | no |
| `k4_v_full_sim` | 4 | full | 18.0 | yes ⚠️ | no |
| `k8_v2_sim` | 8 | 2 | 5.0 | yes ⚠️ | no |
| `k8_v4_sim` | 8 | 4 | 6.0 | yes ⚠️ | no |
| `k8_v_full` | 8 | full | 20.0 | no | yes |
| `k_full_v4_sim` | full | 4 | 18.0 | yes ⚠️ | no |
| `k_full_v8` | full | 8 | 20.0 | no | yes |
**Notes:**
* `full` means full-precision passthrough; that side is not quantised.
* **Average effective bits = (K bits + V bits) / 2**, treating full precision as 32 bits. This is a metadata comparison aid — not a real memory measurement.
* Compressors marked **simulated** store sub-INT8 values in `int8` containers. Do not cite their `compressed_kv_bytes` as evidence of real packed memory savings.


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
* **No real compressor backends.** All compressors in V2/V3/V4 are implemented in PyTorch for research purposes.
* **Sub-INT8 asymmetric compressors are simulated.** V4 compressors with `_sim` suffix (e.g. `k8_v4_sim`, `k4_v8_sim`) quantise K or V to a sub-INT8 numeric range but store values in `int8` containers — no real bit-packing. Do not cite their `compressed_kv_bytes` as evidence of real packed memory savings. `k8_v_full` and `k_full_v8` use only real INT8 and full precision and carry `is_simulated=False`.
* **Average effective bit width is a comparison aid only.** It is defined as (K bits + V bits) / 2, where full precision counts as 32 bits. It is not a real memory measurement.
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
