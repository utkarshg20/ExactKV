# Experiment 004: Workspace-Aware Memory Accounting
_Generated 2026-06-08 by ExactKV. See disclaimers below._

## Experiment Purpose

**Question:** How different are stored KV bytes, materialized working KV bytes, metadata bytes,
and total KV footprint for ExactKV's current materializing compressors?

This is an **accounting experiment, not a performance experiment.**
It does not measure GPU memory, wall-clock time, throughput, or latency.

Key findings from V4 showed that key-side compression severely degrades ExactKV acceptance
while value-side compression has little effect.
Experiment 004 extends this picture by decomposing the memory footprint per compressor into:

- **Stored KV bytes** — the compressed/quantised tensor representation.
- **Materialized working KV bytes** — the full-precision copy required for attention by current compressors.
- **Metadata bytes** — per-tensor scales, zero-points, or similar overhead.
- **Total KV footprint** — accounting sum of all four components (stored + materialized + metadata + temporary).

**Important constraints:**

- `total_kv_footprint_bytes` is a **conservative accounting sum, not a measured peak GPU memory value.**
  It is derived from tensor shapes and dtype widths only.
- Active GPU memory (torch.cuda.memory_reserved or similar) is **not reported in V5.**
  Active GPU measurement is deferred to a later CUDA-specific validation phase.
- Simulated sub-INT8 compressors (`_sim` suffix) store sub-INT8 numeric values in **int8 containers.**
  `stored_kv_bytes` reflects int8 container reality, not hypothetical packed 4-bit or 2-bit storage.
  Do not cite these figures as evidence of real packed-bit memory savings.
- ExactKV does not implement TurboQuant+, KIVI, KVQuant, KVTC, Palu, LMCache, vLLM, or PagedAttention.
  This experiment evaluates ExactKV's own simulated and real-INT8 compressors only.

---

## Experiment Summary
* **Total results:** 340
* **Compressors:** `int4_sim`, `int8`, `k4_v8_sim`, `k4_v_full_sim`, `k8_v2_sim`, `k8_v4_sim`, `k8_v_full`, `k_full_v4_sim`, `k_full_v8`, `noop`
* **Draft lengths:** 4
* **Prompts:** 34
* **Report type:** sweep

## Manifest
_Manifest not available._

## Correctness Summary
| Metric | Value |
|--------|-------|
| Total results | 340 |
| ExactKV failures | **0** (✓ PASS) |
| Lossy divergences | 175 _(expected for lossy compressors)_ |

## Acceptance Leaderboard — by Compressor
| compressor    | simulated | real-bytes | K bits | V bits | avg eff bits | accept_rate | avg_accept_len | drafted | accepted | rejected | corrections | runs | exactkv_fail |
| ------------- | --------- | ---------- | ------ | ------ | ------------ | ----------- | -------------- | ------- | -------- | -------- | ----------- | ---- | ------------ |
| int4_sim      | yes       | no         | 4      | 4      | 4.0          | 0.628       | 2.23           | 655     | 375      | 280      | 114         | 34   | 0            |
| int8          | no        | yes        | 8      | 8      | 8.0          | 0.961       | 3.62           | 501     | 478      | 23       | 11          | 34   | 0            |
| k4_v8_sim     | yes       | no         | 4      | 8      | 6.0          | 0.619       | 2.25           | 664     | 373      | 291      | 116         | 34   | 0            |
| k4_v_full_sim | yes       | no         | 4      | full   | 18.0         | 0.614       | 2.24           | 671     | 373      | 298      | 116         | 34   | 0            |
| k8_v2_sim     | yes       | no         | 8      | 2      | 5.0          | 0.421       | 1.53           | 770     | 303      | 467      | 186         | 34   | 0            |
| k8_v4_sim     | yes       | no         | 8      | 4      | 6.0          | 0.891       | 3.33           | 522     | 465      | 57       | 24          | 34   | 0            |
| k8_v_full     | no        | yes        | 8      | full   | 20.0         | 0.963       | 3.62           | 501     | 479      | 22       | 10          | 34   | 0            |
| k_full_v4_sim | yes       | no         | full   | 4      | 18.0         | 0.909       | 3.41           | 515     | 470      | 45       | 19          | 34   | 0            |
| k_full_v8     | no        | yes        | full   | 8      | 20.0         | 0.990       | 3.79           | 493     | 487      | 6        | 2           | 34   | 0            |
| noop          | no        | no         | full   | full   | 32.0         | 1.000       | 3.85           | 489     | 489      | 0        | 0           | 34   | 0            |

## Acceptance Leaderboard — by Draft Length
| draft_len | accept_rate | avg_accept_len | drafted | accepted | rejected | corrections | runs | exactkv_fail |
| --------- | ----------- | -------------- | ------- | -------- | -------- | ----------- | ---- | ------------ |
| 4         | 0.800       | 2.99           | 5781    | 4292     | 1489     | 598         | 340  | 0            |

## Acceptance Grid — Compressor × Draft Length
| compressor    | draft_len | K bits | V bits | avg eff bits | accept_rate | avg_accept_len | drafted | accepted | rejected | corrections | runs | exactkv_fail |
| ------------- | --------- | ------ | ------ | ------------ | ----------- | -------------- | ------- | -------- | -------- | ----------- | ---- | ------------ |
| int4_sim      | 4         | 4      | 4      | 4.0          | 0.628       | 2.23           | 655     | 375      | 280      | 114         | 34   | 0            |
| int8          | 4         | 8      | 8      | 8.0          | 0.961       | 3.62           | 501     | 478      | 23       | 11          | 34   | 0            |
| k4_v8_sim     | 4         | 4      | 8      | 6.0          | 0.619       | 2.25           | 664     | 373      | 291      | 116         | 34   | 0            |
| k4_v_full_sim | 4         | 4      | full   | 18.0         | 0.614       | 2.24           | 671     | 373      | 298      | 116         | 34   | 0            |
| k8_v2_sim     | 4         | 8      | 2      | 5.0          | 0.421       | 1.53           | 770     | 303      | 467      | 186         | 34   | 0            |
| k8_v4_sim     | 4         | 8      | 4      | 6.0          | 0.891       | 3.33           | 522     | 465      | 57       | 24          | 34   | 0            |
| k8_v_full     | 4         | 8      | full   | 20.0         | 0.963       | 3.62           | 501     | 479      | 22       | 10          | 34   | 0            |
| k_full_v4_sim | 4         | full   | 4      | 18.0         | 0.909       | 3.41           | 515     | 470      | 45       | 19          | 34   | 0            |
| k_full_v8     | 4         | full   | 8      | 20.0         | 0.990       | 3.79           | 493     | 487      | 6        | 2           | 34   | 0            |
| noop          | 4         | full   | full   | 32.0         | 1.000       | 3.85           | 489     | 489      | 0        | 0           | 34   | 0            |

## Histogram Tables
### Accepted-Length Distribution
_Bucket: avg accepted tokens per verification round (floored to int)._

| avg_accept_len_bucket | count   | share |
| --------------------- | ------- | ----- |
| 0                     | 0       | 0.0%  |
| 1                     | 59      | 17.4% |
| 2-3                   | 162     | 47.6% |
| 4-7                   | 119     | 35.0% |
| 8-15                  | 0       | 0.0%  |
| 16+                   | 0       | 0.0%  |
| **Total**             | **340** |       |

### First-Divergence Position Distribution
_Bucket: first token index where lossy output diverged from full output. `no_divergence` = lossy matched full._

| first_div_idx_bucket | count   | share |
| -------------------- | ------- | ----- |
| no_divergence        | 165     | 48.5% |
| 0                    | 0       | 0.0%  |
| 1-4                  | 133     | 39.1% |
| 5-16                 | 42      | 12.4% |
| 17-32                | 0       | 0.0%  |
| 33+                  | 0       | 0.0%  |
| **Total**            | **340** |       |

### Rejection-Count Distribution
_Bucket: total tokens rejected (overridden) by the ExactKV verifier. Non-zero is expected for lossy compressors._

| total_rejected_bucket | count   | share |
| --------------------- | ------- | ----- |
| 0                     | 160     | 47.1% |
| 1-2                   | 25      | 7.4%  |
| 3-5                   | 50      | 14.7% |
| 6-10                  | 44      | 12.9% |
| 11+                   | 61      | 17.9% |
| **Total**             | **340** |       |

## Lossy Divergence Examples
_These examples show prompts where the unverified lossy output differed from full-KV greedy output.  Lossy divergence is **expected** and is **not** an ExactKV failure._

### Example 1 — `int8` | draft_len=4 | category=natural_language

**Prompt ID:** `core_nat_001`  
**First divergence token index:** 5  
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> The capital of France is

**Full-KV output:**
> Paris. It is the largest city in Europe and the second largest in the world

**Lossy output** _(diverges from full — expected)_**:**
> Paris. It is the seat of the government of France. It is the capital

**ExactKV output** _(must match full)_**:**
> Paris. It is the largest city in Europe and the second largest in the world

> _Note: Lossy divergence is expected. The compressor altered the KV cache, causing the unverified lossy output to differ from full-KV greedy. ExactKV corrects this via verification. A non-zero `exactkv_matches_full=False` would be a correctness bug, not a lossy divergence._

### Example 2 — `int4_sim` | draft_len=4 | category=natural_language

**Prompt ID:** `core_nat_001`  
**First divergence token index:** 1  
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> The capital of France is

**Full-KV output:**
> Paris. It is the largest city in Europe and the second largest in the world

**Lossy output** _(diverges from full — expected)_**:**
> Paris, the capital of France is Paris. Paris is the capital of France.

**ExactKV output** _(must match full)_**:**
> Paris. It is the largest city in Europe and the second largest in the world

> _Note: Lossy divergence is expected. The compressor altered the KV cache, causing the unverified lossy output to differ from full-KV greedy. ExactKV corrects this via verification. A non-zero `exactkv_matches_full=False` would be a correctness bug, not a lossy divergence._

### Example 3 — `k8_v4_sim` | draft_len=4 | category=natural_language

**Prompt ID:** `core_nat_001`  
**First divergence token index:** 5  
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> The capital of France is

**Full-KV output:**
> Paris. It is the largest city in Europe and the second largest in the world

**Lossy output** _(diverges from full — expected)_**:**
> Paris. It is the seat of the French government and of the French people.

**ExactKV output** _(must match full)_**:**
> Paris. It is the largest city in Europe and the second largest in the world

> _Note: Lossy divergence is expected. The compressor altered the KV cache, causing the unverified lossy output to differ from full-KV greedy. ExactKV corrects this via verification. A non-zero `exactkv_matches_full=False` would be a correctness bug, not a lossy divergence._

### Example 4 — `k8_v2_sim` | draft_len=4 | category=natural_language

**Prompt ID:** `core_nat_001`  
**First divergence token index:** 2  
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> The capital of France is

**Full-KV output:**
> Paris. It is the largest city in Europe and the second largest in the world

**Lossy output** _(diverges from full — expected)_**:**
> Paris. The capital of France is Paris. The capital of France is Paris.

**ExactKV output** _(must match full)_**:**
> Paris. It is the largest city in Europe and the second largest in the world

> _Note: Lossy divergence is expected. The compressor altered the KV cache, causing the unverified lossy output to differ from full-KV greedy. ExactKV corrects this via verification. A non-zero `exactkv_matches_full=False` would be a correctness bug, not a lossy divergence._

### Example 5 — `k4_v8_sim` | draft_len=4 | category=natural_language

**Prompt ID:** `core_nat_001`  
**First divergence token index:** 2  
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> The capital of France is

**Full-KV output:**
> Paris. It is the largest city in Europe and the second largest in the world

**Lossy output** _(diverges from full — expected)_**:**
> Paris. The capital of the United States is Washington D.C. The capital of

**ExactKV output** _(must match full)_**:**
> Paris. It is the largest city in Europe and the second largest in the world

> _Note: Lossy divergence is expected. The compressor altered the KV cache, causing the unverified lossy output to differ from full-KV greedy. ExactKV corrects this via verification. A non-zero `exactkv_matches_full=False` would be a correctness bug, not a lossy divergence._

## ExactKV Failure Examples
_ExactKV failure means the verified output did NOT match `generate_full_greedy`. This is a correctness bug. Should always be empty._

> ✓ **ExactKV failure count: 0.** The ExactKV loop produced output matching full-KV greedy for every prompt in this report.

## Top Rejection Examples
_Sorted by total rejected tokens descending. High rejection is expected for aggressively lossy compressors and does NOT mean the output is wrong._

### Top-rejection 1 — `k8_v2_sim` | draft_len=4 | category=code

**Prompt ID:** `core_code_001`  
**Acceptance rate:** 0.267  
**Drafted / Accepted / Rejected / Corrections:** 30 / 8 / 22 / 8
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> Write a Python function that computes the factorial of n:

> _High rejection count is expected for aggressively lossy compressors. ExactKV corrects all rejections; exactkv_matches_full must be True._

### Top-rejection 2 — `k8_v2_sim` | draft_len=4 | category=json

**Prompt ID:** `core_json_002`  
**Acceptance rate:** 0.267  
**Drafted / Accepted / Rejected / Corrections:** 30 / 8 / 22 / 8
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> {"name": "Alice", "age": 30, "city":

> _High rejection count is expected for aggressively lossy compressors. ExactKV corrects all rejections; exactkv_matches_full must be True._

### Top-rejection 3 — `k8_v2_sim` | draft_len=4 | category=command

**Prompt ID:** `core_cmd_001`  
**Acceptance rate:** 0.267  
**Drafted / Accepted / Rejected / Corrections:** 30 / 8 / 22 / 8
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> List all files in the current directory sorted by modification time:

> _High rejection count is expected for aggressively lossy compressors. ExactKV corrects all rejections; exactkv_matches_full must be True._

### Top-rejection 4 — `k8_v2_sim` | draft_len=4 | category=command

**Prompt ID:** `core_cmd_002`  
**Acceptance rate:** 0.286  
**Drafted / Accepted / Rejected / Corrections:** 28 / 8 / 20 / 8
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> The git command to create and switch to a new branch named feature/login is:

> _High rejection count is expected for aggressively lossy compressors. ExactKV corrects all rejections; exactkv_matches_full must be True._

### Top-rejection 5 — `k8_v2_sim` | draft_len=4 | category=long_prompt

**Prompt ID:** `core_long_003`  
**Acceptance rate:** 0.310  
**Drafted / Accepted / Rejected / Corrections:** 29 / 9 / 20 / 7
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> The following is a description of a software architecture: The system uses a microservices pattern with three main services: an authentication service, a data processing service, and an API gateway. E…

> _High rejection count is expected for aggressively lossy compressors. ExactKV corrects all rejections; exactkv_matches_full must be True._

## Memory Honesty Notes
> **`int4_sim` memory note:** `int4_sim` is simulated and does **not** claim real packed INT4 memory savings. Values are quantized to the INT4 numeric range but stored in `int8` containers. Memory figures for `int4_sim` reflect `int8` storage only.

| Compressor | Real bytes? | Simulated? | Key note |
|------------|-------------|------------|----------|
| `int4_sim` | no ⚠️ | yes ⚠️ | Simulated compressor ('int4_sim'): sub-INT8 values are stored in int8 containers — no rea… |
| `int8` | yes | no | Real-storage compressor ('int8'): stored_kv_bytes reflects genuine quantised storage. |
| `k4_v8_sim` | no ⚠️ | yes ⚠️ | Simulated compressor ('k4_v8_sim'): sub-INT8 values are stored in int8 containers — no re… |
| `k4_v_full_sim` | no ⚠️ | yes ⚠️ | Simulated compressor ('k4_v_full_sim'): sub-INT8 values are stored in int8 containers — n… |
| `k8_v2_sim` | no ⚠️ | yes ⚠️ | Simulated compressor ('k8_v2_sim'): sub-INT8 values are stored in int8 containers — no re… |
| `k8_v4_sim` | no ⚠️ | yes ⚠️ | Simulated compressor ('k8_v4_sim'): sub-INT8 values are stored in int8 containers — no re… |
| `k8_v_full` | yes | no | Real-storage compressor ('k8_v_full'): stored_kv_bytes reflects genuine quantised storage. |
| `k_full_v4_sim` | no ⚠️ | yes ⚠️ | Simulated compressor ('k_full_v4_sim'): sub-INT8 values are stored in int8 containers — n… |
| `k_full_v8` | yes | no | Real-storage compressor ('k_full_v8'): stored_kv_bytes reflects genuine quantised storage. |
| `noop` | no ⚠️ | no | NoOp compressor: stored_kv_bytes == full_bytes (no compression). |
_For all compressors: `total_kv_footprint_bytes` is a conservative accounting sum, not a measured peak GPU memory value. Current materializing compressors dequantise to full working KV for attention. Active GPU measurement is deferred. See **Workspace-Aware Memory Accounting** below for the per-compressor table and full notes._

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
| `noop` | full | full | 32.0 | no | no |
**Notes:**
* `full` means full-precision passthrough; that side is not quantised.
* **Average effective bits = (K bits + V bits) / 2**, treating full precision as 32 bits. This is a metadata comparison aid — not a real memory measurement.
* Compressors marked **simulated** store sub-INT8 values in `int8` containers. Do not cite their `compressed_kv_bytes` as evidence of real packed memory savings.


## Workspace-Aware Memory Accounting
> **V5 accounting note:** `total_kv_footprint_bytes` is a **conservative accounting sum** (stored KV + materialized working KV + metadata + temporary workspace). It is **NOT** a measured peak GPU memory value. Active GPU memory measurement is deferred to a later CUDA-specific validation phase.

For all current ExactKV compressors, attention requires a full-precision dequantised working copy of the KV cache during each attention call, so `materialized_working_kv_bytes` equals `full_kv_bytes`. The practical peak KV memory footprint during attention is therefore dominated by this working copy, not the stored bytes alone.

For simulated sub-INT8 compressors (`_sim` suffix), `stored_kv_bytes` reflects **int8 container storage** — no real packed 4-bit or 2-bit bit-packing is used. Do not cite these figures as evidence of real packed-bit memory savings.

| Compressor | Stored KV | Materialized KV | Metadata | Temp workspace | Total footprint † | Real bytes? | Simulated? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `int4_sim` | 30.0 KiB | 120.0 KiB | 384 B | 0 B | 150.4 KiB | no ⚠️ | yes ⚠️ |
| `int8` | 30.0 KiB | 120.0 KiB | 384 B | 0 B | 150.4 KiB | yes | no |
| `k4_v8_sim` | 30.0 KiB | 120.0 KiB | 384 B | 0 B | 150.4 KiB | no ⚠️ | yes ⚠️ |
| `k4_v_full_sim` | 75.0 KiB | 120.0 KiB | 192 B | 0 B | 195.2 KiB | no ⚠️ | yes ⚠️ |
| `k8_v2_sim` | 30.0 KiB | 120.0 KiB | 384 B | 0 B | 150.4 KiB | no ⚠️ | yes ⚠️ |
| `k8_v4_sim` | 30.0 KiB | 120.0 KiB | 384 B | 0 B | 150.4 KiB | no ⚠️ | yes ⚠️ |
| `k8_v_full` | 75.0 KiB | 120.0 KiB | 192 B | 0 B | 195.2 KiB | yes | no |
| `k_full_v4_sim` | 75.0 KiB | 120.0 KiB | 192 B | 0 B | 195.2 KiB | no ⚠️ | yes ⚠️ |
| `k_full_v8` | 75.0 KiB | 120.0 KiB | 192 B | 0 B | 195.2 KiB | yes | no |
| `noop` | 120.0 KiB | 120.0 KiB | 0 B | 0 B | 240.0 KiB | no ⚠️ | no |
† Total footprint = stored + materialized + metadata + temp workspace. This is a **conservative accounting sum, NOT a measured peak GPU memory value**.


## Relation to V5 Workspace-Aware Accounting

Experiment 004 is the first ExactKV experiment to populate the V5 workspace-aware memory
fields introduced in V5 Phases A–C:

- **Phase A** added `stored_kv_bytes`, `materialized_working_kv_bytes`, `metadata_bytes`,
  `temporary_workspace_bytes`, and `total_kv_footprint_bytes` to `CompressionStats` and
  `MemorySummary` for every compressor.
- **Phase B** exposed these fields in JSON reports and CSV exports.
- **Phase C** rendered them in Markdown reports and CLI summaries.
- **Phase D (this experiment)** is the first run using these fields on the core prompt suite.

The key observation from this accounting is:
**`materialized_working_kv_bytes` equals `full_kv_bytes` for all current ExactKV compressors.**
All current compressors dequantise the stored cache to full precision for each attention call.
The stored bytes are smaller than full, but the peak KV memory footprint during attention
is dominated by the full-precision working copy, not the stored bytes.
This means `total_kv_footprint_bytes` > `full_kv_bytes` for most compressors —
a conservative and intentionally honest accounting choice.

**This finding does not apply to all real compression backends.**
Systems like KIVI and TurboQuant+ are designed to attend directly to the compressed cache
(or use rotation + quantise-on-the-fly during decode), which may reduce or eliminate
the full-precision working copy requirement.
ExactKV does not implement these backends.
The workspace accounting design is explicitly motivated by this real-world distinction —
see `docs/RELATED_WORK_KV_CACHE_COMPRESSION.md` for details.

## Relation to Related Work

The stored-cache versus working-cache distinction in this experiment is motivated by
how real KV-cache compression backends actually use memory (see
[`docs/RELATED_WORK_KV_CACHE_COMPRESSION.md`](RELATED_WORK_KV_CACHE_COMPRESSION.md)):

- **KIVI** keeps a full-precision residual alongside its 2-bit grouped cache, so
  stored bytes understate the working footprint.
- **KVQuant** carries per-channel scales, non-uniform datatypes, and a sparse outlier
  side-channel — all metadata beyond the quantised tensors.
- **Palu** and **KVTC** reconstruct a dense working cache from low-rank / transform-coded
  forms plus projection matrices and codebooks during decode.
- **TurboQuant+** (community work) claims that aggressive V compression is "free" for
  acceptance — a claim ExactKV V4's simulated compressors are consistent with but do not prove,
  since they use int8 containers and do not implement rotation or real sub-INT8 kernels.

**ExactKV does not implement or claim to reproduce any of the above systems.**
This experiment provides correctness and accounting evidence for ExactKV's own
simulated and real-INT8 compressors only.

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
* **`total_kv_footprint_bytes` is a conservative accounting sum, not a measured peak GPU memory value.** It equals stored KV + materialized working KV + metadata + temporary workspace — all derived from tensor shapes and dtype widths. Active GPU memory measurement (torch.cuda.memory_reserved, etc.) is deferred to a later CUDA-specific validation phase and is not performed in V5.
* **Current materializing compressors dequantise to full precision for attention.** This means `materialized_working_kv_bytes` equals `full_kv_bytes` for all current ExactKV compressors. Stored-byte savings and working-cache footprint are different concepts; the table in the Workspace-Aware Memory Accounting section makes both visible.
* **No external backends implemented.** ExactKV V5 does not implement TurboQuant+, KIVI,
KVQuant, KVTC, Palu, LMCache, vLLM, or PagedAttention. All compressors in this experiment
are ExactKV's own simulated (`_sim`) or real-INT8 implementations.
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
