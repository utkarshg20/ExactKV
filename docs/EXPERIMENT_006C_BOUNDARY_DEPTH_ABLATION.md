# Experiment 006C: Boundary-Depth Ablation for Layer-Aware V Compression

_Generated 2026-06-09 by ExactKV. V7 Phase C ablation. See disclaimers below._

## 1. Purpose

**Question:** Does protecting more boundary V layers (N=1, 2, 4) improve draft
acceptance beyond uniform `k8_v4_sim`, and can deeper boundary protection close the
gap to `k_full_v4_sim`?

This ablation compares **acceptance, rejection, correction, and V5
workspace-memory accounting only**. It does **not** measure throughput, latency,
wall-clock runtime, speedup, or production serving behaviour.

**All compressors in this ablation are simulated layer-aware policies.** They do
**not** use true attention weights. They do **not** implement Sparse V
dequantization. They do **not** implement TurboQuant+, KVQuant, or KIVI. They use
**int8 containers**, not packed INT4 storage.

Reproduce:

```bash
python3 -m exactkv sweep \
  --model Qwen/Qwen2.5-0.5B \
  --suite core \
  --compressors k8_v4_sim,k8_v4_boundary_v8_sim,k8_v4_boundary2_v8_sim,k8_v4_boundary4_v8_sim,k_full_v4_sim \
  --draft-lengths 4 \
  --max-new-tokens 16 \
  --json-out reports/experiment_006c_boundary_depth_ablation.json \
  --csv-out reports/experiment_006c_boundary_depth_ablation.csv

python3 -m exactkv report \
  --report reports/experiment_006c_boundary_depth_ablation.json \
  --markdown-out docs/EXPERIMENT_006C_BOUNDARY_DEPTH_ABLATION.md \
  --title "Experiment 006C: Boundary-Depth Ablation for Layer-Aware V Compression" \
  --max-examples 5
```

Artifacts (gitignored): `reports/experiment_006c_boundary_depth_ablation.json`,
`reports/experiment_006c_boundary_depth_ablation.csv`.

---

## 2. Why Phase C happened after Experiment 006

Experiment 006 ([`docs/EXPERIMENT_006_LAYER_AWARE_V.md`](EXPERIMENT_006_LAYER_AWARE_V.md))
showed `k8_v4_boundary_v8_sim` (N=1) modestly beat `k8_v4_sim` (+0.013 acceptance)
but trailed `k_full_v4_sim` (−0.005). That suggested boundary V protection helps,
but a single-layer boundary might be too shallow.

Phase C was **not skipped** — it was **re-scoped** after Experiment 006 to test
whether **deeper boundary protection** (N=2, N=4) amplifies the gain, without
jumping to real backends (KIVI/KVQuant/TurboQuant) or attention-gated materialization.

---

## 3. Model and prompt suite

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B`, float32, CPU |
| Prompt suite | `core` (34 prompts) |
| `draft_len` | 4 |
| `max_new_tokens` | 16 |
| Total cells | **170** (34 prompts × 5 compressors × 1 draft length) |

---

## 4. Compressor set

| Compressor | Boundary N | Policy |
|---|---:|---|
| `k8_v4_sim` | — | Uniform K8 / V4-sim all layers (baseline) |
| `k8_v4_boundary_v8_sim` | 1 | K8 all layers; V8 first/last 1; V4-sim interior |
| `k8_v4_boundary2_v8_sim` | 2 | K8 all layers; V8 first/last 2; V4-sim interior |
| `k8_v4_boundary4_v8_sim` | 4 | K8 all layers; V8 first/last 4; V4-sim interior |
| `k_full_v4_sim` | — | Full K / V4-sim all layers (conservative sim reference) |

All boundary variants: `is_simulated=True`, `supports_real_bytes_claim=False`,
`value_bit_width=None` with `value_bit_width_label="mixed 8/4-sim"` (mixed per-layer
V precision; int8 containers only).

---

## 5. Exactness result

| Metric | Value |
|---|---|
| Total runs | 170 |
| **ExactKV failures** | **0** ✓ |
| `exactkv_output_ids == full_output_ids` | 170 / 170 |

---

## 6. Acceptance by compressor

| compressor | boundary N | accept_rate | avg_accept_len | drafted | accepted | rejected | corrections |
|---|---:|---:|---:|---:|---:|---:|---:|
| **k8_v4_boundary4_v8_sim** | **4** | **0.954** | 3.58 | 504 | 476 | 28 | 13 |
| k_full_v4_sim | — | 0.909 | 3.41 | 515 | 470 | 45 | 19 |
| k8_v4_boundary2_v8_sim | 2 | 0.906 | 3.40 | 516 | 469 | 47 | 20 |
| k8_v4_boundary_v8_sim | 1 | 0.904 | 3.40 | 516 | 468 | 48 | 21 |
| k8_v4_sim | — | 0.891 | 3.33 | 522 | 465 | 57 | 24 |

---

## 7. Rejection / correction comparison

| Compressor | rejected | corrections | lossy-divergence cells |
|---|---:|---:|---:|
| k8_v4_sim | 57 | 24 | 19 / 34 |
| k8_v4_boundary_v8_sim (N=1) | 48 | 21 | 17 / 34 |
| k8_v4_boundary2_v8_sim (N=2) | 47 | 20 | 17 / 34 |
| **k8_v4_boundary4_v8_sim (N=4)** | **28** | **13** | **11 / 34** |
| k_full_v4_sim | 45 | 19 | 17 / 34 |

Deeper boundary protection monotonically reduces rejections and corrections relative
to uniform `k8_v4_sim`. N=4 also reduces divergence frequency below `k_full_v4_sim`.

---

## 8. Workspace-memory table

Per-compressor accounting totals (example prefill snapshot; values vary by prompt):

| Compressor | Stored KV | Materialized KV | Metadata | Total footprint † | Real bytes? | Simulated? |
|---|---|---|---|---|---|---|
| `k8_v4_sim` | 30.0 KiB | 120.0 KiB | 384 B | 150.4 KiB | no | yes ⚠️ |
| `k8_v4_boundary_v8_sim` | 30.0 KiB | 120.0 KiB | 384 B | 150.4 KiB | no | yes ⚠️ |
| `k8_v4_boundary2_v8_sim` | 30.0 KiB | 120.0 KiB | 384 B | 150.4 KiB | no | yes ⚠️ |
| `k8_v4_boundary4_v8_sim` | 30.0 KiB | 120.0 KiB | 384 B | 150.4 KiB | no | yes ⚠️ |
| `k_full_v4_sim` | 75.0 KiB | 120.0 KiB | 192 B | 195.2 KiB | no | yes ⚠️ |

† `total_kv_footprint_bytes` is a **conservative accounting sum**, not a measured
peak GPU memory value. **Active GPU memory is not reported.**

All boundary-depth variants share the same V5 footprint as `k8_v4_sim` because
interior V4-sim values still occupy int8 containers. `k_full_v4_sim` has higher
stored bytes because K is full precision.

---

## 9. Boundary-depth comparison

| Policy | accept_rate | Δ vs `k8_v4_sim` | Δ vs `k_full_v4_sim` | diverged cells | mean first-div idx |
|---|---:|---:|---:|---:|---:|
| `k8_v4_sim` (uniform) | 0.891 | — | −0.018 | 19 / 34 | ~4.2 |
| boundary N=1 | 0.904 | **+0.013** | −0.005 | 17 / 34 | ~4.5 |
| boundary N=2 | 0.906 | **+0.015** | −0.003 | 17 / 34 | ~5.0 |
| boundary N=4 | 0.954 | **+0.063** | **+0.045** | 11 / 34 | ~6.1 |
| `k_full_v4_sim` | 0.909 | +0.018 | — | 17 / 34 | ~4.7 |

**Key result:** N=1 and N=2 give **incremental** gains over uniform K8/V4. **N=4
substantially improves** acceptance and **beats `k_full_v4_sim`** on this suite,
with fewer divergence cells and lower rejection/correction burden.

---

## 10. What improves as boundary depth increases

- **Acceptance rate** rises monotonically: 0.891 → 0.904 → 0.906 → **0.954**.
- **Rejected tokens** fall: 57 → 48 → 47 → **28**.
- **Corrections** fall: 24 → 21 → 20 → **13**.
- **Lossy-divergence cells** fall: 19 → 17 → 17 → **11**.
- **Mean first-divergence index** shifts later (~4.2 → ~6.1), consistent with
  protecting more early/late layers from aggressive V4-sim.

N=4 is the clear winner among tested boundary depths on acceptance behaviour.

---

## 11. What does not improve

- **V5 `stored_kv_bytes`** — unchanged across N=1/2/4 vs `k8_v4_sim` (int8-container
  simulation hides per-layer precision differences in accounting).
- **Gap to real conservative baselines** — this ablation does not include `k_full_v8`
  or `int8`; N=4 (0.954) may still trail real INT8 (~0.96 in Experiment 006).
- **N=2 vs N=1** — only marginal acceptance gain (+0.002); most benefit requires N=4
  on Qwen2.5-0.5B (24 layers).
- **Attention-aware selection** — boundary depth is structural, not attention-gated.

---

## 12. Recommendation for Phase E and future V7b/V8

**Phase E (v0.7.0 release notes):**

- Highlight `k8_v4_boundary4_v8_sim` as the strongest simulated layer-aware policy
  in V7, with explicit simulated/int8-container labelling.
- Retain Experiment 006 as the broad baseline sweep; cite 006C for boundary-depth
  evidence.
- Do **not** claim TurboQuant+, KVQuant, KIVI, or attention-weighted behaviour.

**Future V7b (optional):**

- Test intermediate N=3 or N=6; correlate with per-layer divergence if attention
  logging is approved (still no fabricated weights).
- Compare N=4 against real `k_full_v8` / `int8` on the same 5-compressor panel.

**V8 / real backends:**

- Real asymmetric adapters (KIVI/KVQuant/TurboQuant) remain **separate approval**;
  Phase C does not substitute for them.
- Any real backend must pass the same `exactkv_failures == 0` gate before Experiment
  006b-style comparison.

---

## 13. What this does not prove

- **No speedup, throughput, latency, runtime, or production-readiness claims.**
- **Simulated policies are not external backends** — TurboQuant+, KVQuant, KIVI
  results are not ExactKV results.
- **No true attention-weighted behaviour** — fixed boundary depth only.
- **No Sparse V dequantization** — full materialized working KV on every attention call.
- **No real packed-bit memory savings** — int8 containers only.
- **N=4 beating `k_full_v4_sim` on one model/suite** does not generalize to all
  models, draft lengths, or `max_new_tokens` settings without further sweeps.
- **Active GPU memory is not reported.**

---

## 14. VeriCache attribution

ExactKV is inspired by the VeriCache paper (Yao et al., arXiv:2605.17613, 2026) and
does **not** claim to have invented the draft-then-verify algorithm. This ablation
evaluates simulated layer-aware V policies inside ExactKV's verification framework —
not VeriCache's system design or reported serving metrics.

---

## Experiment Summary
* **Total results:** 170
* **Compressors:** `k8_v4_boundary2_v8_sim`, `k8_v4_boundary4_v8_sim`, `k8_v4_boundary_v8_sim`, `k8_v4_sim`, `k_full_v4_sim`
* **Draft lengths:** 4
* **Prompts:** 34
* **Report type:** sweep

## Manifest
_Manifest not available._

## Correctness Summary
| Metric | Value |
|--------|-------|
| Total results | 170 |
| ExactKV failures | **0** (✓ PASS) |
| Lossy divergences | 81 _(expected for lossy compressors)_ |

## Acceptance Leaderboard — by Compressor
| compressor             | simulated | real-bytes | K bits | V bits        | avg eff bits | accept_rate | avg_accept_len | drafted | accepted | rejected | corrections | runs | exactkv_fail |
| ---------------------- | --------- | ---------- | ------ | ------------- | ------------ | ----------- | -------------- | ------- | -------- | -------- | ----------- | ---- | ------------ |
| k8_v4_boundary2_v8_sim | yes       | no         | 8      | mixed 8/4-sim | n/a          | 0.906       | 3.40           | 516     | 469      | 47       | 20          | 34   | 0            |
| k8_v4_boundary4_v8_sim | yes       | no         | 8      | mixed 8/4-sim | n/a          | 0.954       | 3.58           | 504     | 476      | 28       | 13          | 34   | 0            |
| k8_v4_boundary_v8_sim  | yes       | no         | 8      | mixed 8/4-sim | n/a          | 0.904       | 3.40           | 516     | 468      | 48       | 21          | 34   | 0            |
| k8_v4_sim              | yes       | no         | 8      | 4             | 6.0          | 0.891       | 3.33           | 522     | 465      | 57       | 24          | 34   | 0            |
| k_full_v4_sim          | yes       | no         | full   | 4             | 18.0         | 0.909       | 3.41           | 515     | 470      | 45       | 19          | 34   | 0            |

## Acceptance Leaderboard — by Draft Length
| draft_len | accept_rate | avg_accept_len | drafted | accepted | rejected | corrections | runs | exactkv_fail |
| --------- | ----------- | -------------- | ------- | -------- | -------- | ----------- | ---- | ------------ |
| 4         | 0.913       | 3.42           | 2573    | 2348     | 225      | 97          | 170  | 0            |

## Acceptance Grid — Compressor × Draft Length
| compressor             | draft_len | K bits | V bits        | avg eff bits | accept_rate | avg_accept_len | drafted | accepted | rejected | corrections | runs | exactkv_fail |
| ---------------------- | --------- | ------ | ------------- | ------------ | ----------- | -------------- | ------- | -------- | -------- | ----------- | ---- | ------------ |
| k8_v4_boundary2_v8_sim | 4         | 8      | mixed 8/4-sim | n/a          | 0.906       | 3.40           | 516     | 469      | 47       | 20          | 34   | 0            |
| k8_v4_boundary4_v8_sim | 4         | 8      | mixed 8/4-sim | n/a          | 0.954       | 3.58           | 504     | 476      | 28       | 13          | 34   | 0            |
| k8_v4_boundary_v8_sim  | 4         | 8      | mixed 8/4-sim | n/a          | 0.904       | 3.40           | 516     | 468      | 48       | 21          | 34   | 0            |
| k8_v4_sim              | 4         | 8      | 4             | 6.0          | 0.891       | 3.33           | 522     | 465      | 57       | 24          | 34   | 0            |
| k_full_v4_sim          | 4         | full   | 4             | 18.0         | 0.909       | 3.41           | 515     | 470      | 45       | 19          | 34   | 0            |

## Histogram Tables
### Accepted-Length Distribution
_Bucket: avg accepted tokens per verification round (floored to int)._

| avg_accept_len_bucket | count   | share |
| --------------------- | ------- | ----- |
| 0                     | 0       | 0.0%  |
| 1                     | 4       | 2.4%  |
| 2-3                   | 95      | 55.9% |
| 4-7                   | 71      | 41.8% |
| 8-15                  | 0       | 0.0%  |
| 16+                   | 0       | 0.0%  |
| **Total**             | **170** |       |

### First-Divergence Position Distribution
_Bucket: first token index where lossy output diverged from full output. `no_divergence` = lossy matched full._

| first_div_idx_bucket | count   | share |
| -------------------- | ------- | ----- |
| no_divergence        | 89      | 52.4% |
| 0                    | 0       | 0.0%  |
| 1-4                  | 37      | 21.8% |
| 5-16                 | 44      | 25.9% |
| 17-32                | 0       | 0.0%  |
| 33+                  | 0       | 0.0%  |
| **Total**            | **170** |       |

### Rejection-Count Distribution
_Bucket: total tokens rejected (overridden) by the ExactKV verifier. Non-zero is expected for lossy compressors._

| total_rejected_bucket | count   | share |
| --------------------- | ------- | ----- |
| 0                     | 92      | 54.1% |
| 1-2                   | 26      | 15.3% |
| 3-5                   | 51      | 30.0% |
| 6-10                  | 1       | 0.6%  |
| 11+                   | 0       | 0.0%  |
| **Total**             | **170** |       |

## Lossy Divergence Examples
_These examples show prompts where the unverified lossy output differed from full-KV greedy output.  Lossy divergence is **expected** and is **not** an ExactKV failure._

### Example 1 — `k8_v4_sim` | draft_len=4 | category=natural_language

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

### Example 2 — `k8_v4_boundary_v8_sim` | draft_len=4 | category=natural_language

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

### Example 3 — `k8_v4_boundary2_v8_sim` | draft_len=4 | category=natural_language

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

### Example 4 — `k8_v4_boundary4_v8_sim` | draft_len=4 | category=natural_language

**Prompt ID:** `core_nat_001`  
**First divergence token index:** 11  
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> The capital of France is

**Full-KV output:**
> Paris. It is the largest city in Europe and the second largest in the world

**Lossy output** _(diverges from full — expected)_**:**
> Paris. It is the largest city in Europe and the third largest city in the

**ExactKV output** _(must match full)_**:**
> Paris. It is the largest city in Europe and the second largest in the world

> _Note: Lossy divergence is expected. The compressor altered the KV cache, causing the unverified lossy output to differ from full-KV greedy. ExactKV corrects this via verification. A non-zero `exactkv_matches_full=False` would be a correctness bug, not a lossy divergence._

### Example 5 — `k_full_v4_sim` | draft_len=4 | category=natural_language

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

## ExactKV Failure Examples
_ExactKV failure means the verified output did NOT match `generate_full_greedy`. This is a correctness bug. Should always be empty._

> ✓ **ExactKV failure count: 0.** The ExactKV loop produced output matching full-KV greedy for every prompt in this report.

## Top Rejection Examples
_Sorted by total rejected tokens descending. High rejection is expected for aggressively lossy compressors and does NOT mean the output is wrong._

### Top-rejection 1 — `k8_v4_sim` | draft_len=4 | category=command

**Prompt ID:** `core_cmd_001`  
**Acceptance rate:** 0.700  
**Drafted / Accepted / Rejected / Corrections:** 20 / 14 / 6 / 2
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> List all files in the current directory sorted by modification time:

> _High rejection count is expected for aggressively lossy compressors. ExactKV corrects all rejections; exactkv_matches_full must be True._

### Top-rejection 2 — `k8_v4_sim` | draft_len=4 | category=natural_language

**Prompt ID:** `core_nat_004`  
**Acceptance rate:** 0.737  
**Drafted / Accepted / Rejected / Corrections:** 19 / 14 / 5 / 2
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> Once upon a time in a land far away, there lived a

> _High rejection count is expected for aggressively lossy compressors. ExactKV corrects all rejections; exactkv_matches_full must be True._

### Top-rejection 3 — `k8_v4_boundary_v8_sim` | draft_len=4 | category=natural_language

**Prompt ID:** `core_nat_004`  
**Acceptance rate:** 0.737  
**Drafted / Accepted / Rejected / Corrections:** 19 / 14 / 5 / 2
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> Once upon a time in a land far away, there lived a

> _High rejection count is expected for aggressively lossy compressors. ExactKV corrects all rejections; exactkv_matches_full must be True._

### Top-rejection 4 — `k8_v4_boundary2_v8_sim` | draft_len=4 | category=natural_language

**Prompt ID:** `core_nat_004`  
**Acceptance rate:** 0.737  
**Drafted / Accepted / Rejected / Corrections:** 19 / 14 / 5 / 2
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> Once upon a time in a land far away, there lived a

> _High rejection count is expected for aggressively lossy compressors. ExactKV corrects all rejections; exactkv_matches_full must be True._

### Top-rejection 5 — `k8_v4_boundary4_v8_sim` | draft_len=4 | category=natural_language

**Prompt ID:** `core_nat_004`  
**Acceptance rate:** 0.737  
**Drafted / Accepted / Rejected / Corrections:** 19 / 14 / 5 / 2
**ExactKV matches full:** ✓ yes

**Prompt excerpt:**
> Once upon a time in a land far away, there lived a

> _High rejection count is expected for aggressively lossy compressors. ExactKV corrects all rejections; exactkv_matches_full must be True._

## Memory Honesty Notes
> **`int4_sim` memory note:** `int4_sim` is simulated and does **not** claim real packed INT4 memory savings. Values are quantized to the INT4 numeric range but stored in `int8` containers. Memory figures for `int4_sim` reflect `int8` storage only.

| Compressor | Real bytes? | Simulated? | Key note |
|------------|-------------|------------|----------|
| `k8_v4_boundary2_v8_sim` | no ⚠️ | yes ⚠️ | Simulated compressor ('k8_v4_boundary2_v8_sim'): sub-INT8 values are stored in int8 conta… |
| `k8_v4_boundary4_v8_sim` | no ⚠️ | yes ⚠️ | Simulated compressor ('k8_v4_boundary4_v8_sim'): sub-INT8 values are stored in int8 conta… |
| `k8_v4_boundary_v8_sim` | no ⚠️ | yes ⚠️ | Simulated compressor ('k8_v4_boundary_v8_sim'): sub-INT8 values are stored in int8 contai… |
| `k8_v4_sim` | no ⚠️ | yes ⚠️ | Simulated compressor ('k8_v4_sim'): sub-INT8 values are stored in int8 containers — no re… |
| `k_full_v4_sim` | no ⚠️ | yes ⚠️ | Simulated compressor ('k_full_v4_sim'): sub-INT8 values are stored in int8 containers — n… |
_For all compressors: `total_kv_footprint_bytes` is a conservative accounting sum, not a measured peak GPU memory value. Current materializing compressors dequantise to full working KV for attention. Active GPU measurement is deferred. See **Workspace-Aware Memory Accounting** below for the per-compressor table and full notes._

## K/V Compression Metadata
_This report includes asymmetric compressors that compress keys and values at different bit-widths.  K bits and V bits are declared by each compressor's capabilities; they are not derived from measured memory. Average effective bits is a comparison aid only._

| Compressor | K bits | V bits | Avg eff bits | Simulated | Real bytes |
|------------|--------|--------|--------------|-----------|------------|
| `k8_v4_boundary2_v8_sim` | 8 | mixed 8/4-sim | n/a | yes ⚠️ | no |
| `k8_v4_boundary4_v8_sim` | 8 | mixed 8/4-sim | n/a | yes ⚠️ | no |
| `k8_v4_boundary_v8_sim` | 8 | mixed 8/4-sim | n/a | yes ⚠️ | no |
| `k8_v4_sim` | 8 | 4 | 6.0 | yes ⚠️ | no |
| `k_full_v4_sim` | full | 4 | 18.0 | yes ⚠️ | no |
**Notes:**
* `full` means full-precision passthrough; that side is not quantised.
* `mixed 8/4-sim` means V uses INT8-range on boundary layers and INT4-range simulation on interior layers (int8 containers; not packed-bit storage).
* **Average effective bits = (K bits + V bits) / 2**, treating full precision as 32 bits. This is a metadata comparison aid — not a real memory measurement. `n/a` means mixed per-layer precision.
* Compressors marked **simulated** store sub-INT8 values in `int8` containers. Do not cite their `compressed_kv_bytes` as evidence of real packed memory savings.


## Workspace-Aware Memory Accounting
> **V5 accounting note:** `total_kv_footprint_bytes` is a **conservative accounting sum** (stored KV + materialized working KV + metadata + temporary workspace). It is **NOT** a measured peak GPU memory value. Active GPU memory measurement is deferred to a later CUDA-specific validation phase.

For all current ExactKV compressors, attention requires a full-precision dequantised working copy of the KV cache during each attention call, so `materialized_working_kv_bytes` equals `full_kv_bytes`. The practical peak KV memory footprint during attention is therefore dominated by this working copy, not the stored bytes alone.

For simulated sub-INT8 compressors (`_sim` suffix), `stored_kv_bytes` reflects **int8 container storage** — no real packed 4-bit or 2-bit bit-packing is used. Do not cite these figures as evidence of real packed-bit memory savings.

| Compressor | Stored KV | Materialized KV | Metadata | Temp workspace | Total footprint † | Real bytes? | Simulated? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `k8_v4_boundary2_v8_sim` | 30.0 KiB | 120.0 KiB | 384 B | 0 B | 150.4 KiB | no ⚠️ | yes ⚠️ |
| `k8_v4_boundary4_v8_sim` | 30.0 KiB | 120.0 KiB | 384 B | 0 B | 150.4 KiB | no ⚠️ | yes ⚠️ |
| `k8_v4_boundary_v8_sim` | 30.0 KiB | 120.0 KiB | 384 B | 0 B | 150.4 KiB | no ⚠️ | yes ⚠️ |
| `k8_v4_sim` | 30.0 KiB | 120.0 KiB | 384 B | 0 B | 150.4 KiB | no ⚠️ | yes ⚠️ |
| `k_full_v4_sim` | 75.0 KiB | 120.0 KiB | 192 B | 0 B | 195.2 KiB | no ⚠️ | yes ⚠️ |
† Total footprint = stored + materialized + metadata + temp workspace. This is a **conservative accounting sum, NOT a measured peak GPU memory value**.


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