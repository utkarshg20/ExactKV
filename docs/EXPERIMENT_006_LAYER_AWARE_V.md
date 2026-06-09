# Experiment 006: Simulated Layer-Aware V Compression

_Generated 2026-06-09 by ExactKV. V7 Phase D sweep. See disclaimers below._

## 1. Purpose

**Question:** Does the simulated layer-aware V policy `k8_v4_boundary_v8_sim`
(K=INT8 all layers; V=INT8 on first/last layer; V=INT4-sim on interior layers)
improve draft acceptance and reduce divergence/rejection/correction burden relative
to uniform simulated K/V baselines — while preserving ExactKV exactness?

This experiment compares **acceptance, divergence, rejection, correction, and V5
workspace-memory accounting only**. It does **not** measure throughput, latency,
wall-clock runtime, speedup, or production serving behaviour.

**`k8_v4_boundary_v8_sim` is simulated.** It uses **int8 containers**, not packed
INT4 storage. It does **not** use true attention weights. It does **not** implement
Sparse V dequantization. It does **not** implement TurboQuant+ or KVQuant.

Reproduce:

```bash
python3 -m exactkv sweep \
  --model Qwen/Qwen2.5-0.5B \
  --suite core \
  --compressors noop,int8,int4_sim,k8_v4_sim,k4_v8_sim,k_full_v4_sim,k4_v_full_sim,k8_v2_sim,k_full_v8,k8_v_full,k8_v4_boundary_v8_sim \
  --draft-lengths 4 \
  --max-new-tokens 16 \
  --json-out reports/experiment_006_layer_aware_v.json \
  --csv-out reports/experiment_006_layer_aware_v.csv

python3 -m exactkv report \
  --report reports/experiment_006_layer_aware_v.json \
  --markdown-out docs/EXPERIMENT_006_LAYER_AWARE_V.md \
  --title "Experiment 006: Simulated Layer-Aware V Compression" \
  --max-examples 5
```

Artifacts (gitignored): `reports/experiment_006_layer_aware_v.json`,
`reports/experiment_006_layer_aware_v.csv`.

---

## 2. Model and prompt suite

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B`, float32, CPU |
| Prompt suite | `core` (34 prompts) |
| `draft_len` | 4 |
| `max_new_tokens` | 16 |
| Total cells | **374** (34 prompts × 11 compressors × 1 draft length) |

---

## 3. Compressor set

| Compressor | Class | Notes |
|---|---|---|
| `noop` | Lossless baseline | Identity |
| `int8` | Real INT8 | `supports_real_bytes_claim=True` |
| `int4_sim` | Simulated uniform INT4 | int8 containers |
| `k8_v4_sim` | Simulated asymmetric K8/V4 | Uniform all layers |
| `k4_v8_sim` | Simulated asymmetric K4/V8 | Uniform all layers |
| `k_full_v4_sim` | Simulated K-full / V4 | Uniform all layers |
| `k4_v_full_sim` | Simulated K4 / V-full | Uniform all layers |
| `k8_v2_sim` | Simulated K8/V2 | Uniform all layers |
| `k_full_v8` | Real K-full / V8 | `is_simulated=False` |
| `k8_v_full` | Real K8 / V-full | `is_simulated=False` |
| **`k8_v4_boundary_v8_sim`** | **Simulated layer-aware V** | **V7 Phase B policy; `boundary_layers=1`** |

ExactKV does **not** implement KIVI, KVQuant, TurboQuant, TurboQuant+, Sparse V,
LMCache, vLLM, or PagedAttention in V7.

---

## 4. Exactness result

| Metric | Value |
|---|---|
| Total runs | 374 |
| **ExactKV failures** | **0** ✓ |
| `exactkv_output_ids == full_output_ids` | 374 / 374 |

ExactKV failure means ExactKV output differs from `generate_full_greedy` — a
correctness bug. **Zero failures** across all 11 compressors including
`k8_v4_boundary_v8_sim`.

---

## 5. Acceptance by compressor

| compressor | simulated | accept_rate | avg_accept_len | drafted | accepted | rejected | corrections | runs | exactkv_fail |
|---|---|---|---|---|---|---|---|---|---|
| noop | no | 1.000 | 3.85 | 489 | 489 | 0 | 0 | 34 | 0 |
| k_full_v8 | no | 0.990 | 3.79 | 493 | 487 | 6 | 2 | 34 | 0 |
| int8 | no | 0.961 | 3.62 | 501 | 478 | 23 | 11 | 34 | 0 |
| k8_v_full | no | 0.963 | 3.62 | 501 | 479 | 22 | 10 | 34 | 0 |
| k_full_v4_sim | yes | 0.909 | 3.41 | 515 | 470 | 45 | 19 | 34 | 0 |
| **k8_v4_boundary_v8_sim** | **yes** | **0.904** | **3.40** | 516 | 468 | 48 | 21 | 34 | 0 |
| k8_v4_sim | yes | 0.891 | 3.33 | 522 | 465 | 57 | 24 | 34 | 0 |
| int4_sim | yes | 0.628 | 2.23 | 655 | 375 | 280 | 114 | 34 | 0 |
| k4_v8_sim | yes | 0.619 | 2.25 | 664 | 373 | 291 | 116 | 34 | 0 |
| k4_v_full_sim | yes | 0.614 | 2.24 | 671 | 373 | 298 | 116 | 34 | 0 |
| k8_v2_sim | yes | 0.421 | 1.53 | 770 | 303 | 467 | 186 | 34 | 0 |

---

## 6. Divergence / rejection / correction summary

| Metric | Value |
|---|---|
| Lossy divergences (lossy ≠ full greedy) | 192 / 374 cells |
| Aggregate drafted tokens | 6,297 |
| Aggregate accepted | 4,760 |
| Aggregate rejected | 1,537 |
| Aggregate corrections | 619 |

**`k8_v4_boundary_v8_sim`:** 17 lossy-divergence cells (50% of prompts), mean
first-divergence index ~4.5, 48 rejected draft tokens, 21 corrections across 34
prompts. Lossy draft divergence is **expected**; it is not an ExactKV failure.

---

## 7. Workspace-memory accounting table

Example prefill snapshot (per-compressor accounting totals; values vary slightly by
prompt length):

| Compressor | Stored KV | Materialized KV | Metadata | Temp | Total footprint † | Real bytes? | Simulated? |
|---|---|---|---|---|---|---|---|
| `noop` | 120.0 KiB | 120.0 KiB | 0 B | 0 B | 240.0 KiB | no | no |
| `int8` | 30.0 KiB | 120.0 KiB | 384 B | 0 B | 150.4 KiB | yes | no |
| `k8_v4_sim` | 30.0 KiB | 120.0 KiB | 384 B | 0 B | 150.4 KiB | no | yes ⚠️ |
| **`k8_v4_boundary_v8_sim`** | **30.0 KiB** | **120.0 KiB** | **384 B** | **0 B** | **150.4 KiB** | **no** | **yes ⚠️** |
| `k_full_v4_sim` | 75.0 KiB | 120.0 KiB | 192 B | 0 B | 195.2 KiB | no | yes ⚠️ |
| `k_full_v8` | 75.0 KiB | 120.0 KiB | 192 B | 0 B | 195.2 KiB | yes | no |

† `total_kv_footprint_bytes` is a **conservative accounting sum**, not a measured
peak GPU memory value. **Active GPU memory is not reported.** Current materializing
compressors dequantise to full working KV for attention, so `materialized_working_kv_bytes`
equals `full_kv_bytes` for all compressors in this sweep.

**Simulated compressors** store sub-INT8 values in int8 containers — `stored_kv_bytes`
reflects int8 container reality, not packed-bit savings. `k8_v4_boundary_v8_sim` does
not show lower stored bytes than uniform `k8_v4_sim` under V5 accounting because
interior V4-sim values still occupy int8 containers.

---

## 8. Direct comparison: `k8_v4_boundary_v8_sim` vs baselines

### vs `k8_v4_sim` (uniform K8/V4)

| Metric | `k8_v4_boundary_v8_sim` | `k8_v4_sim` | Delta |
|---|---|---|---|
| Accept rate | 0.904 | 0.891 | **+0.013** |
| Lossy divergence cells | 17 / 34 | 19 / 34 | **−2** |
| Rejected tokens | 48 | 57 | **−9** |
| Corrections | 21 | 24 | **−3** |
| Mean first-div idx | ~4.5 | ~4.2 | +0.3 (later) |
| Total footprint † | 150.4 KiB | 150.4 KiB | 0 |

Boundary V8 on first/last layer yields a **modest acceptance gain** over uniform
K8/V4 with fewer rejections, at **identical** V5 accounting footprint.

### vs `int4_sim` (uniform INT4)

| Metric | `k8_v4_boundary_v8_sim` | `int4_sim` | Delta |
|---|---|---|---|
| Accept rate | 0.904 | 0.628 | **+0.276** |
| Lossy divergence cells | 17 / 34 | 30 / 34 | **−13** |
| Rejected tokens | 48 | 280 | **−232** |
| Corrections | 21 | 114 | **−93** |
| Mean first-div idx | ~4.5 | ~2.5 | **+2.0** (later) |

Layer-aware K8 + boundary V8 dramatically outperforms uniform INT4 on acceptance
and rejection burden.

### vs `k_full_v4_sim` (full K, uniform V4)

| Metric | `k8_v4_boundary_v8_sim` | `k_full_v4_sim` | Delta |
|---|---|---|---|
| Accept rate | 0.904 | 0.909 | **−0.005** |
| Lossy divergence cells | 17 / 34 | 17 / 34 | 0 |
| Rejected tokens | 48 | 45 | +3 |
| Corrections | 21 | 19 | +2 |
| Total footprint † | 150.4 KiB | 195.2 KiB | **−44.8 KiB** |

Boundary policy **matches** full-K divergence frequency but trails acceptance by a
small margin; accounting footprint is lower because K is INT8-sim rather than full.

### vs `k4_v8_sim` (uniform K4/V8)

| Metric | `k8_v4_boundary_v8_sim` | `k4_v8_sim` | Delta |
|---|---|---|---|
| Accept rate | 0.904 | 0.619 | **+0.285** |
| Lossy divergence cells | 17 / 34 | 28 / 34 | **−11** |
| Rejected tokens | 48 | 291 | **−243** |
| Corrections | 21 | 116 | **−95** |

Protecting boundary V layers while keeping K8 avoids the severe acceptance collapse
from aggressive key compression (`k4_v8_sim`).

---

## 9. What the layer-aware policy improves or fails to improve

**Improves:**

- Draft acceptance vs uniform `k8_v4_sim` (+1.3 pp) and vs aggressive baselines
  (`int4_sim`, `k4_v8_sim`, `k8_v2_sim`) by large margins.
- Rejection and correction counts vs uniform K8/V4 and aggressive compressors.
- Divergence frequency vs `k8_v4_sim` (17 vs 19 cells) and vs aggressive V/K
  policies (fewer early-divergence cells than `int4_sim` / `k4_v8_sim`).
- V5 accounting footprint vs `k_full_v4_sim` (150.4 vs 195.2 KiB) while keeping
  comparable divergence cell count.

**Fails to improve (or marginally trails):**

- Acceptance vs `k_full_v4_sim` (−0.5 pp) and vs real conservative baselines
  (`k_full_v8` 0.990, `int8` 0.961).
- V5 `stored_kv_bytes` vs uniform `k8_v4_sim` — **no accounting difference** under
  int8-container simulation.
- Mean first-divergence index vs `k8_v4_sim` — slightly **later** divergence on
  average does not translate to matching full-K policies.

---

## 10. What this suggests for future V7 work

1. **Boundary V protection is directionally useful** — Phase A's early-divergence
   cluster motivated protecting first/last layers; Experiment 006 shows measurable
   acceptance gain over uniform `k8_v4_sim` without exactness regressions.
2. **Simulated int8 containers hide footprint wins** — any real layer-aware policy
   needs honest `supports_real_bytes_claim` labelling or packed storage before memory
   comparisons are meaningful.
3. **Phase C real adapters remain optional** — `k8_v4_boundary_v8_sim` does not
   close the gap to `k_full_v8`; a real asymmetric backend (KIVI/KVQuant/TurboQuant)
   would require separate approval and Experiment 006b-style comparison.
4. **Sparse V / attention-gated materialization not validated here** — deferred
   directions need deterministic materialize semantics before any experiment cell.

---

## 11. What this does not prove

- **No speedup, throughput, latency, runtime, or production-readiness claims.**
  ExactKV does not measure wall-clock time or tokens/second.
- **`k8_v4_boundary_v8_sim` is not TurboQuant+, KVQuant, KIVI, or PyramidKV.**
  External-paper claims are **not** ExactKV results.
- **No true attention-weighted behaviour** — boundary layers are a fixed structural
  policy, not attention-gated V selection.
- **No real packed-bit memory savings** — int8 containers only.
- **No Sparse V dequantization** — full materialized working KV on every attention call.
- **Active GPU memory is not reported** — accounting sums only.
- **Layer-aware policy does not beat full-K conservative baselines** on acceptance;
  it only narrows the gap among aggressive simulated policies.

---

## 12. Relation to Experiment 006A proxy divergence analysis

Experiment 006A ([`docs/EXPERIMENT_006A_DIVERGENCE_ANALYSIS.md`](EXPERIMENT_006A_DIVERGENCE_ANALYSIS.md))
found that aggressive compressors diverge **early** (mean first-div idx ~1.9–3.1) and
recommended a simulated boundary/layer-aware V policy before real backends.

Experiment 006 **validates that recommendation in part:**

- `k8_v4_boundary_v8_sim` shifts mean first-divergence later (~4.5) vs `int4_sim`
  (~2.5) and `k4_v8_sim` (~2.4), consistent with protecting boundary layers.
- Acceptance ordering remains stable: conservative real (`k_full_v8`) > boundary sim
  ≈ `k_full_v4_sim` > uniform `k8_v4_sim` > aggressive sim.

006A used **proxy analysis only** (no attention weights). Experiment 006 does **not**
add attention weights either — it tests a **structural** layer policy, not
attention-importance correlation.

---

## 13. Relation to related work

PyramidKV, TurboQuant+, and KV-AdaQuant motivate **layer-specific or attention-aware
V budgets**. ExactKV evaluates a **minimal simulated analogue** (`k8_v4_boundary_v8_sim`)
through its own acceptance and exactness metrics.

- **TurboQuant+ themes** (sparse V, layer-aware precision) are **not implemented**.
  ExactKV does not reproduce TurboQuant+ accuracy or speed numbers.
- **KVQuant / KIVI** asymmetric real backends are **not implemented** in V7 Phase D.
- **kvpress KnormPress** (V6) remains a separate token-dropping baseline, not
  comparable to layer-aware quantization without explicit labelling.

Qualitative alignment only: protecting higher-precision V at stack boundaries
improves acceptance vs uniform aggressive V — consistent with layer-heterogeneity
motivation in external work, without claiming reproduction of external results.

---

## 14. VeriCache attribution

ExactKV is inspired by the VeriCache paper (Yao et al., arXiv:2605.17613, 2026) and
does **not** claim to have invented the draft-then-verify algorithm. This experiment
evaluates ExactKV's compressor registry and verification framework on a simulated
layer-aware V policy — not VeriCache's system design or reported serving metrics.

---

## Auto-generated tables and examples

_Below: tables and divergence examples emitted by `python3 -m exactkv report`._

## Experiment Summary
* **Total results:** 374
* **Compressors:** `int4_sim`, `int8`, `k4_v8_sim`, `k4_v_full_sim`, `k8_v2_sim`, `k8_v4_boundary_v8_sim`, `k8_v4_sim`, `k8_v_full`, `k_full_v4_sim`, `k_full_v8`, `noop`
* **Draft lengths:** 4
* **Prompts:** 34
* **Report type:** sweep

## Manifest
_Manifest not available._

## Correctness Summary
| Metric | Value |
|--------|-------|
| Total results | 374 |
| ExactKV failures | **0** (✓ PASS) |
| Lossy divergences | 192 _(expected for lossy compressors)_ |

## Acceptance Leaderboard — by Compressor
| compressor            | simulated | real-bytes | K bits | V bits | avg eff bits | accept_rate | avg_accept_len | drafted | accepted | rejected | corrections | runs | exactkv_fail |
| --------------------- | --------- | ---------- | ------ | ------ | ------------ | ----------- | -------------- | ------- | -------- | -------- | ----------- | ---- | ------------ |
| int4_sim              | yes       | no         | 4      | 4      | 4.0          | 0.628       | 2.23           | 655     | 375      | 280      | 114         | 34   | 0            |
| int8                  | no        | yes        | 8      | 8      | 8.0          | 0.961       | 3.62           | 501     | 478      | 23       | 11          | 34   | 0            |
| k4_v8_sim             | yes       | no         | 4      | 8      | 6.0          | 0.619       | 2.25           | 664     | 373      | 291      | 116         | 34   | 0            |
| k4_v_full_sim         | yes       | no         | 4      | full   | 18.0         | 0.614       | 2.24           | 671     | 373      | 298      | 116         | 34   | 0            |
| k8_v2_sim             | yes       | no         | 8      | 2      | 5.0          | 0.421       | 1.53           | 770     | 303      | 467      | 186         | 34   | 0            |
| k8_v4_boundary_v8_sim | yes       | no         | 8      | full   | 20.0         | 0.904       | 3.40           | 516     | 468      | 48       | 21          | 34   | 0            |
| k8_v4_sim             | yes       | no         | 8      | 4      | 6.0          | 0.891       | 3.33           | 522     | 465      | 57       | 24          | 34   | 0            |
| k8_v_full             | no        | yes        | 8      | full   | 20.0         | 0.963       | 3.62           | 501     | 479      | 22       | 10          | 34   | 0            |
| k_full_v4_sim         | yes       | no         | full   | 4      | 18.0         | 0.909       | 3.41           | 515     | 470      | 45       | 19          | 34   | 0            |
| k_full_v8             | no        | yes        | full   | 8      | 20.0         | 0.990       | 3.79           | 493     | 487      | 6        | 2           | 34   | 0            |
| noop                  | no        | no         | full   | full   | 32.0         | 1.000       | 3.85           | 489     | 489      | 0        | 0           | 34   | 0            |

## Acceptance Leaderboard — by Draft Length
| draft_len | accept_rate | avg_accept_len | drafted | accepted | rejected | corrections | runs | exactkv_fail |
| --------- | ----------- | -------------- | ------- | -------- | -------- | ----------- | ---- | ------------ |
| 4         | 0.809       | 3.02           | 6297    | 4760     | 1537     | 619         | 374  | 0            |

## Acceptance Grid — Compressor × Draft Length
| compressor            | draft_len | K bits | V bits | avg eff bits | accept_rate | avg_accept_len | drafted | accepted | rejected | corrections | runs | exactkv_fail |
| --------------------- | --------- | ------ | ------ | ------------ | ----------- | -------------- | ------- | -------- | -------- | ----------- | ---- | ------------ |
| int4_sim              | 4         | 4      | 4      | 4.0          | 0.628       | 2.23           | 655     | 375      | 280      | 114         | 34   | 0            |
| int8                  | 4         | 8      | 8      | 8.0          | 0.961       | 3.62           | 501     | 478      | 23       | 11          | 34   | 0            |
| k4_v8_sim             | 4         | 4      | 8      | 6.0          | 0.619       | 2.25           | 664     | 373      | 291      | 116         | 34   | 0            |
| k4_v_full_sim         | 4         | 4      | full   | 18.0         | 0.614       | 2.24           | 671     | 373      | 298      | 116         | 34   | 0            |
| k8_v2_sim             | 4         | 8      | 2      | 5.0          | 0.421       | 1.53           | 770     | 303      | 467      | 186         | 34   | 0            |
| k8_v4_boundary_v8_sim | 4         | 8      | full   | 20.0         | 0.904       | 3.40           | 516     | 468      | 48       | 21          | 34   | 0            |
| k8_v4_sim             | 4         | 8      | 4      | 6.0          | 0.891       | 3.33           | 522     | 465      | 57       | 24          | 34   | 0            |
| k8_v_full             | 4         | 8      | full   | 20.0         | 0.963       | 3.62           | 501     | 479      | 22       | 10          | 34   | 0            |
| k_full_v4_sim         | 4         | full   | 4      | 18.0         | 0.909       | 3.41           | 515     | 470      | 45       | 19          | 34   | 0            |
| k_full_v8             | 4         | full   | 8      | 20.0         | 0.990       | 3.79           | 493     | 487      | 6        | 2           | 34   | 0            |
| noop                  | 4         | full   | full   | 32.0         | 1.000       | 3.85           | 489     | 489      | 0        | 0           | 34   | 0            |

## Histogram Tables
### Accepted-Length Distribution
_Bucket: avg accepted tokens per verification round (floored to int)._

| avg_accept_len_bucket | count   | share |
| --------------------- | ------- | ----- |
| 0                     | 0       | 0.0%  |
| 1                     | 60      | 16.0% |
| 2-3                   | 181     | 48.4% |
| 4-7                   | 133     | 35.6% |
| 8-15                  | 0       | 0.0%  |
| 16+                   | 0       | 0.0%  |
| **Total**             | **374** |       |

### First-Divergence Position Distribution
_Bucket: first token index where lossy output diverged from full output. `no_divergence` = lossy matched full._

| first_div_idx_bucket | count   | share |
| -------------------- | ------- | ----- |
| no_divergence        | 182     | 48.7% |
| 0                    | 0       | 0.0%  |
| 1-4                  | 142     | 38.0% |
| 5-16                 | 50      | 13.4% |
| 17-32                | 0       | 0.0%  |
| 33+                  | 0       | 0.0%  |
| **Total**            | **374** |       |

### Rejection-Count Distribution
_Bucket: total tokens rejected (overridden) by the ExactKV verifier. Non-zero is expected for lossy compressors._

| total_rejected_bucket | count   | share |
| --------------------- | ------- | ----- |
| 0                     | 178     | 47.6% |
| 1-2                   | 30      | 8.0%  |
| 3-5                   | 61      | 16.3% |
| 6-10                  | 44      | 11.8% |
| 11+                   | 61      | 16.3% |
| **Total**             | **374** |       |

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

### Example 4 — `k4_v8_sim` | draft_len=4 | category=natural_language

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
| `k8_v4_boundary_v8_sim` | no ⚠️ | yes ⚠️ | Simulated compressor ('k8_v4_boundary_v8_sim'): sub-INT8 values are stored in int8 contai… |
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
| `k8_v4_boundary_v8_sim` | 8 | full | 20.0 | yes ⚠️ | no |
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
| `k8_v4_boundary_v8_sim` | 30.0 KiB | 120.0 KiB | 384 B | 0 B | 150.4 KiB | no ⚠️ | yes ⚠️ |
| `k8_v4_sim` | 30.0 KiB | 120.0 KiB | 384 B | 0 B | 150.4 KiB | no ⚠️ | yes ⚠️ |
| `k8_v_full` | 75.0 KiB | 120.0 KiB | 192 B | 0 B | 195.2 KiB | yes | no |
| `k_full_v4_sim` | 75.0 KiB | 120.0 KiB | 192 B | 0 B | 195.2 KiB | no ⚠️ | yes ⚠️ |
| `k_full_v8` | 75.0 KiB | 120.0 KiB | 192 B | 0 B | 195.2 KiB | yes | no |
| `noop` | 120.0 KiB | 120.0 KiB | 0 B | 0 B | 240.0 KiB | no ⚠️ | no |
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
