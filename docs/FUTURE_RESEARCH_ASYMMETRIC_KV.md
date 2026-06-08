# Future Research: Asymmetric K/V Compression and Workspace-Aware Memory Accounting

> **Scope:** This document describes V4/V5 research directions.
> The asymmetric-compression directions (asymmetric compressor simulator,
> K-only/V-only ablations, K/V bit-width sweeps) have been **implemented in V4**.
> No real compressor backends, no speedup claims, no production-readiness claims.

> **V4 implementation status.** The asymmetric compressor simulator
> (`AsymmetricQuantSimCompressor`), K-only/V-only ablations (`k8_v_full`,
> `k_full_v8`, `k_full_v4_sim`, `k4_v_full_sim`), and the K/V bit-width sweep
> (`k8_v4_sim`, `k8_v2_sim`, `k4_v8_sim`) are all **implemented and tested in
> V4**. Experiment 003 documents the acceptance-behaviour results. See
> [`docs/V4_SCOPE_STATEMENT.md`](V4_SCOPE_STATEMENT.md) and
> [`docs/EXPERIMENT_003_ASYMMETRIC_KV_SWEEP.md`](EXPERIMENT_003_ASYMMETRIC_KV_SWEEP.md).
> The **workspace-aware memory accounting** section (§4) is **not** part of V4
> and remains a **V5** candidate.

---

## 1. Why Asymmetric K/V Compression Matters

Keys and values do not play the same role in the attention mechanism.

**Keys** are used to compute attention scores via the query–key dot product
and the subsequent softmax.  A quantization error in a key changes the
attention routing — which positions receive high weight — and this error
propagates multiplicatively through the softmax.  Small key errors in
high-magnitude positions can shift which context tokens are attended to.

**Values** are aggregated *after* attention weights are formed.  A
quantization error in a value perturbs the weighted sum output, but does not
change which positions are attended to.  The error is additive, not
multiplicative.

This asymmetry means that compressing K and V identically, with the same
bit-width and the same per-tensor scale, is unlikely to be optimal.  Keys
may require higher precision to preserve attention routing, while values may
tolerate more aggressive compression because their errors do not propagate
through the softmax non-linearity.

The practical implication: an aggressive V compressor paired with a
conservative K compressor may achieve a better acceptance rate than a
uniform compressor at the same average bit-width.

### External evidence for the asymmetric-K/V thesis

ExactKV is not alone in this hypothesis. Two external lines of work provide
stronger, independent evidence (see
[`docs/RELATED_WORK_KV_CACHE_COMPRESSION.md`](RELATED_WORK_KV_CACHE_COMPRESSION.md)):

* **KV-AdaQuant** ("More for Keys, Less for Values" / "Quantize What Counts",
  arXiv:2502.15075) gives a *theoretical* grounding: key projections
  systematically have larger spectral and Frobenius norms than value
  projections (the **key–value norm disparity**), and for a fixed bit budget,
  giving keys more bits strictly reduces quantization error
  (**key-prioritized quantization**). Their experiments report **4-bit keys /
  2-bit values ≈ 75% accuracy** versus the reversed **2-bit keys / 4-bit values
  ≈ 55%** — the same direction ExactKV observes in acceptance.

* **KIVI** (arXiv:2402.02750) treats keys and values asymmetrically in
  *granularity* — keys per-channel, values per-token — based on KV
  element-distribution analysis, reflecting that key distributions carry
  channel-structured outliers the softmax is sensitive to.

These are **external claims about accuracy/perplexity**, not ExactKV acceptance
results. ExactKV's contribution is to test whether the same asymmetry shows up
in **acceptance behaviour under full-KV verification**.

---

## 2. Why This Matters for ExactKV

ExactKV V1–V3 evaluated symmetric toy compressors:

| Compressor | K policy | V policy |
|---|---|---|
| `noop` | identity (full precision) | identity (full precision) |
| `int8` | per-tensor INT8 quantization | per-tensor INT8 quantization |
| `int4_sim` | per-tensor INT4 simulation | per-tensor INT4 simulation |
| `debug_noise` | additive Gaussian noise | additive Gaussian noise |

All four treat K and V identically.

**V4 implemented asymmetric compressor policies** for acceptance-rate comparison:

| V4 name | K bit-width | V bit-width | Simulated |
|---|---|---|---|
| `k8_v4_sim` | INT8 | INT4 (simulated) | yes ⚠️ |
| `k8_v2_sim` | INT8 | INT2 (simulated) | yes ⚠️ |
| `k4_v8_sim` | INT4 (simulated) | INT8 | yes ⚠️ |
| `k_full_v4_sim` | full precision | INT4 (simulated) | yes ⚠️ |
| `k4_v_full_sim` | INT4 (simulated) | full precision | yes ⚠️ |
| `k8_v_full` | INT8 | full precision | **no** |
| `k_full_v8` | full precision | INT8 | **no** |

> ⚠️ Compressors marked simulated store sub-INT8 values in `int8` containers.
> Do not cite their memory figures as real packed bit savings.

ExactKV is well-positioned to evaluate these policies because it already
separates the draft-and-verify loop from the compressor implementation.
`AsymmetricQuantSimCompressor` (V4) required only a new compressor class;
the verification engine and analysis layer were unchanged.

See [`docs/EXPERIMENT_003_ASYMMETRIC_KV_SWEEP.md`](EXPERIMENT_003_ASYMMETRIC_KV_SWEEP.md)
for acceptance-rate results across these policies.

### What ExactKV measures

The primary evaluation metric for asymmetric compression should NOT be
reconstruction MSE.  The ExactKV-specific metrics are:

* **acceptance rate** — fraction of drafted tokens accepted by the full-KV
  verifier.
* **average accepted length** — mean number of tokens accepted per
  verification round.
* **first divergence position** — where in the output sequence the lossy
  draft first diverges from full-KV greedy.
* **ExactKV failure count** — must always be 0 in a correct implementation.
* **lossy divergence count** — expected to be non-zero for lossy compressors;
  higher divergence indicates higher compression aggressiveness.

These metrics directly measure the deployment-relevant effect of compression
on generation quality, not the distance between compressed and full tensors.

### Matched-budget caveat: the cleanest V4 comparison

It is tempting to summarise Experiment 003 as "`k_full_v8` is best." That is
true but not the most informative comparison, because `k_full_v8` simply keeps
keys at full precision. The **cleanest matched-budget comparison** is:

| Compressor | K bits | V bits | Acceptance |
|---|---|---|---|
| `k8_v4_sim` ⚠️ | 8 | 4-sim | **0.858** |
| `k4_v8_sim` ⚠️ | 4-sim | 8 | **0.562** |

Both spend the *same* average effective bit budget — one side at 8 bits, the
other at 4 — but **putting the extra bits on the keys (`k8_v4_sim`) is far
better** than putting them on the values (`k4_v8_sim`). This is the comparison
that isolates the asymmetry, and it matches KV-AdaQuant's prediction. (⚠️ both
are simulated sub-INT8 in `int8` containers — not real packed bits.)

### Acceptance behaviour as the main evaluation axis

ExactKV deliberately uses **acceptance behaviour under full-KV verification** as
its primary evaluation axis, not reconstruction MSE. Two compressors at the same
average effective bit width can have very different acceptance rates (e.g.
`k8_v4_sim` vs `k4_v8_sim` above). MSE, computed symmetrically over all
positions, would not predict that gap; acceptance does (§3).

---

## 3. Why MSE May Be Insufficient

Reconstruction MSE (mean squared error between compressed and original K/V
tensors) is the most common evaluation metric for KV quantization in the
literature.  It is easy to compute and hardware-friendly to minimise.

However, MSE is likely to be a poor proxy for acceptance rate in the ExactKV
setting for the following reasons:

1. **Attention is not uniform.**  The softmax concentrates attention on a
   small subset of positions.  A quantization error in a high-attention key
   has far more impact on the output logits than the same error in a
   low-attention key.  MSE spreads error uniformly across all positions and
   does not weight by attention.

2. **The softmax amplifies key errors non-linearly.**  Even a small key error
   in the maximum-logit position can shift the argmax of the attention
   weights, changing which value vector is most strongly aggregated.  This is
   a threshold effect that MSE cannot capture.

3. **Value errors are additive and smoothed.**  After attention-weight
   formation, the value aggregation is a weighted sum.  Quantization noise in
   values is spread across the output dimension by the summation.  High
   bit-width for values may therefore be less important than high bit-width
   for keys.

4. **The relevant test is token correctness.**  In a token-generating system,
   the metric that matters for correctness is whether the argmax of the
   output logit vector matches the full-precision argmax.  This is exactly
   what the ExactKV acceptance rate measures.

**Recommendation:** Future ExactKV experiments should use acceptance rate as
the primary evaluation metric for asymmetric compressor comparisons, with MSE
reported as a secondary diagnostic only.

---

## 4. Workspace-Aware Memory Accounting

The compressed cache byte count reported by the current `MemorySummary` does
not capture the full memory footprint of inference with KV compression.

A compressor may store a tiny quantised cache, but if decoding requires
materialising a dense reconstructed scratch buffer before the attention
forward pass, the true peak memory during decode is:

```
peak_memory ≈ stored_kv_bytes + temporary_workspace_bytes
```

rather than just `stored_kv_bytes`.

### Proposed memory accounting schema (V4/V5)

Future ExactKV reports should distinguish the following memory fields:

| Field | Description |
|---|---|
| `stored_kv_bytes` | Bytes actually stored at rest (compressed format). |
| `materialized_working_kv_bytes` | Bytes needed to reconstruct a full-precision forward-pass cache. |
| `metadata_bytes` | Bytes for scales, zero-points, codebooks, or other compressor metadata. |
| `temporary_workspace_bytes` | Bytes for scratch buffers used during compression or decompression. |
| `active_gpu_kv_bytes` | Peak GPU KV bytes during a decode step (= `materialized_working_kv_bytes` for most quantisers). |
| `total_system_kv_bytes` | Estimated total system memory footprint (`stored + metadata + workspace`). |
| `supports_real_bytes_claim` | Boolean.  `False` for simulated compressors where stored bytes do not reflect real savings. |

This schema is more honest than reporting only `stored_kv_bytes` because it
exposes the hidden workspace cost that makes aggressive compression less
beneficial in practice than the storage ratio suggests.

### Why this matters for `int4_sim`

The current `int4_sim` compressor stores quantised values in `int8`
containers, making `stored_kv_bytes` equal to `int8` storage rather than
true packed 4-bit storage.  Even with real INT4 packing, the
`materialized_working_kv_bytes` during decode would be the full `fp32` or
`fp16` reconstructed cache — equal to the full-KV byte count.  This means
the "memory savings" from INT4 storage may be illusory unless the attention
kernel can operate directly on quantised tensors without dequantisation.

---

## 5. Suggested Future ExactKV Experiments

The following experiments are candidates for V4 or V5:

1. **Asymmetric KV compressor simulator.**
   Implement `AsymmetricQuantCompressor(k_bits, v_bits)` as a simulated
   compressor, reusing the existing per-tensor quantisation logic from
   `Int4SimCompressor` and `Int8Compressor`.

2. **K-only and V-only ablations.**
   Run `K-compressed / V-full` and `K-full / V-compressed` policies to
   isolate the contribution of each tensor type to acceptance-rate degradation.

3. **Asymmetric sweep over K bit-width × V bit-width.**
   Use the sweep runner to produce a grid of acceptance rates for all
   combinations of K ∈ {INT8, INT4-sim} × V ∈ {INT8, INT4-sim}.

4. **Acceptance-rate comparison: symmetric vs. asymmetric.**
   At the same average bit-width (e.g., K8/V4 vs. K6/V6 equivalent),
   compare acceptance rates to test whether asymmetry provides a practical
   benefit.

5. **Workspace-aware memory report schema.**
   Extend `MemorySummary` and the report schema to include
   `materialized_working_kv_bytes`, `metadata_bytes`, and
   `temporary_workspace_bytes`.

6. **Attention-aware divergence analysis.**
   Correlate first-divergence position with the prompt's attention entropy
   or attention-weight concentration to test whether high-entropy positions
   are more sensitive to KV quantisation.

7. **Prompt-category sensitivity analysis.**
   Use the `core`, `structured`, `code`, and `stress` prompt suites to
   measure whether certain prompt categories (e.g., structured JSON,
   code completion) are more or less sensitive to asymmetric compression.

---

## 6. Relation to TurboQuant+

[**TurboQuant+**](https://github.com/TheTom/turboquant_plus) is an external,
community research workspace extending **TurboQuant** (ICLR 2026) for local
inference. It independently explores asymmetric K/V compression and builds
concrete, rotation-based KV formats (`turbo2`/`turbo3`/`turbo4`, via PolarQuant +
Walsh–Hadamard rotation). **ExactKV does not implement TurboQuant or TurboQuant+
and claims none of its results.** All TurboQuant+ figures below are external
claims.

TurboQuant+ reports several findings that *support* ExactKV's asymmetric-K/V
direction:

* K and V need not be compressed symmetrically (independent K/V cache types).
* Quality degradation is dominated by **K** compression; **V** compression is
  reported as nearly free when K precision is maintained.
* MSE alone is an insufficient quality metric.
* Stored bytes are not the full memory story (motivating Sparse-V dequant and
  ExactKV V5 workspace accounting).

### Important nuance: `k8_v2_sim` does not refute TurboQuant-style V methods

ExactKV's `k8_v2_sim` had the lowest acceptance (0.330). This does **not**
contradict TurboQuant+'s "aggressive V is nearly free" finding:

* ExactKV's `k8_v2_sim` is a **naive simulated INT2 numeric quantizer in `int8`
  containers**, per-tensor symmetric, **with no rotation**.
* TurboQuant+'s `turbo2` is a **rotation-based PolarQuant format**: a
  Walsh–Hadamard rotation Gaussianises the distribution (reported kurtosis
  ~900 → ~2.9) *before* quantization, which is what makes 2-bit V viable.
* `k8_v2_sim` is **not** `turbo2`, not Sparse V, not layer-aware V, and not a
  real backend.

So ExactKV's result only shows that **naive aggressive V quantization can hurt
acceptance**; it says nothing against well-designed, rotation-based V-specific
formats.

### Future ExactKV directions inspired by TurboQuant+ (not implemented)

* **Sparse V dequantization** — attention-gated decode that skips low-weight V
  positions. A future *evaluation* target for ExactKV, not a current feature.
* **Layer-aware ("boundary") V policies** — higher V precision on the most
  sensitive layers (e.g. first/last few). A future asymmetric policy to evaluate.
* **Real asymmetric backend comparison** — wrap a real rotation-based format
  behind the `KVCompressor` protocol and evaluate it by acceptance behaviour
  (V6+, behind separate approval).

---

## 7. Scope Boundary

| Item | Status |
|---|---|
| Asymmetric compressor simulator (`AsymmetricQuantSimCompressor`) | **Implemented in V4.** |
| K-only / V-only ablations (`k8_v_full`, `k_full_v8`, `k_full_v4_sim`, `k4_v_full_sim`) | **Implemented in V4.** |
| K/V bit-width sweep (`k8_v4_sim`, `k8_v2_sim`, `k4_v8_sim`) | **Implemented in V4.** See [`docs/EXPERIMENT_003_ASYMMETRIC_KV_SWEEP.md`](EXPERIMENT_003_ASYMMETRIC_KV_SWEEP.md). |
| Workspace-aware memory schema | **Deferred to V5.** Not in V4. |
| Real INT4 packed storage | **Not implemented.** Future work. |
| Real compressor backends (KIVI, KVQuant, SnapKV) | **Out of scope for V4.** V5 candidate. |
| Speedup, throughput, or latency metrics | **Never.** ExactKV measures correctness and acceptance, not performance. |
| Production-readiness claim | **Never.** ExactKV is a research/experimental framework. |

---

## 8. Related External References

For a full survey of KV-cache compression, quantization, eviction, and serving
work — with proper attribution and an explicit statement of what ExactKV does
**not** implement — see
[`docs/RELATED_WORK_KV_CACHE_COMPRESSION.md`](RELATED_WORK_KV_CACHE_COMPRESSION.md).

Key external references for the asymmetric-K/V thesis:

* **KV-AdaQuant** — More for Keys, Less for Values: <https://arxiv.org/abs/2502.15075>
* **KIVI** — Asymmetric 2-bit KV quantization: <https://arxiv.org/abs/2402.02750>
* **KVQuant** — Sub-4-bit, per-channel/pre-RoPE key quant: <https://arxiv.org/abs/2401.18079>
* **TurboQuant** — Data-oblivious vector quantization: <https://arxiv.org/abs/2504.19874>

The following are external research notes referenced as background reading.
These links are not verified to be live; they are listed as directional
pointers for future literature review.

* **Asymmetric K/V Cache Compression:**
  <https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/asymmetric-kv-compression.md>

* **Sparse V Dequantization:**
  <https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/sparse-v-dequant.md>

* **Why MSE Fails for KV Quantization:**
  <https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/why-mse-fails-for-kv-quantization.md>

> These links reference external repositories and papers outside the ExactKV
> project.  ExactKV does not implement or endorse any specific backend
> described in those references.  They are listed as research context only.
