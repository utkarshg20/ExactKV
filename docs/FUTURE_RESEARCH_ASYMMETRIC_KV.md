# Future Research: Asymmetric K/V Compression and Workspace-Aware Memory Accounting

> **Scope:** This document describes V4/V5 research directions.
> None of this is implemented in V3.
> No real compressor backends, no speedup claims, no production-readiness claims.

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

---

## 2. Why This Matters for ExactKV

ExactKV currently evaluates symmetric toy compressors:

| Compressor | K policy | V policy |
|---|---|---|
| `noop` | identity (full precision) | identity (full precision) |
| `int8` | per-tensor INT8 quantization | per-tensor INT8 quantization |
| `int4_sim` | per-tensor INT4 simulation | per-tensor INT4 simulation |
| `debug_noise` | additive Gaussian noise | additive Gaussian noise |

All four treat K and V identically.

A future ExactKV direction is to support and evaluate asymmetric compressor
policies such as:

| Policy label | K bit-width | V bit-width |
|---|---|---|
| K8/V4 | INT8 | INT4 (simulated) |
| K8/V2 | INT8 | INT2 (simulated) |
| K4/V8 | INT4 (simulated) | INT8 |
| K-full/V-int8 | full precision | INT8 |
| K-int8/V-full | INT8 | full precision |

ExactKV is well-positioned to evaluate these policies because it already
separates the draft-and-verify loop from the compressor implementation.
Adding an `AsymmetricQuantCompressor` would require only a new compressor
class and no changes to the verification engine or the analysis layer.

### What ExactKV would measure

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

## 6. Scope Boundary

| Item | Status |
|---|---|
| Asymmetric compressor implementation | **Not in V3.** Candidate for V4. |
| K-only / V-only ablations | **Not in V3.** Candidate for V4. |
| Workspace-aware memory schema | **Not in V3.** Candidate for V4/V5. |
| Real INT4 packed storage | **Not implemented.** Future work. |
| Real compressor backends (KIVI, KVQuant, SnapKV) | **Out of scope for V4.** |
| Speedup, throughput, or latency metrics | **Never.** ExactKV measures correctness and acceptance, not performance. |
| Production-readiness claim | **Never.** ExactKV is a research/experimental framework. |

---

## 7. Related External References

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
