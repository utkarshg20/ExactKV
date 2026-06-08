# ExactKV Research Backlog

> **Status:** Backlog of candidate future experiments. **Nothing here is
> implemented.** This is a planning artifact, grounded in
> [`RELATED_WORK_KV_CACHE_COMPRESSION.md`](RELATED_WORK_KV_CACHE_COMPRESSION.md).
>
> Guardrails for everything below: correctness-first (`exactkv_output_ids ==
> full_output_ids` under greedy decoding), and **no** speedup, throughput,
> latency, runtime, or production-readiness claims. ExactKV's `_sim` compressors
> are not real packed-bit backends. Any real backend is added only behind the
> existing `KVCompressor` protocol and only with separate explicit approval.

---

## How to read this backlog

Each item lists: the idea, why it is interesting for ExactKV (always an
*acceptance-behaviour* question), the roadmap version it most naturally belongs
to, and the hard caveats. Items are **not** scheduled.

| Roadmap | Theme |
|---|---|
| V5 | Workspace-aware memory accounting (in progress / planned) |
| V6 | Real backend adapter interface + first backend candidate |
| V7 | Attention-aware and V-specific backend ideas |
| V8 | Serving-stack integration (evaluation context only) |

---

## Real backend adapters (V6 — design first, implement only on approval)

### B1. Real KIVI adapter
* **Idea:** Wrap a real KIVI-style quantizer (per-channel keys, per-token values,
  full-precision residual) behind the `KVCompressor` protocol.
* **ExactKV question:** Does KIVI's granularity asymmetry translate into higher
  *acceptance* than ExactKV's naive per-tensor quantizers at a comparable budget?
* **Caveats:** Real packed storage and residual handling; `supports_real_bytes_claim`
  must be set honestly. No throughput claims.

### B2. KVQuant-style pre-RoPE key quantization adapter
* **Idea:** Adapter that quantizes keys **before** RoPE and uses per-channel
  scales + dense-and-sparse outlier handling.
* **ExactKV question:** Does pre-RoPE key quantization preserve acceptance better
  than post-RoPE per-tensor key quantization?
* **Caveats:** Requires hooking the key path before rotary embedding; correctness
  of the verification loop must be re-validated.

### B3. TurboQuant-style adapter
* **Idea:** Wrap a real rotation-based format (Walsh–Hadamard rotation +
  Lloyd-Max / PolarQuant scalar quantization) behind `KVCompressor`.
* **ExactKV question:** Does rotation-based 2–4 bit V quantization recover the
  acceptance that ExactKV's naive `k8_v2_sim` lost? (See the `k8_v2_sim` nuance in
  the related-work survey — naive INT2 ≠ `turbo2`.)
* **Caveats:** ExactKV would *evaluate*, not claim, TurboQuant results. No
  TurboQuant performance numbers presented as ExactKV's.

### B4. KVTC-style storage compression
* **Idea:** Transform-coding adapter (PCA decorrelation + adaptive quantization +
  entropy coding) for compact at-rest storage.
* **ExactKV question:** How does entropy-coded storage interact with acceptance,
  and how large is the metadata (codebooks/coder state) under the V5 memory
  schema?
* **Caveats:** Requires calibration; metadata accounting is essential.

---

## Attention-aware and V-specific ideas (V7)

### B5. Sparse V dequantization
* **Idea:** Evaluate attention-gated decode that skips low-weight V positions.
* **ExactKV question:** Does sparsifying V dequantization change acceptance vs a
  dense V cache at the same stored budget?
* **Caveats:** Not TurboQuant-specific; ExactKV would evaluate acceptance, not
  decode speed.

### B6. Layer-aware ("boundary") V policies
* **Idea:** Higher V precision on the most sensitive layers (e.g. first/last few),
  lower precision elsewhere.
* **ExactKV question:** Can a layer-aware V budget match full-V acceptance at a
  lower average bit budget?
* **Caveats:** Per-layer policy plumbing; still a simulated study unless paired
  with a real backend.

### B7. Real asymmetric compressor comparison
* **Idea:** Compare real asymmetric formats against ExactKV's simulated ones on
  acceptance behaviour, with explicit simulated-vs-real labelling.
* **Caveats:** Only meaningful once a real backend (B1–B4) exists.

### B8. Attention-aware divergence analysis
* **Idea:** Correlate first-divergence position with attention entropy / weight
  concentration.
* **ExactKV question:** Are high-entropy positions more sensitive to KV
  quantization? Experiment 003 data is a starting point.
* **Caveats:** Analysis-only; no generation-logic change.

---

## Token eviction methods (V7 — orthogonal to quantization)

### B9. Eviction-policy evaluation (PyramidKV, SnapKV, H2O, StreamingLLM)
* **Idea:** Evaluate token-eviction policies as `KVCompressor`-style strategies
  that drop/retain whole tokens rather than lowering precision.
* **ExactKV question:** How does eviction affect acceptance and first-divergence
  position under full-KV verification? (Eviction is lossy and not designed for
  exact reproduction, so this is an instructive stress test.)
* **Caveats:** These methods change which tokens exist; verification semantics
  must be defined carefully.

---

## Serving-stack integration (V8 — evaluation context only)

### B10. LMCache / vLLM (PagedAttention) integration context
* **Idea:** Use a serving stack only as an *evaluation context* for compressed
  caches, never as a source of performance claims.
* **ExactKV question:** Can ExactKV's acceptance evaluation run against
  caches produced/managed by a real serving stack?
* **Caveats:** No throughput/latency benchmarks; no production-serving claims.
  This is the last and most speculative item; correctness and acceptance must be
  well understood first.

---

## Explicitly out of scope unless separately approved

vLLM integration as a performance claim · LMCache as a performance claim · CUDA /
Triton kernels · CPU offload · batching · sampling · parallel (single-pass)
verification · bonus-token acceptance · throughput / latency / tokens-per-second
benchmarks · speedup claims · production-serving claims · presenting any external
backend's results as ExactKV's own.
