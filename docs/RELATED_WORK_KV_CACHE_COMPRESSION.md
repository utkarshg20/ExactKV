# Related Work: KV Cache Compression, Quantization, Eviction, and Serving

> **Positioning.** ExactKV is a **correctness-first verification and evaluation
> framework** for compressed-KV-cache behaviour. It does **not** implement
> TurboQuant, TurboQuant+, KIVI, KVQuant, KV-AdaQuant, KVTC, Palu, SnapKV, H2O,
> StreamingLLM, PyramidKV, LMCache, vLLM, or PagedAttention. It does **not** make
> speed, throughput, latency, runtime, or production-readiness claims. ExactKV's
> own compressors are research implementations; its sub-INT8 `_sim` compressors
> store values in `int8` containers and are **not** real packed-bit backends.
>
> Any speedup, throughput, perplexity, or memory-reduction numbers in this
> document describe **external work** and are attributed to that work. They are
> not ExactKV results.

This survey grounds ExactKV's roadmap (V5–V8) in the current KV-cache
compression literature. It summarises the major method families, explains why
reconstruction error alone is an insufficient quality metric, draws out the
workspace-memory implications, and states precisely what ExactKV does
differently.

---

## 1. Why related work matters for ExactKV

The KV cache is the dominant memory consumer during long-context LLM inference,
and a large body of work attacks it from several angles: numeric quantization,
asymmetric key/value precision, token eviction, low-rank and transform coding,
and systems-level cache management. ExactKV does not compete with any of these.
Instead, it asks a complementary question:

> *Given a lossy compressed KV cache, how often does drafting from it produce the
> same next token as full-precision greedy decoding — and when it diverges, is
> the divergence detected and corrected so the final output is exactly equal to
> full-KV output?*

This framing matters because most of the literature evaluates compression with
**reconstruction error** (MSE) or **downstream task accuracy / perplexity**.
ExactKV instead measures **acceptance behaviour under full-KV verification**:
acceptance rate, accepted length, first-divergence position, rejection and
correction counts, and exact token-ID equality. A serious related-work review is
necessary to (a) position ExactKV honestly relative to real backends, (b) make
sure ExactKV's asymmetric-K/V findings are framed as *evidence aligned with* the
literature rather than novel proof, and (c) ground the V5 workspace-memory work
in how real backends actually use memory.

---

## 2. Quantization methods

Numeric quantization reduces the bytes per KV element by storing values at lower
precision (INT8, INT4, sub-4-bit), usually with per-tensor, per-token, or
per-channel scales.

| Method | Core idea | Reported by the authors (external claims) |
|---|---|---|
| **KIVI** (Liu et al., ICLR 2024) | Tuning-free 2-bit. **Keys quantized per-channel, values per-token**, based on KV element-distribution analysis. A small full-precision residual is kept for streaming. | ~2.6× less peak memory with near-baseline quality on Llama/Falcon/Mistral. |
| **KVQuant** (Hooper et al., NeurIPS 2024) | Sub-4-bit via **per-channel key quantization**, **pre-RoPE key quantization**, per-layer non-uniform datatypes, and **per-vector dense-and-sparse** outlier isolation. | <0.1 perplexity degradation at 3-bit; long-context serving with custom CUDA kernels. |
| **TurboQuant** (Zandieh et al., ICLR 2026) | **Data-oblivious** vector quantization: random rotation makes coordinates follow a known Beta distribution, then optimal Lloyd-Max scalar quantization per coordinate. Near-optimal distortion without calibration. | Near-neutral quality at ~3–3.5 bits/coordinate on long-context benchmarks. |

Common thread: **keys and values have different statistics**, and naive uniform
quantization leaves accuracy on the table. KIVI and KVQuant both treat keys
specially (per-channel; pre-RoPE), reflecting that key distributions have
channel-structured outliers that the softmax is sensitive to.

ExactKV's own quantizers (`int8`, `int4_sim`) are **deliberately simple**
per-tensor symmetric quantizers. They exist to exercise the verification loop,
not to compete with these methods.

---

## 3. Asymmetric K/V precision methods

A growing line of work argues that **keys should get more precision than
values** at a fixed bit budget.

* **KV-AdaQuant / "Quantize What Counts: More for Keys, Less for Values"**
  (Hariri et al., arXiv:2502.15075). Provides a theoretical foundation:
  1. **Key–Value norm disparity** — key projections systematically have larger
     spectral and Frobenius norms than value projections, implying higher
     information density along the key path.
  2. **Key-prioritized quantization** — for a fixed total bit budget, giving
     keys more bits than values *strictly* reduces quantization error.
  Empirically, **4-bit keys / 2-bit values retained ~75% accuracy** on their
  setup while the reversed **2-bit keys / 4-bit values collapsed to ~55%**.

* **KIVI** (above) is itself asymmetric in *granularity*: per-channel keys,
  per-token values.

* **TurboQuant+** (community repo, see §10) reports the strongest version of
  this claim in practice — "all quality degradation comes from K compression"
  and "V compression is (nearly) free when K precision is maintained" — using a
  rotation-based V quantizer.

**Connection to ExactKV V4.** ExactKV's Experiment 003 (612 runs,
`Qwen/Qwen2.5-0.5B`, simulated compressors, 0 ExactKV failures) is *aligned with
the asymmetric-K/V thesis*: aggressive key compression was far more damaging to
acceptance than aggressive value compression.

| ExactKV V4 compressor | K bits | V bits | Acceptance | Note |
|---|---|---|---|---|
| `k_full_v8` | full | 8 | **0.988** | real bytes |
| `k8_v_full` | 8 | full | 0.953 | real bytes |
| `int8` | 8 | 8 | 0.953 | real bytes |
| `k_full_v4_sim` ⚠️ | full | 4-sim | 0.890 | sim |
| `k8_v4_sim` ⚠️ | 8 | 4-sim | 0.858 | sim |
| `k4_v8_sim` ⚠️ | 4-sim | 8 | 0.562 | sim |
| `int4_sim` ⚠️ | 4-sim | 4-sim | 0.553 | sim |
| `k8_v2_sim` ⚠️ | 8 | 2-sim | 0.330 | sim |

> ⚠️ `_sim` compressors store sub-INT8 values in `int8` containers — not real
> packed bits.

The cleanest matched-budget comparison is **`k8_v4_sim` (0.858) vs `k4_v8_sim`
(0.562)**: both spend the same average effective bit budget (one side 8, one
side 4), but giving the *keys* the extra bits is dramatically better. This is the
same direction KV-AdaQuant predicts. **It is evidence, not universal proof** — it
is one small model with simulated compressors (see §9).

---

## 4. Token eviction and sparse-retention methods

These methods reduce the cache by **dropping or retaining whole tokens** rather
than lowering per-element precision. They are orthogonal to quantization and can
be combined with it.

| Method | Core idea |
|---|---|
| **StreamingLLM** (Xiao et al., 2023) | Keep the first few tokens as **attention sinks** plus a sliding window of recent tokens; enables effectively unbounded streaming. |
| **H2O — Heavy-Hitter Oracle** (Zhang et al., 2023) | Dynamically evict tokens with low **cumulative attention score**; keep "heavy hitters" + recent tokens. |
| **SnapKV** (Li et al., 2024) | Compress the **prompt** KV during prefill using a small observation window at the prompt's end to vote for important prefix positions, then cluster and retain them. |
| **PyramidKV** (Cai et al., 2024) | **Layer-aware budget**: allocate more cache to lower layers (dispersed attention) and less to higher layers (concentrated attention), matching observed attention-sparsity growth. |

Eviction methods change *which tokens exist* in the cache. They are not lossless,
and they generally do not target exact-token reproduction. ExactKV does not
implement any eviction method today; they are a V7 research direction
(§10, backlog).

---

## 5. Transform coding and storage-oriented methods

These methods exploit **redundancy in the hidden dimension** or **temporal
coherence** rather than (or in addition to) per-element bit-width reduction.

| Method | Core idea | External claims |
|---|---|---|
| **Palu** (Chang et al., ICLR 2025) | **Low-rank projection** of the hidden dimension: decompose linear layers, cache smaller intermediate states, reconstruct K/V on the fly. | ~50% KV memory reduction; more when combined with quantization. |
| **KVTC** (Staniszewski & Łańcucki, ICLR 2026) | **Transform coding**: PCA-based feature decorrelation + adaptive quantization + entropy coding; brief calibration, model weights unchanged. | Up to ~20× compression with maintained long-context/reasoning accuracy (their report). |
| **SVD / low-rank family** (e.g. SVD-based KV quantization, linear-compression approaches) | Project K/V onto a low-rank subspace, optionally combined with quantization of the reduced representation. | Memory reduction by exploiting hidden-dimension redundancy. |

These methods are important for the V5 workspace-memory discussion because they
introduce **codebooks, projection matrices, and entropy-coder state** as
metadata, and they **reconstruct a dense working cache on the fly** — exactly the
kind of stored-vs-materialized distinction V5 tracks.

> Note on SVD/low-rank entries: ExactKV treats these as a **family-level**
> directional pointer. Specific named systems (e.g. SVDq, KVLinC) are recorded as
> low-rank / transform-coding approaches to revisit; ExactKV does not reproduce
> their specific numbers.

---

## 6. Serving and KV-cache systems

These systems manage where the cache **lives** and how it is **reused**, rather
than compressing individual tensors.

| System | Core idea | External claims |
|---|---|---|
| **vLLM / PagedAttention** (Kwon et al., SOSP 2023) | OS-style **paging** of the KV cache into fixed-size blocks; near-zero fragmentation; **prefix sharing** across requests. | 2–4× throughput vs prior serving systems at similar latency (their report). |
| **LMCache** (Liu et al., 2025) | External KV-cache layer that **offloads** and **shares** caches across queries/engines (CPU/disk/remote), with prefill–decode disaggregation and a modular connector API. | Up to ~15× throughput improvement combined with vLLM in their workloads (their report). |

Relevance to ExactKV: these systems define the realistic **memory hierarchy** a
real backend lives in (GPU pages, CPU offload, remote storage). ExactKV's V5
workspace-memory schema names the categories (stored / materialized / metadata /
temporary) so that, *if* a real backend is ever added, ExactKV can describe its
footprint honestly. ExactKV does **not** integrate with vLLM or LMCache and makes
no serving claims.

---

## 7. Why MSE and reconstruction error are not enough

Most quantization papers optimise or report **mean-squared error** between the
compressed and original K/V tensors. MSE is convenient and hardware-friendly,
but it is an imperfect proxy for what actually matters in token generation:

1. **Attention is non-uniform.** The softmax concentrates weight on a few
   positions. A quantization error in a high-attention key affects the output
   far more than the same error in a low-attention key. MSE weights all positions
   equally.

2. **The softmax amplifies key errors non-linearly.** A small key perturbation in
   the max-logit position can shift the attention argmax, changing which value is
   aggregated — a threshold effect MSE cannot see. KV-AdaQuant formalises this via
   the spectral-norm/error-amplification argument; KIVI and KVQuant address it via
   per-channel and pre-RoPE key handling.

3. **Value errors are additive and smoothed.** Value quantization noise is spread
   across the output by the attention-weighted sum, so values often tolerate more
   aggressive compression. TurboQuant+ pushes this to "V compression is nearly
   free when K precision is maintained."

4. **The deployment-relevant test is token correctness.** What matters is whether
   the argmax of the output logits matches the full-precision argmax. That is
   exactly what ExactKV's **acceptance rate** measures, and what its **exact
   token-ID equality** gate enforces after correction.

ExactKV therefore uses **acceptance behaviour under full-KV verification** as its
primary axis, with reconstruction error (if ever computed) as a secondary
diagnostic only.

---

## 8. Workspace-memory implications

A recurring lesson across the literature is that **stored compressed bytes are
not the whole memory story**:

* **Materialized working cache.** KIVI keeps a full-precision residual; Palu and
  KVTC reconstruct dense K/V on the fly; quantizers dequantize to the model dtype
  before attention. The peak footprint during a decode step includes this
  **materialized working cache**, which is often the size of the full-precision
  cache regardless of how small the stored form is.

* **Metadata.** Scales, zero-points, codebooks (Lloyd-Max centroids), projection
  matrices (Palu), and entropy-coder tables (KVTC) all consume bytes that a naive
  "compressed bytes" number ignores.

* **Temporary dequantization workspace and dense scratch buffers.** Rotation,
  dequantization, and reconstruction allocate transient buffers. TurboQuant+'s
  "Sparse V dequant" work exists precisely because decode-time dequantization
  cost is real.

This is the motivation for ExactKV V5's **workspace-aware memory accounting**,
which distinguishes `stored_kv_bytes`, `materialized_working_kv_bytes`,
`metadata_bytes`, `temporary_workspace_bytes`, and `total_kv_footprint_bytes`.
For every current ExactKV compressor, `materialized_working_kv_bytes ==
full_kv_bytes`, because all of them dequantize to full precision for the forward
pass — so stored-byte savings do not, by themselves, reduce the decode-time
working footprint. V5 makes that explicit instead of hiding it behind a single
compression ratio.

---

## 9. What ExactKV does differently

* **Verification, not just compression.** ExactKV drafts with a lossy compressed
  cache and **verifies every drafted token against full-KV greedy decoding**,
  correcting any divergence. The final output is exactly equal to full-KV output
  under greedy decoding. None of the methods above provide this exact-equality
  guarantee; they accept some quality loss.

* **Acceptance behaviour as the primary metric.** ExactKV reports acceptance
  rate, accepted length, first-divergence position, rejection and correction
  counts — not perplexity, not MSE, and not throughput.

* **Compressor-agnostic harness.** Any object satisfying the `KVCompressor`
  protocol can be evaluated without touching the verification engine. V4's
  asymmetric compressors required only a new compressor class.

* **No performance claims.** ExactKV measures correctness and acceptance only. It
  makes **no** speedup, throughput, latency, runtime, or production-readiness
  claim.

* **Honest about simulation.** ExactKV's sub-INT8 `_sim` compressors store values
  in `int8` containers and set `supports_real_bytes_claim=False`. They are not
  real packed-bit backends and their stored-byte figures are not real savings.

In short: the literature builds **compression backends**; ExactKV builds a
**verified evaluation layer** that can sit on top of any of them and report
whether — and where — a compressed cache changes the generated tokens.

---

## 10. TurboQuant+ and asymmetric K/V backend ideas

[**TurboQuant+**](https://github.com/TheTom/turboquant_plus) is an external,
community research workspace that integrates and extends **TurboQuant**
(Zandieh et al., ICLR 2026) for `llama.cpp`-style local inference. It builds
concrete KV compression formats (`turbo2`/`turbo3`/`turbo4`, using PolarQuant +
Walsh–Hadamard rotation) and backend/kernel ideas, and independently explores
**asymmetric K/V compression**.

**ExactKV does not implement TurboQuant or TurboQuant+ and does not claim any
TurboQuant+ result.** Every number below is an external TurboQuant+/TurboQuant
claim, not an ExactKV measurement.

### Why the two projects are complementary

* **TurboQuant+ builds compression backends** — real bit-packed formats, rotation
  kernels, and `llama.cpp` integration.
* **ExactKV evaluates compressed-cache policies** — through verified acceptance
  behaviour under full-KV correction, with an exact-token-equality guarantee.

A real TurboQuant-style format could, in principle, be wrapped behind ExactKV's
`KVCompressor` protocol (a V6+ direction, not implemented) and then evaluated by
acceptance behaviour rather than only by perplexity.

### TurboQuant+ themes that support ExactKV's direction

1. **K and V need not be compressed symmetrically.** TurboQuant+ supports
   independent K/V cache types (e.g. `q8_0`-K + `turbo`-V).
2. **Keys drive attention routing.** Because keys feed the query–key dot product
   and softmax, K compression is more fragile — consistent with ExactKV's
   acceptance collapse under aggressive key compression.
3. **Values are aggregated after attention weights form**, so V may tolerate more
   aggressive or more specialised compression — TurboQuant+ reports V compression
   as nearly free when K precision is maintained.
4. **MSE is not sufficient as the only quality metric**, because attention is
   nonlinear and error sensitivity is not uniform across positions (also the
   `why-mse-fails-for-kv-quantization` writeup).
5. **Stored compressed KV bytes are not the full memory story** when decode needs
   materialized working buffers or temporary dequantization workspace (motivating
   ExactKV V5 and TurboQuant+'s Sparse V dequant work).
6. **Sparse V dequantization and layer-aware ("boundary") V compression** are
   future backend ideas — not current ExactKV features.

### Important nuance: ExactKV's `k8_v2_sim` does **not** refute TurboQuant+

ExactKV's `k8_v2_sim` had the lowest acceptance (0.330). This does **not**
contradict TurboQuant+'s "aggressive V is nearly free" finding, because:

* ExactKV's `k8_v2_sim` is a **naive simulated INT2 numeric quantizer stored in
  `int8` containers**, with per-tensor symmetric scaling and **no rotation**.
* TurboQuant+'s `turbo2` is a **rotation-based PolarQuant format** — a
  Walsh–Hadamard rotation Gaussianises the distribution (reported kurtosis ~900 →
  ~2.9) *before* quantization, which is precisely what makes 2-bit V viable.
* `k8_v2_sim` is **not** `turbo2`, not Sparse V, not layer-aware V, and not a real
  backend.

So ExactKV's result only shows that **naive aggressive V quantization can hurt
acceptance**. It says nothing against well-designed, rotation-based V-specific
formats, which remain a promising future direction. Building and verifying such a
format inside ExactKV is explicitly future work, not a current capability.

---

## 11. Source table

| Work | Type | Link |
|---|---|---|
| VeriCache (ExactKV's algorithmic basis) | Draft-with-compressed-KV, verify-with-full-KV | arXiv:2605.17613 |
| KIVI | Asymmetric 2-bit quantization (per-channel K, per-token V) | <https://arxiv.org/abs/2402.02750> · <https://github.com/jy-yuan/KIVI> |
| KVQuant | Sub-4-bit; per-channel + pre-RoPE key quant; dense-and-sparse | <https://arxiv.org/abs/2401.18079> · <https://github.com/SqueezeAILab/KVQuant> |
| KV-AdaQuant ("More for Keys, Less for Values" / "Quantize What Counts") | Asymmetric K/V bit allocation; norm-disparity theory | <https://arxiv.org/abs/2502.15075> · <https://github.com/mohsenhariri/spectral-kv> |
| TurboQuant | Data-oblivious vector quantization (rotation + Lloyd-Max) | <https://arxiv.org/abs/2504.19874> · <https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/> |
| TurboQuant+ | Community TurboQuant integration + asymmetric K/V findings | <https://github.com/TheTom/turboquant_plus> |
| Palu | Low-rank projection KV compression | <https://arxiv.org/abs/2407.21118> · <https://github.com/shadowpa0327/Palu> |
| KVTC | Transform coding (PCA + adaptive quant + entropy coding) | <https://openreview.net/forum?id=aNVKROYpLB> |
| SnapKV | Prompt-KV eviction via observation window | <https://arxiv.org/abs/2404.14469> |
| H2O | Heavy-hitter attention-score eviction | (Zhang et al., 2023) |
| StreamingLLM | Attention sinks + sliding window | (Xiao et al., 2023) |
| PyramidKV | Layer-aware pyramidal cache budget | <https://arxiv.org/abs/2406.02069> |
| vLLM / PagedAttention | Paged KV memory management + prefix sharing | <https://arxiv.org/abs/2309.06180> · <https://github.com/vllm-project/vllm> |
| LMCache | External KV-cache offload/reuse layer | <https://arxiv.org/abs/2510.09665> · <https://github.com/LMCache/LMCache> |
| SVDq / KVLinC (low-rank/transform family) | Low-rank / linear KV compression | directional — revisit in backlog |

> Links are provided for convenience and may change. Inclusion here is **not** an
> endorsement and does **not** imply ExactKV integrates with or reproduces any of
> these systems.

---

## 12. Roadmap implications

The literature supports a staged, correctness-first path:

* **V5 — workspace-aware memory accounting.** Make memory honest (stored vs
  materialized vs metadata vs temporary), motivated by KIVI residuals, Palu/KVTC
  reconstruction, and TurboQuant+ Sparse-V observations. No backend, no
  performance claims.
* **V6 — real backend adapter interface + first backend candidate.** Design a
  `BackendAdapter` so a real format (e.g. a KIVI- or TurboQuant-style quantizer)
  could be wrapped behind `KVCompressor` and evaluated by acceptance behaviour.
  Implementation only behind separate approval.
* **V7 — attention-aware and V-specific backend ideas.** Sparse V
  dequantization, layer-aware ("boundary") V compression, and real asymmetric
  compressor comparisons — evaluated, not just reconstructed.
* **V8 — serving-stack integration.** Only after correctness and acceptance are
  well understood would ExactKV consider any serving-stack (vLLM/LMCache)
  evaluation context — and even then, without adopting their performance claims as
  ExactKV's own.

See [`docs/RESEARCH_BACKLOG.md`](RESEARCH_BACKLOG.md) for the concrete experiment
backlog and [`docs/FUTURE_RESEARCH_ASYMMETRIC_KV.md`](FUTURE_RESEARCH_ASYMMETRIC_KV.md)
for the asymmetric-K/V deep dive.
