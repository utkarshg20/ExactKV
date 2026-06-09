# ExactKV Future Roadmap: V6 → V8

**Status:** Planning document. No code is written here.
**Supersedes pointer:** [`docs/ROADMAP.md`](ROADMAP.md) §V6–V8 stubs (this document expands them).
**Builds on:** V5 (`v0.5.0`) — workspace-aware memory accounting complete,
Experiment 004 (`exactkv_failures == 0` across 340 runs).

> **This is not a schedule.** No version here is approved for implementation.
> Each version must receive separate explicit approval before any code is written.
>
> Hard constraints throughout:
> - No throughput, latency, tokens/sec, speedup, or production-readiness claims.
> - No real backends implemented without explicit per-version approval.
> - No CUDA/Triton kernels, no CPU offload, no batching, no sampling,
>   no parallel verification, no bonus-token acceptance.
> - Simulated (`_sim`) compressors remain int8-container storage only.
> - ExactKV does not implement TurboQuant+, KIVI, KVQuant, KVTC, Palu,
>   LMCache, vLLM, or PagedAttention unless a specific version is approved
>   and describes exactly what will and will not be implemented.

---

## Quick summary

| Version | Main goal | GPU needed? | Public claim allowed? | Primary output |
|---|---|---:|---|---|
| V6 | Real backend adapter interface + first backend | Yes-ish | No speed claims | Backend adapter + Experiment 005 |
| V7 | Attention-aware / V-specific experiments | Yes | No speed claims | Research experiments (006, 006b) |
| V8 | Serving-stack context + final public launch | Yes | Only if carefully measured | Final release package |

---

## Global policies (all versions V6–V8)

### No-performance-claim policy

The following fields must **never** appear as data fields, table columns, or
key-value pairs in any ExactKV output — in code, tests, CLI, or Markdown reports:

```
tokens_per_second
throughput
latency
speedup
runtime_seconds
```

These terms may appear **only in explicit negation prose** (e.g. "This report
does not measure tokens/second", "No latency claim is made here"). Any new
version must pass the existing `_assert_no_forbidden_fields` audit before being
considered complete.

### Global exactness gate

`exactkv_output_ids == full_output_ids` under greedy decoding must hold for
every run in every experiment. `exactkv_failures == 0` is a hard gate for every
phase. Any change to the generation pipeline must be re-validated against this
gate across the full core-suite sweep (≥ 34 prompts × all compressors).

### Global memory-honesty policy

All memory figures must distinguish:

- `stored_kv_bytes` — bytes in the persistent cache representation.
- `materialized_working_kv_bytes` — bytes of the working copy during attention.
- `metadata_bytes` — scales, zero-points, codebooks, projection matrices.
- `temporary_workspace_bytes` — transient scratch during compress/materialize.
- `total_kv_footprint_bytes` — accounting sum (conservative, NOT measured peak).

Any new backend must populate all five V5 workspace fields. Simulated backends
must set `supports_real_bytes_claim=False`. No "compression ratio" or
"memory reduction factor" figure may be presented as a real memory saving unless
`supports_real_bytes_claim=True` **and** `materialized_working_kv_bytes <
full_kv_bytes`.

### Stopping conditions

The roadmap should be paused or revised if:

1. **Integration proves infeasible without breaking exactness.** If wrapping a
   real backend requires changes to the verification loop that invalidate the
   `exactkv_failures == 0` gate, stop, document the barrier, and reassess.
2. **Scope drifts toward performance engineering.** If any version starts
   requiring CUDA kernels, multi-GPU serving, or throughput baselines as primary
   deliverables, stop. That is not ExactKV's goal.
3. **No backend is easy to integrate at V6.** V6 may validly end at
   adapter-interface design + a minimal proof-of-concept (or even design-only)
   if wrapping a real backend turns out to require deep framework changes.
4. **The public-launch story changes.** If the target audience or framing of V8
   changes, reassess what the launch narrative should claim before proceeding.

---

## V6 — Real Backend Adapter Interface and First Backend Candidate

### 1. Why V6 matters after V5

V5 proved that ExactKV's memory accounting is honest: `materialized_working_kv_bytes
== full_kv_bytes` for **all current compressors**, because every current compressor
dequantises to full precision for attention. The stored-byte savings are real for
`int8`, `k8_v_full`, and `k_full_v8`, but they do not reduce the peak working
footprint at all.

V5 also proved that simulated sub-INT8 compressors (`_sim`) use int8 containers.
Their "stored_kv_bytes" figures reflect int8 container reality, not hypothetical
packed 4-bit or 2-bit bytes.

**The natural next question is:** What happens when a *real* compressed backend
plugs into the `KVCompressor` protocol?

- Does a per-channel key quantizer (KIVI-style) improve acceptance versus
  per-tensor key quantization?
- Does a rotation-based V compressor (TurboQuant-style) recover the acceptance
  that naive `k8_v2_sim` lost?
- Does a real packed-4-bit format actually reduce `materialized_working_kv_bytes`,
  or does it still dequantise to full precision?

V6 answers these questions by introducing a **clean adapter boundary** — a
`BackendAdapter` layer that lets a real format plug into `KVCompressor` without
changing the verification engine — and by testing at least one real backend.

### 2. What V6 should add

#### 2.1 `BackendAdapter` interface design

A `BackendAdapter` is a thin wrapper that translates between a real quantisation
format's compress/decompress API and ExactKV's `KVCompressor` protocol.

Responsibilities:
- **`compress(full_kv_state) → CompressedKVState`** — calls the real backend's
  quantise/store operation.
- **`materialize_for_draft(compressed_state) → FullKVState`** — calls the real
  backend's dequantise operation to produce the working cache for attention.
- **`stats(full_kv_state) → CompressionStats`** — returns honest V5 workspace
  fields, including whether the backend uses a real packed format.
- **`capabilities() → CompressorCapabilities`** — declares `is_simulated`,
  `supports_real_bytes_claim`, `key_bit_width`, `value_bit_width`, `asymmetric`.

The `BackendAdapter` must **not** change the verification engine, the
draft-verify-commit loop, or the acceptance/rejection bookkeeping.
The `KVCompressor` protocol is the only interface the verification engine sees.

#### 2.2 Required `CompressorCapabilities` metadata for real backends

| Field | Real backend requirement |
|---|---|
| `is_simulated` | Must be `False` for all real backends |
| `supports_real_bytes_claim` | Must be `True` only when `stored_kv_bytes` reflects actual packed bytes |
| `key_bit_width` | Actual bits used per key element (or `full`) |
| `value_bit_width` | Actual bits used per value element (or `full`) |
| `asymmetric` | `True` if K and V use different compression policies |
| `backend_name` | New: name of the external backend (e.g. `"kivi"`, `"turboquant"`) |
| `backend_version` | New: version string of the external backend library |

#### 2.3 V5 workspace fields for real backends

A real backend with genuine packed storage may be the first to set
`materialized_working_kv_bytes < full_kv_bytes`, if and only if it uses a
fused dequantise-and-attend kernel that avoids materialising a full-precision
working copy. If a backend still dequantises to full precision for attention
(the common case), `materialized_working_kv_bytes` remains equal to
`full_kv_bytes`.

All five V5 fields must be populated. `metadata_bytes` must include:
- Per-channel or per-token scales (not just per-tensor)
- Codebook or projection metadata if applicable
- Any residual tensors kept alongside the compressed cache

`total_kv_footprint_bytes` remains a conservative accounting sum, NOT measured
peak GPU memory. Active GPU profiling is still deferred.

#### 2.4 Experiment 005

**Name:** First Real Backend Adapter Comparison

**Goal:** Run a core-suite sweep with at least one real backend adapter
alongside ExactKV's simulated and real-INT8 compressors, reporting:

- `exactkv_failures` (must be 0)
- acceptance rate by compressor (real backend vs simulated equivalents)
- first divergence position distribution
- rejection and correction counts
- V5 workspace fields (stored, materialized, metadata, total) for the real backend
- `supports_real_bytes_claim` and `is_simulated` values

**Honesty requirements:**
- If the real backend's `stored_kv_bytes` is genuinely smaller than int8
  container bytes, say so — and say whether `materialized_working_kv_bytes` is
  also smaller or still equals `full_kv_bytes`.
- Do not cite the external backend's perplexity or benchmark numbers as ExactKV
  results.
- Do not claim the real backend's acceptance rate proves anything about
  production speedup or throughput.

### 3. What V6 explicitly should not add

- ❌ Real backend throughput, latency, or speedup benchmarks.
- ❌ CUDA/Triton kernels (unless the backend brings them from its own library;
  in that case, ExactKV still does not write or claim any kernel).
- ❌ CPU offload, batching, sampling, or parallel verification.
- ❌ Active GPU memory profiling (`torch.cuda.memory_reserved` etc.) — this is
  still deferred from V5.
- ❌ New compressors beyond what the first backend candidate requires.
- ❌ Changes to the draft-verify-commit loop or verification engine.
- ❌ Any claim that a real backend outperforms simulated ones in terms of
  generation speed.

### 4. Candidate first backends

#### Option A: KIVI
- **What it is:** Per-channel INT8/INT2 key quantization + per-token INT2/INT4
  value quantization + a small full-precision residual.
  (Liu et al., ICLR 2024; arXiv:2402.02750; github.com/jy-yuan/KIVI)
- **Why it is a good first candidate:** KIVI is widely cited, has a published
  implementation, treats K and V asymmetrically (aligned with ExactKV's V4
  findings), and the residual approach is a clear example of why
  `stored_kv_bytes` alone understates the footprint.
- **Integration challenge:** KIVI's per-channel/per-token scales differ from
  ExactKV's current per-tensor model; the adapter would need to translate
  scale granularity into honest `metadata_bytes` accounting.
- **GPU requirement:** KIVI's original kernels are CUDA. The adapter must be
  tested on a CUDA device. A CPU-only reference path may not exist in the
  KIVI repo.

#### Option B: kvpress
- **What it is:** A Hugging Face-native library for KV cache compression,
  with multiple supported compressors via a hooks API.
  (github.com/NVIDIA/kvpress)
- **Why it is a good first candidate:** kvpress is designed for HF compatibility
  and has minimal new infrastructure requirements. It uses existing HF
  `DynamicCache`-style hooks. An ExactKV `BackendAdapter` wrapping kvpress
  would be the minimal-friction real-backend experiment.
- **Integration challenge:** kvpress uses hooks that modify the forward pass
  directly; verifying that the draft-verify loop still works correctly with
  these hooks is non-trivial and must be fully re-validated.
- **GPU requirement:** CPU-compatible for smaller compressors, but real
  savings require a GPU.

#### Option C: KVQuant-style adapter
- **What it is:** Per-channel key quantization + pre-RoPE key quantization +
  per-vector dense-and-sparse outlier handling.
  (Hooper et al., NeurIPS 2024; arXiv:2401.18079)
- **Why it is interesting:** KVQuant represents the state of the art in key
  quantization granularity, and it raises the question of whether pre-RoPE
  key quantization improves acceptance over ExactKV's current post-RoPE
  per-tensor key quantization.
- **Integration challenge:** Requires hooking the key path before the rotary
  embedding application. This is a deeper verification loop change and
  correctness re-validation is more complex. Not recommended as the first
  backend.

#### Option D: TurboQuant-style adapter
- **What it is:** Data-oblivious rotation-based vector quantization
  (Walsh–Hadamard rotation → Lloyd-Max / PolarQuant scalar quantization).
  (Zandieh et al., ICLR 2026; arXiv:2504.19874)
- **Why it is interesting:** Rotation-based V quantization is the key claim
  behind "V compression is nearly free when K precision is maintained"
  (TurboQuant+). ExactKV's `k8_v2_sim` lost significant acceptance because
  its naive INT2 quantization has no rotation; a TurboQuant-style adapter
  would test whether rotation recovers this acceptance.
- **Integration challenge:** Rotation adds compute overhead and changes the
  materialize path significantly. ExactKV would need to ensure the rotation
  is applied consistently in compress and materialize to preserve correctness.
  Recommended as a V7 candidate, not V6.

### 5. Recommended first backend for V6

**Recommendation: kvpress** (Option B), with KIVI as the alternate.

**Reasoning:** kvpress is Hugging Face-native and designed for compatibility
with HF model APIs (the same APIs ExactKV uses). It requires no C++/CUDA
extensions by default and covers multiple compressor styles under one
consistent hook interface. The integration challenge (validating the
draft-verify loop under hooks) is substantial but well-scoped.

If kvpress integration proves infeasible (e.g. the hooks invalidate the
verification loop), KIVI is the next choice because of its clear academic
grounding and implementation quality, though it requires a CUDA device.

**A TurboQuant-style adapter is recommended for V7, not V6**, because it
requires rotation infrastructure that raises correctness re-validation
complexity substantially.

### 6. V6 risks and unknowns

| Risk | Severity | Mitigation |
|---|---|---|
| No backend wraps cleanly behind `KVCompressor` | High | V6 can validly end at adapter design + minimal PoC if integration is infeasible |
| Real backend's `materialize_for_draft` still dequantises to full precision | Medium | Document this honestly; it does not block V6 |
| GPU access required for meaningful real-backend testing | Medium | Use a cloud instance or small GPU; CPU smoke test only for CI |
| Real backend library API changes | Low | Pin exact version; record `backend_version` in report |
| Correctness re-validation scope expands unexpectedly | Medium | Define a minimal adapter path that touches only `compress()` and `materialize()` |

### 7. V6 exit criteria

V6 is complete when:

- [ ] `BackendAdapter` interface is designed, documented in `docs/`, and
  reviewed.
- [ ] At least one real backend adapter is implemented and passes the
  exactness gate (`exactkv_failures == 0`) on the core-suite sweep.
- [ ] The adapter correctly populates all V5 workspace fields with honest
  `supports_real_bytes_claim` values.
- [ ] Experiment 005 report is generated, including a clear comparison of
  acceptance behaviour between the real backend and ExactKV's simulated/INT8
  compressors.
- [ ] No forbidden performance fields anywhere in V6 code, tests, or docs.
- [ ] Full prior test suite remains green.

**If no backend integrates cleanly, V6 can deliver:**
- `BackendAdapter` interface design document
- An explanation of why integration failed
- A recommended V6b path

---

## V7 — Attention-Aware and V-Specific Experiments

### 1. Why V7 matters after V6

V6 establishes that at least one real backend can plug into ExactKV and be
evaluated by acceptance behaviour. V7 asks a deeper question:

> Can we go beyond bit-width-only compression and evaluate policies that are
> *aware* of how attention uses the KV cache?

The related-work survey identifies several directions that are
attention-motivated rather than purely data-rate-motivated:

- **Sparse V dequantization** — skip dequantising value positions with very
  low attention weight.
- **Layer-aware V precision** — allocate higher V precision to sensitive
  layers (first/last), lower precision elsewhere.
- **Pre-RoPE key quantization** — quantise keys before rotary embedding to
  avoid channel outliers introduced by RoPE.
- **Real asymmetric compressor comparison** — compare a real backend's K/V
  separation against ExactKV's simulated asymmetric policies.

V7 is also the natural home for **attention-aware divergence analysis** —
correlating first-divergence position and rejection patterns with attention
entropy, and asking whether compressors that diverge early tend to diverge at
high-entropy positions.

### 2. What V7 should add

#### 2.1 Sparse V dequantization evaluation

An adapter (or adapter extension) that skips dequantising value positions
below an attention-weight threshold. This changes `materialize_for_draft()` to
be attention-aware.

**ExactKV question:** Does attention-gated V dequantization improve acceptance
(by avoiding low-weight noise) or hurt it (by dropping positions the
verification engine would have accepted)?

**Key correctness concern:** Sparse V materialize must produce a working KV
cache that is deterministically reproducible — otherwise the draft and verify
passes may use different effective KV caches, breaking the verification
guarantee. This must be validated before any experiment is run.

#### 2.2 Layer-aware (boundary) V compression policy

A policy that assigns different compression levels to different transformer
layers, motivated by PyramidKV's finding that attention sparsity grows with
layer depth.

**ExactKV question:** Does giving the first and last layers higher V precision
improve acceptance compared to uniform V compression at the same average budget?

**Metrics (same as all V7 experiments):**
- `exactkv_failures` (must be 0)
- Acceptance rate by compressor and layer policy
- Average accepted length
- First-divergence position distribution
- Rejection and correction counts
- V5 workspace fields

#### 2.3 Pre-RoPE key quantization evaluation

A KVQuant-style adapter (from V6 Option C) that quantises keys before the
rotary embedding is applied, using per-channel scales.

**ExactKV question:** Does pre-RoPE key quantization preserve acceptance
better than ExactKV's current post-RoPE per-tensor key quantization?

This is included in V7 (not V6) because it requires hooking the key path
earlier in the forward pass, which needs deeper correctness re-validation.

#### 2.4 Attention-weighted divergence analysis

A pure analysis extension (no new compressor, no generation change) that
computes attention entropy at each token position and correlates it with
first-divergence index and rejection position from existing experiment
reports.

**ExactKV question:** Are high-entropy token positions more sensitive to KV
quantization errors? Are rejections more common at positions where attention
is concentrated versus dispersed?

This is analysis-only. It does not require re-running the model.

#### 2.5 Real asymmetric compressor comparison

Once V6 establishes at least one real backend, V7 can compare a real
asymmetric format (e.g. KIVI's per-channel K + per-token V) against ExactKV's
simulated asymmetric compressors (`k8_v4_sim`, `k4_v8_sim`, etc.) side by side.

The comparison must explicitly label which compressors are real and which are
simulated, and must not conflate their `stored_kv_bytes` figures.

#### 2.6 Proposed experiments

**Experiment 006: Attention-aware V compression sweep**

Run the core suite with:
- A sparse V dequantization adapter (if correctness validation passes)
- A layer-aware V policy adapter
- Existing `k_full_v4_sim`, `k_full_v8`, `int8`, `noop` as baselines

Report: acceptance by policy, first-divergence position distribution,
V5 workspace fields, memory honesty notes.

**Experiment 006b (conditional on V6): Real vs simulated asymmetric comparison**

Run the core suite with the V6 real backend alongside the V4 simulated
asymmetric compressors, at matched nominal bit budgets. Report acceptance
with explicit simulated/real labelling.

### 3. What V7 explicitly should not add

- ❌ ExactKV does not claim any TurboQuant+, KVQuant, or KIVI result.
  ExactKV *evaluates* — any comparison must attribute external systems honestly.
- ❌ No throughput, latency, speedup, or production-serving claim, even if a
  V6 real backend enables faster decoding.
- ❌ No CUDA/Triton kernel implementation (if a real backend uses kernels,
  ExactKV wraps them but does not write them).
- ❌ No attention-aware divergence analysis result should be cited as evidence
  about which compressor is "better" in a deployment sense — only about
  acceptance behaviour under full-KV verification.
- ❌ No eviction-policy implementation (SnapKV, H2O, StreamingLLM,
  PyramidKV) without a separate scope review that defines how "which tokens
  exist" interacts with the verification protocol.

### 4. How V7 relates to external work

| External work | V7 relation |
|---|---|
| **TurboQuant+** "V compression is nearly free when K precision is maintained" | V7 can evaluate whether a rotation-based V adapter recovers `k8_v2_sim`'s collapsed acceptance. ExactKV would evaluate, not claim TurboQuant+ results. |
| **KV-AdaQuant** norm-disparity theory (keys need more bits) | V4 Experiment 003 and V7 experiments are *evidence aligned with* this theory on ExactKV's compressors. Not a re-derivation or reproduction. |
| **KIVI** per-channel K granularity | V7 Experiment 006b (if V6 adapter exists) tests whether KIVI-style granularity improves acceptance. KIVI perplexity numbers are not cited as ExactKV results. |
| **KVQuant** pre-RoPE key quantization | V7 pre-RoPE adapter evaluation. KVQuant CUDA kernel numbers are not cited. |
| **PyramidKV** layer-aware budgets | V7 layer-aware V policy experiment is directionally motivated by PyramidKV. ExactKV measures acceptance, not perplexity or memory usage. |

### 5. V7 risks and unknowns

| Risk | Severity | Mitigation |
|---|---|---|
| Sparse V dequantization is non-deterministic or breaks verification | High | Full correctness re-validation before any experiment |
| Pre-RoPE key quantization requires deep forward-pass hook changes | Medium | Scope carefully; may need to defer to V7b or drop |
| V6 real backend does not exist yet, blocking V7 real-vs-simulated comparison | Medium | V7 can proceed with attention-aware analysis and layer-aware policy experiments independently |
| Attention entropy computation requires re-running the model with activation logging | Medium | Design as a one-pass analysis addition; do not re-introduce timing metrics |
| Scope drifts toward "best compressor" competition rather than verification evaluation | High | All results framed as "acceptance behaviour under full-KV verification", never as "best compressor" |

### 6. V7 exit criteria

V7 is complete when:

- [ ] At least one attention-aware experiment (Experiment 006 or sub-experiment)
  completes with `exactkv_failures == 0`.
- [ ] Results include honest V5 workspace fields for all new compressors/policies.
- [ ] No V7 result is cited as a real-system performance claim.
- [ ] Attention-aware divergence analysis (if implemented) is clearly labelled
  as analysis-only, with no generation logic change.
- [ ] Full prior test suite remains green.
- [ ] No forbidden performance fields anywhere in V7 code, tests, or docs.

---

## V8 — Serving-Stack Context and Final Public Launch

### 1. Why V8 matters after V6/V7

After V6 and V7, ExactKV will have:
- A correctness-first verification engine (V1–V3)
- Honest workspace-aware memory accounting (V5)
- At least one real backend adapter evaluated by acceptance behaviour (V6)
- Attention-aware and/or V-specific experimental results (V7)

V8 asks the final question before a serious public release:

> What is ExactKV's relationship to the serving infrastructure that real
> production LLM systems use?

**This does not mean ExactKV becomes a serving system.** V8's goal is to either:
(a) show that ExactKV's acceptance evaluation can run against caches produced by
a serving stack (e.g. vLLM with PagedAttention), or
(b) clearly document *why* deep serving-stack integration is out of scope and
what the appropriate final release story is without it.

The default story for V8 — even if full serving integration is never achieved —
remains: **exactness, acceptance behaviour, divergence correction, memory
honesty, and real-backend evaluation**. That story is complete and honest
without any throughput number.

### 2. What V8 should add

#### 2.1 Serving-context evaluation (evaluation context only)

If technically feasible, use vLLM or LMCache only as an **evaluation harness** —
a way to produce KV caches that ExactKV can then evaluate for acceptance
behaviour. ExactKV does not run the serving stack; it evaluates caches the
serving stack produces.

**Hard requirement:** Any serving-context evaluation must preserve the
`exactkv_failures == 0` gate. If the serving stack's caching semantics change
which KV states are available for ExactKV's verification loop, correctness must
be explicitly re-validated.

**No throughput benchmark.** Even if vLLM is used as the cache source, ExactKV
does not measure, report, or imply any throughput, latency, or speedup from the
serving system.

#### 2.2 Optional GPU-memory profiling

If a real backend on a CUDA device is available, V8 can introduce measured
peak GPU memory tracking (`torch.cuda.memory_reserved` or a similar profiler),
deferred from V5.

This is the first place `active_gpu_kv_bytes` (or a named equivalent) could
be populated with real measurements. Requirements before any such measurement:

- A real backend with genuine packed storage (not int8 containers) must exist.
- The profiling methodology must be documented: model type, GPU, batch size
  (if batching is introduced separately), isolation from activation memory.
- Any GPU memory figure must be presented as a **measured value with caveats**,
  not as a universal claim.

**If no GPU-memory profiling is performed in V8, this is acceptable** — the
conservative accounting sum from V5 remains the honest figure.

#### 2.3 Final documentation package

V8 produces the final docs suite needed for a public release:

| Document | Purpose |
|---|---|
| Final `README.md` | Public-facing project overview with complete experiment index |
| `docs/ARCHITECTURE_FINAL.md` | Definitive architecture: runtime, verification engine, adapter interface, reporting pipeline |
| `docs/PROJECT_STATUS_V1.0.md` | Replaces `PROJECT_STATUS_V0.4.0.md` with V1.0 status |
| `docs/EXPERIMENT_INDEX.md` | One-line entry for every experiment (001–007) with result summary |
| Updated `docs/RELATED_WORK_KV_CACHE_COMPRESSION.md` | Any new external systems relevant after V6/V7 |
| `docs/LAUNCH_NARRATIVE_DRAFT.md` | Private draft of the public launch narrative (for review, not for publishing) |
| `docs/RELEASE_NOTES_V1.0.md` | Full V1–V8 changelog |

#### 2.4 Proposed Experiment 007

**Experiment 007: Serving-stack compatibility or final public demo**

Scope depends on V8 integration outcome:

- **If serving-context evaluation succeeds:** Run the core suite against a
  real serving stack as the cache source, report acceptance behaviour, and
  confirm `exactkv_failures == 0`.
- **If serving-stack integration is infeasible:** Run a final comprehensive
  sweep (all compressors including V6 real backend if available, all prompt
  suites, multiple draft lengths) as the definitive ExactKV public demo.
  Render a complete Markdown report suitable for the public launch narrative.

Either way, Experiment 007 must produce a report that:
- Has `exactkv_failures == 0`
- Covers all major compressors (simulated, real-INT8, and first real backend
  if available)
- Includes V5 workspace-aware memory accounting for all compressors
- Makes no throughput, latency, speedup, or production-readiness claim

### 3. What V8 explicitly should not add (by default)

The following require separate approval and a new scope document:

- ❌ **Throughput or latency benchmarks.** If ever added, they require:
  real packed-bit storage, a CUDA device, careful baseline isolation, an
  explicit methodology document, and at least three independent warm runs.
  They may not appear in any ExactKV report without a methodology section.
- ❌ **Multi-GPU or distributed inference.** Not an ExactKV goal.
- ❌ **Production-serving claims.** ExactKV never implies production readiness
  unless V8 explicitly adds a scope-approved production readiness section.
- ❌ **Sampling, beam search, or parallel (single-pass) verification.**
  Still deferred unless separately approved with correctness re-validation.

### 4. What would be required before any performance claim

If a future version (V8 or beyond) ever adds throughput or latency measurements,
the following conditions must all hold:

1. A real packed-bit backend exists (not int8 containers) — `is_simulated=False`,
   `supports_real_bytes_claim=True` for all reported compressors.
2. Measurements are taken on a CUDA device with documented hardware (GPU model,
   VRAM, driver version).
3. Baseline is defined: "compared to ExactKV's sequential full-KV generation on
   the same hardware" (not compared to external systems unless those external
   comparisons carry their own attribution and methodology).
4. Warm-up runs are documented and discarded.
5. A methodology section appears in the report with the exact measurement code.
6. An explicit disclaimer states: "This is not a claim about production serving
   performance under real workloads, batching, or system-level scheduling."
7. The `exactkv_failures == 0` gate must still pass on the same run.

**By default, no version should add performance measurements without this
checklist being complete.**

### 5. V8 risks and unknowns

| Risk | Severity | Mitigation |
|---|---|---|
| Serving-stack integration is technically infeasible within ExactKV's HF-centric architecture | High | Accept this; V8's default story (correctness + acceptance + memory honesty + real backend) is complete without it |
| Launch narrative drifts toward performance claims under community pressure | High | The launch narrative must be reviewed against the no-performance-claim policy before any public posting |
| GPU-memory profiling requires hardware not always available | Medium | Default to V5 accounting sum; profiling is optional |
| V6/V7 not yet complete, blocking V8 | High | V8 must not begin until V6 exit criteria are met |
| Final docs require updates to many files | Medium | Use `docs/EXPERIMENT_INDEX.md` as the single source of truth; update incrementally |

### 6. V8 exit criteria

V8 is complete when:

- [ ] Final documentation package is complete (README, architecture, status,
  experiment index, related work, release notes).
- [ ] Experiment 007 completes with `exactkv_failures == 0`.
- [ ] No forbidden performance fields anywhere.
- [ ] The launch narrative draft exists and has been reviewed for honesty.
- [ ] `docs/RELEASE_NOTES_V1.0.md` is written.
- [ ] A `v1.0.0` git tag is assigned.

---

## Cross-version risks

| Risk | Affects | Mitigation |
|---|---|---|
| HF `DynamicCache` internals change between transformers versions | V6+ | Pin `transformers` version; re-validate on upgrade |
| Real backend libraries have unstable APIs | V6+ | Pin exact backend version; record `backend_version` in report |
| Scope creep toward throughput engineering | V6–V8 | The no-performance-claim policy is a hard global constraint |
| Community expectations push toward "speedup" story | V6–V8 | The public launch story must be correctness + acceptance + memory honesty |
| Real backends require GPU hardware unavailable in CI | V6–V8 | CPU smoke test for integration correctness; GPU sweep for real acceptance numbers |
| V5 workspace-memory accounting needs revision once real backends exist | V6+ | V5 schema has `materialized_working_kv_bytes` and `metadata_bytes` already — real backends populate them honestly |

---

## Attribution and related work

All external system claims cited in this document are attributed to their
respective authors. ExactKV does not implement, reproduce, or claim the results
of any external system. Citations:

- KIVI: Liu et al., ICLR 2024, arXiv:2402.02750
- KVQuant: Hooper et al., NeurIPS 2024, arXiv:2401.18079
- KV-AdaQuant: Hariri et al., arXiv:2502.15075
- TurboQuant: Zandieh et al., ICLR 2026, arXiv:2504.19874
- TurboQuant+: community repo, github.com/TheTom/turboquant_plus
- Palu: Chang et al., ICLR 2025, arXiv:2407.21118
- KVTC: Staniszewski & Łańcucki, ICLR 2026, openreview.net/forum?id=aNVKROYpLB
- PyramidKV: Cai et al., 2024, arXiv:2406.02069
- vLLM / PagedAttention: Kwon et al., SOSP 2023, arXiv:2309.06180
- LMCache: Liu et al., 2025, arXiv:2510.09665
- kvpress: NVIDIA, github.com/NVIDIA/kvpress
- VeriCache (ExactKV's algorithmic basis): Yao et al., arXiv:2605.17613, 2026

See [`docs/RELATED_WORK_KV_CACHE_COMPRESSION.md`](RELATED_WORK_KV_CACHE_COMPRESSION.md)
for the full survey, and [`docs/RESEARCH_BACKLOG.md`](RESEARCH_BACKLOG.md) for
the concrete experiment backlog behind V6–V8.
