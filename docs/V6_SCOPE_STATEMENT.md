# V6 Scope Statement: Real Backend Adapter Interface and First Backend Candidate

**Status:** V6 complete (Phases 0–E). See [`docs/RELEASE_NOTES_V0.6.0.md`](RELEASE_NOTES_V0.6.0.md).
**Builds on:** `v0.5.0` — workspace-aware memory accounting complete; Experiment 004
(340 runs, `exactkv_failures == 0`).
**Expands:** [`docs/FUTURE_ROADMAP_V6_V8.md`](FUTURE_ROADMAP_V6_V8.md) §V6 into an
approvable, phased scope.

> No real compressor backends are implemented in this document.
> No performance, throughput, latency, speedup, or production-readiness claims.
> ExactKV does **not** implement kvpress, KIVI, KVQuant, TurboQuant, TurboQuant+,
> KVTC, Palu, LMCache, vLLM, PagedAttention, CUDA/Triton kernels, or any serving
> integration. External systems named below are **candidates and motivation**,
> not current capabilities.

---

## 1. V6 goal

Design a **real-backend adapter interface** that lets a real KV-cache compression
format plug into ExactKV's existing `KVCompressor` protocol, and **select the
first realistic backend candidate** — without turning ExactKV into a serving
stack.

V6 evaluates any real backend strictly through ExactKV's existing metrics:

- exact token equality (`exactkv_output_ids == full_output_ids`)
- acceptance rate
- average accepted length
- first-divergence position
- rejection count
- correction count
- workspace-aware memory fields (V5)

V6 is an **integration and evaluation** version, not a performance or serving
version. If real-backend integration proves too risky, **V6 may validly stop at
adapter-interface design plus a minimal proof-of-concept** (see §14).

---

## 2. Why V6 matters after V5

V5 established two honest facts about every current ExactKV compressor:

1. **`materialized_working_kv_bytes == full_kv_bytes` for all current
   compressors.** Every current compressor dequantises to full precision for
   attention, so stored-byte savings do not reduce the working footprint.
2. **Simulated sub-INT8 compressors (`_sim`) use int8 containers.** Their
   `stored_kv_bytes` reflects int8 container reality, not packed 4-bit or 2-bit
   storage. `supports_real_bytes_claim=False` is enforced for all of them.

V4's Experiment 003 and V5's Experiment 004 showed that **key compression is far
more damaging to acceptance than value compression**, aligned with (but not a
reproduction of) KV-AdaQuant and KIVI. But all of that evidence comes from
**simulated** per-tensor quantizers with no rotation, no per-channel scaling, and
no real packing.

The natural and unanswered question after V5 is:

> **What happens when a *real* compressed backend plugs into ExactKV?**

- Does a per-channel key quantizer (KIVI-style) improve acceptance versus
  ExactKV's naive per-tensor key quantization?
- Does a real packed format actually reduce `materialized_working_kv_bytes`, or
  does it still dequantise to full precision for attention?
- Does a real backend's `metadata_bytes` (per-channel/per-token scales,
  residuals, codebooks) materially change the `total_kv_footprint_bytes`
  accounting picture?

V6 answers these by introducing a clean adapter boundary and testing at least one
real backend — **measured only by acceptance behaviour and honest memory
accounting, never by speed.**

---

## 3. What V6 should add

### 3.1 `BackendAdapter` interface (design + minimal implementation)

A thin wrapper that translates a real backend's compress/decompress API into
ExactKV's `KVCompressor` protocol, **without changing the verification engine,
the draft-verify-commit loop, or the accept/reject bookkeeping**. The
`KVCompressor` protocol remains the only interface the verification engine sees.

### 3.2 Honest capability + workspace-memory population for real backends

Each adapter must declare `CompressorCapabilities` (§8) and populate all five V5
workspace-memory fields (§9) honestly, including a correct
`supports_real_bytes_claim` value.

### 3.3 First real backend candidate behind the adapter

One real backend (recommended candidate in §6) wrapped behind `BackendAdapter`,
passing the exactness gate on the core suite.

### 3.4 Experiment 005

A core-suite sweep comparing the first real backend against ExactKV's simulated
and real-INT8 compressors by acceptance behaviour and workspace memory (§12).

### 3.5 Documentation

An adapter-interface design document, plus the Experiment 005 report and V6
release notes when the phase completes.

---

## 4. What V6 explicitly does not add

* ❌ **No throughput, latency, tokens/sec, speedup, or `runtime_seconds`.**
* ❌ **No production-serving claims.**
* ❌ **No vLLM, LMCache, or PagedAttention integration** (V8 evaluation context
  at most; not promised).
* ❌ **No CUDA/Triton kernels written by ExactKV.** If a backend library brings
  its own kernels, ExactKV wraps them but writes and claims none.
* ❌ **No CPU offload, batching, sampling, parallel (single-pass) verification,
  or bonus-token acceptance.**
* ❌ **No changes to generation logic or the draft-verify-commit loop.**
* ❌ **No changes to existing report schemas** beyond additive capability fields
  if an adapter genuinely needs them (see §8; additive and backward-compatible
  only).
* ❌ **No active GPU memory profiling** (`torch.cuda.memory_reserved`, etc.).
  Still deferred to V8 at the earliest.
* ❌ **No relaxation of `supports_real_bytes_claim=False` for `_sim`
  compressors.** Simulated compressors remain int8-container storage.
* ❌ **No new simulated compressors.**

---

## 5. Candidate backend comparison

> All systems below are **external work**. ExactKV does not implement or claim
> any of them. Any quality/memory figure attributed to them is the authors' own
> claim, not an ExactKV measurement.

| Candidate | What it is | Integration friction | HF-native? | GPU needed for real savings? | First-backend fit |
|---|---|---|---|---|---|
| **kvpress** | HF-native KV-cache compression library with multiple compressors via a hooks API (NVIDIA) | Medium — hooks modify the forward pass; draft-verify loop must be re-validated under hooks | Yes | Partial (CPU works for some; real savings need GPU) | **Best first candidate** |
| **KIVI** | Tuning-free 2-bit: per-channel keys, per-token values, small full-precision residual (Liu et al., ICLR 2024) | Medium–High — per-channel/per-token scale granularity differs from ExactKV's per-tensor model; original kernels are CUDA | Partial | Yes (CUDA kernels) | Strong alternate |
| **KVQuant-style** | Sub-4-bit: per-channel + **pre-RoPE** key quant + dense-and-sparse outliers (Hooper et al., NeurIPS 2024) | High — requires hooking the key path **before** rotary embedding; deeper correctness re-validation | No | Yes | Defer to **V7** |
| **TurboQuant-style** | Data-oblivious rotation (Walsh–Hadamard) + Lloyd-Max / PolarQuant scalar quant (Zandieh et al., ICLR 2026) | High — rotation infrastructure in both compress and materialize; correctness re-validation complex | No | Yes | Defer to **V7** |

### Candidate detail

**kvpress.** Designed for Hugging Face compatibility — the same model APIs
ExactKV already uses. No C++/CUDA extension required for the base library. The
main risk is that its forward-pass hooks must be shown not to invalidate the
draft-verify loop's correctness guarantee. Lowest-friction path to a *real*
(non-simulated) compressed cache behind ExactKV.

**KIVI.** Academically clean, widely cited, asymmetric in granularity
(per-channel K, per-token V) — directly relevant to ExactKV's V4/V5 finding that
keys are more fragile. The residual is a textbook example of why
`stored_kv_bytes` understates the working footprint. Friction: scale-granularity
translation into honest `metadata_bytes`, and CUDA-only kernels in the reference
implementation.

**KVQuant-style.** State-of-the-art key-quantization granularity, but pre-RoPE
key quantization needs the key path hooked before rotary embedding — a deeper
change with heavier correctness re-validation. Recommended for **V7**, not V6.

**TurboQuant-style.** The rotation-based V quantizer is the key idea behind "V
compression is nearly free when K precision is maintained." It would directly
test whether rotation recovers the acceptance that ExactKV's naive `k8_v2_sim`
lost. But rotation must be applied consistently in compress and materialize to
preserve correctness — recommended for **V7**.

---

## 6. Recommended first backend and why

**Recommendation: kvpress.** Alternate: **KIVI**.

**Reasoning:**

1. **HF-native.** kvpress targets the same Hugging Face model interfaces ExactKV
   is built on, minimizing new infrastructure.
2. **No mandatory CUDA/Triton.** The base library can exercise the adapter
   boundary on CPU for correctness, reserving GPU for real-savings evaluation.
3. **Multiple compressors under one interface.** A single adapter can expose
   several real compressors, maximizing the acceptance-behaviour comparison
   surface for Experiment 005.
4. **Well-scoped main risk.** The principal hazard — hooks possibly invalidating
   the draft-verify loop — is concrete and testable up front (§10).

**If kvpress integration proves infeasible** (e.g. its hooks cannot coexist with
ExactKV's verification loop without breaking the exactness gate), fall back to
**KIVI** for its clear academic grounding and asymmetric granularity, accepting
the CUDA dependency.

**A TurboQuant-style or KVQuant-style adapter is explicitly deferred to V7**,
because both require deeper forward-pass changes (rotation; pre-RoPE hooking) and
heavier correctness re-validation.

---

## 7. Adapter interface requirements

The `BackendAdapter` must:

1. **Satisfy the existing `KVCompressor` protocol** so the verification engine
   needs no changes. The engine only ever sees `KVCompressor`.
2. **Implement `compress(full_kv_state) → CompressedKVState`** by delegating to
   the real backend's quantise/store operation.
3. **Implement `materialize_for_draft(compressed_state) → working KV`** by
   delegating to the real backend's dequantise/reconstruct operation, producing
   a working cache usable for attention.
4. **Implement `stats(full_kv_state) → CompressionStats`** returning honest V5
   workspace fields (§9).
5. **Implement `capabilities() → CompressorCapabilities`** (§8).
6. **Be deterministic.** Given the same input KV state, `compress` then
   `materialize_for_draft` must produce the same working cache on the draft pass
   and any re-materialization, so the verification guarantee holds. Any
   stochastic step (e.g. randomized rotation seed) must be fixed and recorded.
7. **Not modify** the draft-verify-commit loop, acceptance bookkeeping, or
   generation logic.
8. **Pin the backend version.** Record the exact external library version used so
   experiments are reproducible.
9. **Fail loudly, not silently.** If the backend cannot honestly populate a
   required field, the adapter must raise rather than emit a misleading default.

---

## 8. Capability metadata requirements

Each real-backend adapter must declare `CompressorCapabilities`:

| Field | Requirement for a real backend |
|---|---|
| `is_simulated` | **`False`** for all real backends |
| `supports_real_bytes_claim` | **`True` only when `stored_kv_bytes` reflects actual packed/quantised bytes** held between decode steps; otherwise `False` |
| `key_bit_width` | Actual bits per key element (or `full`) |
| `value_bit_width` | Actual bits per value element (or `full`) |
| `asymmetric` | `True` if K and V use different policies/granularities |
| `backend_name` *(new, additive)* | External backend identifier (e.g. `"kvpress"`, `"kivi"`) |
| `backend_version` *(new, additive)* | Pinned version string of the external library |

> Any new capability field (`backend_name`, `backend_version`) is **additive and
> backward-compatible**: existing V1–V5 compressors and reports must continue to
> load and validate without these fields. This is the only schema-adjacent change
> V6 permits, and it adds fields — it changes no existing field.

**Honesty rule:** `is_simulated=False` does **not** by itself authorize a memory
saving claim. A saving may be cited only when `supports_real_bytes_claim=True`
**and** the relevant workspace field is genuinely smaller than `full_kv_bytes`
(§9).

---

## 9. Workspace-memory requirements

Every adapter must populate all five V5 fields:

| Field | Real-backend requirement |
|---|---|
| `stored_kv_bytes` | Actual bytes of the persistent compressed representation. For a real packed format this may finally be below int8-container size; for a backend that stores int8 + scales it is int8-level. |
| `materialized_working_kv_bytes` | Bytes of the working copy during attention. **May be `< full_kv_bytes` only if** the backend uses a fused dequantise-and-attend path that avoids a full-precision working copy. If it dequantises to full precision (the common case), this **equals `full_kv_bytes`** and must be reported as such. |
| `metadata_bytes` | Must include per-channel/per-token scales, zero-points, **residual tensors** (e.g. KIVI residual), codebooks, and projection matrices — not just a per-tensor scale. |
| `temporary_workspace_bytes` | Conservative transient scratch during compress/materialize. |
| `total_kv_footprint_bytes` | Conservative **accounting sum**, **NOT** a measured peak GPU memory value. |

**Required framing (unchanged from V5):**

- `total_kv_footprint_bytes` is a conservative accounting sum derived from tensor
  shapes and dtype widths. It is **not** a measured peak GPU memory value.
- Active GPU memory measurement remains **deferred** (V8 at the earliest).
- A real backend's `stored_kv_bytes` is the **only** place a genuine at-rest
  saving can appear, and only when `supports_real_bytes_claim=True`.

**Real vs simulated separation (hard requirement):** Experiment 005 and all V6
reports must clearly distinguish real-backend memory claims from simulated
compressor figures. Simulated `_sim` figures (int8 containers,
`supports_real_bytes_claim=False`) must never be presented alongside a real
backend's figures in a way that implies the `_sim` numbers are real packed-bit
savings.

---

## 10. Test and gate plan

### Global gates (every V6 phase)

- All prior tests remain green (1263 at `v0.5.0`).
- Primary correctness criterion unchanged: `exactkv_output_ids ==
  full_output_ids` under greedy decoding; `exactkv_failures == 0`.
- No forbidden performance fields anywhere: `tokens_per_second`, `throughput`,
  `latency`, `speedup`, `runtime_seconds`.

### Adapter correctness gates

| Test | Assertion |
|---|---|
| Adapter satisfies `KVCompressor` | Registry resolves it; verification engine runs unchanged |
| Determinism | `compress`→`materialize` reproducible across draft and re-materialize |
| Exactness under adapter | `exactkv_failures == 0` on a smoke subset, then full core suite |
| Capability honesty | `is_simulated=False`; `supports_real_bytes_claim` matches actual storage |
| Workspace fields populated | All five V5 fields present and non-negative |
| Metadata completeness | `metadata_bytes` includes scales + residual/codebook bytes, not just a per-tensor scale |
| Backward compatibility | Existing V1–V5 compressors/reports load without `backend_name`/`backend_version` |
| No real-bytes claim without basis | Saving cited only when `supports_real_bytes_claim=True` and field `< full_kv_bytes` |
| No forbidden performance fields | Pattern audit passes on adapter output, reports, and docs |

### Hook-safety gate (kvpress-specific)

If the chosen backend uses forward-pass hooks, an explicit test must show the
draft pass and the verify pass operate on the intended KV states and that the
exactness gate still holds. **If this cannot be guaranteed, the backend is
rejected and V6 falls back (KIVI) or stops at design (§14).**

Pre-Phase C research (`docs/KVPRESS_INTEGRATION_RESEARCH.md`) adds these
**required gates** before kvpress integration code:

1. **Hook isolation:** no kvpress forward hooks registered during
   `verify_sequential` or `_commit`; assert via `verification_mode()`.
2. **Global attention patch:** `import kvpress` permanently patches
   `ALL_ATTENTION_FUNCTIONS`; must not break verify or existing tests.
3. **Transformers version isolation:** kvpress 0.5.3 requires
   `transformers>=4.56,<5.3`; ExactKV currently runs 5.8.x — Phase C must use
   an isolated optional extra or separate gate environment.
4. **Initial press restriction:** Phase C starts with prefill-only `KnormPress`
   only; no `DecodingPress`, `PrefillDecodingPress`, or `AdaKVPress`.
5. **Model-reference extension:** kvpress needs `ModelRuntime.model` access;
   `_backend_compress(k_tensors, …)` alone is insufficient.

**Phase C scaffold (implemented):**

- Optional `[kvpress]` extra in `pyproject.toml` (`kvpress==0.5.3`,
  `transformers>=4.56,<5.3`) — not on the default install path.
- `BackendAdapter.verification_mode()` default no-op context manager.
- `ExactKVGenerator` wraps verification inside `verification_mode()` when present.
- No `import kvpress` in default ExactKV module loading.

**Phase C restricted adapter (implemented — experimental, KnormPress only):**

- `exactkv/compressors/kvpress_knorm.py` — `KVPressKnormAdapter` using
  `KnormPress` only; **not** registered in the default compressor registry.
- Lazy `import kvpress` inside adapter construction only.
- Replay prefill: `with press(compression_model): model(input_ids, DynamicCache())`.
- **Hook isolation gate:** `verification_mode()` asserts zero attention forward
  hooks on the verification `ModelRuntime.model`; compression replay runs on an
  isolated `deepcopy` of the model by default (`isolate_compression_model=True`)
  because kvpress does not restore `rotary_emb` assignments on context exit.
- **Full-state immutability gate:** verification uses authoritative full KV;
  compress/verify tests assert `full_state.past_key_values` bytes and tensor
  values unchanged after `verify_sequential`.
- **Logical vs physical sequence length:** `CompressedKVState.logical_seq_len`
  equals the full prefill length; physical `kv_seq_len` on the pruned
  `DynamicCache` may be shorter under KnormPress.
- **Isolated `[kvpress]` environment:** run adapter tests in `.venv-kvpress`
  (`pip install -e ".[kvpress]"`); default env stays on `transformers==5.8.x`
  without kvpress.
- **Python 3.13:** `fire>=0.7.1` workaround required (`kvpress` pins `fire<0.7`
  which imports removed `pipes`); install manually in the kvpress venv only.

**Phase C core-suite validation (2026-06-09, `docs/KVPRESS_KNORM_VALIDATION.md`):**

- Full `core` suite (34 prompts), `Qwen/Qwen2.5-0.5B`, `draft_len=4`,
  `max_new_tokens=16`, `compression_ratio=0.5`.
- `exactkv_failures == 0`; `exactkv_output_ids == full_output_ids` on all prompts.
- Hook-safety, full-state immutability, physical/logical seq, and workspace gates pass.
- Lossy draft divergences expected (~39% draft acceptance); final output exact.

**Phase D Experiment 005 (2026-06-09, `docs/EXPERIMENT_005_KVPRESS_KNORM.md`):**

- 272 cells (34 core prompts × 8 compressors), `draft_len=4`, `max_new_tokens=16`.
- `exactkv_failures == 0` on all compressors including `kvpress_knorm_restricted`.
- kvpress hook-safety and workspace-memory gates pass; lossy draft acceptance ~41%.
- Artifacts: `reports/experiment_005_kvpress_knorm.{json,csv}` (gitignored).

**Phase E release (2026-06-09, `docs/RELEASE_NOTES_V0.6.0.md`):**

- V6 release notes, README/ROADMAP updates, git hygiene and no-performance-claim audits.
- Default env remains kvpress-free; `kvpress_knorm_restricted` not in default registry.
- Ready to tag `v0.6.0`.

**Empirical research pass (2026-06-09, `docs/KVPRESS_INTEGRATION_RESEARCH.md`):**

- Dedicated `.venv-kvpress` install succeeds with `transformers==5.2.0`, `kvpress==0.5.3`.
- Python 3.13 requires `fire>=0.7.1` workaround (`kvpress` pins `fire<0.7`); document in kvpress CI.
- Hooks: 0 → 24 → 0 across `with press(model):` on Qwen2.5-0.5B; global attention patch on `import kvpress`.
- `dynamic_v5` cache compatible; physical seq shrinks; logical seq must be set separately.
- Recommendation unchanged: **proceed with kvpress — only with restrictions**.

---

## 11. GPU requirements

- **Adapter correctness:** CPU is sufficient for the exactness gate on a small
  model (`Qwen/Qwen2.5-0.5B`, float32), provided the backend has a CPU path.
- **Real-savings evaluation:** A CUDA device is required to exercise a backend's
  real packed storage / kernels and to observe genuine `stored_kv_bytes` below
  int8-container size. KIVI's reference kernels are CUDA-only.
- **CI:** CPU-only smoke correctness for the adapter; the full real-backend
  acceptance sweep (Experiment 005) runs on GPU and is documented, not gated in
  CPU CI.
- **No GPU memory profiling** is performed in V6 (deferred). GPU is used to run a
  real backend, not to measure peak device memory.

---

## 12. Experiment 005 plan

**Name:** First Real Backend Adapter Comparison

**Goal:** Compare the first real backend (recommended: kvpress) against ExactKV's
simulated and real-INT8 compressors by acceptance behaviour and workspace memory.

**Configuration (illustrative; finalized at implementation time):**

- Model: `Qwen/Qwen2.5-0.5B` (and a GPU-class model if the backend requires GPU)
- Suite: `core` (34 prompts)
- Compressors: the real backend adapter + `noop`, `int8`, `k_full_v8`,
  `k8_v_full`, and one or two `_sim` baselines for contrast
- Draft length(s): a small set (e.g. 4)
- `max_new_tokens`: small but meaningful (e.g. 16)

**Reported (per compressor):**

- `exactkv_failures` (must be 0)
- acceptance rate, average accepted length
- first-divergence position distribution
- rejection and correction counts
- all five V5 workspace fields
- `supports_real_bytes_claim`, `is_simulated`, `backend_name`, `backend_version`

**Required wording in the report:**

- "Simulated compressors store sub-INT8 values in int8 containers;
  `stored_kv_bytes` reflects int8 storage, not packed-bit savings."
- "`materialized_working_kv_bytes` equals `full_kv_bytes` unless the backend uses
  a fused dequantise-and-attend path."
- "`total_kv_footprint_bytes` is a conservative accounting sum, not a measured
  peak GPU memory value. Active GPU measurement is deferred."
- "ExactKV evaluates this backend by acceptance behaviour and memory honesty. It
  does not claim the backend's speedup, throughput, latency, runtime, or
  production readiness, and does not cite the backend's external benchmark
  numbers as ExactKV results."
- VeriCache attribution.

**Not reported:** tokens/second, throughput, latency, speedup, wall-clock
runtime.

---

## 13. Risks and unknowns

| Risk | Severity | Mitigation |
|---|---|---|
| Backend hooks invalidate the draft-verify correctness guarantee | High | Hook-safety gate (§10); reject backend if unmet |
| kvpress `transformers<5.3` vs ExactKV 5.8.x | High | Isolated optional extra; pin versions; see research doc |
| kvpress global `patch_attention_functions` on import | High | Attention-patch gate; reject if verify/regression breaks |
| No backend wraps cleanly behind `KVCompressor` | High | V6 may stop at adapter design + minimal PoC (§14) |
| Real backend still dequantises to full precision (`materialized == full`) | Medium | Report honestly; this does not block V6 |
| Scale-granularity (per-channel/per-token) translation into honest `metadata_bytes` | Medium | Document the accounting; raise rather than emit misleading defaults |
| GPU required for meaningful real-savings numbers | Medium | CPU smoke for correctness; documented GPU sweep for acceptance/memory |
| External backend API/version churn | Low | Pin and record `backend_version` |
| Correctness re-validation scope expands unexpectedly | Medium | Keep the adapter minimal — touch only `compress`/`materialize`/`stats`/`capabilities` |
| Pressure to add a speedup story once a real backend runs | High | No-performance-claim policy (§15) is a hard gate |

---

## 14. Exit criteria

V6 is complete when **either** the full path or the design-only fallback is met.

**Full path:**

- [x] `BackendAdapter` interface designed, documented, and reviewed.
- [x] At least one real backend adapter implemented, passing the exactness gate
      (`exactkv_failures == 0`) on the core suite.
- [x] Adapter populates all five V5 workspace fields with an honest
      `supports_real_bytes_claim`.
- [x] Capability fields (`backend_name`, `backend_version`) added additively;
      backward compatibility preserved.
- [x] Experiment 005 report generated, clearly comparing the real backend to
      simulated/INT8 compressors by acceptance and workspace memory, with real
      vs simulated memory claims kept distinct.
- [x] No forbidden performance fields anywhere in V6 code, tests, or docs.
- [x] Full prior test suite remains green.

**Design-only fallback (acceptable V6 outcome):** if no backend integrates
cleanly without risking the exactness gate, V6 may deliver:

- [ ] The `BackendAdapter` interface design document.
- [ ] A minimal proof-of-concept (e.g. a trivial pass-through "real" adapter that
      exercises the boundary without a heavyweight external dependency).
- [ ] A written explanation of why full integration was deferred and a
      recommended V6b path.

---

## 15. No-performance-claim policy (unchanged from V1)

The following may **never** appear as data fields, table columns, or key-value
pairs in any ExactKV output — code, tests, CLI, or Markdown:

```
tokens_per_second
throughput
latency
speedup
runtime_seconds
```

They may appear **only in explicit negation prose or methodology caveats**, e.g.
"ExactKV does not measure tokens/second", "No latency claim is made", or the V8
methodology checklist describing what *would* be required before any such
measurement.

Wrapping a real backend does **not** authorize adopting that backend's
performance numbers. ExactKV evaluates a backend by **acceptance behaviour and
memory honesty only**, and never presents an external system's speedup,
throughput, latency, or perplexity as an ExactKV result.

The V5 workspace-memory fields are honesty fields, not performance claims. A real
backend's `stored_kv_bytes` saving is a memory fact (when
`supports_real_bytes_claim=True`), not a speed claim.

---

## 16. Relationship to V7 and V8

- **V7 — attention-aware and V-specific experiments.** Once V6 establishes that a
  real backend can plug in and pass the exactness gate, V7 evaluates
  attention-aware policies (sparse V dequantization, layer-aware V precision,
  pre-RoPE key quantization, attention-weighted divergence analysis) and a real
  vs simulated asymmetric comparison. The **TurboQuant-style and KVQuant-style
  adapters deferred from V6 land in V7**, where their deeper forward-pass changes
  can be validated. Still no performance claims.
- **V8 — serving-stack context and final public launch.** Serving stacks
  (vLLM/LMCache) appear, if at all, only as an **evaluation context** for caches
  ExactKV then verifies — never as a performance-claim source. V8 is also the
  earliest point active GPU memory profiling could be introduced, and only under
  the strict methodology checklist in
  [`docs/FUTURE_ROADMAP_V6_V8.md`](FUTURE_ROADMAP_V6_V8.md) §V8.

See [`docs/FUTURE_ROADMAP_V6_V8.md`](FUTURE_ROADMAP_V6_V8.md) for the full V6–V8
arc, [`docs/RESEARCH_BACKLOG.md`](RESEARCH_BACKLOG.md) for the concrete backend
backlog (B1–B4), and
[`docs/RELATED_WORK_KV_CACHE_COMPRESSION.md`](RELATED_WORK_KV_CACHE_COMPRESSION.md)
for the external-systems survey and attribution.

---

## V6 phases

| Phase | Scope | Deliverable | Status |
|---|---|---|---|
| **Phase 0** (this document) | Scope statement only; no code | `docs/V6_SCOPE_STATEMENT.md` committed and reviewed | ✅ Complete |
| **Phase A** | `BackendAdapter` interface design document; capability/workspace field plan | `docs/BACKEND_ADAPTER_INTERFACE.md`; no backend yet | ✅ Complete |
| **Phase B** | Minimal proof-of-concept adapter (trivial real/pass-through) exercising the boundary; exactness gate on smoke | PoC adapter + tests | ✅ Complete |
| **Phase C** | kvpress safety scaffold + first real backend (recommended: kvpress, **restricted** — see `docs/KVPRESS_INTEGRATION_RESEARCH.md`); hook-safety + version-isolation + exactness gates | `KVPressKnormAdapter` + core-suite validation (`docs/KVPRESS_KNORM_VALIDATION.md`) | ✅ Complete |
| **Phase D** | Experiment 005 (acceptance + workspace memory comparison); report rendering | [`docs/EXPERIMENT_005_KVPRESS_KNORM.md`](EXPERIMENT_005_KVPRESS_KNORM.md) | ✅ Complete |
| **Phase E** | V6 release notes; README/ROADMAP updates; audit; tag | [`docs/RELEASE_NOTES_V0.6.0.md`](RELEASE_NOTES_V0.6.0.md) | ✅ Complete |

> Phases B–E require **separate explicit approval** before any code is written.
> Phases 0 and A (this document and `BACKEND_ADAPTER_INTERFACE.md`) are
> design-only and introduce no code and no behaviour change.

---

## Attribution

The draft-then-verify compressed-KV algorithm is from:

> **VeriCache: Turning Lossy KV Cache into Lossless LLM Inference.**
> Yao et al., arXiv:2605.17613, 2026.

ExactKV does not claim to have invented this algorithm. V6 extends the framework
with a real-backend adapter boundary and evaluates any wrapped backend by
acceptance behaviour and honest memory accounting — never by performance.

External candidate systems cited (none implemented by ExactKV): kvpress
(NVIDIA, github.com/NVIDIA/kvpress); KIVI (Liu et al., ICLR 2024,
arXiv:2402.02750); KVQuant (Hooper et al., NeurIPS 2024, arXiv:2401.18079);
TurboQuant (Zandieh et al., ICLR 2026, arXiv:2504.19874); TurboQuant+
(community, github.com/TheTom/turboquant_plus).
