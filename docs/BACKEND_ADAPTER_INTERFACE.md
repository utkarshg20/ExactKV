# BackendAdapter Interface Design

**Status:** V6 Phase A design + Phase B implementation reference.
**Implementation:** `exactkv/compressors/backend_adapter.py` (Phase B complete).
**Pre-Phase C research:** `docs/KVPRESS_INTEGRATION_RESEARCH.md`.
**Scope context:** [`docs/V6_SCOPE_STATEMENT.md`](V6_SCOPE_STATEMENT.md).

> ExactKV does not implement kvpress, KIVI, KVQuant, TurboQuant, TurboQuant+,
> KVTC, Palu, LMCache, vLLM, or PagedAttention. This document designs an adapter
> boundary so a real backend *could* plug in, without implementing any of them.
> No performance, throughput, latency, speedup, or production-readiness claim.

---

## 1. Purpose of the BackendAdapter

ExactKV's existing `KVCompressor` protocol (V1) was designed for simple
research compressors implemented entirely in pure PyTorch. All V1–V5
compressors own their compress/dequantise logic end-to-end.

A real compression backend — KIVI, kvpress, a TurboQuant-style quantizer — is
typically a third-party library with its own compress and dequantise API, its own
cache format, and potentially its own hooks into the model's forward pass. Simply
subclassing or rewriting such a backend to satisfy `KVCompressor` directly would
require duplicating or reimplementing it, which is both fragile and dishonest
(ExactKV would effectively be implementing the backend).

The **`BackendAdapter`** is a *thin wrapper layer* between a real backend's API
and ExactKV's `KVCompressor` protocol. It:

- Translates `FullKVState` into the backend's input format.
- Delegates compress/dequantise to the real backend's own code.
- Translates the backend's output back into `CompressedKVState`.
- Populates `CompressorCapabilities` and `CompressionStats` (including V5
  workspace-memory fields) with **honest** values reflecting what the backend
  actually stores and how it actually materializes the working cache.

The verification engine, generation loop, acceptance bookkeeping, and all
ExactKV metrics remain untouched. The adapter's job is purely transliteration.

---

## 2. Relationship to the existing `KVCompressor` protocol

The `KVCompressor` protocol is defined in `exactkv/compressors/base.py` as a
structural `typing.Protocol`. Any object that provides the correct methods and
`name` attribute satisfies it — no inheritance required.

```
KVCompressor protocol                    BackendAdapter (proposed)
──────────────────────────────────────   ─────────────────────────────────────────────
name: str                            →   name: str  (delegates to backend identity)
compress(full_state)                 →   calls backend.encode(extracted KV tensors)
materialize_for_draft(compressed)    →   calls backend.decode(compressed data)
update_after_commit(compressed, new) →   calls compress(new_full_state)
stats(compressed)                    →   reads backend's byte counts + V5 fields
capabilities                         →   built from backend metadata + adapter metadata
```

A `BackendAdapter` instance **is** a `KVCompressor`. The verification engine and
`ExactKVGenerator` never need to know which side of the adapter boundary they
are on.

---

## 3. Why the verification engine must remain unchanged

`VerificationEngine.verify_sequential` (`exactkv/verification/engine.py`) takes
a `FullKVState` and a list of draft token IDs. It runs the **full model** on the
full state — it never touches the compressor.

**The correctness guarantee** — `exactkv_output_ids == full_output_ids` — rests
on the fact that verification always uses the authoritative full-precision KV
cache, not the compressed one. Introducing a real backend cannot change this:

- The backend compresses and dequantises on the **draft path** only.
- The verify path is entirely full-precision, regardless of what the backend does.
- Any attempt to make the verify path backend-aware would break the guarantee.

Therefore: the `BackendAdapter` interacts exclusively with `ExactKVGenerator`'s
**draft path** (`compress`, `materialize_for_draft`, `update_after_commit`) and
**never with `VerificationEngine`**.

---

## 4. Proposed class / interface shape

The following is a **design sketch in Python pseudocode**. This is not
executable code — it describes the intended structure for Phase B
implementation.

```python
# exactkv/compressors/backend_adapter.py  (V6 Phase B — not yet written)

class BackendAdapter:
    """
    Wraps a real KV-cache compression backend behind the KVCompressor protocol.

    Subclass this for each real backend (e.g. KVPressAdapter, KIVIAdapter).
    Subclasses must override _backend_compress and _backend_materialize.
    They must NOT override compress, materialize_for_draft, update_after_commit,
    or stats — those are sealed here to protect the correctness invariants.

    No throughput, latency, speedup, or production-readiness claim.
    """

    # ── Identity ────────────────────────────────────────────────────────────
    name: str                    # registry name, e.g. "kvpress_snapkv"
    backend_name: str            # external library name, e.g. "kvpress"
    backend_version: str         # pinned library version, e.g. "0.3.1"
    adapter_name: str            # this adapter class name
    adapter_version: str         # adapter's own version (for experiment tracking)

    # ── Capabilities (declared at construction, immutable) ───────────────────
    capabilities: CompressorCapabilities

    # ── Public protocol methods (sealed — do not override) ───────────────────

    def compress(self, state: FullKVState) -> CompressedKVState:
        """
        Extract KV tensors from full_state, call _backend_compress,
        wrap result in CompressedKVState.
        Must NOT mutate full_state or its past_key_values.
        """

    def materialize_for_draft(self, compressed: CompressedKVState) -> Any:
        """
        Call _backend_materialize to produce past_key_values for the draft model.
        Return value is deep-copied by ExactKVGenerator._draft before use.
        Must return a HF-compatible past_key_values object (tuple, DynamicCache v4/v5).
        """

    def update_after_commit(
        self,
        compressed: CompressedKVState,
        new_full_state: FullKVState,
    ) -> CompressedKVState:
        """
        V1 strategy: recompress from the new authoritative full state.
        Default implementation calls self.compress(new_full_state).
        May be overridden by a backend that supports efficient incremental update,
        but the result must still satisfy the alignment invariant:
          result.logical_seq_len == new_full_state.seq_len
        """

    def stats(self, compressed: CompressedKVState) -> CompressionStats:
        """
        Return CompressionStats with all V5 workspace-aware fields populated
        by _backend_workspace_bytes. No forbidden performance fields.
        """

    # ── Abstract backend methods (override in each adapter subclass) ──────────

    def _backend_compress(
        self,
        k_tensors: list[torch.Tensor],
        v_tensors: list[torch.Tensor],
        cache_format: str,
    ) -> Any:
        """
        Call the real backend's quantise/encode operation.
        Return value is backend-specific; stored as CompressedKVState.data.
        Must be deterministic for a given input (fixed seeds if stochastic).
        Must NOT modify the input tensors in place.
        """
        raise NotImplementedError

    def _backend_materialize(
        self,
        backend_data: Any,
        cache_format: str,
    ) -> Any:
        """
        Call the real backend's dequantise/reconstruct operation.
        Return a HF-compatible past_key_values object.
        Must be deterministic for a given backend_data (fixed seeds if stochastic).
        """
        raise NotImplementedError

    def _backend_workspace_bytes(
        self,
        full_state: FullKVState,
        backend_data: Any,
    ) -> dict[str, int]:
        """
        Return a dict with keys:
          stored_kv_bytes, materialized_working_kv_bytes,
          metadata_bytes, temporary_workspace_bytes, total_kv_footprint_bytes.
        All values must be honest counts for THIS backend's actual storage,
        not hypothetical packed-bit estimates.
        """
        raise NotImplementedError
```

---

## 5. Required methods

### 5.1 `compress(full_state: FullKVState) → CompressedKVState`

**Purpose:** Create the backend's compressed representation from the
authoritative full-precision KV state.

**Sequence:**
1. Extract `(k_tensors, v_tensors, cache_format)` from
   `full_state.past_key_values` using `exactkv.cache.utils.extract_kv_tensors`.
2. Delegate to `_backend_compress(k_tensors, v_tensors, cache_format)`.
3. Run a materialize to get the compressed model's next-token prediction
   (needed for `CompressedKVState.metadata["next_token_id"]`).
4. Wrap result in `CompressedKVState`:
   ```
   CompressedKVState(
       data=backend_data,
       metadata={"next_token_id": compressed_next_token_id, ...},
       compressor_name=self.name,
       logical_seq_len=full_state.seq_len,
       generated_ids=full_state.generated_ids,
       device=full_state.device,
   )
   ```

**Invariants:**
- `full_state.past_key_values` must be identical before and after the call.
- `result.logical_seq_len == full_state.seq_len` (alignment invariant).
- `result.metadata["next_token_id"]` must be the compressed model's prediction
  for the next token, not the full model's.

### 5.2 `materialize_for_draft(compressed: CompressedKVState) → Any`

**Purpose:** Produce a `past_key_values` object the draft model can consume.

**Sequence:**
1. Retrieve `backend_data` from `compressed.data`.
2. Delegate to `_backend_materialize(backend_data, cache_format)`.
3. Return the resulting `past_key_values`. `ExactKVGenerator._draft` will
   deep-copy this before use, so the adapter need not clone.

**Invariants:**
- Must return a format detectable by `exactkv.cache.utils._detect_format`:
  either a `tuple`, a `DynamicCache` with `.layers` (v5), or a `DynamicCache`
  with `.key_cache` / `.value_cache` (v4).
- Must NOT mutate `compressed.data` in place.
- Must be deterministic: same `backend_data` → same returned cache.

### 5.3 `update_after_commit(compressed, new_full_state) → CompressedKVState`

**Purpose:** Re-synchronize the compressed state after tokens are committed to
the authoritative full state.

**V1 strategy (default, always correct):** Call `self.compress(new_full_state)`.
This recompresses from scratch after every commit. It is correct but
computationally wasteful for backends that support incremental update.

**Incremental strategy (allowed, must preserve invariant):** If the backend
supports an efficient incremental update (e.g. appending new KV entries without
recompressing the whole cache), the subclass may override this method.
**Required invariant:** `result.logical_seq_len == new_full_state.seq_len`.
**Required test:** Any incremental strategy must produce the same
`next_token_id` as the full-recompress strategy on a smoke subset. This is
verified by the Phase B adapter-equivalence test (§19).

### 5.4 `stats(compressed: CompressedKVState) → CompressionStats`

**Purpose:** Return byte-level statistics including all V5 workspace-aware
fields.

The base `BackendAdapter.stats` implementation:
1. Retrieves `full_state` from `compressed.metadata["full_state_ref"]` (a
   reference stored at compress time — see §9 for details).
2. Calls `_backend_workspace_bytes(full_state, compressed.data)`.
3. Builds and returns `CompressionStats` with all V5 fields, `full_bytes` from
   `kv_total_bytes(full_state.past_key_values)`, and a correct
   `compression_ratio` / `memory_reduction_factor` pair.

**Honesty invariants** (see also §9):
- `supports_real_bytes_claim` is taken from `self.capabilities` and never
  inflated.
- If `is_simulated=True`, `stored_kv_bytes` must reflect actual container size,
  not a hypothetical packed-bit size.
- `total_kv_footprint_bytes` is a conservative accounting sum, not a measured
  peak GPU value.

### 5.5 `capabilities` attribute

A `CompressorCapabilities` instance (see §7) constructed at adapter
initialization and immutable thereafter. Read-only access; must not be mutated
after `__init__`.

---

## 6. Backend identity metadata

Each `BackendAdapter` instance declares four identity strings at construction.
These are stored in `CompressorCapabilities.notes` (existing field) and in the
JSON report's `compressor_capabilities` dict via two new additive fields
(`backend_name`, `backend_version`).

| Field | Type | Example | Purpose |
|---|---|---|---|
| `name` | `str` | `"kvpress_snapkv"` | Registry name; must be unique; used in all report fields |
| `backend_name` | `str` | `"kvpress"` | External library identifier |
| `backend_version` | `str` | `"0.3.1"` | Pinned version of the external library |
| `adapter_name` | `str` | `"KVPressSnapKVAdapter"` | Python class name of this adapter |
| `adapter_version` | `str` | `"0.1.0"` | Version of the adapter code itself |

**Why pinning matters:** A backend's compress/decompress semantics may change
across library versions, making experiment results non-reproducible if the
version is not recorded. `backend_version` appears in JSON reports and in the
Experiment 005 document so the comparison can be reproduced.

**Schema impact:** `backend_name` and `backend_version` are **additive** fields
in `CompressorCapabilities` (with `default=""`) and, when serialised, in JSON
reports. Old V1–V5 compressors and reports load safely without them (they default
to empty string). No existing field is changed.

---

## 7. Capability metadata requirements

`CompressorCapabilities` must be fully declared at adapter construction time.
All fields are immutable after initialization.

| Field | Existing? | Real-backend requirement |
|---|---|---|
| `name` | Yes | Registry name; unique; matches `self.name` |
| `compressor_type` | Yes | `"quantization"` for quantizing backends; `"token_dropping"` for eviction backends; `"mixed"` if both |
| `is_simulated` | Yes | **`False`** for all real backends; `True` only if the adapter wraps a simulated backend (unlikely in V6) |
| `supports_real_bytes_claim` | Yes | **`True` only when `stored_kv_bytes` reflects actual compressed storage bytes**, not int8-container bytes; otherwise `False` |
| `supports_token_dropping` | Yes | `True` if the backend discards KV tokens (e.g. eviction methods) |
| `supports_quantization` | Yes | `True` if the backend quantises KV values |
| `key_bit_width` | Yes | Effective bits per key element (`8`, `4`, `2`) or `None` if full precision or unquantized |
| `value_bit_width` | Yes | Effective bits per value element; same semantics |
| `asymmetric` | Yes | `True` if K and V use different compression policies or granularities |
| `notes` | Yes | Human-readable description including `backend_name`, `backend_version`, `adapter_name` |
| `backend_name` | **New (additive)** | External library name (e.g. `"kvpress"`) |
| `backend_version` | **New (additive)** | Pinned library version string |

**`is_simulated=False` does not imply `supports_real_bytes_claim=True`.** A real
backend may still not be able to claim real packed-bit savings if it stores
sub-INT8 values in wider containers. The two fields are independent. A backend
must set `supports_real_bytes_claim=True` only when `stored_kv_bytes` provably
reflects the bytes actually held in persistent storage between decode steps.

---

## 8. Workspace-memory requirements

Every `BackendAdapter` must populate all five V5 workspace-aware fields in
`CompressionStats` via `_backend_workspace_bytes`. All values are byte counts
(`int`). All are conservative estimates; none claim to be measured peak GPU
memory.

| Field | What to measure | Honesty rule |
|---|---|---|
| `stored_kv_bytes` | Bytes of the compressed representation as held between decode steps (not during attention). For a real packed 4-bit backend: actual packed tensor bytes. For a backend that stores int8 + residual: int8 bytes + residual bytes. | Must NOT include materialized working copy bytes. Must NOT hypothesize smaller bit-packing than actually performed. |
| `materialized_working_kv_bytes` | Bytes of the working KV cache during an attention step. For all current materializing backends: equals `full_kv_bytes` (dequantised to model dtype). For a hypothetical fused dequantise-and-attend backend: may be smaller. | Must be reported honestly. Must equal `full_kv_bytes` unless the backend genuinely avoids a full-precision working copy during attention. |
| `metadata_bytes` | Bytes of per-tensor, per-channel, per-token, or per-vector scales, zero-points, codebooks, residual tensors, projection matrices, or entropy-coder tables. Must be complete — not just a per-tensor scale. | For a backend with a full-precision residual (e.g. KIVI-style): residual bytes are `metadata_bytes`, not `stored_kv_bytes`. For a transform-coded backend: codebook / projection-matrix bytes count here. |
| `temporary_workspace_bytes` | Conservative transient allocation during `_backend_compress` or `_backend_materialize`. Minimum: the size of one intermediate tensor (e.g. the dequantised output during materialize). | Report as a conservative lower bound. Do not report 0 if scratch allocation genuinely occurs. |
| `total_kv_footprint_bytes` | Conservative accounting sum: `stored_kv_bytes + materialized_working_kv_bytes + metadata_bytes + temporary_workspace_bytes`. | **NOT a measured peak GPU memory value.** Frame as an accounting total, not a device-memory measurement. Active GPU profiling is deferred. |

### 8.1 Required framing in all V6 output

Any report, CLI summary, or Markdown section involving a real backend adapter
must include:

- "`total_kv_footprint_bytes` is a conservative accounting sum derived from
  tensor shapes and byte widths. It is **not** a measured peak GPU memory value."
- "Active GPU memory measurement (`torch.cuda.memory_reserved`, etc.) is
  deferred and is not performed in V6."
- If the backend genuinely reduces `stored_kv_bytes` below int8-container size:
  "This backend claims real packed-bit storage (`supports_real_bytes_claim=True`).
  `materialized_working_kv_bytes` is [still / no longer] equal to `full_kv_bytes`."
- If the backend does NOT reduce `materialized_working_kv_bytes`: "This backend
  dequantises to full precision for attention. `materialized_working_kv_bytes ==
  full_kv_bytes`. Stored-byte savings do not reduce the working footprint."

---

## 9. Real vs simulated memory distinction

This is the most critical honesty boundary V6 introduces.

| | Simulated (`_sim`) compressors (V1–V5) | Real backend adapter (V6) |
|---|---|---|
| `is_simulated` | `True` | `False` |
| `stored_kv_bytes` | int8-container bytes (not packed bits) | Actual compressed bytes in persistent storage |
| `supports_real_bytes_claim` | `False` | `True` only if actual compressed bytes are smaller than full-precision bytes |
| `materialized_working_kv_bytes` | == `full_kv_bytes` (always) | == `full_kv_bytes` for materializing backends; may be smaller for fused kernels |
| Cite as memory saving? | **No** — int8 container is not a packed-bit saving | **Only if `supports_real_bytes_claim=True` and the field < `full_kv_bytes`** |

**Experiment 005 presentation rule:** In any table or report comparing a real
backend against simulated compressors, a column header or footnote must
distinguish: "⚠️ Simulated: stored_kv_bytes reflects int8 container, not
packed-bit savings" vs. "Real: stored_kv_bytes reflects actual compressed
storage."

---

## 10. Hook-safety requirements (kvpress-like backends)

Some backends — kvpress in particular — register hooks into the model's forward
pass to intercept and compress KV cache entries at each layer during generation.
This is architecturally different from ExactKV's post-prefill compress/materialize
pattern.

A hook-based backend introduces **hook-safety concerns** that must be resolved
before the adapter passes the Phase C gate:

### 10.1 Hooks must not run during the verify path

`VerificationEngine.verify_sequential` runs the full model on the full KV state.
If backend hooks are globally active, they may:
(a) compress the full-state cache during verification, corrupting it.
(b) change which past_key_values object the full model sees.
(c) silently skip running the hook (if the hook is registered on the draft model
only) — acceptable.

**Required behaviour:** Backend hooks must be inactive or no-op during all
verification-engine forward passes. The `BackendAdapter` must provide a context
manager or flag mechanism that the ExactKVGenerator can use:

```python
# Pseudocode — not yet implemented
with self.compressor.verification_mode():
    acceptance = self.engine.verify_sequential(full_state, draft_tokens)
```

### 10.2 Hooks must not mutate `FullKVState.past_key_values`

The deep-copy guard in `ExactKVGenerator._draft` and
`VerificationEngine.verify_sequential` protects against `DynamicCache` in-place
mutation. But a backend hook registered on the model itself could intercept a
forward pass on `full_state.past_key_values` before the deep copy runs.

**Required behaviour:** Backend hooks must either (a) be registered only on the
draft model's forward path, not the full model's, or (b) check a flag that is
off during all full-model forward calls.

### 10.3 Global monkeypatch risk (kvpress-specific)

Pre-Phase C research found that `import kvpress` calls
`patch_attention_functions()`, which **permanently** wraps every entry in
`transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS`. This is distinct from the
removable forward hooks registered by `with press(model):`.

The patch is mostly a pass-through when `module.masked_key_indices` is `None`
(default for `KnormPress` and most ScorerPress subclasses), but it still executes
on every attention call in the process. Phase C must include a **global
attention-patch gate** confirming verify and regression tests remain correct
after `import kvpress`.

`AdaKVPress` and other head-wise presses set `masked_key_indices` and rely on
this patch — they are **excluded** from V6 Phase C initial integration.

### 10.4 `verification_mode()` (required for hook-based backends)

kvpress hooks are scoped via `with press(model):` and removed on exit, but
ExactKV still needs an explicit guard because verify and commit run forwards on
the same `ModelRuntime.model` object.

**Implemented API (Phase C scaffold — default no-op):**

```python
@contextmanager
def verification_mode(self):
    """Default: no-op yield. Hook-based subclasses may assert hooks inactive."""
    yield
```

`BackendAdapter.verification_mode()` is implemented as a no-op context manager.
Hook-based subclasses (e.g. future `KVPressKnormAdapter`) may override to assert
no forward hooks are active or to disable hooks during verification.

`ExactKVGenerator._verify_draft_tokens` wraps `verify_sequential` inside
`compressor.verification_mode()` when the method exists (`callable` check).
Legacy compressors without `verification_mode` are unchanged.

### 10.5 Model-reference requirement (hook-based backends)

kvpress has no standalone `compress(cache)` API. Compression happens during a
model forward with `with press(model):` or via per-layer `press.compress()`
calls that need the attention `module`, `hidden_states`, and `kwargs`.

A `KVPressAdapter` must hold a `ModelRuntime` (or model) reference. The current
`_backend_compress(k_tensors, v_tensors, cache_format)` signature is
**insufficient** for full kvpress fidelity. Phase C options:

1. **Replay path (recommended):** re-run prefill with `with press(model):` on
   sequence tokens; store resulting `DynamicCache` in `backend_data`.
2. **Offline path (KnormPress only):** call `press.compress()` per layer with
   extracted keys/values and `model.model.layers[i].self_attn` as `module`.

### 10.6 Hook-safety gate (Phase C prerequisite)

Before the Phase C real-backend integration is considered passing, dedicated
tests must confirm:

```
Given: KVPressAdapter used press(model) during compress(); hooks now removed
When:  VerificationEngine.verify_sequential runs
Then:  full_state.past_key_values unchanged (shapes, values, kv_total_bytes)
       compressed_state.data unchanged
       no forward hooks remain on model.model.layers[*].self_attn
```

Additionally: after `import kvpress`, all existing ExactKV regression tests pass
(attention-patch gate).

**If hook-safety cannot be guaranteed, the backend is rejected for V6** and the
design-only fallback path is taken (§14 of V6_SCOPE_STATEMENT.md).

---

## 11. Determinism requirements

`_backend_compress` and `_backend_materialize` must be deterministic:

- **Same input → same output.** For a given `(k_tensors, v_tensors)` and fixed
  random seed, every call to `_backend_compress` must produce a `backend_data`
  object that, when passed to `_backend_materialize`, produces the same
  `past_key_values`. A stochastic step (e.g. Walsh–Hadamard randomized rotation,
  random pruning) must use a fixed, recorded seed.
- **Seed recording.** If the backend is stochastic, the seed used must be stored
  in `CompressedKVState.metadata["backend_seed"]` so that experiments are
  reproducible.
- **Why determinism matters for verification.** In `ExactKVGenerator._draft`,
  the materialized cache is deep-copied and used in a forward pass to produce
  draft tokens. If `_backend_materialize` is non-deterministic, two calls to it
  could produce different working caches, making the draft path non-reproducible
  and potentially breaking the alignment invariant.

---

## 12. No-mutation requirements

The adapter must observe the same no-mutation invariants as all existing
compressors:

1. **`compress` must not mutate `full_state.past_key_values`.** Use
   `extract_kv_tensors` which returns references; clone before passing to
   `_backend_compress` if the backend may modify its input.
2. **`materialize_for_draft` must not mutate `compressed.data`.** The
   `ExactKVGenerator._draft` deep-copies the returned cache, not `compressed.data`.
3. **`update_after_commit` may create a new `CompressedKVState` but must not
   modify the old one.** The old compressed state may still be referenced by
   the engine or bookkeeping code briefly after commit.
4. **`stats` must not run any forward pass.** It derives byte counts from tensor
   sizes only.

These match the invariants already documented in `exactkv/verification/engine.py`
and `exactkv/runtime/exactkv_generator.py`. Violating them causes the same
`DynamicCache` mutation bugs the existing deep-copy guards defend against.

---

## 13. DynamicCache compatibility requirements

`exactkv/cache/utils.py` currently supports three `past_key_values` formats:

| Format name | Condition | Used by |
|---|---|---|
| `"tuple"` | `isinstance(past_key_values, tuple)` | Legacy HF, some test fixtures |
| `"dynamic_v5"` | `hasattr(past_key_values, "layers")` | transformers ≥ 5.x |
| `"dynamic_v4"` | `hasattr(past_key_values, "key_cache")` | transformers 4.36–4.x |

`_backend_materialize` must return a cache object detectable by
`_detect_format`. That means:

- **Preferred:** Return the same format as the input. Pass `cache_format` from
  `extract_kv_tensors` through `_backend_compress` into `_backend_materialize`
  and use `rebuild_cache` to reconstruct.
- **Acceptable:** Return any of the three supported formats, with a comment in
  the adapter explaining the format choice.
- **Not acceptable:** Return a backend-specific custom object that `_detect_format`
  cannot handle. If a backend returns such an object, the adapter must wrap it
  in a supported format.

If a backend requires a new cache format, that format must be added to
`cache/utils.py` in the Phase B implementation — along with tests — before the
adapter is considered passing.

---

## 14. Error handling requirements

The adapter must fail loudly rather than silently emit misleading defaults:

| Situation | Required behaviour |
|---|---|
| Backend's `_backend_compress` raises | Re-raise, wrapped in a descriptive `RuntimeError` naming the backend and version |
| `_backend_workspace_bytes` cannot determine a field | Raise `NotImplementedError("...")` rather than returning 0 for a field that is known to be non-zero |
| `logical_seq_len` would violate alignment after compress | Assert immediately: `assert result.logical_seq_len == full_state.seq_len` |
| Hook registration fails at construction | Raise, not silently skip; adapter must not be partially initialized |
| Backend version mismatch at import time | Emit a `warnings.warn` with the expected and found versions; do not silently proceed if the API has changed |
| `_backend_materialize` returns an undetectable format | Raise with a description of the returned type and guidance for extending `cache/utils.py` |

---

## 15. Backend version pinning requirements

Reproducibility requires that every experiment record which exact external
library version was used.

**At adapter construction:**
```python
import importlib.metadata
_version = importlib.metadata.version("kvpress")  # or "kivi", etc.
if _version != EXPECTED_KVPRESS_VERSION:
    warnings.warn(
        f"KVPressSnapKVAdapter expects kvpress=={EXPECTED_KVPRESS_VERSION}, "
        f"found {_version}. Results may not be reproducible."
    )
self.backend_version = _version
```

**In every JSON report result** (via `compressor_capabilities`):
```json
"compressor_capabilities": {
    "name": "kvpress_snapkv",
    "backend_name": "kvpress",
    "backend_version": "0.3.1",
    "adapter_name": "KVPressSnapKVAdapter",
    "adapter_version": "0.1.0",
    "is_simulated": false,
    "supports_real_bytes_claim": true,
    ...
}
```

**In Experiment 005 docs:** The exact backend and adapter versions appear in
the experiment manifest section, analogous to the model name and prompt suite.

---

## 16. Pass-through proof-of-concept adapter

Before wiring in a heavyweight real backend (Phase C), Phase B introduces a
**`PassThroughAdapter`** — a trivial `BackendAdapter` subclass that stores the
full-precision KV tensors unchanged. It:

- Satisfies the `KVCompressor` protocol.
- Exercises every code path in `BackendAdapter`: `compress`, `materialize_for_draft`,
  `update_after_commit`, `stats`, `capabilities`.
- Declares `is_simulated=False`, `supports_real_bytes_claim=False` (no
  compression performed), `stored_kv_bytes == full_kv_bytes`,
  `materialized_working_kv_bytes == full_kv_bytes`.
- Passes the exactness gate trivially: since it returns full-precision KV,
  its compressed model predictions equal the full model's, so `exactkv_failures`
  should be 0 and acceptance rate should be 1.0.
- Is **not** registered in the compressor registry by default (test-only).

```python
# Pseudocode for PassThroughAdapter (Phase B — not yet written)
class PassThroughAdapter(BackendAdapter):
    name = "pass_through_adapter"
    backend_name = "none"
    backend_version = "0.0.0"
    adapter_name = "PassThroughAdapter"
    adapter_version = "0.1.0"

    def _backend_compress(self, k_tensors, v_tensors, cache_format):
        # Clone tensors to satisfy no-mutation requirement.
        return {"k": [t.clone() for t in k_tensors],
                "v": [t.clone() for t in v_tensors],
                "cache_format": cache_format}

    def _backend_materialize(self, backend_data, cache_format):
        return rebuild_cache(backend_data["k"], backend_data["v"], cache_format, 0)

    def _backend_workspace_bytes(self, full_state, backend_data):
        full_bytes = kv_total_bytes(full_state.past_key_values)
        return {
            "stored_kv_bytes": full_bytes,
            "materialized_working_kv_bytes": full_bytes,
            "metadata_bytes": 0,
            "temporary_workspace_bytes": full_bytes,
            "total_kv_footprint_bytes": full_bytes + 0 + full_bytes + full_bytes,
        }
```

`PassThroughAdapter` is the gate-keeper for Phase B: if it passes all tests, the
adapter boundary is sound and Phase C can proceed with a real backend.

---

## 17. How a kvpress adapter would fit at a high level

> ExactKV does not implement kvpress. This section describes how such an adapter
> would fit the design if kvpress integration is approved in Phase C.

kvpress registers **forward hooks** on attention layers via a context manager.
There is no `setup_hook()` API. The correct pattern is:
```python
from kvpress import KnormPress
press = KnormPress(compression_ratio=0.5)
with press(model):
    output = model(input_ids, past_key_values=cache)  # hooks compress in-place
compressed_cache = output.past_key_values
```

A `KVPressAdapter` would:

1. **During compress (replay path):** Hold a `ModelRuntime` reference; re-run
   prefill with `with press(model):` on sequence tokens; store the resulting
   `DynamicCache` as `backend_data`. Do not mutate `full_state.past_key_values`.
2. **`backend_name`:** `"kvpress"`. `backend_version`: pinned from
   `importlib.metadata.version("kvpress")`.
3. **Hook-safety (§10):** Hooks must be removed before verify/commit
   (`with press(model):` scope). `verification_mode()` asserts no hooks remain
   active. `import kvpress` also patches attention globally — test gate required.
4. **Version pin:** `kvpress==0.5.3` with `transformers>=4.56,<5.3` — conflicts
   with ExactKV's default 5.8.x; use isolated optional extra (see research doc).
5. **Phase C initial press:** `KnormPress` only; no DecodingPress/AdaKVPress.
6. **`_backend_materialize`:** Return `backend_data` directly (kvpress's
   compressed cache is already a `DynamicCache`).
7. **`supports_real_bytes_claim`:** Depends on which kvpress compressor is used.
   SnapKV (token dropping) returns a sparse cache that is smaller by token count
   but still stores full-precision tensors for retained tokens, so
   `stored_kv_bytes < full_kv_bytes` (real) but no bit-width reduction.
   A quantizing kvpress compressor would need per-compressor analysis.
8. **`materialized_working_kv_bytes`:** For SnapKV and similar, retained tokens
   are already at full precision; attention uses them directly.
   `materialized_working_kv_bytes == stored_kv_bytes` (no separate dequantize
   step), which is < `full_kv_bytes`.

This is the **only** V6-candidate adapter where `materialized_working_kv_bytes`
might genuinely differ from `full_kv_bytes` — and it only applies to
token-dropping compressors (fewer tokens, same precision), not quantizing ones.

---

## 18. How a KIVI adapter would differ at a high level

> ExactKV does not implement KIVI. This section describes how such an adapter
> would fit the design if KIVI integration is approved as the alternate V6 path.

KIVI (arXiv:2402.02750) stores keys per-channel at low bit-width (e.g. INT2/INT4)
and values per-token, plus a small full-precision residual stream. Its reference
implementation provides a `compress(k, v) → (quantized_k, quantized_v, residual)`
interface and a corresponding `decompress` call.

A `KIVIAdapter` would:

1. **In `_backend_compress`:** Call KIVI's per-channel key quantizer and
   per-token value quantizer on the extracted `k_tensors`, `v_tensors`. Store
   quantized tensors + residual as `backend_data`.
2. **`backend_name`:** `"kivi"`. `backend_version`: pinned.
3. **No hooks:** KIVI quantizes offline (not via forward-pass hooks), so there
   is no hook-safety concern. `verification_mode()` is a no-op.
4. **`_backend_materialize`:** Call KIVI's dequantize on `backend_data` to
   produce full-precision `k_tensors`, `v_tensors`, then call `rebuild_cache`.
   **`materialized_working_kv_bytes == full_kv_bytes`** (KIVI dequantizes to
   full precision for attention in its reference implementation).
5. **`stored_kv_bytes`:** quantized_k bytes + quantized_v bytes.
6. **`metadata_bytes`:** per-channel scales (K) + per-token scales (V) +
   **residual tensor bytes** — this is the key KIVI bookkeeping item that a
   naive `compressed_kv_bytes` would miss.
7. **`supports_real_bytes_claim`:** `True` if KIVI uses real packed 2-bit/4-bit
   storage; depends on the reference implementation's actual storage format.
   If the reference implementation stores INT2/INT4 in `int8` containers
   (as many do), then `supports_real_bytes_claim=False` for the container-level
   figure.
8. **GPU dependency:** KIVI's reference kernels are CUDA-only. A CPU path
   may not exist; the adapter correctness test would require a CUDA device.

**Key difference from kvpress:** KIVI has no hooks; it is a pure
compress-offline / dequantize-for-attention backend. This simplifies hook-safety
but requires careful residual accounting in `metadata_bytes`.

---

## 19. Tests and gates for Phase B and Phase C

### Phase B gates — `PassThroughAdapter`

| Test | Assertion |
|---|---|
| `PassThroughAdapter` satisfies `KVCompressor` | `isinstance`-like structural check; registry resolves name |
| `compress` does not mutate full_state | `kv_total_bytes` and `kv_seq_len` identical before and after |
| Alignment invariant | `result.logical_seq_len == full_state.seq_len` |
| `materialize_for_draft` returns detectable format | `_detect_format(materialized)` does not raise |
| `update_after_commit` alignment | `result.logical_seq_len == new_full_state.seq_len` |
| All V5 workspace fields present and non-negative | `stats().stored_kv_bytes >= 0` etc. |
| `total_kv_footprint_bytes == stored + materialized + metadata + temporary` | Exact equality |
| `exactkv_failures == 0` on smoke suite | Integration gate |
| Acceptance rate == 1.0 (pass-through is lossless) | Integration gate |
| No forbidden performance fields in `stats()` dict | Pattern audit |
| `backend_name`, `backend_version` present in capabilities | Field presence check |
| Backward-compat: old compressors load without `backend_name` | Field-absence test on existing fixtures |

### Phase C gates — first real backend adapter (kvpress or KIVI)

All Phase B gates, plus:

| Test | Assertion |
|---|---|
| Hook-safety (if applicable) | `full_state.past_key_values` bit-identical after `verify_sequential`; compressed.data unchanged |
| Determinism | Two calls to `compress(same_full_state)` → identical `next_token_id` from materialize |
| `supports_real_bytes_claim` accuracy | If `True`: `stored_kv_bytes < full_kv_bytes` on at least one real input; if `False`: flag is `False` |
| `metadata_bytes` completeness | Residual / scales / codebook bytes included; not just a per-tensor scale |
| `stored_kv_bytes` honesty | Does not hypothesize smaller packing than the backend actually performs |
| `materialized_working_kv_bytes` honesty | Reports `== full_kv_bytes` unless backend proves otherwise |
| `exactkv_failures == 0` on core suite | Hard gate; run before any Experiment 005 report |
| Experiment 005 runs end-to-end | Report generated; real vs simulated distinction present |
| No `tokens_per_second`, `throughput`, `latency`, `speedup`, `runtime_seconds` | Pattern audit on report and CLI output |
| `backend_version` in JSON report | Field present and matches pinned version |

---

## 20. What is explicitly out of scope for BackendAdapter

- ❌ **No throughput, latency, speedup, tokens/sec, or `runtime_seconds`** in
  any field, column, or output format.
- ❌ **No production-readiness claims.** An adapter passing the exactness gate
  is an *evaluation* result, not a deployment result.
- ❌ **No CUDA/Triton kernel writing by ExactKV.** If a backend brings kernels,
  ExactKV wraps them; ExactKV does not author them.
- ❌ **No active GPU memory profiling** (`torch.cuda.memory_reserved` etc.).
  `total_kv_footprint_bytes` is an accounting sum. GPU profiling is deferred.
- ❌ **No changes to `VerificationEngine` or the draft-verify-commit loop.**
- ❌ **No new simulation compressors.** V6 is real-backend integration, not more
  simulation.
- ❌ **No relaxation of `supports_real_bytes_claim=False` for `_sim` compressors.**
- ❌ **No batching, sampling, CPU offload, parallel verification, or
  bonus-token acceptance.**
- ❌ **No serving-stack integration** (vLLM, LMCache, PagedAttention). Those are
  V8 at the earliest.
- ❌ **No claiming external backend benchmark numbers as ExactKV results.** A
  real backend's speedup, perplexity, or memory numbers from its authors' papers
  must never appear without explicit attribution as external claims, and must
  never appear as ExactKV measurements.

---

## Appendix A: Notation used in this document

| Symbol | Meaning |
|---|---|
| `FullKVState` | `exactkv.cache.full_state.FullKVState` |
| `CompressedKVState` | `exactkv.cache.compressed_state.CompressedKVState` |
| `KVCompressor` | `exactkv.compressors.base.KVCompressor` (Protocol) |
| `CompressorCapabilities` | `exactkv.compressors.base.CompressorCapabilities` |
| `CompressionStats` | `exactkv.compressors.base.CompressionStats` |
| `extract_kv_tensors` | `exactkv.cache.utils.extract_kv_tensors` |
| `rebuild_cache` | `exactkv.cache.utils.rebuild_cache` |
| `kv_total_bytes` | `exactkv.cache.utils.kv_total_bytes` |
| `VerificationEngine` | `exactkv.verification.engine.VerificationEngine` |
| `ExactKVGenerator` | `exactkv.runtime.exactkv_generator.ExactKVGenerator` |

## Appendix B: V5 workspace-field formula (from `V5_SCOPE_STATEMENT.md` §5.6)

```
total_kv_footprint_bytes =
    stored_kv_bytes
    + metadata_bytes
    + max(materialized_working_kv_bytes, temporary_workspace_bytes)
```

For all current compressors and for most materializing real backends,
`materialized_working_kv_bytes == full_kv_bytes` and
`temporary_workspace_bytes ≈ stored_kv_bytes`, so `max(...)` equals
`full_kv_bytes` and the formula simplifies to:

```
total_kv_footprint_bytes ≈ stored_kv_bytes + metadata_bytes + full_kv_bytes
```

A real backend that avoids materializing a full-precision working copy would
populate `materialized_working_kv_bytes < full_kv_bytes`, which would change
this formula meaningfully. That is one of the key questions Experiment 005 is
designed to answer.
