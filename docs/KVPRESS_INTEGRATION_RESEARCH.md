# kvpress Integration Research (V6 Pre-Phase C)

**Status:** Research-only. ExactKV does **not** implement kvpress.
**Purpose:** Determine whether kvpress can safely serve as the first real backend
candidate for V6 Phase C, and define hook-safety requirements before any
integration code is written.
**Sources:** [NVIDIA/kvpress](https://github.com/NVIDIA/kvpress) (v0.5.3,
2026-04-09), [arXiv:2510.00636](https://arxiv.org/abs/2510.00636), ExactKV
adapter design (`docs/BACKEND_ADAPTER_INTERFACE.md`, `exactkv/compressors/backend_adapter.py`).

> ExactKV does not implement kvpress, KIVI, or any external compression backend.
> No performance, throughput, latency, speedup, runtime, or production-readiness
> claim. External kvpress benchmark numbers are **not** ExactKV results.

---

## 1. What kvpress is

**kvpress** is an NVIDIA-maintained, PyPI-distributed Python library
(`pip install kvpress`, current version **0.5.3**) that implements a large
family of **training-free KV-cache compression methods** called **presses**.
Each press reduces KV-cache memory by either:

- **Token dropping** — pruning low-importance key-value pairs along the sequence
  dimension (e.g. `KnormPress`, `SnapKVPress`, `ExpectedAttentionPress`).
- **Dimension pruning** — zeroing low-importance key channels (e.g. `ThinKPress`;
  no real byte savings without additional packing).
- **Quantization** — via Hugging Face `QuantizedCache` (opt-in; requires
  `optimum-quanto`).

kvpress is a **research and benchmarking framework**, not a serving stack. Its
stated goal is to simplify development and fair comparison of KV-cache compression
methods on Hugging Face transformer models. It ships a leaderboard and evaluation
CLI; those external results must not be cited as ExactKV measurements.

---

## 2. How kvpress integrates with Hugging Face models

### 2.1 Primary integration path

kvpress integrates at the **Hugging Face transformer layer**, not as a standalone
tensor compressor:

1. Load a `PreTrainedModel` (e.g. `Qwen2ForCausalLM`).
2. Create a `BasePress` subclass instance with a `compression_ratio`.
3. Wrap model forward passes in the press context manager: `with press(model):`.
4. Run forward passes; hooks compress the `DynamicCache` in-place per layer.

The recommended high-level API is `KVPressTextGenerationPipeline`, registered as
a transformers pipeline (`"kv-press-text-generation"`) on `import kvpress`.
The pipeline:

- Compresses context during **prefill** (backbone `model.model()` call with hooks).
- Generates answers with **greedy decoding** on the compressed cache.
- Separates context tokens from question tokens so compression applies only to
  context (important for fair evaluation).

### 2.2 Supported models

`SUPPORTED_MODELS` in `kvpress/presses/base_press.py`:

- `LlamaForCausalLM`, `MistralForCausalLM`, `Phi3ForCausalLM`
- `Qwen2ForCausalLM`, `Qwen3ForCausalLM`
- `Gemma3ForConditionalGeneration` (partial — sliding-window layers skipped)

`Qwen/Qwen2.5-0.5B` (ExactKV's gate model) maps to `Qwen2ForCausalLM` and is
architecturally compatible, though not explicitly listed in `SUPPORTED_MODELS`
(kvpress logs a warning for untested models).

### 2.3 Cache format

kvpress **requires** Hugging Face `DynamicCache` with `.layers[i].keys/.values`.
Its `forward_hook` accesses `cache.layers[module.layer_idx]` directly. Legacy
tuple caches and `dynamic_v4` (`key_cache`/`value_cache`) are **not** supported
by kvpress hooks. ExactKV's `cache/utils.py` supports all three formats, but a
kvpress adapter must materialize `dynamic_v5` caches.

### 2.4 Global side effect on import

**Critical finding:** `import kvpress` calls `patch_attention_functions()`, which
**globally monkeypatches** every function in `transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS`.
The patch wraps attention to support head-wise key masking (`module.masked_key_indices`)
for presses like `AdaKVPress`. The patch is **not scoped** to a context manager
and **cannot be undone** without process restart.

When `masked_key_indices` is `None` (default for most presses), the patch is
mostly a pass-through during decode, but it still runs on every attention call.
This is a hook-safety concern distinct from the removable forward hooks.

---

## 3. What kvpress compressor APIs exist

### 3.1 Press class hierarchy

| Class | Role |
|---|---|
| `BasePress` | Abstract base; defines `compress()`, `forward_hook()`, `__call__()` context manager |
| `ScorerPress` | Score-based token pruning; `compression_ratio` ∈ [0, 1) |
| Concrete presses | 25+ implementations (Knorm, SnapKV, ExpectedAttention, StreamingLLM, ThinK, …) |
| Wrapper presses | `ComposedPress`, `AdaKVPress`, `DecodingPress`, `PrefillDecodingPress`, `KeyRerotationPress`, … |

### 3.2 Core methods

**`compress(module, hidden_states, keys, values, attentions, kwargs)`**
Layer-local compression logic. Returns pruned `(keys, values)` tensors. Can be
called directly (offline) if all inputs are available.

**`forward_hook(module, input, kwargs, output)`**
Post-attention hook registered on `layer.self_attn`. Extracts keys/values from
`past_key_values`, calls `compress()`, writes back to `cache.layers[layer_idx]`.
Skips compression when `cache_position[-1] > q_len` (decode phase for prefill-only
presses).

**`__call__(model)` → context manager**
Registers forward hooks on all attention layers; removes them in `finally`.
Also assigns `layer.self_attn.rotary_emb = language_model.rotary_emb` per layer
(not restored on exit).

### 3.3 No standalone "compress this cache" API

kvpress has **no** public function of the form `compress_past_key_values(cache)`.
Compression is designed to happen **during a model forward pass** with hooks
active, or by calling `press.compress()` per layer with full layer context
(module, hidden_states, attentions, kwargs).

ExactKV's `BackendAdapter._backend_compress` currently receives only cloned
`(k_tensors, v_tensors, cache_format)` — **insufficient** for presses that need
`hidden_states` (ExpectedAttention, SnapKV, ThinK) or attentions (ObservedAttention).

### 3.4 Quantization path

Optional `QuantizedCache` (via `optimum-quanto`) stores quantized keys/values in
`cache_layer._quantized_keys/_quantized_values`. This is a separate path from
token-dropping presses and requires additional dependencies.

---

## 4. Hook, monkeypatch, and model-modification model

kvpress uses **three distinct mechanisms**:

| Mechanism | Scope | Reversible? | When active |
|---|---|---|---|
| **Forward hooks** (`register_forward_hook`) | Per attention layer | Yes — removed on context-manager exit | Only inside `with press(model):` |
| **Global attention patch** (`patch_attention_functions`) | All models in process | **No** — permanent after `import kvpress` | Always, after import |
| **Model attribute mutation** (`rotary_emb` assignment) | Per-layer `self_attn` | **No** — not restored on exit | During `with press(model):` setup |

There is **no** `setup_hook()` API. The correct pattern is:

```python
with press(model):
    outputs = model(input_ids, past_key_values=cache)
```

`DecodingPress` and `PrefillDecodingPress` register hooks that fire during
decode-phase forwards — **incompatible** with ExactKV's per-round draft loop
without careful scoping. These are **out of scope** for V6 Phase C.

`ComposedPress` chains multiple `forward_hook` calls per layer. Order-sensitive;
documented failure modes when presses depend on different inputs.

---

## 5. How kvpress stores compressed KV

After compression, kvpress writes pruned tensors back into the **same**
`DynamicCache` object in-place:

```python
cache_layer.keys = keys      # shape: [bsz, num_kv_heads, n_kept, head_dim]
cache_layer.values = values  # n_kept < original seq_len for token-dropping presses
```

For `QuantizedCache`, it stores quantized tensors and sets `keys`/`values` to
empty placeholders with `cumulative_length` tracking.

**Storage characteristics:**

- Retained tokens remain at **full precision** (fp16/bf16/fp32) — no bit-width
  reduction for token-dropping presses.
- Physical sequence length decreases; logical token positions of retained tokens
  are re-packed contiguously (0 … n_kept−1).
- RoPE position semantics may be wrong after pruning unless `KeyRerotationPress`
  is used — relevant for acceptance behaviour, not for ExactKV correctness
  (which compares against full KV, not kvpress's own quality claims).

---

## 6. How kvpress materializes or uses compressed KV during generation

kvpress does **not** dequantize or reconstruct a full-length cache. The compressed
`DynamicCache` **is** the working cache for attention:

- **Prefill-only presses:** hooks fire during prefill; during single-token decode
  steps, `cache_position[-1] > q_len` causes hooks to return without compressing.
- **Draft forwards** in ExactKV pass the compressed cache directly to
  `model.forward()` — same as kvpress's own `generate_answer()` loop.
- **No separate materialize step** for token-dropping presses: the pruned cache
  is used as-is.

Implication for ExactKV workspace memory:

- `stored_kv_bytes` = bytes of pruned cache tensors (< `full_bytes`).
- `materialized_working_kv_bytes` = same as `stored_kv_bytes` (no dequant step).
- This is the **first case** where `materialized_working_kv_bytes < full_kv_bytes`
  among ExactKV compressors.

---

## 7. Whether kvpress can fit ExactKV's KVCompressor protocol

**Partial fit — requires adapter extensions beyond current `BackendAdapter` shape.**

| KVCompressor method | kvpress fit | Gap |
|---|---|---|
| `compress(full_state)` | Indirect | Needs model reference + forward replay or per-layer offline `press.compress()` |
| `materialize_for_draft(compressed)` | Direct | Return stored `DynamicCache` from `backend_data` |
| `update_after_commit(compressed, new_full)` | Indirect | Re-compress from new full state (replay or offline) |
| `stats(compressed)` | Direct | Byte counts from pruned cache tensors |
| `capabilities` | Direct | Map press type to `CompressorCapabilities` |

**Alignment invariant:** ExactKV requires `compressed.logical_seq_len == full_state.seq_len`
(logical token count). kvpress physically shortens the cache. The adapter must
set `logical_seq_len = state.seq_len` while storing a shorter physical cache —
this is valid (alignment tracks logical tokens, not `kv_seq_len`).

**Determinism:** ScorerPress pruning uses `topk` — deterministic for fixed inputs.
`RandomPress` is non-deterministic and must be excluded from Phase C.

---

## 8. Required adapter boundary if kvpress is used

A `KVPressAdapter` (not yet implemented) would need:

### 8.1 Model reference

The adapter must hold a reference to `ModelRuntime.model` (or receive it at
construction). `_backend_compress` alone cannot integrate kvpress — the adapter
needs either:

- **Replay path (recommended for fidelity):** Re-run prefill forward with
  `with press(model):` on the full sequence tokens, capturing the compressed
  `DynamicCache`. Most faithful to kvpress semantics.
- **Offline path (restricted):** Call `press.compress()` per layer on extracted
  tensors. Only viable for presses whose `score()` uses keys alone (e.g.
  `KnormPress`). Requires `module` reference from `model.model.layers[i].self_attn`.

### 8.2 Strict hook scoping

```python
# Pseudocode — not implemented
with press(model):
    compressed_cache = replay_prefill(tokens, starting_cache=cloned_full_cache)
# hooks removed here — safe for verify/draft/commit on full path
```

Hooks must **never** be registered during:
- `VerificationEngine.verify_sequential`
- `ExactKVGenerator._commit` (full-model path)
- Any forward on `full_state.past_key_values`

Prefill-only presses naturally skip compression during single-token decode
(`cache_position[-1] > q_len`), so draft forwards with an already-compressed
cache do not re-trigger hooks **if hooks are not registered**.

### 8.3 `verification_mode()` context manager

Required on `BackendAdapter` for hook-based backends (see §11).

### 8.4 Cache format constraint

Materialized cache must be `dynamic_v5` (`DynamicCache` with `.layers`). The
adapter should call `rebuild_cache(..., "dynamic_v5", ...)` or store the
`DynamicCache` object directly.

### 8.5 Press selection restrictions for Phase C

| Allowed (Phase C initial) | Excluded from Phase C |
|---|---|
| `KnormPress` (keys-only, simplest offline path) | `DecodingPress`, `PrefillDecodingPress` |
| `SnapKVPress` (if replay path used) | `AdaKVPress` (requires global attention patch side effects) |
| `ExpectedAttentionPress` (if replay path used) | `RandomPress` (non-deterministic) |
| `StreamingLLMPress` | `KVzipPress` (multi-forward; expensive) |
| | `ComposedPress` (ordering fragility) |
| | `ThinKPress` alone (no byte savings; dimension zeroing) |

Start with **`KnormPress`** as the Phase C gate press.

---

## 9. Hook-safety risks

| Risk | Severity | Detail |
|---|---|---|
| Hooks active during `verify_sequential` | **Critical** | Would compress full-state cache in-place, corrupting authoritative KV |
| Hooks active during `_commit` | **Critical** | Same — mutates full path cache |
| Global `patch_attention_functions` | **High** | Permanent after `import kvpress`; affects all attention calls in process |
| `rotary_emb` reassignment | **Medium** | Model state mutation during `with press(model):`; not restored |
| `DecodingPress` hooks during draft | **High** | Would compress during ExactKV draft loop |
| Hook leak (not removed on exception) | **High** | kvpress uses `try/finally` in `__call__` — should be safe if context manager used correctly |
| Deep-copy insufficient | **Medium** | Hooks mutate cache in-place; deep-copy of `full_state.past_key_values` before verify protects against DynamicCache mutation, but not against hooks registered on the model |
| `logical_seq_len` vs physical mismatch | **Low** | By design for token-dropping; adapter must set logical correctly |

**Most important invariant:** `VerificationEngine.verify_sequential` must run
with **zero active kvpress forward hooks** and must not have
`full_state.past_key_values` modified. The existing `copy.deepcopy` guard in
the verification engine protects against DynamicCache in-place growth but **does
not** protect against hook-driven in-place pruning of the authoritative cache if
hooks are registered during verify.

---

## 10. Exact hook-safety tests needed before Phase C

These tests must pass before any kvpress adapter is merged:

### 10.1 Hook isolation gate

```
Given: KVPressAdapter with KnormPress, hooks were used during compress()
When:  VerificationEngine.verify_sequential(full_state, draft_tokens) runs
Then:  full_state.past_key_values is unchanged (tensor shapes, values, kv_total_bytes)
       compressed_state.data is unchanged
       no forward hooks remain registered on model.model.layers[*].self_attn
```

### 10.2 Verify-path forward gate

```
Given: press was used in compress(); hooks now removed
When:  runtime.forward(token, past_key_values=deepcopy(full_state.past_key_values)) runs
Then:  returned past_key_values seq_len matches expectation
       full_state.past_key_values unchanged
```

### 10.3 Draft-path gate

```
Given: compressed cache with n_kept < full_seq_len
When:  ExactKVGenerator._draft runs
Then:  draft completes without exception
       no hooks fire during draft (assert hook call count == 0)
```

### 10.4 Exactness gate

```
Given: backend_kvpress_knorm (or similar) on Qwen/Qwen2.5-0.5B
When:  ExactKV runs 2 prompts × 2 lengths × 2 draft lengths
Then:  exactkv_output_ids == generate_full_greedy.generated_ids
       (Same gate as NoOp/pass-through — lossy compression may fail this; that is a valid outcome)
```

### 10.5 Global attention-patch gate

```
Given: import kvpress has patched ALL_ATTENTION_FUNCTIONS
When:  verify_sequential runs without any press hooks active
Then:  full_state.past_key_values unchanged
       (Confirms patch is no-op when masked_key_indices is None)
```

### 10.6 Import side-effect documentation gate

```
Given: kvpress imported in test process
When:  ExactKV pass-through adapter tests run
Then:  all pass (no regression from global attention patch)
```

---

## 11. Whether `verification_mode()` is needed and what it should do

**Yes — required for hook-based backends.**

The current `BackendAdapter` base class does not implement `verification_mode()`.
Research confirms it is needed even though kvpress hooks are context-manager-scoped,
because:

1. The adapter must **assert** no hooks are registered before verify/commit.
2. The generator should wrap verify in `verification_mode()` as a belt-and-suspenders
   guard (minimal generation-logic change: one context manager wrap).
3. It provides a documented, testable contract for Phase C.

**Proposed behaviour (design only):**

```python
@contextmanager
def verification_mode(self):
    """Assert no backend hooks are active; yield; assert full_state unchanged."""
    if self._hook_handles:  # adapter tracks handles if it ever registers globally
        raise RuntimeError("Backend hooks active during verification")
    yield
    # Post-verify: optional full_state integrity check in tests
```

For kvpress specifically, if hooks are correctly scoped to `compress()` only,
`verification_mode()` is primarily an **assertion guard**, not a hook-disabler.
If a future press requires persistent hooks, `verification_mode()` must disable them.

**Minimal `ExactKVGenerator` change for Phase C:** wrap only the
`verify_sequential` call:

```python
with self._verification_guard():  # calls compressor.verification_mode() if present
    acceptance = self.engine.verify_sequential(full_state, draft_tokens)
```

Use `hasattr(compressor, 'verification_mode')` to preserve backward compatibility
with existing compressors.

---

## 12. Whether `full_state.past_key_values` can remain authoritative and unmodified

**Yes — if and only if hooks are not registered during verify or commit.**

ExactKV already protects the verify path with `copy.deepcopy(full_state.past_key_values)`.
The commit path (`_commit`) runs forwards on `full_state.past_key_values` directly
— hooks **must not** be active here.

The compress path may use hooks on a **clone** or replay forward, storing the
result in `compressed_state.data` without touching `full_state.past_key_values`.
The current `BackendAdapter.compress()` clones KV tensors before `_backend_compress`
— a replay-based kvpress adapter should clone the full `DynamicCache` or replay
from tokens without mutating the authoritative object.

---

## 13. Whether `compressed_state` can store kvpress objects without breaking cache alignment

**Yes.**

`CompressedKVState.data` can store a `DynamicCache` (or a dict wrapping one).
Fields:

- `logical_seq_len = full_state.seq_len` (logical alignment invariant)
- `data` = pruned `DynamicCache` (physical `kv_seq_len` may be smaller)
- `metadata["next_token_id"]` = draft prediction from compressed path (requires
  `_get_next_token_id` override: materialize + one forward, or replay capture)

Cache alignment in traces compares `full_state.seq_len` vs `compressed.logical_seq_len`,
not physical `kv_seq_len`. Token-dropping compressors are compatible with this
invariant.

---

## 14. Workspace-memory implications

For token-dropping kvpress presses (e.g. KnormPress, SnapKV):

| Field | Expected value |
|---|---|
| `stored_kv_bytes` | `kv_total_bytes(pruned_cache)` < `full_bytes` |
| `materialized_working_kv_bytes` | == `stored_kv_bytes` (no dequant; pruned cache used directly) |
| `metadata_bytes` | 0 (no scales/zero-points for token-dropping) |
| `temporary_workspace_bytes` | 0 conservatively; replay forward may use scratch — document honestly |
| `total_kv_footprint_bytes` | conservative accounting sum; it is NOT a measured peak GPU memory value |
| `supports_real_bytes_claim` | `True` — fewer tokens at full precision is genuine storage reduction |
| `compression_ratio` | < 1.0 |
| `is_simulated` | `False` |

For `QuantizedCache` presses: additional `metadata_bytes` for quantizer state;
requires separate per-press analysis in Phase C.

**First ExactKV compressor where `materialized_working_kv_bytes < full_kv_bytes`.**
Reports and Markdown rendering must handle this without implying GPU peak measurement.

---

## 15. CPU/GPU requirements

| Scenario | CPU | GPU |
|---|---|---|
| KnormPress correctness gate (0.5B, fp32) | **Works** — press is tensor math only | Optional |
| ExpectedAttentionPress | Works on CPU; slower | Recommended |
| flash_attention_2 | Not required | Used in kvpress examples for speed (out of scope for ExactKV claims) |
| QuantizedCache | Requires `optimum-quanto` | Either |
| kvpress evaluation CLI / leaderboard | — | External; not ExactKV |

ExactKV V6 policy unchanged: CPU suffices for exactness gate; GPU for Experiment
005 if desired, not for peak-memory profiling.

---

## 16. Version pinning recommendation

| Package | Pin | Notes |
|---|---|---|
| `kvpress` | `==0.5.3` | Current PyPI; record in `backend_version` |
| `transformers` | `>=4.56.0,<5.3` | **kvpress constraint** |
| `torch` | `>=2.3.1,<3` | kvpress constraint |
| `numpy` | `>=2.0.0,<3` | kvpress requires numpy 2.x |

### Transformers version conflict (blocking risk)

| Component | transformers version |
|---|---|
| ExactKV dev environment (measured) | **5.8.1** |
| kvpress 0.5.3 requirement | **>=4.56.0, <5.3** |
| ExactKV `pyproject.toml` | `>=4.40` (no upper bound) |

**These ranges do not overlap at 5.8.1.** Phase C must either:

1. Run kvpress integration in an **isolated optional extra** with
   `transformers<5.3` (separate CI job or manual gate), or
2. Wait for kvpress to lift the `<5.3` cap, or
3. Vendor/fork kvpress with transformers 5.8 compatibility patches.

Option 1 is the pragmatic Phase C path. Document the pin in Experiment 005;
do not change ExactKV's default transformers pin for the main test suite until
compatibility is confirmed.

---

## 17. Failure modes that should reject kvpress for V6

| Failure mode | Action |
|---|---|
| transformers version incompatibility cannot be isolated | Reject kvpress; fall back to KIVI or design-only stop |
| Global attention patch breaks verify or pass-through tests | Reject kvpress |
| Hooks cannot be guaranteed inactive during verify/commit | Reject kvpress |
| `full_state.past_key_values` mutated after verify (any press) | Reject kvpress |
| Only viable presses require DecodingPress / AdaKVPress | Defer those presses; if none remain viable, reject |
| Offline compression impossible and replay forward mutates full state | Reject kvpress |
| `exactkv_failures > 0` on core suite with all reasonable press configs | Not a rejection of kvpress — valid experimental outcome; adapter still valid if exactness gate is testable |
| Import kvpress breaks existing 1309 tests | Reject until isolated optional dependency |

---

## 18. Recommendation

### **Proceed with kvpress — only with restrictions**

kvpress remains the **best first backend candidate** for V6 Phase C, but research
reveals tighter constraints than the Phase A design assumed:

1. **Transformers version isolation is mandatory** before any integration code.
2. **Start with `KnormPress` only** — simplest offline/replay path, keys-only scoring.
3. **No `DecodingPress` / `PrefillDecodingPress` / `AdaKVPress`** in V6.
4. **Adapter needs model reference** — extend `BackendAdapter` or `KVPressAdapter`
   with `ModelRuntime` access; `_backend_compress` alone is insufficient.
5. **Implement `verification_mode()`** and wrap `verify_sequential` in the generator.
6. **Treat `import kvpress` global attention patch as a test gate**, not optional.
7. **Do not use kvpress pipeline** for ExactKV integration — use press context
   manager + ExactKV's own prefill/draft/verify loop.
8. **Exactness gate may fail** for lossy presses — that is a valid Experiment 005
   outcome, not an integration failure.

**Do not defer to KIVI yet.** KIVI remains the documented fallback if hook-safety
or transformers isolation fails. KIVI's advantage (no hooks) does not outweigh
kvpress's HF-native multi-press surface **unless** the blockers above cannot be
resolved.

**Do not stop V6 at adapter design.** Phase B pass-through PoC is sound; Phase C
can proceed after the hook-safety and version-isolation gates are defined (this
document).

---

## Appendix A: kvpress API summary (quick reference)

```
# Installation
pip install kvpress  # 0.5.3; pulls transformers>=4.56,<5.3

# Import side effect
import kvpress  # patches ALL_ATTENTION_FUNCTIONS globally

# Press usage
from kvpress import KnormPress
press = KnormPress(compression_ratio=0.5)
with press(model):
    out = model(input_ids, past_key_values=DynamicCache())

# Press hierarchy
BasePress → ScorerPress → KnormPress, SnapKVPress, ExpectedAttentionPress, ...
BasePress → ThinKPress, DecodingPress, ComposedPress, ...

# Pipeline (NOT for ExactKV integration)
pipe = pipeline("kv-press-text-generation", model=model, device="cuda:0")
answer = pipe(context, question=question, press=press)["answer"]
```

---

## Appendix B: ExactKV constraint checklist

| Constraint | kvpress compliance |
|---|---|
| `verify_sequential` not corrupted | Requires hook isolation — testable |
| Full KV authoritative | Yes, if hooks scoped to compress only |
| Greedy decoding only | Compatible |
| No throughput/latency claims | Compatible — do not use kvpress benchmark docs |
| Evaluate via exactness/acceptance/memory fields | Compatible |
| No external benchmark as ExactKV result | Compatible with discipline |
| `BackendAdapter` protocol | Needs model-reference extension |
| Backward-compatible reports | Compatible — `backend_name`/`backend_version` additive |

---

*Research completed 2026-06-09. No kvpress code implemented in ExactKV.*
