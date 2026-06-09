# kvpress Integration Research (V6 Phase C)

**Status:** Research-only — empirical validation completed 2026-06-09.
ExactKV does **not** implement kvpress, `KVPressKnormAdapter`, or any kvpress compressor registration.

**Purpose:** Determine whether kvpress can safely serve as the first real backend
candidate for V6 Phase C, and define hook-safety requirements before adapter
implementation.

**Methods:** Source review of [NVIDIA/kvpress](https://github.com/NVIDIA/kvpress)
v0.5.3, plus hands-on experiments in a dedicated `.venv-kvpress` environment
(`pip install -e ".[kvpress]"`) using `Qwen/Qwen2.5-0.5B` on CPU/float32.

> No performance, throughput, latency, speedup, runtime, or production-readiness
> claim. External kvpress leaderboard numbers are **not** ExactKV results.

---

## 1. kvpress summary

**kvpress** (v0.5.3, NVIDIA, Apache 2.0) is a research framework for
training-free KV-cache compression on Hugging Face transformer models. Compression
is implemented via **presses** — dataclass objects that register **forward hooks**
on attention layers during `with press(model):` and prune KV tokens (or key
channels) in-place inside a `DynamicCache`.

kvpress is **not** a standalone tensor compressor. There is no public
`compress_past_key_values(cache)` API. Compression happens during model forward
passes with hooks active, or via per-layer `press.compress(module, hidden_states,
keys, values, attentions, kwargs)` when full layer context is available.

ExactKV evaluates backends by exactness, acceptance, divergence, rejection,
correction, and workspace-memory fields — never by external speed or serving claims.

---

## 2. Version and dependency constraints

### 2.1 Pinned optional extra (ExactKV `pyproject.toml`)

```toml
kvpress = [
    "kvpress==0.5.3",
    "transformers>=4.56,<5.3",
]
```

Install only in a dedicated environment: `pip install -e ".[kvpress]"`.
**Not** on the default ExactKV install path.

### 2.2 Empirical install results (2026-06-09)

| Environment | Python | transformers | kvpress import | Notes |
|---|---|---|---|---|
| Default ExactKV | 3.13.3 | **5.8.1** | Not installed | `import exactkv.compressors` does **not** load kvpress |
| `.venv-kvpress` (dedicated) | 3.13.3 | **5.2.0** | **0.5.3** | Requires `fire>=0.7.1` workaround (see below) |
| `.venv-kvpress312` (attempted) | 3.12.3 | — | **Failed** | pyenv 3.12 missing `_lzma` module |

### 2.3 Install blockers discovered

**Python 3.13 + kvpress 0.5.3 default pins:** `fire==0.6.0` imports removed stdlib
module `pipes` → `ModuleNotFoundError` on `import kvpress`.

**Workaround (research venv only):** `pip install 'fire>=0.7.1'` after
`pip install -e ".[kvpress]"`. Conflicts with kvpress's `fire<0.7` pin but import
succeeds. Document this in Phase C CI; do not add to default `pyproject.toml`
until upstream fixes or ExactKV documents the override in the `[kvpress]` extra.

**Transformers split:** Default ExactKV (5.8.x) and kvpress extra (≤5.2.x) **cannot
coexist** in one environment. Phase C adapter development and kvpress-specific tests
must run in the isolated `[kvpress]` venv. Main CI stays on default transformers.

---

## 3. API surface discovered

### 3.1 Entry point

```python
import kvpress  # side effect: patch_attention_functions() runs globally
from kvpress import KnormPress
```

`kvpress.__all__` exports **29** `*Press` classes (empirical, 2026-06-09).

### 3.2 BasePress public API

| Method / attribute | Role |
|---|---|
| `compression_ratio` | Fraction of KV pairs to remove (ScorerPress); ∈ [0, 1) |
| `compress(module, hidden_states, keys, values, attentions, kwargs)` | Layer-local prune logic; returns `(keys, values)` |
| `forward_hook(module, input, kwargs, output)` | Post-attention hook; mutates `DynamicCache` in-place |
| `__call__(model)` | Context manager: register hooks on all attention layers; remove in `finally` |
| `post_init_from_model(model)` | Optional model-dependent setup |

There is **no** `setup_hook()`. Correct pattern:

```python
with press(model):
    out = model(input_ids, past_key_values=cache)
```

### 3.3 KnormPress (Phase C initial candidate)

```python
KnormPress(compression_ratio: float = 0.0)
```

- MRO: `KnormPress → ScorerPress → BasePress`
- Scoring: `score()` returns `-keys.norm(dim=-1)` (keys only; no hidden_states required for scoring)
- Still uses `forward_hook` during forward; offline `compress()` alone is insufficient without `module` reference
- `compression_ratio=0.5` on 5-token prefill → keeps `int(5 * 0.5) = 2` tokens per layer

### 3.4 Internal state

Prefill-only presses do not persist hook handles on the press object. Hook handles
live on `layer.self_attn._forward_hooks` during the context manager. `DecodingPress`
maintains `hidden_states_buffer` and `layer_step_counts` — **excluded from V6 Phase C**.

---

## 4. Press classes relevant to V6

| Press | V6 Phase C | Reason |
|---|---|---|
| **KnormPress** | **Initial gate press** | Simplest scorer; keys-only; prefill-only |
| SnapKVPress | Phase C+ (if replay path works) | Needs hidden_states / attention context |
| ExpectedAttentionPress | Phase C+ | Needs query statistics, RoPE, hidden_states |
| StreamingLLMPress | Later | Sink+recent policy; still prefill-only |
| RandomPress | **Never** | Non-deterministic |
| DecodingPress | **Never in V6** | Hooks fire during decode |
| PrefillDecodingPress | **Never in V6** | Combined decode hooks |
| AdaKVPress | **Never in V6** | Requires global attention patch side effects |
| ComposedPress | **Never in V6** | Ordering fragility |
| ThinKPress | **Never in V6** | Dimension zeroing; no byte savings |
| KVPressTextGenerationPipeline | **Never** | Bypasses ExactKV draft-verify-commit |

---

## 5. Hugging Face integration model

1. Load `PreTrainedModel` (e.g. `Qwen2ForCausalLM` for `Qwen/Qwen2.5-0.5B`).
2. Create `DynamicCache()` (kvpress requires `.layers[i].keys/.values`).
3. `with press(model): model(input_ids, past_key_values=cache)`.
4. Hooks fire per layer during prefill; cache tensors shortened in-place.
5. Subsequent single-token decode steps: prefill-only hooks skip when
   `cache_position[-1] > q_len` (no re-compression during decode **if hooks not registered**).

**ExactKV integration must NOT use `KVPressTextGenerationPipeline`.** Use ExactKV's
own prefill (`prefill_to_full_state`), draft, verify, commit loop with compression
only inside adapter `compress()` replay.

**Supported model types** (`SUPPORTED_MODELS`): Llama, Mistral, Phi3, Qwen2, Qwen3,
Gemma3 (partial). `Qwen/Qwen2.5-0.5B` works empirically (maps to `Qwen2ForCausalLM`;
kvpress logs a warning for untested models).

---

## 6. Hook and global-patch behavior

### 6.1 Forward hooks (`with press(model):`)

**Empirical (Qwen2.5-0.5B, 24 layers):**

| Phase | `_forward_hooks` per layer |
|---|---|
| Before `with press(model):` | 0 |
| During | 1 per layer (24 total) |
| After context exit | 0 |

Hooks are **removed** on context exit (`try/finally` in `BasePress.__call__`).
They are registered on `layer.self_attn`, not globally on the module tree root.

**Side effect not restored:** `layer.self_attn.rotary_emb = language_model.rotary_emb`
is assigned during hook setup and not reverted.

### 6.2 Global attention monkeypatch (`import kvpress`)

`kvpress/__init__.py` calls `patch_attention_functions()`, which wraps every entry
in `transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS` with `attention_patch`.
This is **permanent** for the process lifetime.

**Empirical:** `attention_fn.__wrapped__` exists after import — patch is active.
For `KnormPress` (no `masked_key_indices`), the wrapper is mostly pass-through but
still executes on every attention call.

### 6.3 Hook inspection for `verification_mode()`

Future `KVPressKnormAdapter.verification_mode()` can assert:

```python
for layer in model.model.layers:
    assert len(layer.self_attn._forward_hooks) == 0
```

This is testable and sufficient for removable forward hooks. It does **not** undo
the global attention patch (no API for that).

### 6.4 Can hooks be disabled during ExactKV verification?

**Yes, if designed correctly:**

- Register hooks only inside `compress()` replay (`with press(model):`).
- Never register during `verify_sequential`, `_commit`, or `_draft`.
- `ExactKVGenerator` already wraps verify in `verification_mode()` (scaffold).
- Prefill-only presses do not fire during single-token decode when hooks are absent.

**Risk:** If hooks leak (exception before `finally`, or bug), verify path corrupts
full KV. The `verification_mode()` override must assert zero hooks.

---

## 7. Cache format behavior

### 7.1 Empirical (Qwen/Qwen2.5-0.5B, KnormPress 0.5)

| Metric | Full prefill | After KnormPress 0.5 |
|---|---|---|
| `cache/utils._detect_format` | `dynamic_v5` | `dynamic_v5` |
| Physical `kv_seq_len` | 5 | 2 |
| `kv_total_bytes` | 122,880 | 49,152 |
| Draft forward on compressed cache | — | **OK** (seq grows to 3 after one token) |

### 7.2 Compatibility with ExactKV `cache/utils.py`

- `_detect_format`: **compatible** (`dynamic_v5`)
- `extract_kv_tensors` / `rebuild_cache`: work on stored `DynamicCache`
- Tuple and `dynamic_v4` caches: **not** produced by kvpress hooks; adapter must
  store/materialize `dynamic_v5` only

### 7.3 Logical vs physical sequence length

**Critical:** `compressed.logical_seq_len` must equal `full_state.seq_len` (5) while
physical `kv_seq_len` may be shorter (2). ExactKV alignment invariant uses logical
length, not physical cache length.

### 7.4 Greedy output under compression

Empirical: compressed-cache greedy decode **diverges** from `generate_full_greedy`
on the same prompt (expected for lossy compression). ExactKV's job is to verify
and correct so final output still matches full KV — acceptance rate may be < 1.0.

---

## 8. Whether kvpress fits BackendAdapter

**Partial fit — requires adapter extension beyond current `_backend_compress` signature.**

| `KVCompressor` method | kvpress fit |
|---|---|
| `compress(full_state)` | Needs **replay forward** with `with press(model):` + `input_ids`; cannot use tensor-only `_backend_compress` |
| `materialize_for_draft(compressed)` | Return stored `DynamicCache` directly |
| `update_after_commit` | Re-replay from new full state |
| `stats(compressed)` | `kv_total_bytes` on pruned cache |
| `capabilities` | Map press metadata + `backend_name="kvpress"` |
| `verification_mode()` | Override to assert zero `_forward_hooks` |
| `_get_next_token_id` | Override: one forward on materialized cache |

---

## 9. Required BackendAdapter changes

The current `BackendAdapter._backend_compress(k_tensors, v_tensors, cache_format)`
is **insufficient** for kvpress. Phase C adapter implementation needs:

1. **`ModelRuntime` reference** (or `model` + `device`) on `KVPressKnormAdapter`.
2. **`input_ids` for replay** — from `full_state.full_sequence_ids` or prompt+gen.
3. **New compress path** — either:
   - override sealed `compress()` on a `KVPressBackendAdapter` subclass, or
   - add protected `_backend_compress_from_full_state(full_state)` hook called from
     `compress()` when `runtime` is set.
4. **`verification_mode()` override** — assert `len(layer.self_attn._forward_hooks)==0`
   for all layers before/after verify.
5. **Hook handle tracking (optional)** — adapter stores active handles only during
   `compress()` for defensive checks.
6. **Do not register** adapter in default `exactkv.compressors` import path until
   kvpress extra is installed (lazy registration or separate optional module).

**No change needed** to `VerificationEngine` or report schemas for the adapter itself.

---

## 10. Hook-safety risks

| Risk | Severity | Mitigation |
|---|---|---|
| Hooks active during `verify_sequential` | **Critical** | Scope hooks to `compress()` only; `verification_mode()` assert |
| Hooks active during `_commit` | **Critical** | Never register outside `compress()` |
| Global `patch_attention_functions` | **High** | Run attention-patch gate after `import kvpress`; process-isolated CI job |
| `rotary_emb` reassignment | Medium | Accept for Phase C; document |
| `DynamicCache` in-place mutation during draft | Medium | `ExactKVGenerator._draft` already deep-copies materialized cache |
| Physical/logical seq_len mismatch | Low | Set `logical_seq_len = full_state.seq_len` explicitly |
| Python 3.13 `fire` import failure | **High** | Document workaround; prefer Python 3.10–3.12 for kvpress CI |
| transformers version split | **High** | Isolated `[kvpress]` venv; never merge pins into default |

---

## 11. Required Phase C tests

1. **Hook isolation:** zero `_forward_hooks` during `verify_sequential`.
2. **Full-state immutability:** `full_state.past_key_values` unchanged after verify.
3. **Compressed-state immutability:** `compressed.data` unchanged after verify.
4. **Hook register/remove:** count 0 → N → 0 across `with press(model):`.
5. **Attention-patch gate:** full default pytest suite passes in a subprocess that
   `import kvpress` first (or kvpress-env job).
6. **Lazy import gate:** `import exactkv.compressors` does not load kvpress (existing).
7. **`verification_mode()` gate:** override asserts when hooks leaked (unit test with mock).
8. **Alignment gate:** `logical_seq_len == full_state.seq_len` after compress.
9. **Workspace gate:** `stored_kv_bytes < full_bytes`; footprint reconciles.
10. **Exactness gate:** ExactKV output vs `generate_full_greedy` (may fail for lossy
    press — valid experimental outcome; adapter correctness ≠ 100% acceptance).

---

## 12. Workspace-memory implications

For token-dropping presses (KnormPress, SnapKV, etc.) — **empirical KnormPress 0.5:**

| Field | Value / rule |
|---|---|
| `stored_kv_bytes` | `kv_total_bytes(pruned DynamicCache)` — **< full_bytes** |
| `materialized_working_kv_bytes` | **== stored_kv_bytes** (pruned cache used directly; no dequant) |
| `metadata_bytes` | 0 (no scales/zero-points for token-dropping) |
| `temporary_workspace_bytes` | 0 conservatively; replay forward may use scratch — document if measured |
| `total_kv_footprint_bytes` | accounting sum; **NOT a measured peak GPU memory value** |
| `supports_real_bytes_claim` | **True** — fewer tokens at full precision is genuine storage reduction |
| `compression_ratio` | < 1.0 |
| `is_simulated` | False |

**First ExactKV compressor where `materialized_working_kv_bytes < full_kv_bytes`.**

`QuantizedCache` presses (opt-in, `optimum-quanto`): separate analysis deferred.

---

## 13. CPU/GPU requirements

| Scenario | CPU | GPU |
|---|---|---|
| KnormPress gate (0.5B, fp32) | **Verified** | Optional |
| ExpectedAttentionPress | Works; slower | Recommended in kvpress examples |
| kvpress evaluation CLI / leaderboard | — | External; not ExactKV |

ExactKV policy: CPU suffices for adapter correctness gate; GPU optional for
Experiment 005 (not in scope of this research pass).

---

## 14. Failure modes that reject kvpress for V6

| Failure mode | Action |
|---|---|
| `import kvpress` breaks default pytest after global patch | Reject or isolate to kvpress CI subprocess |
| Cannot install `[kvpress]` on supported Python without workarounds | Document; defer adapter |
| Hooks cannot be guaranteed inactive during verify | Reject; fall back to KIVI |
| `full_state.past_key_values` mutated after verify | Reject |
| Only viable presses require DecodingPress/AdaKVPress | Restrict press set; reject if none remain |
| transformers isolation impossible in CI | Reject |

**Not a rejection:** lossy press yields `acceptance_rate < 1.0` or exactness gate
failure — valid Experiment 005 outcome.

---

## 15. Recommendation

### **Proceed with kvpress — only with restrictions**

kvpress remains the **best first backend candidate** after empirical validation:

- `with press(model):` hook lifecycle is **clean** (0 → 24 → 0 hooks on Qwen2.5-0.5B).
- `dynamic_v5` cache is **compatible** with ExactKV `cache/utils.py`.
- Physical cache shrinks honestly; workspace fields are meaningful.
- KnormPress runs on CPU with gate model.
- Scaffold (`verification_mode()`, optional extra) is in place.

**Restrictions before `KVPressKnormAdapter` implementation:**

1. Develop and test only in `.venv-kvpress` (`pip install -e ".[kvpress]"`).
2. Document Python 3.13 `fire>=0.7.1` workaround or use Python 3.10–3.12.
3. Implement `KVPressKnormAdapter` with `ModelRuntime` + replay prefill — **not**
   tensor-only `_backend_compress`.
4. Override `verification_mode()` to assert zero forward hooks.
5. Do not register adapter on default import path.
6. Run attention-patch gate in kvpress CI job.
7. Start with `KnormPress` only.

**Do not defer to KIVI yet** unless install/hook gates fail in CI.

**Do not stop V6** — scaffold + research support proceeding to adapter implementation.

---

## Appendix A: Empirical session log (2026-06-09)

```
Dedicated venv: .venv-kvpress (Python 3.13.3)
  kvpress==0.5.3, transformers==5.2.0, fire==0.7.1 (workaround)

Default env: transformers==5.8.1, kvpress not installed
  import exactkv.compressors → kvpress not in sys.modules

Hook counts (Qwen2.5-0.5B): before=0, during=24, after=0
Global attention patch after import kvpress: wrapped=True

KnormPress(0.5) on 5-token prefill:
  full_bytes=122880, compressed_bytes=49152, physical_seq=2, logical=5
  draft forward: OK
  greedy vs full: diverges (expected)

kvpress scaffold tests in kvpress venv: 8 passed
kvpress lazy-import test: passed (no longer skipped when kvpress installed)
```

---

*ExactKV does not implement kvpress. This document informs Phase C adapter work only.*
