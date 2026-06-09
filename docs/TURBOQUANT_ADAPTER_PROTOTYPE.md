# TurboQuant Python Adapter Prototype (V9 Phase B)

**Status:** Phase B prototype — restricted Python path only. **Not** in the default
compressor registry. **Not** Experiment 008.

> ExactKV does not implement llama.cpp, MLX, GGUF, or production TurboQuant serving.
> No throughput, latency, tokens/sec, speedup, `runtime_seconds`, `active_gpu_kv_bytes`,
> or production-readiness claims.

**Prerequisite:** [`TURBOQUANT_INTEGRATION_RESEARCH.md`](TURBOQUANT_INTEGRATION_RESEARCH.md) (Phase A restricted go).

---

## 1. Purpose

Provide a thin `BackendAdapter` wrapper around the upstream dev-only Python
`KVCacheCompressor` so ExactKV can:

1. Compress authoritative HF KV tensors offline (NumPy path).
2. Materialise dequantised `past_key_values` for draft generation.
3. Keep full-precision verification unchanged (`exactkv_output_ids == full_output_ids`).

Phase B validates the adapter boundary and smoke exactness — not a serving integration.

---

## 2. Environment

| Item | Value |
|---|---|
| Virtualenv | `.venv-turboquant` (isolated) |
| ExactKV extra | `pip install -e ".[dev,turboquant]"` (adds `scipy` only) |
| Upstream repo | Clone `https://github.com/TheTom/turboquant_plus` |
| PYTHONPATH | Set to turboquant_plus repo root (``turboquant`` is not on PyPI) |
| Default ExactKV env | **No turboquant import**; default registry unchanged |

Example setup:

```bash
git clone https://github.com/TheTom/turboquant_plus.git vendor/turboquant_plus
python3 -m venv .venv-turboquant
source .venv-turboquant/bin/activate
pip install -e ".[dev,turboquant]"
export PYTHONPATH=$PWD/vendor/turboquant_plus
```

Factory (not registry):

```python
from exactkv.compressors.turboquant_adapter import create_turboquant_python_adapter
adapter = create_turboquant_python_adapter(runtime, head_dim=64, k_bits=3, v_bits=3)
```

---

## 3. What was implemented

| Component | Path |
|---|---|
| Adapter class | `exactkv/compressors/turboquant_adapter.py` |
| Factory | `create_turboquant_python_adapter(...)` |
| Optional extra | `[turboquant]` in `pyproject.toml` (`scipy>=1.10`) |
| Tests | `tests/test_turboquant_adapter.py` |
| Scratch inspector | `scripts/research/turboquant_phase_a_inspect.py` |

Adapter obligations:

- Lazy-import `turboquant` only in `__init__` / factory.
- `_backend_compress` → `KVCacheCompressor.compress`.
- `_backend_materialize` → `decompress` + `rebuild_cache`.
- `_get_next_token_id` → forward on materialised compressed KV (lossy draft path).
- V5 workspace fields from actual numpy payload + fixed quantizer metadata.

---

## 4. What was not implemented

- llama.cpp / GGUF `turbo2`/`turbo3`/`turbo4` packed formats
- MLX `TurboKVCache`
- REFRACT / vLLM / SGLang backends
- Default registry entry
- Experiment 008 script or reports
- Boundary V, sparse V, asymmetric layer policies
- KIVI, KVQuant, KVTC, Palu
- Any generation or verification logic changes

---

## 5. Adapter path

```
FullKVState.past_key_values
  → extract_kv_tensors (clone in BackendAdapter.compress)
  → torch → numpy stack (L, H, S, D)
  → KVCacheCompressor.compress → CompressedKVCache in backend_data
  → materialize: decompress → torch per layer → rebuild_cache
  → ExactKVGenerator draft path only
VerificationEngine → FullKVState only (unchanged)
```

---

## 6. Cache shape conversion

HF per-layer tensors: `(batch=1, num_kv_heads, seq_len, head_dim)`

TurboQuant stack: `(num_layers, num_kv_heads, seq_len, head_dim)`

- Batch dimension must be 1; otherwise `ValueError`.
- `head_dim` must match adapter config (64 for `Qwen/Qwen2.5-0.5B`).
- All layers must share the same `(H, S, D)` shape.

---

## 7. Stored-bytes accounting

| Field | Source |
|---|---|
| `stored_kv_bytes` | Recursive `nbytes` of numpy arrays in `CompressedKVCache` |
| `metadata_bytes` | Fixed rotation matrices + Lloyd-Max centroids (+ QJL projection on K) |
| `materialized_working_kv_bytes` | Full dequantised KV bytes (`__full_bytes__`) |
| `temporary_workspace_bytes` | Conservative `full_bytes // 4` scratch estimate |
| `total_kv_footprint_bytes` | Sum of the four components |

`supports_real_bytes_claim=False` because the Python prototype stores int64 indices
and float metadata — not packed-bit llama.cpp formats — and may exceed fp32 footprint
at small sequence lengths. Accounting still counts actual numpy `nbytes` honestly.
External llama.cpp compression ratios are **not** used.

---

## 8. ExactKV smoke gate result

Gate (when `turboquant` importable + model cached):

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B`, float32, CPU |
| Prompts | 2 |
| `draft_len` | 2, 4 |
| `max_new_tokens` | 8 |
| Cells | 4 |

**Required:** `exactkv_output_ids == full_output_ids`; acceptance/rejection/correction
counts reconcile; cache alignment each round.

Run:

```bash
export PYTHONPATH=$PWD/vendor/turboquant_plus
.venv-turboquant/bin/pytest tests/test_turboquant_adapter.py -v
```

---

## 9. Limitations

1. **Python prototype only** — not byte-compatible with llama.cpp packed turbo formats.
2. **K path includes QJL** via upstream `KVCacheCompressor` (production llama.cpp drops QJL).
3. **Dense Haar rotation** in Python vs WHT in production C kernels.
4. **Per-round full recompress** — no incremental turboquant update.
5. **CPU-first** — NumPy quantize/dequantize may be slow on large models (acceptable for evaluation).
6. **Factory-only** — must construct adapter explicitly; not `get_compressor(...)`.

---

## 10. Phase C / Experiment 008 readiness

Phase B provides:

- [x] Adapter module + factory
- [x] Isolated optional extra
- [x] Unit tests + smoke exactness gate
- [x] Honest capabilities + byte accounting

Phase C still requires separate approval:

- `scripts/run_experiment_008_turboquant.py`
- Full `core` suite (34 prompts)
- Gitignored `reports/experiment_008_turboquant.{json,csv}`
- `docs/EXPERIMENT_008_TURBOQUANT.md`

---

## Related documents

| Document | Relevance |
|---|---|
| [`TURBOQUANT_INTEGRATION_RESEARCH.md`](TURBOQUANT_INTEGRATION_RESEARCH.md) | Phase A feasibility |
| [`V9_SCOPE_STATEMENT.md`](V9_SCOPE_STATEMENT.md) | V9 phases |
| [`BACKEND_ADAPTER_INTERFACE.md`](BACKEND_ADAPTER_INTERFACE.md) | Adapter contract |
| [`EXPERIMENT_005_KVPRESS_KNORM.md`](EXPERIMENT_005_KVPRESS_KNORM.md) | Isolated env precedent |
