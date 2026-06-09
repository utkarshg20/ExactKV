# KIVI Offline Adapter Prototype (V9 Phase D2)

**Status:** Phase D2 complete — restricted offline adapter + smoke exactness gate.
**ExactKV does not implement KIVI production CUDA/Triton inference.**

**Date:** 2026-06-09  
**Module:** `exactkv/compressors/kivi_adapter.py`  
**Factory:** `create_kivi_offline_adapter(runtime, k_bits=2, v_bits=2)`  
**Compressor name:** `kivi_offline_k2_v2` (not in default registry)

> Guardrails: no throughput, latency, tokens/sec, speedup, `runtime_seconds`,
> `active_gpu_kv_bytes`, or production-serving claims as ExactKV results.
> External KIVI paper/README claims are **upstream only**, not ExactKV measurements.

---

## 1. Purpose

V9 Phase D2 implements a **restricted offline KIVI adapter** behind
`BackendAdapter`, following the TurboQuant Python prototype pattern (Phase B) and
the Phase D1 feasibility recommendation in
[`KIVI_KVQUANT_INTEGRATION_RESEARCH.md`](KIVI_KVQUANT_INTEGRATION_RESEARCH.md).

Goal: wrap upstream KIVI `models/utils_quant.py` simulate quant/dequant on
ExactKV-extracted **post-RoPE** HF K/V tensors, materialise for draft generation,
and preserve `exactkv_output_ids == full_output_ids`.

---

## 2. Environment

| Requirement | Detail |
|---|---|
| **Isolation** | `PYTHONPATH=/path/to/kivi_research` or dedicated `.venv-kivi` |
| **Upstream** | Clone [jy-yuan/KIVI](https://github.com/jy-yuan/KIVI) — **no full pip install required** for simulate path |
| **ExactKV default** | `pip install -e ".[dev]"` unchanged; no KIVI in default deps |
| **Import rule** | `import exactkv.compressors` must **not** load `models.utils_quant` |
| **Registry** | Factory-only; `kivi_offline_k2_v2` **not** in `list_compressors()` |

Example:

```bash
git clone https://github.com/jy-yuan/KIVI.git /tmp/kivi_research
cd /path/to/ExactKV
PYTHONPATH=/tmp/kivi_research pytest tests/test_kivi_adapter.py -q
```

---

## 3. What was implemented

| Item | Detail |
|---|---|
| `KIVIOfflineAdapter` | `BackendAdapter` subclass in `exactkv/compressors/kivi_adapter.py` |
| `create_kivi_offline_adapter(...)` | Lazy-import factory (not registered) |
| K quant | Upstream `quantize_by_channel_and_pack_cache(..., simulate=True)` |
| V quant | CPU-safe port of KIVI `quantize_and_pack` simulate math (per-token on flattened `(H×S, D)`) |
| Materialize | Dequant → per-layer torch tensors → `rebuild_cache` |
| Draft path | `_get_next_token_id` override for lossy materialised cache |
| Tests | `tests/test_kivi_adapter.py` — import isolation + unit + smoke exactness |
| Default config | `k_bits=2`, `v_bits=2`, `group_size=32` → name `kivi_offline_k2_v2` |

---

## 4. What was not implemented

| Item | Status |
|---|---|
| KIVI production CUDA/Triton (`quant/new_pack.py`, `cuda_bmm_fA_qB_outer`, `kivi_gemv`) | **Not implemented** |
| `LlamaForCausalLM_KIVI` / `MistralForCausalLM_KIVI` | **Not implemented** |
| `flash-attn` integration | **Not implemented** |
| Packed-bit storage / `supports_real_bytes_claim=True` | **Not implemented** |
| Residual fp16 window (`residual_length`) | **Deferred** — name is `kivi_offline_k2_v2` without `r32` suffix |
| KVQuant | **Not implemented** |
| Experiment 009 | **Not run** |
| Default registry entry | **Not added** |

---

## 5. Adapter path

```
FullKVState.past_key_values
    → extract_kv_tensors (clone in compress)
    → per layer:
        K: quantize_by_channel_and_pack_cache (simulate)
        V: _kivi_simulate_quantize_per_token (CPU-safe KIVI math)
    → store quant codes + scales in backend_data
    → materialize: dequant → rebuild_cache
    → draft forward (_get_next_token_id)
VerificationEngine → always FullKVState (unchanged)
```

---

## 6. Cache shape and post-RoPE behavior

- Input tensors: HF layout `(batch=1, num_kv_heads, seq_len, head_dim)` from
  `extract_kv_tensors` — **post-RoPE** keys and values.
- Output materialised cache: same layout via `rebuild_cache`.
- `logical_seq_len` preserved; `FullKVState` never mutated.

---

## 7. K quant path

Uses upstream KIVI helpers directly:

- `quantize_by_channel_and_pack_cache(k_tensor, group_size, k_bits, simulate=True)`
- `dequantize_by_channel_and_unpack_cache(..., simulate=True)`

Per-channel quantization along flattened `(heads × head_dim)` features per sequence
position — matches KIVI offline research path in Phase D1.

---

## 8. V quant path and CPU/GPU restriction

Upstream `quantize_and_pack(..., simulate=True)` hardcodes `device='cuda'` for an
internal bit-width tensor. Phase D2 implements `_kivi_simulate_quantize_per_token`
and `_kivi_simulate_dequantize_per_token` in `kivi_adapter.py`:

- Same math as KIVI simulate branch (`process_input` grouping + scale/round/dequant).
- Runs on `data.device` (CPU-safe for ExactKV default sweeps).
- Documented in `CompressorCapabilities.notes` — **not a different quantizer algorithm**.

V tensors are reshaped to `(H×S, head_dim)` before per-token grouping along
`head_dim`.

---

## 9. Residual-window status

KIVI production keeps the last `residual_length` tokens in fp16 before quantizing
older tokens. **Phase D2 does not implement the residual window.** All tokens are
quantized uniformly. A future phase may add `residual_length` and rename to
`kivi_offline_k2_v2_r32`.

---

## 10. Stored-bytes accounting

| Field | Behaviour |
|---|---|
| `stored_kv_bytes` | Sum of `element_size × numel` for stored `k_quant`, `k_scale`, `k_mn`, `v_quant`, `v_scale`, `v_mn` tensors (CPU copies in `backend_data`) |
| `metadata_bytes` | `0` (no separate fixed metadata in Phase D2) |
| `materialized_working_kv_bytes` | `full_kv_bytes` — adapter materialises dense KV for attention |
| `temporary_workspace_bytes` | Conservative `full_kv_bytes // 4` |
| `total_kv_footprint_bytes` | Accounting sum only — **not** measured peak GPU memory |
| `supports_real_bytes_claim` | **`False`** — unpacked quant codes, not packed-bit KIVI CUDA format |

No upstream KIVI compression-ratio claims are used.

---

## 11. ExactKV smoke gate result

**Environment:** `PYTHONPATH=/tmp/kivi_research`, `.venv-turboquant`, `Qwen/Qwen2.5-0.5B`,
`float32`, `draft_len ∈ {2, 4}`, `max_new_tokens=8`, 2 prompts.

| Gate | Result |
|---|---|
| `exactkv_output_ids == full_output_ids` | ✅ All 4 cells |
| `exactkv_failures == 0` | ✅ |
| Acceptance/rejection/correction reconcile | ✅ |
| `full_seq_len_after == compressed_seq_len_after` | ✅ |
| pytest `tests/test_kivi_adapter.py` (KIVI env) | **17 passed** (unit + smoke) |
| pytest default env | **4 passed**, **13 skipped** (KIVI unavailable) |
| `tests/test_backend_adapter_poc.py` | **47 passed** (both envs) |

---

## 12. Limitations

1. **Not production KIVI** — no CUDA/Triton kernels, no fused quant attention.
2. **Not packed-bit storage** — simulate path stores float/int quant containers.
3. **No residual window** — may differ from upstream KIVI streaming behaviour.
4. **No Qwen-specific upstream model** — bridge is tensor-only on HF cache.
5. **V quant CPU port** — required because upstream simulate branch assumes CUDA.
6. **Acceptance not evaluated at scale** — smoke gate only; Experiment 009 is Phase D3.
7. **KVQuant deferred** — see Phase D1 research doc.

---

## 13. Phase D3 / Experiment 009 readiness

| Prerequisite | Status |
|---|---|
| Phase D2 adapter + smoke exactness | ✅ Complete |
| Factory isolation + capabilities honest | ✅ |
| `supports_real_bytes_claim=False` | ✅ |
| Experiment 009 script | **Not created** — Phase D3 |
| RunPod packed-bit validation | Optional stretch — not required for Exp 009 CPU path |
| KVQuant adapter | **Deferred** pending RunPod |

Phase D3 may add `scripts/run_experiment_009_kivi_offline.py` and
`docs/EXPERIMENT_009_KIVI_OFFLINE.md` using the same panel as Experiment 008.

---

## Capability metadata (reference)

```python
CompressorCapabilities(
    name="kivi_offline_k2_v2",
    backend_name="kivi",
    adapter_name="KIVIOfflineAdapter",
    adapter_version="0.1.0",
    is_simulated=False,
    supports_real_bytes_claim=False,
    supports_quantization=True,
    key_bit_width_label="kivi_k2_offline",
    value_bit_width_label="kivi_v2_offline",
    ...
)
```

---

## Related documents

| Document | Relevance |
|---|---|
| [`KIVI_KVQUANT_INTEGRATION_RESEARCH.md`](KIVI_KVQUANT_INTEGRATION_RESEARCH.md) | Phase D1 feasibility |
| [`TURBOQUANT_ADAPTER_PROTOTYPE.md`](TURBOQUANT_ADAPTER_PROTOTYPE.md) | Adapter pattern precedent |
| [`BACKEND_ADAPTER_INTERFACE.md`](BACKEND_ADAPTER_INTERFACE.md) | Contract |
| [`V9_SCOPE_STATEMENT.md`](V9_SCOPE_STATEMENT.md) | Phase D2 scope |

## Attribution

**KIVI:** Liu et al., ICML 2024, [arXiv:2402.02750](https://arxiv.org/abs/2402.02750) —
external claims only. ExactKV does not reproduce upstream KIVI accuracy or memory claims.
