# KVQuant Adapter Prototype (V9 Phase D5)

**Status:** Phase D5 — restricted faithful `KVQuantSimAdapter` prototype (factory-only).  
**Prerequisite:** Phase D4b RunPod validation ([`KVQUANT_RUNPOD_VALIDATION.md`](KVQUANT_RUNPOD_VALIDATION.md)).

> This is a **restricted KVQuant simquant adapter**. It uses KVQuant’s **pre-RoPE**
> `k_proj`/`v_proj` path. This is **not** post-RoPE tensor approximation. This is
> **not** KVQuant deployment CUDA. This is **not** forked transformers deployment.
> This is **not** in the default registry. This requires an **external quantizers
> pickle** (not committed). `supports_real_bytes_claim=False`. ExactKV does **not**
> claim upstream KVQuant results. ExactKV does **not** claim speedup, throughput,
> latency, runtime, tokens/sec, active GPU memory, or production readiness.

---

## 1. Purpose

Implement a faithful ExactKV `BackendAdapter` for KVQuant **simquant** after:

- D4b GPU validation (quantizer artifact + `QuantLinearSim` forward + isolation)
- KIVI offline adapter (D2) and Experiment 009 (accept **0.012**)
- TurboQuant Python adapter (D1) and Experiment 008 (accept **0.435**)

Phase D5 delivers a **factory-only** prototype and smoke exactness gate — **not**
Experiment 010.

---

## 2. Environment

| Requirement | Value |
|---|---|
| KVQuant | `pip install -e KVQuant/quant` |
| transformers | **~=4.44** (5.x breaks Qwen2 calibration/replay) |
| torch + CUDA | Required (`QuantLinearSim` uses CUDA) |
| Quantizers | `EXACTKV_KVQUANT_QUANTIZERS=/path/to/quantizers_qwen05b.pickle` |
| Default ExactKV | **No** KVQuant import on `import exactkv.compressors` |

RunPod reference artifact: `/workspace/kvquant_d4/quantizers_qwen05b.pickle` (151745 B, 48 keys).

---

## 3. What was implemented

| Item | Location |
|---|---|
| `KVQuantSimAdapter` | [`exactkv/compressors/kvquant_adapter.py`](../exactkv/compressors/kvquant_adapter.py) |
| Factory | `create_kvquant_sim_adapter(runtime, quantizers_path=...)` |
| Tests | [`tests/test_kvquant_adapter.py`](../tests/test_kvquant_adapter.py) |
| Compressor name | `kvquant_sim_qwen05b` |

Adapter path:

1. Load `quantizers.pickle`
2. `deepcopy(runtime.model)` → draft model (CUDA fp16 when available)
3. Scoped Qwen **bias patch** + `make_quant_sim` on `k_proj`/`v_proj` only
4. `_compresses_via_full_state()` → replay prefill through quantized draft
5. Store `past_key_values` + `__compressed_next_token_id__` in `backend_data`
6. `materialize_for_draft` moves cache to runtime device for drafting
7. Verification uses unmodified `runtime.model` only

---

## 4. What was not implemented

- Default registry registration
- Experiment 010 sweep
- KVQuant `deployment/` CUDA kernels
- Forked transformers deployment path
- Fisher / NUQ calibration inside ExactKV
- Post-RoPE tensor bridge (KIVI/TurboQuant-style)
- Performance or production-serving claims

---

## 5. Adapter path

```text
FullKVState
  → compress() [_compresses_via_full_state]
      → draft_model(full_sequence_ids, use_cache=True)
      → past_key_values (post-RoPE, from quantized projectors)
  → materialize_for_draft()
      → cache on runtime.device for ExactKVGenerator._draft
  → VerificationEngine
      → runtime.model + authoritative FullKVState only
```

---

## 6. Quantizer artifact handling

- Path via factory arg or `EXACTKV_KVQUANT_QUANTIZERS`
- Pickle **not committed** (`.gitignore`: `quantizers*.pickle`)
- `stored_kv_bytes` = pickle file size (+ small metadata estimate)
- Per-model calibration required for other checkpoints

---

## 7. Draft model clone and verifier isolation

- `make_quant_sim` mutates modules **in place** → draft is always `deepcopy`
- `runtime.model` must have **zero** `QuantLinearSim` modules (asserted at init)
- Tests: `test_verifier_model_unmodified`, `test_draft_is_deepcopy`

---

## 8. Qwen bias patch

D4b: Qwen2.5 `k_proj`/`v_proj` have **bias**. Upstream passes `tmp.bias is not None`
(bool) into `QuantLinearSim`, which breaks on tensor bias.

**Adapter-local scoped patch** (`_scoped_kvquant_qwen_bias_fix`):

- `make_quant_sim` passes `tmp.bias` (tensor or None)
- `QuantLinearSim.__init__` wrapper routes tensor bias safely
- Patches **restored** after draft setup — no permanent KVQuant mutation

---

## 9. Pre-RoPE KVQuant behavior

Quantization applies to **linear projector outputs** before rotary embedding inside
attention. ExactKV stores **post-RoPE** `past_key_values` from the quantized draft
forward replay — faithful to KVQuant simquant semantics, not a tensor-only shortcut.

---

## 10. Stored-bytes accounting

| Field | Meaning |
|---|---|
| `stored_kv_bytes` | Quantizer pickle size (external artifact) |
| `metadata_bytes` | Conservative estimate for in-memory quantizer metadata |
| `materialized_working_kv_bytes` | Full-precision KV byte count when cache is materialized |
| `temporary_workspace_bytes` | Conservative `full_bytes // 4` |
| `total_kv_footprint_bytes` | Sum of above — **not** measured peak GPU memory |

`supports_real_bytes_claim=False` — no packed-bit KV storage claim.

---

## 11. ExactKV smoke gate result

**Gate:** Qwen/Qwen2.5-0.5B, 2 prompts × 2 draft lengths, `max_new_tokens=8`.

**RunPod (L40S, KVQuant venv, 2026-06-09):** **PASS**

| Check | Result |
|---|---|
| `exactkv_output_ids == full_output_ids` | ✅ 4/4 cells (2 prompts × 2 draft lengths) |
| `exactkv_failures` | ✅ 0 |
| Acceptance / rejection / correction reconcile | ✅ |
| Cache alignment (`full_seq_len_after == compressed_seq_len_after`) | ✅ |

```bash
export EXACTKV_KVQUANT_QUANTIZERS=/workspace/kvquant_d4/quantizers_qwen05b.pickle
pytest tests/test_kvquant_adapter.py tests/test_backend_adapter_poc.py -q
# 65 passed (RunPod KVQuant env); 47 passed, 18 skipped (default env without KVQuant)
```

---

## 12. Limitations

- CUDA required for simquant path
- transformers 4.44 pin in isolated venv
- wikitext2 calibration may fail (`HfUriError`); synthetic calibration used for D4b artifact
- Lossy draft — acceptance may be below int8/KIVI/TurboQuant (unknown until Exp 010)
- Single model/artifact pair validated (Qwen2.5-0.5B)

---

## 13. Phase D6 / Experiment 010 readiness

| Gate | Status |
|---|---|
| D4b GPU validation | ✅ |
| D5 adapter + smoke gate | ✅ RunPod pytest 65 passed |
| Experiment 010 approval | **Separate** — not started in D5 |
| Registry / default deps | **No** |

Experiment 010 remains a distinct milestone after D5 smoke passes.

---

## Related

- [`KVQUANT_RUNPOD_VALIDATION.md`](KVQUANT_RUNPOD_VALIDATION.md) — D4b GPU results
- [`KIVI_ADAPTER_PROTOTYPE.md`](KIVI_ADAPTER_PROTOTYPE.md) — offline tensor bridge precedent
- [`BACKEND_ADAPTER_INTERFACE.md`](BACKEND_ADAPTER_INTERFACE.md) — adapter contract
