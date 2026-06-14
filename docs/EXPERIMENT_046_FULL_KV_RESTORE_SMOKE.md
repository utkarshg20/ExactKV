# Experiment 046: Full-KV Restore Smoke (Phase 12A)

**Status:** Real HF `past_key_values` storage round-trip smoke — **not** wired into default runtime.

> This is a **full-KV restore smoke**, not a serving runtime.  
> **vLLM and LMCache are not integrated.**  
> **Remote prefix caching is not implemented.**  
> No speedup, latency improvement, throughput improvement, active memory savings, or production-serving claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput results.  
> **Current ExactKV generation and verification behavior is unchanged.**

Companion: [`FULL_KV_STORAGE_MANAGER.md`](FULL_KV_STORAGE_MANAGER.md) · [`VERICACHE_PARITY_CLAIM_GATE.md`](VERICACHE_PARITY_CLAIM_GATE.md) · `exactkv/cache/hf_kv_restore.py`

---

## 1. Purpose

Prove that ExactKV can:

1. Capture real Hugging Face `past_key_values` after prompt prefill
2. Store the snapshot through `KVStorageBackend` (in-memory and file)
3. Reload and rebuild the cache
4. Continue greedy generation and match the pre-store continuation **token-for-token**

This is **storage/restore smoke only** — not vLLM, LMCache, remote prefix runtime, or serving.

---

## 2. Setup

| Parameter | Default |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` |
| Device | `cpu` |
| dtype | `float32` |
| Prompts | 4 tiny deterministic prompts |
| `max_new_tokens` | 8 |
| Backends | `InMemoryKVStorageBackend`, `FileKVStorageBackend` |

Reproduce:

```bash
python3 scripts/research/run_exp046_full_kv_restore_smoke.py
```

Report (gitignored): `reports/experiment_046_full_kv_restore_smoke.json`

Unit tests (no model download):

```bash
pytest tests/test_hf_kv_restore.py -q
```

Model integration (optional):

```bash
EXACTKV_RUN_MODEL_SMOKE=1 pytest tests/test_exp046_full_kv_restore_smoke.py -q
```

---

## 3. Cache format detected

Supports via `exactkv/cache/utils.py`:

| Format | Detection |
|---|---|
| Legacy tuple | `tuple` of `(K, V)` per layer |
| `DynamicCache` v4 | `.key_cache` / `.value_cache` |
| `DynamicCache` v5 | `.layers[i].keys/.values` |

Unsupported formats fail with an explicit `HfKvRestoreError` blocker.

---

## 4. Storage backends tested

| Backend | Residency metadata |
|---|---|
| `InMemoryKVStorageBackend` | `CPU` |
| `FileKVStorageBackend` | `DISK` (`torch.save` payload + JSON sidecar) |

Uses existing `build_verifier_storage_metadata` — **not** a production storage format.

---

## 5. Results

Fill after running the script. Template fields:

| Field | Meaning |
|---|---|
| `token_exact_match_count` | Prompt×backend cells where live == restored |
| `failures_count` | Mismatches or restore blockers |
| `cache_format_detected` | HF cache format on model prefill |
| `per_prompt[]` | Live vs restored token ids and decoded text |

---

## 6. Token exactness

Each cell compares:

- **Live continuation:** greedy decode from cloned prefill KV (pre-persistence snapshot)
- **Restored continuation:** greedy decode from backend reload

Match criteria: identical token id lists and decoded strings; `first_divergence_idx` if any.

---

## 7. Restore blockers

Recorded in `restore_blockers[]` when:

- Unsupported cache format
- Model load failure
- Storage validation failure
- Tensor device/shape mismatch after reload

Failures are **not hidden**.

---

## 8. What this proves

- Real HF full-KV tensors can round-trip through the Phase 11C storage manager
- Reloaded KV supports identical greedy continuation on a tiny panel
- Both in-memory and file backends work for this smoke payload

---

## 9. What this does not prove

| Claim | Status |
|---|---|
| vLLM / LMCache integration | **Not shown** |
| Remote prefix cache runtime | **Not shown** |
| Throughput or speed benefit | **Not shown** (Exp 030: ExactKV slower on diagnostic panel) |
| Active memory savings | **Not shown** (Exp 031: no VRAM savings at tested scale) |
| Production serving | **Not shown** |
| Full VeriCache reproduction | **Not shown** |
| Universal model/prompt coverage | **Not shown** — tiny panel only |

---

## 10. Relation to VeriCache parity

VeriCache separates full-KV verifier residency from compressed draft paths. Phase 12A is the **first real restore smoke** toward Stage 2 storage — still isolated from `ExactKVGenerator`. The Phase 11K claim gate keeps throughput/memory/serving/full-parity claims **forbidden**.

---

## 11. Next step

- See [`EXPERIMENT_047_FULL_KV_RESTORE_PANEL.md`](EXPERIMENT_047_FULL_KV_RESTORE_PANEL.md) (Phase 12B) for multi-prompt panel hardening
- Thread restore plan into offline verifier experiments (still **not** default runtime)
- Do **not** claim LMCache/vLLM/remote prefix until respective contract gates clear
