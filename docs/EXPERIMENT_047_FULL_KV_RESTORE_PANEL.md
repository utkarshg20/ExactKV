# Experiment 047: Full-KV Restore Panel (Phase 12B)

**Status:** Multi-prompt HF `past_key_values` storage round-trip panel — **not** wired into default runtime.

> This is a **full-KV restore panel**, not a serving runtime.  
> **vLLM and LMCache are not integrated.**  
> **Remote prefix caching is not implemented.**  
> No speedup, latency improvement, throughput improvement, active memory savings, or production-serving claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput results.  
> **Current ExactKV generation and verification behavior is unchanged.**

Companion: [`EXPERIMENT_046_FULL_KV_RESTORE_SMOKE.md`](EXPERIMENT_046_FULL_KV_RESTORE_SMOKE.md) · [`FULL_KV_STORAGE_MANAGER.md`](FULL_KV_STORAGE_MANAGER.md) · `exactkv/cache/hf_kv_restore.py`

---

## 1. Purpose

Harden the Phase 12A full-KV restore path across:

- A larger deterministic prompt panel (8–16 prompts)
- Both `InMemoryKVStorageBackend` and `FileKVStorageBackend`
- Required CPU float32 plus optional CUDA float16/bfloat16 when available

Each cell captures real HF `past_key_values`, stores through a backend, reloads, and compares greedy continuation token-for-token against the live cache path.

This is **storage/restore panel hardening only** — not vLLM, LMCache, remote prefix runtime, or serving.

---

## 2. Setup

| Parameter | Default |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` |
| Required device/dtype | `cpu` / `float32` |
| Optional device/dtype | `cuda` / `float16`, `cuda` / `bfloat16` (when CUDA available) |
| Prompts | 12 deterministic panel prompts |
| `max_new_tokens` | 12 |
| Backends | `InMemoryKVStorageBackend`, `FileKVStorageBackend` |

Reproduce:

```bash
python3 scripts/research/run_exp047_full_kv_restore_panel.py
```

Report (gitignored): `reports/experiment_047_full_kv_restore_panel.json`

Unit tests (no model download):

```bash
pytest tests/test_hf_kv_restore.py tests/test_exp047_full_kv_restore_panel.py -q
```

Model integration (optional):

```bash
EXACTKV_RUN_MODEL_SMOKE=1 pytest tests/test_exp047_full_kv_restore_panel.py -q
```

---

## 3. Panel composition

| Category | Prompt IDs | Intent |
|---|---|---|
| Short natural continuation | `panel_001`, `panel_002` | Everyday completion |
| Structured JSON | `panel_003`, `panel_004` | JSON object/array continuation |
| Retrieval-copy | `panel_005`, `panel_006` | Quote or repeat from inline source |
| Code-like completion | `panel_007`, `panel_008` | Python snippet continuation |
| Long-context-ish summary | `panel_009`, `panel_010` | Short background + summary ask |
| Tool-call style | `panel_011`, `panel_012` | Function/tool JSON continuation |

Each prompt is tested with **both** storage backends for every **tested** device/dtype config.

---

## 4. Device/dtype configs

| Config | Required | When skipped |
|---|---|---|
| CPU float32 | **Yes** | Never — blocks experiment if model load fails |
| CUDA float16 | No | CUDA unavailable or model load failure |
| CUDA bfloat16 | No | CUDA unavailable, unsupported, or model load failure |

Skipped CUDA configs are recorded in `device_dtype_configs_tested[]` with `skip_reason`. GPU is **not** required to pass this experiment.

---

## 5. Storage backends

| Backend | Residency metadata |
|---|---|
| `InMemoryKVStorageBackend` | `CPU` |
| `FileKVStorageBackend` | `DISK` (`torch.save` payload + JSON sidecar) |

Uses existing `build_verifier_storage_metadata` — **not** a production storage format.

---

## 6. Results

Fill after running the script. Template fields:

| Field | Meaning |
|---|---|
| `total_cells` | Prompt × backend × tested device/dtype configs |
| `passed_cells` | Cells with `cell_status=passed` |
| `failed_cells` | Mismatch or restore blocker |
| `skipped_cells` | Reserved for per-cell skips (usually 0) |
| `aggregate_exactness.token_exact_match_count` | Cells with identical live vs restored tokens |
| `cache_formats_detected` | HF cache formats observed |
| `per_cell[]` | Full metadata per cell |

---

## 7. Token exactness

Each cell compares:

- **Live continuation:** greedy decode from cloned prefill KV (pre-persistence snapshot)
- **Restored continuation:** greedy decode from backend reload

Match criteria: identical token id lists and decoded strings; `first_divergence_idx` if any.

---

## 8. Skips/blockers

**Skips:** CUDA configs when CUDA is unavailable — recorded in `device_dtype_configs_tested`, not counted as failures.

**Blockers:** Recorded in `restore_blockers[]` when:

- Unsupported cache format
- Required CPU model load failure
- Storage validation failure
- Tensor device/shape mismatch after reload

Failures are **not hidden**.

---

## 9. What this proves

- Full-KV capture → store → reload → continuation equivalence holds on a **larger small panel**
- Both in-memory and file backends remain stable across prompt categories
- Optional CUDA dtype variants can be exercised when hardware permits

---

## 10. What this does not prove

| Claim | Status |
|---|---|
| vLLM / LMCache integration | **Not shown** |
| Remote prefix cache runtime | **Not shown** |
| Throughput or speed benefit | **Not shown** (Exp 030: ExactKV slower on diagnostic panel) |
| Active memory savings | **Not shown** (Exp 031: no VRAM savings at tested scale) |
| Production serving | **Not shown** |
| Full VeriCache reproduction | **Not shown** |
| Universal model/prompt coverage | **Not shown** — small panel only |

---

## 11. Relation to VeriCache parity

Phase 12B extends Phase 12A’s first real restore smoke toward Stage 2 full-KV storage — still isolated from `ExactKVGenerator`. The Phase 11K claim gate keeps throughput/memory/serving/full-parity claims **forbidden**.

---

## 12. Next step

- See [`EXPERIMENT_048_OFFLINE_VERIFIER_RESTORE_SMOKE.md`](EXPERIMENT_048_OFFLINE_VERIFIER_RESTORE_SMOKE.md) (Phase 12C) for offline verifier restore integration
- Optional CUDA dtype panel for offline verifier (same exactness gate)
- Do **not** claim LMCache/vLLM/remote prefix until respective contract gates clear
