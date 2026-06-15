# Experiment 058: Expanded GPU Memory Accounting Panel (Phase 14C)

**Status:** Expanded CUDA memory accounting panel for explicit experimental restored-verifier runtime — **not** a memory savings claim.

> This is an **expanded GPU memory accounting diagnostic**, not a memory savings claim.  
> **Active GPU memory savings are not claimed.**  
> Speedup, latency improvement, throughput improvement, active memory savings, and production serving are **not** claimed.  
> Restored full KV is used **only through the explicit experimental path**.  
> **Current ExactKV default generation behavior is unchanged.**  
> **vLLM and LMCache are not integrated.**  
> **Remote prefix caching is not implemented.**  
> ExactKV still does **not** reproduce VeriCache throughput results.

Companion: [`EXPERIMENT_057_GPU_MEMORY_ACCOUNTING.md`](EXPERIMENT_057_GPU_MEMORY_ACCOUNTING.md) · `exactkv/metrics/gpu_memory_accounting.py`

---

## 1. Purpose

Phase 14B recorded diagnostic CUDA memory on a **2-prompt** panel. Phase 14C expands to a broader **exactness-gated panel** to test whether Phase 14B memory observations are **stable** across:

- more prompts (4)
- all three smoke compressors
- draft lengths 4 and 8
- dtypes float16 and bfloat16
- storage backends `in_memory_kv_storage` and `file_kv_storage`

This does **not** claim memory savings. Default ExactKV generation behavior is unchanged.

---

## 2. CUDA setup

| Parameter | Default |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` |
| Device | `cuda` |
| Prompts | 4 (`offline_001`–`offline_004`) |
| `draft_len` | 4, 8 |
| `max_new_tokens` | 12 |
| Compressors | `int4_sim`, `k8_v4_sim`, `int8` |
| Storage backends | `in_memory_kv_storage`, `file_kv_storage` |
| dtypes | `float16`, `bfloat16` when supported |
| Runtime path | `run_experimental_restored_verifier` |

On RunPod A5000 use **system Python only**:

```bash
ssh runpod-a5000
cd /workspace/ExactKV
/usr/bin/python3 scripts/research/run_exp058_expanded_gpu_memory_panel.py
```

Report (gitignored): `reports/experiment_058_expanded_gpu_memory_panel.json`

---

## 3. Gate dependencies

Requires **Exp 056 CUDA exactness gate passed**. Optionally compares peaks against **Exp 057** baseline when that report exists.

**Stops on exactness failure** — does not continue the panel if any slice reports `exactkv_failures > 0`.

---

## 4. Measurement methodology

Per dtype:

1. `model_loaded` and `full_greedy` baselines (once per dtype)
2. For each storage backend: `kv_capture_store_reload`
3. For each `(storage_backend, draft_len)` slice: `restored_verifier_runtime` via explicit experimental path

Aggregate min/max/mean peaks recorded across slices. Compared against Exp 057 baseline peaks when available.

---

## 5. Panel dimensions

| Dimension | Exp 057 (14B) | Exp 058 (14C) |
|---|---|---|
| Prompts | 2 | **4** |
| Draft lengths | 4 | **4, 8** |
| Storage backends | in_memory only | **in_memory + file** |
| dtypes | float16, bfloat16 | float16, bfloat16 |
| Compressors | 3 | 3 |

Expected exactness cells (if all slices run): 2 dtypes × 2 backends × 2 draft_lens × 4 prompts × 3 compressors = **96 cells**.

---

## 6. Results

**CUDA expanded panel on RunPod RTX A5000 (2026-06-15):**

| Field | Result |
|---|---|
| status | **pass** |
| slices | 8 (2 dtypes × 2 backends × 2 draft_lens) |
| exactness | **96/96**, `exactkv_failures=0` |
| dtypes | `float16`, `bfloat16` |
| prompt_count | 4 |
| draft_lens | 4, 8 |
| storage_backends | `in_memory_kv_storage`, `file_kv_storage` |

**Aggregate peak allocated (bytes):**

| Label | min | max | mean |
|---|---|---|---|
| full_greedy | 1,008,657,920 | 1,008,854,528 | ~1,008,756,224 |
| restored_verifier_runtime | 2,005,513,216 | 2,005,916,672 | ~2,005,714,944 |
| kv_capture_store_reload | 1,008,657,920 | 1,008,854,528 | ~1,008,756,224 |

**Stability vs Exp 057:** full_greedy and restored_verifier peaks stay within the Phase 14B baseline range across the expanded panel. Restored-verifier peaks remain **above** full-greedy peaks — diagnostic only, **not** a memory savings claim.

Report (gitignored): `reports/experiment_058_expanded_gpu_memory_panel.json`

---

## 7. Active GPU memory observations

Report includes `aggregate_peak_stats` for `full_greedy`, `restored_verifier_runtime`, and `kv_capture_store_reload`.

Phase 14B observed (~float16):

- full_greedy peak ~1.01 GB allocated
- restored_verifier_runtime peak ~2.01 GB allocated

Phase 14C tests whether this pattern holds across the expanded panel. **Higher restored-verifier peaks are diagnostic observations, not failures to hide.**

---

## 8. Stored/offloaded KV accounting

Per-slice `full_kv_payload_bytes` and `stored_kv_payload_bytes` recorded separately from active GPU peaks.

---

## 9. Exactness result

Pass criterion: `exactkv_failures == 0`, `token_exact_match_count == total_cells`, `exactness_gate_passed: true`.

---

## 10. Blockers / skips

| Condition | Behavior |
|---|---|
| Exp 056 gate failed | `status: blocked` |
| CUDA unavailable | `status: blocked` |
| Exactness failure in any slice | **stop**, `status: failed` |
| Unsupported dtype | skipped with reason |

---

## 11. What this proves

- Phase 14B memory observations can be compared across a broader exactness-gated panel
- Memory peaks vary by storage backend and draft length (diagnostic)
- Default runtime unchanged

---

## 12. What this does not prove

| Claim | Status |
|---|---|
| Active GPU memory savings | **Not claimed** |
| Speed / latency / throughput | **Not shown** |
| Production serving | **Not shown** |
| vLLM / LMCache | **Not shown** |
| Full VeriCache reproduction | **Not shown** |

---

## 13. Relation to VeriCache parity

Phase 14C extends diagnostic memory accounting only. Throughput/memory/serving parity claims remain **forbidden** (Phase 11K).

---

## 14. Next step

- Phase 14D: optional drift-prone prompt subset under same diagnostic framework — still no savings claims
- Human-reviewed gate before any default-runtime discussion remains required
