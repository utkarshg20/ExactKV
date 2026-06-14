# Experiment 050: Offline Restored-Verifier Drift Stress (Phase 12E)

**Status:** Drift-prone panel with reloaded full-KV verifier and lossy compressor drafts — **not** default runtime.

> This is an **offline restored-verifier drift stress experiment**, not default runtime integration.  
> Restored full KV is used only in an isolated experiment path.  
> Lossy draft source uses **existing compressor logic**.  
> **vLLM and LMCache are not integrated.**  
> **Remote prefix caching is not implemented.**  
> No speedup, latency improvement, throughput improvement, active memory savings, or production-serving claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput results.  
> **Current ExactKV generation and verification behavior is unchanged.**

Companion: [`EXPERIMENT_049_OFFLINE_VERIFIER_LOSSY_DRAFT.md`](EXPERIMENT_049_OFFLINE_VERIFIER_LOSSY_DRAFT.md) · `exactkv/cache/offline_verifier.py`

---

## 1. Purpose

Phase 12D proved exactness with lossy drafts but often showed high acceptance without stressing real draft divergence. Phase 12E reruns the **same reloaded full-KV verifier path** on a drift-prone prompt/compressor panel with longer generation and larger `draft_len` values to observe **real lossy draft divergence** and verifier correction.

---

## 2. Setup

| Parameter | Default |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` |
| Device / dtype | `cpu` / `float32` |
| Prompts | 12 drift-targeted prompts |
| `max_new_tokens` | 32 |
| `draft_len` values | 4, 8 |
| Compressors | `int4_sim`, `k8_v4_sim`, `k8_v4_boundary4_v8_sim`, `int8` (baseline) |
| Backends | `InMemoryKVStorageBackend`, `FileKVStorageBackend` |

Reproduce:

```bash
python3 scripts/research/run_exp050_offline_restored_verifier_drift_stress.py
```

Report (gitignored): `reports/experiment_050_offline_restored_verifier_drift_stress.json`

---

## 3. Drift stress settings

Increased divergence pressure via:

- **`draft_len` 4 and 8** (more tokens per verify round)
- **`max_new_tokens` 32** (longer decode horizon)
- **Drift-prone compressors** (`int4_sim`, `k8_v4_sim`, `k8_v4_boundary4_v8_sim`)
- **Prompt panel** from Exp 034b/037-style categories: pharmacy semantic (`pharm_001`-style), LongBench-style multi-doc QA, tool JSON, retrieval-copy, code-like, long-context summary

If no divergence appears, the report sets `no_real_drift_observed: true` — **divergence is never fabricated**.

---

## 4. Storage backend

Same Phase 12A–12D capture → store → reload path for verifier `FullKVState`.

---

## 5. Lossy draft sources

Existing built-in compressor lifecycle only (`compress` → `materialize_for_draft` → greedy draft → `update_after_commit`). No registry changes.

---

## 6. Verifier source

**Type:** `reloaded_full_kv` — unchanged from Phase 12C–12D.

---

## 7. Results

Fill after running the script. Key aggregate fields:

| Field | Meaning |
|---|---|
| `draft_divergence_count` | Total verify rounds with lossy draft ≠ verifier prefix |
| `semantic_divergence_count` | Correction rounds on semantic-tagged prompt categories |
| `mean_acceptance` | Aggregate acceptance across cells |
| `no_real_drift_observed` | `true` when `draft_divergence_count == 0` |

---

## 8. Exactness result

Pass criterion: `exactkv_failures == 0` and final offline output matches live full greedy for every cell.

---

## 9. Acceptance / correction behavior

When lossy drafts diverge, the reloaded full-KV verifier accepts matching prefix tokens and commits a correction token at the first mismatch, then recompresses the draft state.

---

## 10. Real drift examples if any

Per-cell `round_traces[]` record rounds where `all_matched: false`, with `draft_tokens`, `verifier_tokens`, and `correction_token`. Pharmacy and LongBench-style prompts are included to increase the chance of human-meaningful corrections (e.g. semantic tool-call drift).

---

## 11. If no drift, say no real drift was observed

When `no_real_drift_observed: true`, the panel did not produce any verify round with lossy draft mismatch under tested settings. Exactness may still pass — report honestly without inflating drift metrics.

---

## 12. Restore blockers

Capture/storage/reload failures.

---

## 13. Draft blockers

Compressor compress/materialize/update failures.

---

## 14. Verification blockers

Sequential verification or alignment failures.

Failures are **not hidden**.

---

## 15. What this proves

- Restored full-KV verifier can **correct real lossy draft divergence** on a drift-stress panel when divergence occurs
- Exact full-greedy output is preserved despite corrections (`exactkv_failures == 0`)
- Drift metrics are reported honestly (`no_real_drift_observed` when applicable)

---

## 16. What this does not prove

| Claim | Status |
|---|---|
| Default runtime integration | **Not shown** |
| Compressor ranking / leaderboard | **Not shown** |
| vLLM / LMCache integration | **Not shown** |
| Throughput or speed benefit | **Not shown** |
| Active memory savings | **Not shown** |
| Production serving | **Not shown** |
| Full VeriCache reproduction | **Not shown** |

---

## 17. Relation to VeriCache parity

Phase 12E stress-tests the dual-cache verify loop with stored full-KV verifier residency — still isolated from `ExactKVGenerator`. Phase 11K keeps throughput/memory/serving/full-parity claims **forbidden**.

---

## 18. Next step

- See [`RESTORED_VERIFIER_RUNNER.md`](RESTORED_VERIFIER_RUNNER.md) and [`EXPERIMENT_052_RESTORED_VERIFIER_RUNNER_SMOKE.md`](EXPERIMENT_052_RESTORED_VERIFIER_RUNNER_SMOKE.md) (Phase 12G) for consolidated isolated runner
- See [`EXPERIMENT_051_OFFLINE_VERIFIER_CUDA_DRIFT_PANEL.md`](EXPERIMENT_051_OFFLINE_VERIFIER_CUDA_DRIFT_PANEL.md) (Phase 12F) for CUDA float16/bfloat16 drift panel
