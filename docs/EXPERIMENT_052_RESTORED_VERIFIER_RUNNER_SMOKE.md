# Experiment 052: Restored-Verifier Runner Smoke (Phase 12G)

**Status:** Runner consolidation smoke — **not** default runtime.

> This is an **isolated restored-verifier runner**, not default runtime integration.  
> Restored full KV is used only in an isolated experiment path.  
> **vLLM and LMCache are not integrated.**  
> **Remote prefix caching is not implemented.**  
> No speedup, latency improvement, throughput improvement, active memory savings, or production-serving claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput results.  
> **Current ExactKV generation and verification behavior is unchanged.**

Companion: [`RESTORED_VERIFIER_RUNNER.md`](RESTORED_VERIFIER_RUNNER.md) · `exactkv/cache/restored_verifier_runner.py`

---

## 1. Purpose

Phases 12C–12E proved the offline restored-verifier path across controlled, lossy, and drift-stress panels. Phase 12G consolidates that logic into a **reusable isolated runner API** and validates it with a small smoke run — not a new benchmark claim.

---

## 2. Setup

| Parameter | Default |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` |
| Device / dtype | `cpu` / `float32` |
| Prompts | 4 (`offline_001`–`offline_004`) |
| `max_new_tokens` | 12 |
| `draft_len` | 4 |
| Compressors | `int4_sim`, `k8_v4_sim`, `int8` |
| Storage backend | `InMemoryKVStorageBackend` (File optional via `--include-file-backend`) |

Reproduce:

```bash
python3 scripts/research/run_exp052_restored_verifier_runner_smoke.py
```

Model smoke:

```bash
EXACTKV_RUN_MODEL_SMOKE=1 pytest tests/test_exp052_restored_verifier_runner_smoke.py -q
```

Report (gitignored): `reports/experiment_052_restored_verifier_runner_smoke.json`

---

## 3. Runner config

Uses `RestoredVerifierRunConfig` from `exactkv/cache/restored_verifier_runner.py`:

- `model_id`, `device`, `dtype`
- `prompt_ids`, `compressor_names`, `draft_len`, `max_new_tokens`
- `storage_backend_name`, `namespace_prefix`
- `verifier_source`: `reloaded_full_kv`

---

## 4. Storage backend

Default: `InMemoryKVStorageBackend`.

Optional: `FileKVStorageBackend` with `--include-file-backend` (namespace `exp052/`).

Same capture → store → reload path as Phases 12A–12E.

---

## 5. Lossy draft sources

Built-in compressors only:

- `int4_sim`
- `k8_v4_sim`
- `int8` (baseline)

Existing compressor logic — no registry changes.

---

## 6. Verifier source

**Type:** `reloaded_full_kv` — unchanged from Phase 12C–12F.

---

## 7. Results

Fill after running the script. Key aggregate fields:

| Field | Meaning |
|---|---|
| `total_cells` | prompt_count × compressor_count × backend_count |
| `token_exact_match_count` | Cells matching live full greedy |
| `draft_divergence_count` | Total lossy draft correction rounds |
| `mean_acceptance` | Aggregate acceptance across cells |
| `phase12f_gate` | Phase 12F exactness gate outcome |

Prior CPU evidence (Phase 12E): 192/192 exact, 264 draft divergence rounds.

---

## 8. Exactness result

Pass criterion: `exactkv_failures == 0` for all cells. Phase 12G is **blocked** if Phase 12F ran cells and reported exactness failures.

---

## 9. Acceptance / correction behavior

When lossy drafts diverge, the reloaded full-KV verifier accepts matching prefix tokens and commits a correction token at the first mismatch — same as Phase 12E.

---

## 10. Blockers

- `restore_blockers` — capture/storage/reload failures
- `draft_blockers` — compressor failures
- `verification_blockers` — sequential verification failures

Failures are **not hidden**.

---

## 11. What this proves

- A clean `run_restored_verifier()` API wraps Phase 12C–12E logic without duplicating verifier code
- Small smoke panel preserves exact full-greedy output when prior exactness gates pass
- Runner reports serialize to a stable Exp 052 schema

---

## 12. What this does not prove

| Claim | Status |
|---|---|
| Default runtime integration | **Not shown** |
| New benchmark / leaderboard claim | **Not shown** |
| vLLM / LMCache integration | **Not shown** |
| Throughput or speed benefit | **Not shown** |
| Active memory savings | **Not shown** |
| Production serving | **Not shown** |
| Full VeriCache reproduction | **Not shown** |

---

## 13. Relation to VeriCache parity

Phase 12G packages the dual-cache verify loop into an isolated runner — still separate from `ExactKVGenerator`. Phase 11K keeps throughput/memory/serving/full-parity claims **forbidden**.

---

## 14. Next step

- See [`EXPERIMENT_053_RESTORED_VERIFIER_RUNNER_PANEL.md`](EXPERIMENT_053_RESTORED_VERIFIER_RUNNER_PANEL.md) (Phase 12H) for runner-backed drift panel
- Use `run_restored_verifier()` as canonical path for future offline panels
