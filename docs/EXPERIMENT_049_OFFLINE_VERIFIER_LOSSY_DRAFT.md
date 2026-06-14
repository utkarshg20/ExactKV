# Experiment 049: Offline Verifier Restore with Lossy Draft (Phase 12D)

**Status:** Isolated draft/verify loop with lossy compressor drafts and reloaded full-KV verifier — **not** default runtime.

> This is an **offline verifier restore experiment**, not default runtime integration.  
> Restored full KV is used only in an isolated experiment path.  
> Lossy draft source uses **existing compressor logic**.  
> **vLLM and LMCache are not integrated.**  
> **Remote prefix caching is not implemented.**  
> No speedup, latency improvement, throughput improvement, active memory savings, or production-serving claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput results.  
> **Current ExactKV generation and verification behavior is unchanged.**

Companion: [`EXPERIMENT_048_OFFLINE_VERIFIER_RESTORE_SMOKE.md`](EXPERIMENT_048_OFFLINE_VERIFIER_RESTORE_SMOKE.md) · `exactkv/cache/offline_verifier.py`

---

## 1. Purpose

Replace Phase 12C’s controlled draft source with **real lossy compressor drafts** while keeping the verifier source as **reloaded full-KV** from storage.

Each cell answers: can lossy compressed/materialized draft tokens be verified and corrected against stored full KV while preserving exact full-greedy output?

---

## 2. Setup

| Parameter | Default |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` |
| Device / dtype | `cpu` / `float32` |
| Prompts | 6 deterministic prompts |
| `max_new_tokens` | 12 |
| `draft_len` | 4 |
| Lossy compressors | `int8`, `int4_sim`, `k8_v4_sim` (registry built-ins) |
| Backends | `InMemoryKVStorageBackend`, `FileKVStorageBackend` |

Reproduce:

```bash
python3 scripts/research/run_exp049_offline_verifier_lossy_draft.py
```

Report (gitignored): `reports/experiment_049_offline_verifier_lossy_draft.json`

Unit tests (no model download):

```bash
pytest tests/test_exp049_offline_verifier_lossy_draft.py tests/test_offline_verifier_restore.py -q
```

Model integration (optional):

```bash
EXACTKV_RUN_MODEL_SMOKE=1 pytest tests/test_exp049_offline_verifier_lossy_draft.py -q
```

---

## 3. Storage backend

Same Phase 12A–12C storage path: capture prefill KV → `KVStorageBackend` → reload into `FullKVState` for verification.

---

## 4. Lossy draft source

Uses existing built-in compressor lifecycle (no registry changes):

1. `compress(full_state)` after prefill / each commit
2. `materialize_for_draft(compressed)` + greedy forwards (isolated copy of `ExactKVGenerator._draft`)
3. `update_after_commit(compressed, full_state)` after each commit

Default compressors: **`int8`**, **`int4_sim`**, **`k8_v4_sim`**.

This is **offline verifier integration smoke**, not compressor ranking.

---

## 5. Verifier source

**Type:** `reloaded_full_kv`

Sequential verification via unchanged `VerificationEngine.verify_sequential` on the reloaded `FullKVState` — not the original live cache object.

---

## 6. Results

Fill after running the script. Template fields:

| Field | Meaning |
|---|---|
| `cells` | Per prompt×backend×compressor results |
| `exactkv_failures` | Cells where offline output ≠ live reference |
| `token_exact_match_count` | Cells with identical token lists |
| `mean_acceptance` | Aggregate mean acceptance across cells |
| `per_cell_mean_acceptance` | Per-cell acceptance rates |

---

## 7. Exactness result

Pass criterion: `offline_output_token_ids == live_reference_token_ids` for every cell with `exactkv_failures == 0`.

---

## 8. Acceptance / correction behavior

Lossy drafts may mismatch full-KV predictions. Verifier accepts matching prefix, commits correction token, recompresses draft state, and continues until `max_new_tokens` or EOS.

---

## 9. Divergence examples

Recorded in `first_divergences[]` when offline output diverges from live full greedy (should be empty on pass).

---

## 10. Restore blockers

Capture/storage/reload failures (unsupported cache format, validation error, etc.).

---

## 11. Draft blockers

Compressor `compress`, `materialize_for_draft`, or `update_after_commit` failures.

---

## 12. Verification blockers

Sequential verification or alignment assertion failures during the offline loop.

Failures are **not hidden**.

---

## 13. What this proves

- Real lossy compressor drafts can be verified against **reloaded** full-KV payloads
- Accept/correct semantics preserve exact full-greedy output on the smoke panel
- Works across `int8`, `int4_sim`, and `k8_v4_sim` with both storage backends

---

## 14. What this does not prove

| Claim | Status |
|---|---|
| Default runtime integration | **Not shown** |
| Compressor leaderboard ranking | **Not shown** |
| vLLM / LMCache integration | **Not shown** |
| Remote prefix cache runtime | **Not shown** |
| Throughput or speed benefit | **Not shown** |
| Active memory savings | **Not shown** |
| Production serving | **Not shown** |
| Full VeriCache reproduction | **Not shown** |

---

## 15. Relation to VeriCache parity

Phase 12D extends Phase 12C toward VeriCache’s dual-cache verify loop — lossy draft + stored full-KV verifier — still isolated from `ExactKVGenerator`. Phase 11K keeps throughput/memory/serving/full-parity claims **forbidden**.

---

## 16. Next step

- Optional CUDA dtype panel for offline lossy verifier (same exactness gate)
- Broader prompt panel or additional built-in compressors only with explicit blockers
- Do **not** wire into default runtime without separate approval
