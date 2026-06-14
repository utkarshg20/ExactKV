# Experiment 048: Offline Verifier Restore Smoke (Phase 12C)

**Status:** Isolated draft/verify loop with reloaded full-KV verifier — **not** default runtime.

> This is an **offline verifier restore smoke**, not default runtime integration.  
> Restored full KV is used only in an isolated experiment path.  
> **vLLM and LMCache are not integrated.**  
> **Remote prefix caching is not implemented.**  
> No speedup, latency improvement, throughput improvement, active memory savings, or production-serving claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput results.  
> **Current ExactKV generation and verification behavior is unchanged.**

Companion: [`EXPERIMENT_047_FULL_KV_RESTORE_PANEL.md`](EXPERIMENT_047_FULL_KV_RESTORE_PANEL.md) · `exactkv/cache/offline_verifier.py`

---

## 1. Purpose

Prove that stored/reloaded full-KV payloads can serve as the **authoritative verifier source** in a tiny ExactKV-style draft/verify loop while preserving exact full-KV greedy output.

Flow per cell:

1. Prefill and capture real HF `past_key_values`
2. Store through `KVStorageBackend` (in-memory or file)
3. Reload into a `FullKVState`
4. Propose controlled draft tokens (with optional injected mismatch)
5. Verify drafts using **reloaded** full KV via sequential verification
6. Commit accepted prefix + correction
7. Compare final output to live full-KV greedy reference

---

## 2. Setup

| Parameter | Default |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` |
| Device / dtype | `cpu` / `float32` |
| Prompts | 6 deterministic prompts |
| `max_new_tokens` | 12 |
| `draft_len` | 4 |
| Backends | `InMemoryKVStorageBackend`, `FileKVStorageBackend` |

Reproduce:

```bash
python3 scripts/research/run_exp048_offline_verifier_restore_smoke.py
```

Report (gitignored): `reports/experiment_048_offline_verifier_restore_smoke.json`

Unit tests (no model download):

```bash
pytest tests/test_offline_verifier_restore.py -q
```

Model integration (optional):

```bash
EXACTKV_RUN_MODEL_SMOKE=1 pytest tests/test_exp048_offline_verifier_restore_smoke.py -q
```

---

## 3. Storage backend

| Backend | Role |
|---|---|
| `InMemoryKVStorageBackend` | Round-trip prefill KV in process memory |
| `FileKVStorageBackend` | Round-trip via `torch.save` + JSON sidecar |

Uses Phase 11C / 12A storage helpers — **not** a production format.

---

## 4. Draft source

**Type:** `controlled_draft_with_injected_mismatch`

Each round proposes up to `draft_len` tokens taken from the live full-greedy reference continuation. On odd rounds, token index 1 is intentionally corrupted to exercise accept/correct behavior.

This is a **restore-verifier integration smoke**, not compressor evaluation.

---

## 5. Verifier source

**Type:** `reloaded_full_kv`

Sequential verification uses `VerificationEngine.verify_sequential` on a `FullKVState` built from the **reloaded** storage payload — not the original live cache object.

---

## 6. Results

Fill after running the script. Template fields:

| Field | Meaning |
|---|---|
| `cells` | Per prompt×backend results |
| `exactkv_failures` | Cells where offline output ≠ live reference |
| `token_exact_match_count` | Cells with identical token lists |
| `accepted_prefix_lengths` | Per-cell per-round accepted prefix lengths |

---

## 7. Exactness result

Pass criterion: `offline_output_token_ids == live_reference_token_ids` for every cell, with `exactkv_failures == 0`.

---

## 8. Acceptance / correction behavior

On injected mismatch rounds:

- Verifier accepts matching prefix tokens
- Commits correction token at first mismatch
- Resumes drafting from corrected reloaded state

Odd rounds typically show `accepted_prefix_length == 1` with a non-null `correction_token`.

---

## 9. Restore blockers

Recorded when capture, storage, or reload fails (unsupported cache format, validation error, etc.).

---

## 10. Verification blockers

Recorded when sequential verification or commit raises an exception during the offline loop.

Failures are **not hidden**.

---

## 11. What this proves

- Reloaded full-KV payloads can back sequential verification in an isolated draft/verify loop
- Accept/correct semantics preserve exact full-greedy output on the smoke panel
- Both in-memory and file backends work as verifier KV sources

---

## 12. What this does not prove

| Claim | Status |
|---|---|
| Default runtime integration | **Not shown** — isolated experiment path only |
| vLLM / LMCache integration | **Not shown** |
| Remote prefix cache runtime | **Not shown** |
| Throughput or speed benefit | **Not shown** |
| Active memory savings | **Not shown** |
| Production serving | **Not shown** |
| Full VeriCache reproduction | **Not shown** |
| Compressor ranking | **Not shown** — controlled draft only |

---

## 13. Relation to VeriCache parity

VeriCache separates stored full-KV verifier residency from compressed draft paths. Phase 12C is the **first offline verifier restore integration smoke** toward using stored full KV as verifier source — still isolated from `ExactKVGenerator`. Phase 11K keeps throughput/memory/serving/full-parity claims **forbidden**.

---

## 14. Next step

- Optional CUDA dtype panel for offline verifier (same exactness gate)
- Thread offline verifier into broader offline experiments (still **not** default runtime)
- Do **not** claim LMCache/vLLM/remote prefix until respective contract gates clear
