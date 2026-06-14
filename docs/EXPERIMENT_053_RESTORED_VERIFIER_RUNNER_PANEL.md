# Experiment 053: Runner-Backed Restored-Verifier Drift Panel (Phase 12H)

**Status:** Exp 050-style drift panel via `run_restored_verifier()` — **not** default runtime.

> This is a **runner-backed offline restored-verifier panel**, not default runtime integration.  
> Restored full KV is used only in an isolated experiment path.  
> **vLLM and LMCache are not integrated.**  
> **Remote prefix caching is not implemented.**  
> No speedup, latency improvement, throughput improvement, active memory savings, or production-serving claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput results.  
> **Current ExactKV generation and verification behavior is unchanged.**

Companion: [`RESTORED_VERIFIER_RUNNER.md`](RESTORED_VERIFIER_RUNNER.md) · `exactkv/cache/restored_verifier_runner.py`

---

## 1. Purpose

Phase 12G consolidated the offline restored-verifier path into `run_restored_verifier()`. Phase 12H reruns the **Exp 050-style drift panel** through that runner only — proving the consolidated API can reproduce meaningful drift-stress results without duplicating experiment-specific loop logic in the script.

---

## 2. Why runner-backed panel matters

Exp 048–051 each embedded their own panel loops. Phase 12H makes the runner the **canonical path**: one `RestoredVerifierRunConfig`, one `run_restored_verifier()` call, one `RestoredVerifierRunReport`. The script does not call `run_offline_drift_stress_cell` directly.

---

## 3. Setup

| Parameter | Default |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` |
| Device / dtype | `cpu` / `float32` |
| Prompts | 8 drift-targeted (`--full-panel` for 12) |
| `max_new_tokens` | 32 |
| `draft_len` values | 4, 8 |
| Compressors | `int4_sim`, `k8_v4_sim`, `k8_v4_boundary4_v8_sim`, `int8` |
| Backends | `InMemoryKVStorageBackend` (File via `--include-file-backend`) |

Reproduce:

```bash
python3 scripts/research/run_exp053_restored_verifier_runner_panel.py
```

Larger run:

```bash
python3 scripts/research/run_exp053_restored_verifier_runner_panel.py --include-file-backend --full-panel
```

CUDA (skips cleanly if unavailable):

```bash
python3 scripts/research/run_exp053_restored_verifier_runner_panel.py --cuda
```

Report (gitignored): `reports/experiment_053_restored_verifier_runner_panel.json`

---

## 4. Runner config

Uses `default_panel_config()` → `RestoredVerifierRunConfig` with:

- `namespace_prefix`: `exp053`
- `draft_len_values`: `[4, 8]`
- `verifier_source`: `reloaded_full_kv`

Full config is embedded in report `config` field.

---

## 5. Storage backend

Default: `InMemoryKVStorageBackend`.

Optional: `FileKVStorageBackend` with `--include-file-backend`.

---

## 6. Lossy draft sources

Built-in compressors only — existing registry, no changes:

- `int4_sim`
- `k8_v4_sim`
- `k8_v4_boundary4_v8_sim`
- `int8` (baseline)

---

## 7. Verifier source

**Type:** `reloaded_full_kv` — unchanged from Phase 12C–12G.

---

## 8. Results

Fill after running the script. Key aggregate fields:

| Field | Meaning |
|---|---|
| `total_cells` | prompts × compressors × draft_lens × backends |
| `draft_divergence_count` | Total lossy draft correction rounds |
| `semantic_divergence_count` | Corrections on semantic-tagged categories |
| `no_real_drift_observed` | `true` when `draft_divergence_count == 0` |
| `mean_acceptance` | Aggregate acceptance across cells |

Prior Phase 12E reference (direct loop, not runner): 192/192 exact, 264 draft divergence rounds.

---

## 9. Exactness result

Pass criterion: `exactkv_failures == 0` and final output matches live full greedy for every cell. Failures are **not hidden**.

---

## 10. Acceptance / correction behavior

When lossy drafts diverge, the reloaded full-KV verifier accepts matching prefix tokens and commits a correction token at the first mismatch — same behavior as Phase 12E, now via runner.

---

## 11. Real drift examples or no-drift statement

When `no_real_drift_observed: true`, **no real drift was observed** — the panel did not produce any verify round with lossy draft mismatch under tested settings. When drift occurs, per-cell `draft_divergence_count` and `semantic_divergence_count` record honest metrics — **not fabricated**.

---

## 12. Blockers

- `restore_blockers` — capture/storage/reload failures
- `draft_blockers` — compressor failures
- `verification_blockers` — sequential verification failures

---

## 13. What this proves

- `run_restored_verifier()` can reproduce a meaningful Exp 050-style drift panel
- No duplicated Exp 048–051 loop logic in the Exp 053 script
- Exact full-greedy output preserved when `exactkv_failures == 0`
- Drift metrics reported honestly (`no_real_drift_observed` when applicable)

---

## 14. What this does not prove

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

## 15. Relation to VeriCache parity

Phase 12H exercises the dual-cache verify loop through the consolidated runner — still isolated from `ExactKVGenerator`. Phase 11K keeps throughput/memory/serving/full-parity claims **forbidden**.

---

## 16. Next step

- Deprecate direct-loop scripts for new panels; use `run_restored_verifier()` as canonical path
- Optional CUDA panel via `--cuda` when hardware available
- Documentation-only public threading — still **not** default runtime wiring
