# Experiment 054: Experimental Restored-Verifier Runtime (Phase 13A)

**Status:** Explicit opt-in experimental runtime smoke — **not** default ExactKVGenerator.

> This is a **non-default experimental restored-verifier runtime path**.  
> Restored full KV is used **only when explicitly enabled**.  
> **Default ExactKV generation behavior is unchanged.**  
> **vLLM and LMCache are not integrated.**  
> **Remote prefix caching is not implemented.**  
> No speedup, latency improvement, throughput improvement, active memory savings, or production-serving claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput results.

Companion: [`EXPERIMENTAL_RESTORED_VERIFIER_RUNTIME.md`](EXPERIMENTAL_RESTORED_VERIFIER_RUNTIME.md) · `exactkv/runtime/experimental.py`

---

## 1. Purpose

Phase 12H proved the restored-verifier runner can reproduce drift panels. Phase 13A adds a **non-default experimental runtime entry point** so users can explicitly opt into the restored full-KV verifier path without changing `ExactKVGenerator` or default CLI behavior.

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
| Storage backend | `in_memory_kv_storage` |
| Mode | `RESTORED_VERIFIER_OFFLINE` with `enabled=True` |

Reproduce:

```bash
python3 scripts/research/run_exp054_experimental_restored_verifier_runtime.py
```

Disabled smoke:

```bash
python3 scripts/research/run_exp054_experimental_restored_verifier_runtime.py --disabled
```

Report (gitignored): `reports/experiment_054_experimental_restored_verifier_runtime.json`

---

## 3. Experimental runtime config

Uses `ExperimentalRestoredVerifierConfig`:

- `enabled: bool` — must be `True` to invoke runner
- `mode: ExperimentalRuntimeMode.RESTORED_VERIFIER_OFFLINE`
- `verifier_source: reloaded_full_kv`
- `claim_note` — must include experimental/non-default caveats

---

## 4. Explicit opt-in behavior

- `enabled=False` → `status="disabled"`, `runner_called=False`
- `enabled=True` + invalid config → `status="invalid"`, runner not called
- `enabled=True` + valid config → calls `run_restored_verifier()` only
- **No** environment-variable activation

---

## 5. Storage backend

Default: `InMemoryKVStorageBackend` via `storage_backends=["in_memory_kv_storage"]`.

---

## 6. Lossy draft sources

Built-in compressors: `int4_sim`, `k8_v4_sim`, `int8`.

---

## 7. Verifier source

**Type:** `reloaded_full_kv` — required when enabled.

---

## 8. Results

Fill after running the script. Key fields:

| Field | Meaning |
|---|---|
| `enabled` | Whether experimental mode was requested |
| `runner_called` | Whether `run_restored_verifier()` was invoked |
| `runtime_mode` | `restored_verifier_offline` when enabled |
| `draft_divergence_count` | Lossy draft correction rounds |

---

## 9. Exactness result

Pass criterion: `exactkv_failures == 0` when runner executes. Failures are preserved in report — **not hidden**.

---

## 10. Acceptance / correction behavior

Same as Phase 12G–12H offline runner: reloaded full-KV verifier accepts prefix / corrects mismatch.

---

## 11. Blockers

`restore_blockers`, `draft_blockers`, `verification_blockers` aggregated from runner report.

---

## 12. What this proves

- Explicit opt-in experimental runtime API exists
- Disabled configs do not call the runner
- Default `ExactKVGenerator` behavior is unchanged
- Experimental path delegates to `run_restored_verifier()` only

---

## 13. What this does not prove

| Claim | Status |
|---|---|
| Default runtime integration | **Not shown** |
| CLI wiring / automatic activation | **Not shown** |
| vLLM / LMCache integration | **Not shown** |
| Throughput or speed benefit | **Not shown** |
| Production serving | **Not shown** |
| Full VeriCache reproduction | **Not shown** |

---

## 14. Relation to VeriCache parity

Phase 13A exposes an opt-in experimental path toward VeriCache’s stored-verifier concept — still isolated from `ExactKVGenerator`. Phase 11K keeps throughput/memory/serving/full-parity claims **forbidden**.

---

## 15. Next step

- See [`EXPERIMENT_055_EXPERIMENTAL_RESTORED_VERIFIER_CLI.md`](EXPERIMENT_055_EXPERIMENTAL_RESTORED_VERIFIER_CLI.md) (Phase 13B) for explicit CLI flag
- Optional CLI flag on additional research scripts only — still non-default
