# Experiment 055: Explicit CLI for Experimental Restored-Verifier Runtime (Phase 13B)

**Status:** Explicit `--experimental-restored-verifier` CLI flag — **not** default CLI behavior.

> This is an **explicit CLI opt-in** for a non-default experimental restored-verifier runtime path.  
> The restored-verifier path is **not** activated unless `--experimental-restored-verifier` is passed.  
> **Environment variables do not activate this mode.**  
> **Default ExactKV generation behavior is unchanged.**  
> **vLLM and LMCache are not integrated.**  
> **Remote prefix caching is not implemented.**  
> No speedup, latency improvement, throughput improvement, active memory savings, or production-serving claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput results.

Companion: [`EXPERIMENTAL_RESTORED_VERIFIER_RUNTIME.md`](EXPERIMENTAL_RESTORED_VERIFIER_RUNTIME.md) · `exactkv/runtime/experimental_cli.py`

---

## 1. Purpose

Phase 13A added `run_experimental_restored_verifier()`. Phase 13B exposes it through an **explicit CLI flag** so users can invoke the restored full-KV verifier path from the command line only when they opt in.

---

## 2. CLI opt-in behavior

| Flag | Behavior |
|---|---|
| `--experimental-restored-verifier` absent | Disabled; runner not called; model not loaded |
| `--experimental-restored-verifier` present | Builds enabled `ExperimentalRestoredVerifierConfig` and calls runtime wrapper |

Supporting args (only meaningful when flag present): `--model-id`, `--device`, `--dtype`, `--prompt-ids`, `--storage-backends`, `--compressors`, `--draft-lens`, `--max-new-tokens`, `--output`.

---

## 3. Disabled behavior when flag absent

```bash
python3 scripts/research/run_exp055_experimental_restored_verifier_cli.py
```

- `cli_flag_present: false`
- `enabled: false`
- `runner_called: false`
- No model load
- Report written with disabled status

---

## 4. Enabled behavior when flag present

```bash
python3 scripts/research/run_exp055_experimental_restored_verifier_cli.py --experimental-restored-verifier
```

- `cli_flag_present: true`
- `enabled: true`
- Calls `run_experimental_restored_verifier()` → `run_restored_verifier()`
- `verifier_source: reloaded_full_kv`

---

## 5. Setup

| Parameter | Default (when flag present) |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` |
| Device / dtype | `cpu` / `float32` |
| Prompts | 4 (`offline_001`–`offline_004`) |
| `max_new_tokens` | 12 |
| `draft_len` | 4 |
| Compressors | `int4_sim`, `k8_v4_sim`, `int8` |
| Storage backend | `in_memory_kv_storage` |

Report (gitignored): `reports/experiment_055_experimental_restored_verifier_cli.json`

---

## 6. Results

Fill after running. Key fields: `cli_flag_present`, `runner_called`, `exactkv_failures`, `draft_divergence_count`.

---

## 7. Exactness result

Pass criterion when flag enabled: `exactkv_failures == 0`. Failures preserved in report.

---

## 8. Acceptance / correction behavior

Same as Phase 12G–13A: reloaded full-KV verifier accepts prefix / corrects mismatch.

---

## 9. Blockers

`restore_blockers`, `draft_blockers`, `verification_blockers` from runner report when enabled.

---

## 10. What this proves

- Users can invoke restored-verifier path from CLI **only** with explicit flag
- Default CLI/script path does not load model or call runner
- Default `ExactKVGenerator` behavior unchanged

---

## 11. What this does not prove

| Claim | Status |
|---|---|
| Default CLI integration | **Not shown** |
| Env-var activation | **Explicitly forbidden** |
| Production serving | **Not shown** |
| vLLM / LMCache integration | **Not shown** |
| Throughput or speed benefit | **Not shown** |
| Full VeriCache reproduction | **Not shown** |

---

## 12. Relation to VeriCache parity

Phase 13B adds CLI opt-in toward stored-verifier experimentation — still isolated from `ExactKVGenerator`. Phase 11K keeps throughput/memory/serving/full-parity claims **forbidden**.

---

## 13. Next step

- Phase 14A: CUDA restored-verifier runtime gate — [`EXPERIMENT_056_CUDA_RESTORED_VERIFIER_RUNTIME_GATE.md`](EXPERIMENT_056_CUDA_RESTORED_VERIFIER_RUNTIME_GATE.md)
- Wire flag into additional research scripts only (never default main CLI)
- Phase 13C+: human-reviewed gate before any default-runtime discussion
