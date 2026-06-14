# Experiment 056: CUDA Restored-Verifier Runtime Gate (Phase 14A)

**Status:** CUDA float16/bfloat16 exactness gate for explicit experimental restored-verifier runtime — **not** default runtime.

> This is a **CUDA exactness gate for the explicit experimental restored-verifier runtime path**.  
> Restored full KV is used **only when explicitly enabled** via `run_experimental_restored_verifier()`.  
> **Default ExactKV generation behavior is unchanged.**  
> **vLLM and LMCache are not integrated.**  
> **Remote prefix caching is not implemented.**  
> No speedup, latency improvement, throughput improvement, active memory savings, or production-serving claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput results.  
> Passing this gate would be **CUDA exactness evidence**, not a performance result.

Companion: [`EXPERIMENTAL_RESTORED_VERIFIER_RUNTIME.md`](EXPERIMENTAL_RESTORED_VERIFIER_RUNTIME.md) · [`EXPERIMENT_055_EXPERIMENTAL_RESTORED_VERIFIER_CLI.md`](EXPERIMENT_055_EXPERIMENTAL_RESTORED_VERIFIER_CLI.md) · `exactkv/runtime/experimental.py`

---

## 1. Purpose

Phase 13B exposed the restored full-KV verifier through an explicit CLI opt-in flag (`--experimental-restored-verifier`). Phase 14A runs the **same explicit experimental runtime path** on CUDA float16 and bfloat16 (when supported) to answer:

> Does the opt-in restored full-KV verifier runtime path remain exact on CUDA float16 / bfloat16?

This is the **first GPU-required restored-verifier runtime gate**. It does **not** change `ExactKVGenerator`, `VerificationEngine`, or default CLI/runtime behavior.

---

## 2. CUDA setup

| Parameter | Default |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` |
| Device | `cuda` (blocked report if unavailable) |
| dtypes | `float16`; `bfloat16` when `torch.cuda.is_bf16_supported()` |
| Prompts | 4 (`offline_001`–`offline_004`) |
| `max_new_tokens` | 12 (8–16 range) |
| `draft_len` | 4 |
| Compressors | `int4_sim`, `k8_v4_sim`, `int8` |
| Storage backend | `in_memory_kv_storage` |

Reproduce:

```bash
python3 scripts/research/run_exp056_cuda_restored_verifier_runtime_gate.py
```

CUDA smoke test:

```bash
EXACTKV_RUN_CUDA_SMOKE=1 pytest tests/test_exp056_cuda_restored_verifier_runtime_gate.py -q
```

Report (gitignored): `reports/experiment_056_cuda_restored_verifier_runtime_gate.json`

`configure_cuda_determinism()` sets `cudnn.deterministic=True` and `cudnn.benchmark=False` before CUDA runs.

---

## 3. Experimental runtime path

Exp 056 calls **`run_experimental_restored_verifier()`** only — not a duplicated offline loop or direct `run_offline_drift_stress_cell` calls.

Flow per supported dtype:

1. Build enabled `ExperimentalRestoredVerifierConfig` via `default_cuda_gate_experimental_config(dtype)`
2. Call `run_experimental_restored_verifier(config, experiment_id=exp056_cuda_restored_verifier_runtime_gate)`
3. Runtime wrapper calls `run_restored_verifier()` with `verifier_source: reloaded_full_kv`

`runtime_path` in report: `run_experimental_restored_verifier`.

---

## 4. Explicit opt-in behavior

| Mechanism | Exp 056 behavior |
|---|---|
| `ExperimentalRestoredVerifierConfig.enabled` | `true` (gate builds enabled config internally) |
| CLI flag `--experimental-restored-verifier` | **Not required** for Exp 056 script — gate uses runtime API directly |
| Environment variables | **Do not** activate experimental mode |
| Default `ExactKVGenerator` | **Unchanged** |
| `cli_opt_in_required` in report | `true` — documents that production/default paths still require explicit opt-in |

The gate proves the **runtime API path** on CUDA; it does not make experimental mode default.

---

## 5. Dtype configs

| dtype | When tested |
|---|---|
| `float16` | CUDA available |
| `bfloat16` | CUDA available **and** `torch.cuda.is_bf16_supported()` |

Unsupported dtypes are listed in `skipped_configs` with explicit `skip_reason`. When CUDA is unavailable, **all** dtype configs are skipped and status is `blocked`.

---

## 6. Storage backend

`in_memory_kv_storage` only for this gate — same capture → store → reload path as Phase 12A–13B, namespace `exp056/{dtype}/`.

---

## 7. Lossy draft sources

Built-in compressors only:

- `int4_sim`
- `k8_v4_sim`
- `int8` (baseline)

Draft path uses existing compressor logic; no registry changes.

---

## 8. Verifier source

**Type:** `reloaded_full_kv`

For each cell: live full greedy reference on CUDA; experimental runtime verifies draft tokens against **reloaded** full KV; accepts prefix / corrects mismatch.

---

## 9. Results

Fill after running on CUDA hardware. When CUDA is unavailable locally, report status is `blocked` with `cuda_available: false`, `total_cells: 0`, `token_exact_match_count: 0`, `exactkv_failures: 0` (no cells ran — **not** a pass).

Key aggregate fields:

| Field | Meaning |
|---|---|
| `cuda_available` | Whether CUDA was present at run time |
| `dtype_configs` | dtypes actually tested |
| `dtype_supported` | Per-dtype hardware support map |
| `skipped_configs` | Dtype configs not tested |
| `total_cells` | Prompt × compressor cells per tested dtype |
| `exactkv_failures` | Mismatches vs live full greedy |
| `token_exact_match_count` | Cells with exact final output |
| `draft_divergence_count` | Draft/verifier divergence events (expected under lossy draft) |
| `mean_acceptance` | Mean accepted prefix length across cells |
| `cuda_blockers` | CUDA/dtype/exactness blockers |

---

## 10. Exactness result

Pass criterion when CUDA cells run: `exactkv_failures == 0` and `token_exact_match_count == total_cells`.

Blocked (no CUDA): `status: blocked` — **not** counted as CUDA exactness evidence.

Failures are preserved in report; exactness is **not** silently relaxed.

---

## 11. Acceptance / correction behavior

Same as Phase 12G–13B: reloaded full-KV verifier accepts matching draft prefix lengths and corrects on mismatch. `accepted_prefix_lengths` and `first_divergences` recorded per cell.

---

## 12. CUDA blockers / skips

| Condition | Report behavior |
|---|---|
| CUDA unavailable | `status: blocked`, `cuda_blockers: ["CUDA unavailable"]` |
| bfloat16 unsupported | dtype in `skipped_configs` with reason |
| dtype load/runtime exception | dtype skipped, reason in `cuda_blockers` |
| exactness failure | `status: failed`, dtype noted in `cuda_blockers` |

---

## 13. What this proves

- Explicit experimental restored-verifier **runtime API** can run on CUDA float16/bfloat16
- Restored full KV verifier remains exact vs live full greedy when explicitly enabled on GPU
- Unsupported dtype configs are skipped explicitly
- Default `ExactKVGenerator` / `VerificationEngine` / default CLI behavior unchanged

---

## 14. What this does not prove

| Claim | Status |
|---|---|
| Default runtime integration | **Not shown** |
| CLI flag required for gate script | Gate uses runtime API; default CLI still unchanged |
| Production serving | **Not shown** |
| vLLM / LMCache integration | **Not shown** |
| Remote prefix cache runtime | **Not shown** |
| Speed, latency, throughput, active memory savings | **Not shown** |
| Full VeriCache reproduction | **Not shown** |
| Batching or custom CUDA kernels | **Not shown** |

---

## 15. Relation to VeriCache parity

Phase 14A extends Phase 13A–13B experimental opt-in toward **CUDA exactness evidence** for the restored-verifier runtime path. Phase 11K keeps throughput/memory/serving/full-parity claims **forbidden**. This gate is exactness-only.

---

## 16. Next step

- **Phase 14B:** Optional `--experimental-restored-verifier --device cuda --dtype float16` on Exp 055 CLI helper; or expanded CUDA panel (more prompts/backends) — still non-default, no performance claims
- Human-reviewed gate before any default-runtime discussion remains required
