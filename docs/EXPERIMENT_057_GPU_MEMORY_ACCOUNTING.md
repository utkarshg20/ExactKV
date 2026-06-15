# Experiment 057: GPU Memory Accounting Diagnostic (Phase 14B)

**Status:** CUDA memory accounting diagnostic for explicit experimental restored-verifier runtime — **not** a memory savings claim.

> This is a **GPU memory accounting diagnostic**, not a memory savings claim.  
> **Active GPU memory savings are not claimed.**  
> Speedup, latency improvement, throughput improvement, active memory savings, and production serving are **not** claimed.  
> Restored full KV is used **only through the explicit experimental path**.  
> **Current ExactKV default generation behavior is unchanged.**
> **vLLM and LMCache are not integrated.**  
> **Remote prefix caching is not implemented.**  
> ExactKV still does **not** reproduce VeriCache throughput results.

Companion: [`EXPERIMENT_056_CUDA_RESTORED_VERIFIER_RUNTIME_GATE.md`](EXPERIMENT_056_CUDA_RESTORED_VERIFIER_RUNTIME_GATE.md) · `exactkv/metrics/gpu_memory_accounting.py`

---

## 1. Purpose

Phase 14A validated CUDA exactness for `run_experimental_restored_verifier()`. Phase 14B records **diagnostic active CUDA memory observations** for:

- model loaded baseline
- full greedy reference
- KV capture / store / reload
- restored-verifier runtime (explicit experimental path)
- optional per-compressor restored-verifier peaks

This answers: *What active CUDA memory is observed under a tiny exactness-gated panel?* It does **not** claim memory savings. Default ExactKV generation behavior is unchanged.

---

## 2. CUDA setup

| Parameter | Default |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` |
| Device | `cuda` |
| dtypes | `float16`; `bfloat16` when supported |
| Prompts | 2 (`offline_001`, `offline_002`) |
| `max_new_tokens` | 12 |
| `draft_len` | 4 |
| Compressors | `int4_sim`, `k8_v4_sim`, `int8` |
| Storage backend | `in_memory_kv_storage` |
| Verifier source | `reloaded_full_kv` |
| Runtime path | `run_experimental_restored_verifier` |

Reproduce (requires passing Exp 056 gate):

```bash
python3 scripts/research/run_exp057_gpu_memory_accounting.py
```

With automatic Exp 056 gate if report missing:

```bash
python3 scripts/research/run_exp057_gpu_memory_accounting.py --run-exp056-if-missing
```

CUDA smoke:

```bash
EXACTKV_RUN_CUDA_SMOKE=1 pytest tests/test_exp057_gpu_memory_accounting.py -q
```

Report (gitignored): `reports/experiment_057_gpu_memory_accounting.json`

---

## 3. Exactness gate dependency

Exp 057 **refuses to proceed** unless Exp 056 report shows:

- `cuda_available: true`
- `status: pass`
- `exactkv_failures: 0`
- `token_exact_match_count == total_cells`
- no restore/draft/verification blockers

Exp 057 also runs a **tiny exactness-gated memory panel** and records `exactkv_failures` / `token_exact_match_count` in its own report.

---

## 4. Measurement methodology

Helpers in `exactkv/metrics/gpu_memory_accounting.py`:

- `snapshot_cuda_memory()` — point-in-time allocated/reserved bytes
- `reset_cuda_peak_memory()` — reset peak counters before each region
- `synchronize_cuda()` — `torch.cuda.synchronize()` before reads
- `measure_cuda_memory(label, fn)` — peak allocated/reserved during `fn()`

Rules:

- No wall-clock or throughput metrics
- No inferred memory savings
- Peak stats reset before each labeled region
- Raw deltas (e.g. restored-verifier peak minus full-greedy peak) are **diagnostic only**

---

## 5. What was measured

| Label | Description |
|---|---|
| `model_loaded` | After `ModelRuntime` init |
| `full_greedy` | Live full greedy on first prompt |
| `kv_capture_store_reload` | Prefill capture, in-memory store, reload |
| `restored_verifier_runtime` | Full tiny panel via `run_experimental_restored_verifier()` |
| `restored_verifier_{compressor}` | Single-compressor experimental run (optional) |

Stored/offloaded payload bytes (`full_kv_payload_bytes`, `stored_kv_payload_bytes`) are recorded separately from active GPU peak memory.

---

## 6. Results

**CUDA diagnostic collected on RunPod RTX A5000 (2026-06-15):**

| Field | Result |
|---|---|
| status | **pass** |
| dtypes tested | `float16`, `bfloat16` |
| prompt_count | 2 |
| exactness_gate_passed | true |
| exactkv_failures | **0** |
| token_exact_match | **12/12** (6 per dtype panel) |
| full_kv_payload_bytes | 147,552 |
| stored_kv_payload_bytes | 147,552 |

**float16 peak allocated (bytes):**

| Label | peak_allocated | peak_reserved |
|---|---|---|
| model_loaded | 1,005,038,592 | 1,061,158,912 |
| full_greedy | 1,008,854,528 | 1,063,256,064 |
| kv_capture_store_reload | 1,008,854,528 | 1,061,158,912 |
| restored_verifier_runtime | 2,005,916,672 | 2,103,443,456 |

These are **diagnostic measurements only** — not memory savings claims.

Report (gitignored): `reports/experiment_057_gpu_memory_accounting.json`

---

## 7. Active GPU memory observations

Report includes per-label `peak_allocated_bytes` and `peak_reserved_bytes`. Raw differences such as:

`restored_verifier_peak_allocated_bytes - full_greedy_peak_allocated_bytes`

are **diagnostic measurements only**. Current materializing paths may show **no active GPU memory reduction**.

---

## 8. Stored/offloaded KV accounting

- `full_kv_payload_bytes` — tensor payload size of captured full KV
- `stored_kv_payload_bytes` — backend metadata `total_payload_bytes` after store

These describe **stored payload accounting**, not active GPU peak memory savings.

---

## 9. Exactness result

Pass criterion for Exp 057 panel: `exactkv_failures == 0` and `token_exact_match_count == total_cells`.

Memory diagnostics do not relax exactness.

---

## 10. Blockers / skips

| Condition | Behavior |
|---|---|
| Exp 056 gate failed | `status: blocked`, no memory panel |
| CUDA unavailable | `status: blocked` |
| dtype unsupported | skipped with reason in `blockers` |
| exactness failure in panel | `status: failed` |

---

## 11. What this proves

- Active CUDA memory can be observed at restored-verifier lifecycle points
- Measurements are tied to an exactness-gated experimental runtime panel
- Stored KV payload bytes can be accounted separately from GPU peaks
- Default `ExactKVGenerator` / `VerificationEngine` unchanged

---

## 12. What this does not prove

| Claim | Status |
|---|---|
| Active GPU memory savings | **Not claimed** |
| Speed / latency / throughput benefit | **Not shown** |
| Production serving | **Not shown** |
| vLLM / LMCache integration | **Not shown** |
| Remote prefix cache runtime | **Not shown** |
| Full VeriCache reproduction | **Not shown** |

---

## 13. Relation to VeriCache parity

Phase 14B adds **diagnostic memory accounting** toward understanding restored-verifier resource use — still isolated from default runtime. Phase 11K keeps throughput/memory/serving/full-parity claims **forbidden**.

---

## 14. Next step

- Phase 14C: expanded memory panel (more prompts/backends) — still diagnostic, no savings claims
- Human-reviewed gate before any default-runtime discussion remains required
