# Experiment 065: Idle-GPU vLLM Object-Level KV/Cache Probe (Phase 15E)

**Status:** **deferred** on auto-serving RunPod vLLM template (`happy_blush_scallop`) — `blocked_by_running_server`; idle GPU required for object-level pass. Report: `reports/experiment_065_idle_vllm_object_kv_probe.json` (generate on idle pod).

> This is an **idle-GPU vLLM object-level probe**, not ExactKV-vLLM integration. This is **not vLLM integration**.  
> **No vLLM runtime integration is implemented.**  
> Any visible **private** vLLM attributes are **not** treated as stable APIs.  
> **Raw KV export is not claimed** unless a clear public/stable API is found.  
> **ExactKV default runtime is unchanged.**  
> No serving, batching, throughput, latency, speedup, memory-saving, or production claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput results.

Companion: [`EXPERIMENT_064_VLLM_KV_VISIBILITY_PROBE.md`](EXPERIMENT_064_VLLM_KV_VISIBILITY_PROBE.md) · `exactkv/integrations/vllm_kv_visibility.py`

---

## 1. Purpose

Phase 15E runs **object-level** metadata-only KV/cache inspection after tiny `LLM` init on an **idle GPU** (no auto-started vLLM server). Answers whether ExactKV can safely inspect vLLM cache/engine/scheduler metadata after initializing a small model.

---

## 2. Why an idle GPU is required

Exp 064 showed object-level inspection is **blocked** when the RunPod vLLM template auto-starts `vllm serve Qwen/Qwen3-8B` (~23 GiB VRAM). Stopping that supervised server often drops SSH and the server respawns.

Phase 15E requires:

- vLLM-compatible CUDA 13 image/container
- **No** model server auto-started
- Enough free VRAM for `Qwen/Qwen2.5-0.5B` via vLLM

If a running server is detected and cannot be disabled, the probe **stops and reports blocked** (default: does not kill the server).

---

## 3. Environment

| Field | Recommended |
|---|---|
| Image | vLLM CUDA-13 container without auto-serve entrypoint |
| Python | `/usr/bin/python3` |
| torch | `2.11.0+cu130` (expected) |
| CUDA | 13.0 |
| GPU | A5000 / A40 / L40S / 4090 |
| vLLM | `0.23.0` |

**Do not run on the auto-serving Qwen3-8B template unless the server is fully disabled and does not respawn.**

### Pod run (auto-serving template — expected blocked)

| Field | Value |
|---|---|
| Pod | RunPod vLLM template `happy_blush_scallop` (disposable) |
| Python | `/usr/bin/python3` (3.12.13) |
| torch | `2.11.0+cu130` |
| CUDA | 13.0 |
| GPU | NVIDIA RTX A5000 |
| vLLM | `0.23.0` |
| gpu_memory_before | `free=0.88GiB used=23.11GiB total=23.99GiB` |
| gpu_memory_after | same as before (no LLM init) |
| running_server_detected | **true** (`vllm serve Qwen/Qwen3-8B` on port 8000) |
| stopped_processes | `[]` (default: server not stopped) |
| llm_object_initialized | **false** |
| generation_smoke_attempted | **false** |
| generation_smoke_passed | **false** |
| kv_cache_visibility_status | **`blocked_by_running_server`** |
| raw_kv_export_status | **`blocked_by_running_server`** |
| experiment status | **`blocked`** |

**Operational note:** Use a vLLM CUDA-13 pod **without** auto-started serve for a pass run with object-level surfaces.

---

## 4. LLM object initialization

| Setting | Value |
|---|---|
| model | `Qwen/Qwen2.5-0.5B` |
| max_model_len | `256` |
| dtype | `float16` |
| gpu_memory_utilization | `0.35` |

Skipped when a running server is detected (unless `--stop-server` succeeds) or VRAM is insufficient.

---

## 5. Generation smoke

| Setting | Value |
|---|---|
| prompt | default smoke prompt |
| max_tokens | `8` |
| sampling | greedy (`temperature=0.0`) |

---

## 6. Object-level surfaces

Bounded recursive `dir()` inspection (depth 2, max 200 attrs/object) for names containing: `cache`, `kv`, `block`, `paged`, `scheduler`, `engine`, `executor`, `gpu`.

Reported categories:

- `visible_llm_attrs`
- `visible_engine_attrs`
- `visible_model_executor_attrs`
- `visible_scheduler_attrs`
- `visible_cache_attrs`
- `visible_block_attrs`
- `cache_config_summary` (metadata fields only)

---

## 7. Cache/engine/scheduler visibility

After successful `LLM` init, nested inspection walks `llm_engine` / `engine` and `model_executor` when visible.

---

## 8. KV/cache visibility status

Conservative `kv_cache_visibility_status` values:

| Status | Meaning |
|---|---|
| `blocked_by_running_server` | Server detected; idle GPU required |
| `blocked_by_oom` | Insufficient VRAM or OOM during init/smoke |
| `llm_init_failed` | Init attempted but failed (non-OOM) |
| `llm_init_success_no_cache_surface` | Init + smoke OK; no cache surfaces found |
| `cache_config_visible` | Cache config metadata readable |
| `private_cache_attrs_visible` | Cache-like private attrs visible |
| `engine_cache_metadata_visible` | Engine + cache metadata visible |
| `public_kv_export_visible` | Public KV hook name found (unvalidated) |
| `raw_kv_export_not_available` | No stable public export path |

---

## 9. Raw KV export status

Separate from visibility classification: `not_probed`, `public_api_candidate_found`, `raw_kv_export_not_available`, `blocked_by_oom`, `blocked_by_running_server`.

---

## 10. Possible adapter path

Hypothesis only — e.g. `potential adapter path — engine/cache metadata visible; private attrs require validation; raw KV export not available`.

**Not** an integration claim.

---

## 11. Blockers

Typical blockers (documented, not hidden):

- `Running vLLM/OpenAI/model server detected — idle GPU required`
- OOM during LLM init or generation smoke
- `llm_init_failed` (engine core init errors)
- Inspection exceptions (captured)

Report (gitignored): `reports/experiment_065_idle_vllm_object_kv_probe.json`

---

## 12. What this proves

- Whether object-level cache/engine/scheduler metadata is visible after tiny `LLM` init on idle GPU
- Conservative adapter-path notes for future prototype work
- ExactKV default runtime unchanged

---

## 13. What this does not prove

| Claim | Status |
|---|---|
| ExactKV integrated with vLLM | **Not shown** |
| vLLM serving supported by ExactKV | **Not shown** |
| Throughput / memory improved | **Not shown** |
| Full VeriCache reproduction | **Not shown** |
| Private vLLM attrs are stable APIs | **Explicitly not claimed** |
| Raw KV export works | **Not claimed unless public API found** |

---

## 14. Relation to VeriCache parity

Maps object-level vLLM cache/engine visibility against ExactKV compressed-draft + restored-verifier adapter needs — design input only. Auto-serving template run blocked; idle-GPU pass still required for surface enumeration.

---

## 15. Next step

- **Idle pod:** re-run `run_exp065_idle_vllm_object_kv_probe.py` on vLLM CUDA-13 image without auto-serve
- **Phase 15F:** isolated KV export prototype spike with explicit private-API validation — still no default-runtime integration

---

## Setup

```bash
# On idle vLLM CUDA-13 pod (no auto-started serve):
python3 scripts/research/run_exp065_idle_vllm_object_kv_probe.py

# Only if server can be stopped without respawn (discouraged on supervised template):
python3 scripts/research/run_exp065_idle_vllm_object_kv_probe.py --stop-server
```

Report (gitignored): `reports/experiment_065_idle_vllm_object_kv_probe.json`
