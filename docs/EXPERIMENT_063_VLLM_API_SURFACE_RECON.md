# Experiment 063: vLLM API Surface and KV-Cache Reconnaissance (Phase 15C)

**Status:** **pass** (import-level recon) — object-level recon and generation smoke **skipped** (`blocked_by_running_server`, ~0.88 GiB free VRAM).

> This is **vLLM API surface reconnaissance**, not ExactKV-vLLM integration. This is **not vLLM integration**.  
> **No vLLM runtime integration is implemented.**  
> Any visible **private** vLLM attributes are **not** treated as stable APIs.  
> **ExactKV default runtime is unchanged.**  
> No serving, batching, throughput, latency, speedup, memory-saving, or production claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput results.

Companion: [`EXPERIMENT_062_VLLM_CONTAINER_FEASIBILITY.md`](EXPERIMENT_062_VLLM_CONTAINER_FEASIBILITY.md) · `exactkv/integrations/vllm_surface_recon.py`

---

## 1. Purpose

Phase 15C inspects vLLM API surfaces and KV-cache visibility safely after Exp 062 confirmed import works on the CUDA 13 vLLM template image. No ExactKV wiring.

---

## 2. Environment

| Field | Value |
|---|---|
| Pod | RunPod vLLM template `happy_blush_scallop` |
| Python | `/usr/bin/python3` |
| torch | `2.11.0+cu130` |
| CUDA | 13.0 |
| GPU | NVIDIA RTX A5000 |
| vLLM | `0.23.0` |

---

## 3. Why Exp 062 generation smoke was blocked

Qwen3-8B vLLM server occupied GPU VRAM (~0.24 GiB free). Generation smoke failed with OOM — not a CUDA 13 install failure.

---

## 4. Preflight GPU/server check

| Field | Result |
|---|---|
| gpu_memory_summary | `free=0.88GiB used=23.11GiB total=23.99GiB` |
| running_server_detected | **true** (`vllm serve Qwen/Qwen3-8B`) |
| stopped_processes | `[]` (default: did not stop server) |

---

## 5. Import-level surfaces

| Category | Visible |
|---|---|
| top-level modules | `vllm`, `vllm.config`, `vllm.engine`, `vllm.engine.llm_engine`, `vllm.model_executor`, `vllm.v1`, `vllm.entrypoints.*` |
| config | `CacheConfig`, `EngineArgs`, `ModelConfig`, `SchedulerConfig`, `VllmConfig` |
| engine | `LLMEngine` |
| scheduler | (none at import level in this run) |
| cache modules | `CacheConfig` (via config module) |

---

## 6. Optional object-level surfaces

| Field | Result |
|---|---|
| llm_object_initialized | **false** (skipped — GPU busy) |
| generation_smoke_attempted | **false** |
| object_level_attr_names | `[]` |

---

## 7. KV/cache visibility classification

**`blocked_by_running_server`** — import-level config symbols visible; object-level cache attrs not inspected while server holds GPU.

---

## 8. Possible adapter path

`potential adapter path requires idle GPU prototype validation; blocked by running server`

Compressed-draft + restored-verifier adapter wiring **not implemented**; prototype validation required on idle GPU.

---

## 9. Blockers

- `GPU busy — vLLM/OpenAI server detected; skipped LLM object init`

Report (gitignored): `reports/experiment_063_vllm_api_surface_recon.json`

---

## 10. What this proves

- Which vLLM modules/symbols are visible at import level
- Whether object-level cache/engine attrs are discoverable on idle GPU
- Conservative adapter-path notes for future prototype work
- ExactKV default runtime unchanged

---

## 11. What this does not prove

| Claim | Status |
|---|---|
| ExactKV integrated with vLLM | **Not shown** |
| vLLM serving supported by ExactKV | **Not shown** |
| Throughput / memory improved | **Not shown** |
| Full VeriCache reproduction | **Not shown** |
| Private vLLM attrs are stable APIs | **Explicitly not claimed** |

---

## 12. Relation to VeriCache parity

Maps visible vLLM cache/engine surfaces against ExactKV compressed-draft + restored-verifier adapter needs — design input only.

---

## 13. Next step

- **Phase 15D:** isolated KV export prototype spike on idle GPU — still no default-runtime integration

---

## Setup

```bash
ssh -tt -i ~/.ssh/runpod_exactkv 0be11ieuly11lh-644110e0@ssh.runpod.io
cd /workspace/ExactKV
python3 scripts/research/run_exp063_vllm_api_surface_recon.py
# disposable pod only:
python3 scripts/research/run_exp063_vllm_api_surface_recon.py --stop-template-server --allow-llm-init
```

Report (gitignored): `reports/experiment_063_vllm_api_surface_recon.json`
