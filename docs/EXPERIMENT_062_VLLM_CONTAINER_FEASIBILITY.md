# Experiment 062: vLLM Container / CUDA-13 Feasibility (Phase 15C-env)

**Status:** **pass (import/CUDA)** — generation smoke **failed** when attempted (GPU memory held by Qwen3-8B vLLM server); **pass** with `--skip-generation-smoke` for import-only confirmation.

> This is a **vLLM environment feasibility probe**, not ExactKV↔vLLM integration.  
> Passing this phase does **not** mean vLLM integration exists. This is **not vLLM integration**.  
> **ExactKV default runtime is unchanged.**  
> No serving, batching, throughput, latency, speedup, memory-saving, or production claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput results.

Companion: [`EXPERIMENT_061_VLLM_VERSION_SWEEP.md`](EXPERIMENT_061_VLLM_VERSION_SWEEP.md) · `exactkv/integrations/vllm_probe.py`

---

## 1. Purpose

Exp 061 showed recent pip vLLM wheels fail on the cu128 A5000 pod (`libcudart.so.13` missing). Phase 15C-env probes a **vLLM-compatible CUDA 13 image/container** using its native Python — no ExactKV integration.

---

## 2. Why the previous cu128 pod was blocked

| Phase | Result |
|---|---|
| 15B | vLLM 0.23.0 pip install OK; `LLM` import failed — `libcudart.so.13` |
| 15B-unblock / Exp 061 | Versions 0.22.1–0.20.1 all failed same blocker |

The cu128 RunPod image cannot load recent vLLM CUDA extensions.

---

## 3. New environment

| Field | Value |
|---|---|
| Pod | `happy_blush_scallop` (`0be11ieuly11lh`) |
| Template | RunPod vLLM template (`pvcdqlwm9r`) |
| CWD | `/vllm-workspace` |
| Python | `/usr/bin/python3` (3.12.13) |
| torch | `2.11.0+cu130` |
| CUDA runtime | **13.0** |
| GPU | NVIDIA RTX A5000 |
| vLLM | **0.23.0** (preinstalled in image) |
| Serving | Qwen3-8B on port 8000 (left running) |

---

## 4. vLLM import result

| Field | Result |
|---|---|
| vLLM package | **importable** (`0.23.0`) |
| `LLM` | **importable** |
| `SamplingParams` | **importable** |
| Blocker | none (no `libcudart.so.13` error) |

---

## 5. Tiny generation smoke

| Run | Result |
|---|---|
| Default (`Qwen/Qwen2.5-0.5B`) | **failed** — GPU OOM: ~0.24 GiB free; Qwen3-8B server holds VRAM |
| `--skip-generation-smoke` | import probe **pass** (smoke not attempted) |

Generation failure is **resource contention**, not CUDA 13 / vLLM install failure. Re-run smoke with serving stopped or on idle GPU for full pass.

---

## 6. Visible integration surfaces

| Surface | Status |
|---|---|
| model_loading_surface | accessible |
| generation_call_surface | accessible |
| sampling_greedy_config_surface | accessible |
| kv_cache_access_surface | unknown |
| scheduler_cache_api_surface | unknown |
| restored_full_kv_verifier_path | blocked — KV cache APIs not visible at probe time |

---

## 7. KV/cache access status

`unknown — requires deeper prototype inspection` (Phase 15C scope)

---

## 8. Blockers

| Run | Blocker |
|---|---|
| Full smoke | `generation_smoke: RuntimeError` — insufficient free GPU memory while Qwen3-8B serving |
| Import-only | none |

---

## 9. What this proves

- CUDA 13 vLLM template resolves the `libcudart.so.13` blocker from cu128 pip wheels
- `LLM` and `SamplingParams` import successfully in the native container Python
- Minimum environment exists for Phase 15C API reconnaissance
- ExactKV default runtime on cu128 pod unchanged

---

## 10. What this does not prove

| Claim | Status |
|---|---|
| ExactKV integrated with vLLM | **Not shown** |
| vLLM serving supported | **Not shown** |
| Throughput / latency / memory improved | **Not shown** |
| Full VeriCache reproduction | **Not shown** |
| Generation smoke under concurrent serving | **Not shown** (OOM when server holds GPU) |

---

## 11. Relation to VeriCache parity

VeriCache depends on a working vLLM stack. This phase validates the container/image path after cu128 pip wheels failed.

---

## 12. Next step

- **Phase 15C:** vLLM API surface reconnaissance — [`EXPERIMENT_063_VLLM_API_SURFACE_RECON.md`](EXPERIMENT_063_VLLM_API_SURFACE_RECON.md)
- **Phase 15D:** isolated KV export prototype spike on idle GPU — still no default-runtime integration

---

## Setup

```bash
ssh -tt -i ~/.ssh/runpod_exactkv 0be11ieuly11lh-644110e0@ssh.runpod.io
cd /workspace/ExactKV
VLLM_ENVIRONMENT_LABEL="RunPod vLLM template happy_blush_scallop" \
  python3 scripts/research/run_exp062_vllm_container_feasibility.py
```

Report (gitignored): `reports/experiment_062_vllm_container_feasibility.json`
