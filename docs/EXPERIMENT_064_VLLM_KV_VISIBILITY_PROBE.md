# Experiment 064: vLLM KV/Cache Visibility Probe (Phase 15D)

**Status:** **blocked** (object-level) — import/preflight pass; tiny `LLM` init blocked while Qwen3-8B template server holds GPU.

> This is a **vLLM KV/cache visibility probe**, not ExactKV-vLLM integration. This is **not vLLM integration**.  
> **No vLLM runtime integration is implemented.**  
> Any visible **private** vLLM attributes are **not** treated as stable APIs.  
> **Raw KV export is not claimed** unless a clear public/stable API is found.  
> **ExactKV default runtime is unchanged.**  
> No serving, batching, throughput, latency, speedup, memory-saving, or production claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput results.

Companion: [`EXPERIMENT_063_VLLM_API_SURFACE_RECON.md`](EXPERIMENT_063_VLLM_API_SURFACE_RECON.md) · `exactkv/integrations/vllm_kv_visibility.py`

---

## 1. Purpose

Phase 15D runs **object-level** metadata-only KV/cache visibility inspection after optional tiny `LLM` init on an idle GPU. Builds on Exp 063 import-level recon. No ExactKV wiring.

---

## 2. Environment

| Field | Value |
|---|---|
| Pod | RunPod vLLM template `happy_blush_scallop` (disposable) |
| Python | `/usr/bin/python3` (3.12.13) |
| torch | `2.11.0+cu130` |
| CUDA | 13.0 |
| GPU | NVIDIA RTX A5000 |
| vLLM | `0.23.0` |

**Do not run on the cu128 ExactKV pod.**

---

## 3. Server stop / idle GPU setup

Preflight records `nvidia-smi`, detects vLLM/OpenAI server processes, and optionally stops them with `--stop-template-server`.

| Field | Default run (server running) | Full probe (`--stop-template-server --allow-llm-init`) |
|---|---|---|
| gpu_memory_before | `free=0.88GiB used=23.11GiB total=23.99GiB` | _requires idle GPU after stop_ |
| gpu_memory_after | same as before | _freed after stop_ |
| running_server_detected | **true** | **false** (expected after stop) |
| stopped_processes | `[]` | server PIDs/commands |

**Operational note:** On the RunPod vLLM template, stopping the pre-running `vllm serve Qwen/Qwen3-8B` process can terminate the container SSH session (docker-init supervises the server). Reconnect and run `--allow-llm-init` on a briefly idle GPU, or stop the server manually before probing.

---

## 4. Tiny vLLM object initialization

| Setting | Value |
|---|---|
| model | `Qwen/Qwen2.5-0.5B` |
| max_model_len | `256` |
| dtype | `float16` |
| gpu_memory_utilization | `0.35` (when init attempted) |

Skipped when GPU busy or free VRAM below threshold (~4 GiB) unless `--allow-llm-init`.

---

## 5. Generation smoke

| Setting | Value |
|---|---|
| prompt | default smoke prompt |
| max_tokens | `8` |
| sampling | greedy (`temperature=0.0`) |

---

## 6. Visible engine/cache/scheduler surfaces

Bounded recursive `dir()` inspection (depth 2, max 200 attrs/object) searches for names containing: `cache`, `kv`, `block`, `paged`, `scheduler`, `engine`, `executor`, `gpu`.

**This run:** object-level inspection not reached — `LLM` init blocked.

---

## 7. KV/cache visibility classification

**`blocked_by_running_server`** — Qwen3-8B server occupied GPU; tiny `LLM` init failed (`Engine core initialization failed` / insufficient free VRAM while server loading or loaded).

With `--allow-llm-init` on a briefly idle GPU (server stopped, ~24 GiB free at preflight), init still failed when the template server process respawned and reclaimed VRAM during `LLM` construction.

---

## 8. Raw KV export status

**`blocked_by_running_server`** — no `LLM` object; raw KV export not probed.

---

## 9. Possible adapter path

`adapter blocked pending stable hook — running server occupied GPU; stop server and re-probe on idle GPU`

Hypothesis only — **not** an integration claim.

---

## 10. Blockers

- `GPU busy — vLLM/OpenAI server detected; skipped LLM object init` (default run)
- `RuntimeError: Engine core initialization failed` (allow-llm-init while server present or respawning)
- `ValueError: Free memory on device cuda:0 ... less than desired GPU memory utilization` (vLLM 0.23 worker during contested GPU)

Report (gitignored): `reports/experiment_064_vllm_kv_visibility_probe.json`

---

## 11. What this proves

- Preflight GPU/server detection and conservative reporting work on the vLLM template pod
- Object-level KV/cache probe is **blocked** while the template server holds GPU — blockers are recorded, not hidden
- ExactKV default runtime unchanged

---

## 12. What this does not prove

| Claim | Status |
|---|---|
| ExactKV integrated with vLLM | **Not shown** |
| vLLM serving supported by ExactKV | **Not shown** |
| Object-level cache/engine attrs after `LLM` init | **Not shown** (blocked) |
| Throughput / memory improved | **Not shown** |
| Full VeriCache reproduction | **Not shown** |
| Private vLLM attrs are stable APIs | **Explicitly not claimed** |
| Raw KV export works | **Not claimed** |

---

## 13. Relation to VeriCache parity

Maps intended object-level vLLM cache/engine visibility against ExactKV compressed-draft + restored-verifier adapter needs — design input only. Object-level attrs remain unverified until idle-GPU probe succeeds.

---

## 14. Next step

- **Phase 15E:** idle-GPU object-level probe — [`EXPERIMENT_065_IDLE_VLLM_OBJECT_KV_PROBE.md`](EXPERIMENT_065_IDLE_VLLM_OBJECT_KV_PROBE.md) (blocked on auto-serving template; idle pod required for pass)
- **Phase 15F:** isolated KV export prototype spike with explicit private-API validation — still no default-runtime integration

---

## Setup

```bash
ssh -tt -i ~/.ssh/runpod_exactkv 0be11ieuly11lh-644110e0@ssh.runpod.io
cd /workspace/ExactKV

# server still running (expect blocked):
python3 scripts/research/run_exp064_vllm_kv_visibility_probe.py

# disposable pod — stop server + object init (may drop SSH; reconnect if needed):
python3 scripts/research/run_exp064_vllm_kv_visibility_probe.py --stop-template-server --allow-llm-init
```

Report (gitignored): `reports/experiment_064_vllm_kv_visibility_probe.json`
