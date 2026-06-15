# Experiment 059: vLLM Feasibility Probe (Phase 15A)

**Status:** Install-safe vLLM environment probe — **not** vLLM integration.

> This is a **vLLM feasibility probe**, not vLLM integration.  
> **No vLLM runtime integration is implemented.**  
> No serving, batching, throughput, latency, speedup, or memory-saving claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput results.  
> **Default ExactKV generation behavior is unchanged.**  
> vLLM import failure is a **dependency/environment blocker**, not an ExactKV correctness failure.

Companion: [`VLLM_PROTOTYPE_PATH.md`](VLLM_PROTOTYPE_PATH.md) · `exactkv/integrations/vllm_probe.py`

---

## 1. Purpose

Phase 14 showed CUDA exactness for the HF restored-verifier path but **no active GPU memory savings**. The next systems track toward VeriCache reproduction requires serving-runtime feasibility, starting with vLLM.

Phase 15A answers: *Is vLLM importable in the current GPU environment, and what is the minimum safe next step toward vLLM integration?*

This probe does **not** install vLLM or modify torch/CUDA packages.

---

## 2. Environment

Run on RunPod RTX A5000 with **system Python only**:

```bash
ssh runpod-a5000
cd /workspace/ExactKV
/usr/bin/python3 scripts/research/run_exp059_vllm_feasibility_probe.py
```

| Rule | Phase 15A |
|---|---|
| Python | `/usr/bin/python3` |
| Install vLLM | **Forbidden** |
| Modify torch/CUDA | **Forbidden** |
| Use broken venv | **Forbidden** |

Report (gitignored): `reports/experiment_059_vllm_feasibility_probe.json`

---

## 3. vLLM availability

The probe uses `importlib.util.find_spec("vllm")` first, then optional lazy import inside `probe_vllm_availability()`.

| Outcome | status |
|---|---|
| vLLM not installed | `blocked` |
| vLLM import raises | `blocked` |
| vLLM imports; smoke fails | `failed` |
| vLLM imports; smoke passes or skipped | `pass` or `blocked` |

---

## 4. Import result

Report fields: `vllm_importable`, `vllm_version`, `import_error`, `llm_class_importable`, `sampling_params_importable`.

---

## 5. Optional generation smoke

Only when vLLM is **already importable** and CUDA is available:

- Model: `Qwen/Qwen2.5-0.5B`
- One prompt, greedy (`temperature=0`), `max_tokens=8`
- Records `generation_smoke_attempted`, `generation_smoke_passed`, `generation_smoke_error`
- **No throughput comparison**

---

## 6. Visible integration surfaces

`visible_integration_surfaces` documents probe-time accessibility:

| Surface | Meaning |
|---|---|
| `model_loading_surface` | `LLM` class visible |
| `generation_call_surface` | `LLM.generate` visible |
| `sampling_greedy_config_surface` | `SamplingParams` visible |
| `kv_cache_access_surface` | Cache-related symbols detected |
| `scheduler_cache_api_surface` | Scheduler symbols detected |
| `restored_full_kv_verifier_path` | Whether restored full-KV verifier wiring appears possible |

Values: `accessible`, `unknown`, or `blocked`.

---

## 7. KV cache access status

Summary string in `kv_cache_access_status` — design reconnaissance only, not implementation.

---

## 8. Blockers

All import and smoke failures recorded in `blockers`. Not hidden.

---

## 9. Results

**RunPod RTX A5000 / system Python (2026-06-15):**

| Field | Result |
|---|---|
| status | **blocked** |
| python | `/usr/bin/python3` |
| torch | `2.8.0+cu128` |
| CUDA | true |
| GPU | NVIDIA RTX A5000 |
| vLLM importable | **false** |
| import_error | `ModuleNotFoundError: No module named 'vllm'` |
| generation smoke | not attempted |
| kv_cache_access_status | blocked |

**Minimum safe next step:** Phase 15B — install vLLM in an **isolated venv** (not system Python), re-run Exp 059. Do not modify system torch.

Report (gitignored): `reports/experiment_059_vllm_feasibility_probe.json`

---

## 10. What this proves

- Whether vLLM is importable without installing packages
- Which integration surfaces are visible at probe time
- Minimum safe next step (isolated vLLM venv install vs blocked)

---

## 11. What this does not prove

| Claim | Status |
|---|---|
| vLLM runtime integration exists | **Not shown** |
| Speed / latency / throughput | **Not shown** |
| Memory savings | **Not shown** |
| Production serving | **Not shown** |
| Full VeriCache reproduction | **Not shown** |
| Restored verifier wired to vLLM | **Not shown** |

---

## 12. Relation to VeriCache parity

VeriCache uses vLLM + LMCache for serving throughput panels. Phase 15A is the **first install-safe feasibility gate** before any isolated vLLM prototype environment (Phase 15B+).

---

## 13. Next step

- **Phase 15B:** Isolated vLLM install in a **separate venv** (not system Python) on RunPod, re-run Exp 059
- Do not wire ExactKV restored verifier until KV export surfaces are understood

Default ExactKV generation behavior is unchanged.
