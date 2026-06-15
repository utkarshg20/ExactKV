# Experiment 060: Isolated vLLM Venv Feasibility (Phase 15B)

**Status:** **blocked** — vLLM pip install succeeded in isolated venv; functional import blocked by `libcudart.so.13` mismatch on cu128 pod.

> This is an **isolated vLLM environment feasibility test**, not vLLM integration.  
> **vLLM is not installed into system Python.**  
> **ExactKV default runtime is unchanged.**  
> No serving, batching, throughput, latency, speedup, memory-saving, or production claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput results.  
> Passing this phase only means a vLLM environment is available for **future integration work**.

Companion: [`EXPERIMENT_059_VLLM_FEASIBILITY_PROBE.md`](EXPERIMENT_059_VLLM_FEASIBILITY_PROBE.md) · `exactkv/integrations/vllm_probe.py`

---

## 1. Purpose

Phase 15A showed vLLM is absent from system Python. Phase 15B creates an **isolated venv** at `.venv-vllm`, installs vLLM there, and reruns the feasibility probe without touching `/usr/bin/python3`.

---

## 2. Why isolated venv is required

| Risk | Mitigation |
|---|---|
| Breaking system torch cu128 | vLLM deps installed only in venv |
| Polluting ExactKV system Python | `/usr/bin/python3` never receives `pip install vllm` |
| cu130 torch mismatch (seen in broken venv) | Setup pins torch cu128 in venv before vLLM |

---

## 3. Setup

On RunPod A5000:

```bash
ssh runpod-a5000
cd /workspace/ExactKV
bash scripts/setup/setup_vllm_venv_runpod.sh
/usr/bin/python3 scripts/research/run_exp060_vllm_venv_feasibility.py
```

Venv path: `/workspace/ExactKV/.venv-vllm`

---

## 4. System Python baseline

| Field | Expected |
|---|---|
| Python | `/usr/bin/python3` |
| torch | `2.8.0+cu128` |
| CUDA | true |
| GPU | NVIDIA RTX A5000 |

System Python is probed but **never modified**.

---

## 5. vLLM venv result

| Field | Result (RunPod A5000) |
|---|---|
| venv path | `/workspace/ExactKV/.venv-vllm` |
| venv created | yes |
| pip install vLLM | **succeeded** (vLLM 0.23.0 wheels) |
| system torch unchanged | `2.8.0+cu128` |
| venv torch | `2.11.0+cu128` (cu128 index) |
| vLLM in system Python | **no** |

---

## 6. vLLM import result

| Field | Result |
|---|---|
| vLLM package on disk | yes (`0.23.0`) |
| `LLM` / `SamplingParams` import | **failed** |
| import_error | `libcudart.so.13: cannot open shared object file` |
| vllm_importable (functional) | **false** |

vLLM 0.23.0 wheels require CUDA 13 runtime libraries; this pod exposes CUDA 12.8 (`cu128`).

---

## 7. CUDA result inside venv

| Field | Result |
|---|---|
| venv `torch.cuda.is_available()` | **true** |
| GPU | NVIDIA RTX A5000 |
| vLLM native extension load | **failed** (`libcudart.so.13` missing) |

Torch CUDA works in the venv; vLLM compiled extensions do not match the host CUDA runtime.

---

## 8. Tiny generation smoke result

| Field | Result |
|---|---|
| attempted | **false** (blocked before smoke) |
| passed | false |

---

## 9. Results

| Field | Result |
|---|---|
| experiment_id | `exp060_vllm_venv_feasibility` |
| status | **blocked** |
| system_python | `/usr/bin/python3` |
| venv_python | `/workspace/ExactKV/.venv-vllm/bin/python` |
| install_attempted | true |
| install_success | false (functional import) |

Report (gitignored): `reports/experiment_060_vllm_venv_feasibility.json`

---

## 10. Blockers

1. `LLM class import failed: libcudart.so.13: cannot open shared object file: No such file or directory`
2. `SamplingParams import failed: libcudart.so.13: cannot open shared object file: No such file or directory`

These are **environment blockers**, not ExactKV correctness failures. System Python and default ExactKV runtime were not modified.

---

## 11. What this proves

- vLLM can be isolated from ExactKV system Python
- Whether vLLM import + tiny generation work in that venv on A5000
- Minimum safe next step for integration **design** (not implementation)

Default ExactKV generation behavior is unchanged.

---

## 12. What this does not prove

| Claim | Status |
|---|---|
| ExactKV integrated with vLLM | **Not shown** |
| vLLM serving supported | **Not shown** |
| Throughput / latency / memory improved | **Not shown** |
| Full VeriCache reproduction | **Not shown** |

---

## 13. Relation to VeriCache parity

VeriCache uses vLLM for serving throughput. Phase 15B establishes whether a **separate vLLM environment** can exist on the pod before any ExactKV↔vLLM wiring (Phase 15C+).

---

## 14. Next step

- **Phase 15B-unblock (environment):** Pin vLLM to a **cu128-compatible** wheel set (or upgrade pod CUDA 13 runtime) and re-run Exp 060 — still no ExactKV integration
- **Phase 15C:** KV cache surface reconnaissance — only after functional vLLM import + smoke pass
