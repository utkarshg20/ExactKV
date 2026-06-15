# Experiment 061: vLLM Version Compatibility Sweep (Phase 15B-unblock)

**Status:** **blocked** — all 5 pip candidates installed but failed functional import (`libcudart.so.13` on cu128 pod). No winning candidate.

> This is a **vLLM environment compatibility sweep**, not vLLM integration.  
> **vLLM is not installed into system Python.**  
> **ExactKV default runtime is unchanged.**  
> No serving, batching, throughput, latency, speedup, memory-saving, or production claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput results.  
> Passing this phase only identifies a candidate vLLM environment for **future integration work**.

Companion: [`EXPERIMENT_060_VLLM_VENV_FEASIBILITY.md`](EXPERIMENT_060_VLLM_VENV_FEASIBILITY.md) · `exactkv/integrations/vllm_probe.py`

---

## 1. Purpose

Phase 15B installed latest vLLM (0.23.0) in an isolated venv but functional import failed (`libcudart.so.13`). Phase 15B-unblock tests up to **five** older vLLM versions in separate venvs to find one compatible with the pod's CUDA 12.8 runtime.

---

## 2. Why version sweep is needed

| Issue | Approach |
|---|---|
| Latest vLLM wheel targets CUDA 13 | Try descending pip versions in isolated venvs |
| Must not touch system Python | One venv per candidate under `.venv-vllm-sweep/` |
| Evidence over assumptions | Sweep records install/import/CUDA/smoke per version |

---

## 3. System Python baseline

| Field | Expected |
|---|---|
| Python | `/usr/bin/python3` |
| torch | `2.8.0+cu128` |
| CUDA | true |
| GPU | NVIDIA RTX A5000 |

System Python is never modified.

---

## 4. Candidate selection

| Field | Result |
|---|---|
| Excluded | `0.23.0` (known bad from Exp 060) |
| Tested | `0.22.1`, `0.22.0`, `0.21.0`, `0.20.2`, `0.20.1` |
| Source | `pip index versions vllm` (descending, max 5) |

---

## 5. Candidate results

| Version | Install | Import | CUDA (torch) | Classification |
|---|---|---|---|---|
| 0.22.1 | pass | fail | pass | `import_failed` |
| 0.22.0 | pass | fail | pass | `import_failed` |
| 0.21.0 | pass | fail | pass | `import_failed` |
| 0.20.2 | pass | fail | pass | `import_failed` |
| 0.20.1 | pass | fail | pass | `import_failed` |

All failures: `libcudart.so.13: cannot open shared object file` on `LLM` / `SamplingParams` import.

Report (gitignored): `reports/experiment_061_vllm_version_sweep.json`

---

## 6. Winning candidate

**None.** `any_candidate_passed: false`, `winning_candidate: null`

---

## 7. Generation smoke result

| Field | Result |
|---|---|
| attempted | **false** (blocked before smoke on all candidates) |
| passed | false |

---

## 8. Blockers

All five candidates share the same environment blocker: vLLM wheels require **CUDA 13** runtime (`libcudart.so.13`); this pod exposes **CUDA 12.8** (`cu128`). Not an ExactKV correctness failure.

Failed venvs were removed after each candidate to stay within pod disk quota (~19 GB per vLLM venv).

---

## 9. What this proves

- Whether any vLLM wheel version works on this cu128 A5000 pod in isolation
- Which version (if any) is a candidate for Phase 15C API reconnaissance
- System Python and ExactKV default runtime remain unchanged

---

## 10. What this does not prove

| Claim | Status |
|---|---|
| ExactKV integrated with vLLM | **Not shown** |
| vLLM serving supported | **Not shown** |
| Throughput / latency / memory improved | **Not shown** |
| Full VeriCache reproduction | **Not shown** |

---

## 11. Relation to VeriCache parity

VeriCache depends on a working vLLM serving stack. This sweep finds an environment compatible with the pod before any ExactKV↔vLLM wiring.

---

## 12. Next step

- **Phase 15C-env complete:** Exp 062 on RunPod vLLM template — see [`EXPERIMENT_062_VLLM_CONTAINER_FEASIBILITY.md`](EXPERIMENT_062_VLLM_CONTAINER_FEASIBILITY.md)
- **Phase 15C:** API surface reconnaissance on CUDA-13 vLLM image (still no ExactKV integration)

---

## Setup

```bash
ssh runpod-a5000
cd /workspace/ExactKV
bash scripts/setup/sweep_vllm_versions_runpod.sh
/usr/bin/python3 scripts/research/run_exp061_vllm_version_sweep.py
```
