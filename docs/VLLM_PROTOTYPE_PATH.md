# vLLM Prototype Path (Phase 11F)

**Status:** vLLM prototype contract only — **vLLM is not imported or required by ExactKV.**

> This is a **vLLM prototype contract**, not a vLLM integration.  
> **vLLM is not imported or required by ExactKV.**  
> **ExactKV still does not implement production serving.**  
> No speedup, latency improvement, throughput improvement, active memory savings, or production-serving claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput results.

Companion: [`VERICACHE_SYSTEMS_ROADMAP.md`](VERICACHE_SYSTEMS_ROADMAP.md) · [`VERICACHE_PARITY_AUDIT.md`](VERICACHE_PARITY_AUDIT.md) · [`DUAL_CACHE_ABSTRACTION.md`](DUAL_CACHE_ABSTRACTION.md) · [`EXTENDED_VERIFICATION_SCHEDULER.md`](EXTENDED_VERIFICATION_SCHEDULER.md) · `exactkv/integrations/vllm_contract.py`

---

## 1. Why vLLM is needed for VeriCache parity

VeriCache's systems layer uses a **serving engine** (vLLM + LMCache) for practical inference: paged KV blocks, batch scheduling, and measured throughput panels. ExactKV today runs on a **Hugging Face correctness harness** with algorithmic draft/verify semantics only.

Stage 5 documents **what must be true** before ExactKV can honestly start a vLLM prototype — without claiming integration exists today.

---

## 2. What is implemented now

| Layer | Phase | Status |
|---|---|---|
| Dual-cache roles (`DRAFT` / `VERIFIER`) | 11B | Contract only |
| Full-KV storage manager | 11C | Tiny payload smoke |
| Materialized compressed-draft backend | 11D | Metadata + synthetic smoke |
| Verification scheduler (`FUTURE_VLLM` placeholder) | 11E | Policy metadata only |
| **vLLM prototype gates** | **11F** | **`VLLMPrototypePlan` + validators** |

**Not implemented:** vLLM import, paged-KV adapter, scheduler hooks, prototype runtime, serving tests.

---

## 3. What is not implemented

- vLLM package dependency or import
- PagedAttention block export for authoritative full-KV verify
- vLLM batch scheduler integration
- LMCache / remote prefix cache
- Multi-request serving or batching
- Throughput or active memory benefit claims

Exp 007/017 concluded **no-go** for direct vLLM integration until authoritative full-KV export is solved.

---

## 4. Cache contract mapping (future vLLM path)

| ExactKV contract (today) | Future vLLM target (design only) |
|---|---|
| `CacheRole.DRAFT` / `CompressedKVState` | Compressed or materialized draft blocks in paged layout |
| `CacheRole.VERIFIER` / `FullKVState` | Authoritative full-precision KV for per-step verify |
| `DualCacheState` | Dual residency: lossy draft pages + full verify pages |
| `KVStorageBackend` (11C) | Optional host/disk tier behind verifier role |
| `MaterializedDraftMetadata` (11D) | Draft-side byte accounting before hot-path integration |
| `VerificationPolicy` + `FUTURE_VLLM` (11E) | Scheduler policy when verify runs on vLLM worker |

This mapping is **documentation and gate metadata** — not a working adapter.

---

## 5. Why this phase does not integrate vLLM

Phase 11F answers: *What must be true before prototype code?* It does **not** write prototype code because:

1. Exp 017 reaffirmed **no-go** for unsafe paged-KV export.
2. Prior contract layers (11B–11E) must be stable before runtime wiring.
3. Claim firewall requires gates before any integration or performance language.

---

## 6. Integration gates

| Gate | Required | Default (11F) | Purpose |
|---|---|---|---|
| `optional_dependency_isolation` | yes | satisfied | vLLM stays optional extra, not core dep |
| `no_required_vllm_import` | yes | satisfied | Default install never imports vLLM |
| `cache_api_mapping_identified` | yes | satisfied | Dual-cache ↔ paged KV mapping documented |
| `draft_cache_role_mapping` | yes | satisfied | Draft role mapped to compressed path |
| `verifier_cache_role_mapping` | yes | satisfied (blocked) | Verifier role mapped; Exp 017 blocker noted |
| `scheduler_mapping` | yes | satisfied | 11E `FUTURE_VLLM` policy placeholder |
| `exactness_test_plan` | yes | satisfied | `exactkv_failures == 0` panel before perf claims |
| `rollback_fallback_path` | yes | **unsatisfied** | HF generator remains fallback until prototype exists |
| `no_speed_claim_before_benchmark` | yes | satisfied | Stage 8 harness gate |
| `no_memory_claim_before_measurement` | yes | satisfied | Active measurement gate (Exp 031) |
| `no_production_claim_before_serving_tests` | yes | satisfied | Serving test gate |

`PROTOTYPE_READY` status is **blocked** until all required gates pass validation.

---

## 7. Claim gates

| Gate type | Rule |
|---|---|
| **Exactness** | Bounded panel with `exactkv_failures == 0` before any performance comparison |
| **Memory** | Active memory measurement before any savings claim |
| **Serving** | Multi-request serving tests before production-serving claim |

---

## 8. JSON schema (plan)

```json
{
  "status": "CONTRACT_ONLY",
  "capabilities_required": ["PAGED_KV_CACHE", "CUSTOM_CACHE_MANAGER"],
  "gates": [
    {
      "gate_name": "rollback_fallback_path",
      "required": true,
      "satisfied": false,
      "blocker": "No prototype runtime"
    }
  ],
  "allowed_claims": ["vLLM prototype contract metadata exists"],
  "forbidden_claims": ["speedup", "vLLM integrated"],
  "claim_note": "...",
  "dependency_import_attempted": false
}
```

---

## 11. Phase 15A feasibility probe

Exp 059 (`exactkv/integrations/vllm_probe.py`) performs an **install-safe** vLLM import probe on the target GPU environment. It does **not** install vLLM into system Python or advance integration status beyond `CONTRACT_ONLY` unless a future phase explicitly gates prototype work.

See [`EXPERIMENT_059_VLLM_FEASIBILITY_PROBE.md`](EXPERIMENT_059_VLLM_FEASIBILITY_PROBE.md).

---

## 12. Phase 15B isolated vLLM venv

Exp 060 installs vLLM only in `.venv-vllm` (not system Python) and reruns the feasibility probe via subprocess. Passing means a vLLM environment is available for **future integration work** — not that ExactKV is integrated with vLLM.

Setup: `scripts/setup/setup_vllm_venv_runpod.sh` · Probe: `scripts/research/run_exp060_vllm_venv_feasibility.py`

See [`EXPERIMENT_060_VLLM_VENV_FEASIBILITY.md`](EXPERIMENT_060_VLLM_VENV_FEASIBILITY.md).

---

## 13. Phase 15B-unblock version sweep

Exp 061 tests up to five vLLM versions in isolated venvs under `.venv-vllm-sweep/` to find a cu128-compatible wheel. Passing identifies a candidate environment for Phase 15C — **not** ExactKV integration.

Setup: `scripts/setup/sweep_vllm_versions_runpod.sh` · Report: `scripts/research/run_exp061_vllm_version_sweep.py`

See [`EXPERIMENT_061_VLLM_VERSION_SWEEP.md`](EXPERIMENT_061_VLLM_VERSION_SWEEP.md).

---

## 15. Phase 15C-env container feasibility

Exp 062 probes the native Python of a RunPod vLLM template (CUDA 13). Passing means import + smoke work in that image — **not** ExactKV integration.

Report: `scripts/research/run_exp062_vllm_container_feasibility.py`

See [`EXPERIMENT_062_VLLM_CONTAINER_FEASIBILITY.md`](EXPERIMENT_062_VLLM_CONTAINER_FEASIBILITY.md).

---

## 16. Phase 15C API surface recon

Exp 063 (`exactkv/integrations/vllm_surface_recon.py`) inspects vLLM modules, config/engine/scheduler/cache symbols, and optional object-level attrs — **not** ExactKV integration.

Report: `scripts/research/run_exp063_vllm_api_surface_recon.py`

See [`EXPERIMENT_063_VLLM_API_SURFACE_RECON.md`](EXPERIMENT_063_VLLM_API_SURFACE_RECON.md).

---

## 17. Phase 15D KV/cache visibility probe

Exp 064 (`exactkv/integrations/vllm_kv_visibility.py`) runs metadata-only object inspection after optional tiny `LLM` init — **not** ExactKV integration.

Report: `scripts/research/run_exp064_vllm_kv_visibility_probe.py`

See [`EXPERIMENT_064_VLLM_KV_VISIBILITY_PROBE.md`](EXPERIMENT_064_VLLM_KV_VISIBILITY_PROBE.md).

---

## 18. Phase 15E idle-GPU object KV probe

Exp 065 (`run_exp065_idle_vllm_object_kv_probe.py`) requires an idle GPU (no auto-started serve) for object-level cache/engine metadata inspection — **not** ExactKV integration.

See [`EXPERIMENT_065_IDLE_VLLM_OBJECT_KV_PROBE.md`](EXPERIMENT_065_IDLE_VLLM_OBJECT_KV_PROBE.md).

---

## 19. How Stage 6+ build on this

| Stage | Connection |
|---|---|
| **Stage 6** — LMCache | Verifier-tier backing; separate contract after vLLM gates clear |
| **Stage 8** — Throughput harness | Required before speed/latency claims on any backend |
| **Phase 15A** | Install-safe vLLM feasibility probe — [`EXPERIMENT_059_VLLM_FEASIBILITY_PROBE.md`](EXPERIMENT_059_VLLM_FEASIBILITY_PROBE.md) |
| **Phase 15B** | Isolated vLLM venv feasibility — [`EXPERIMENT_060_VLLM_VENV_FEASIBILITY.md`](EXPERIMENT_060_VLLM_VENV_FEASIBILITY.md) |
| **Phase 15B-unblock** | vLLM version compatibility sweep — [`EXPERIMENT_061_VLLM_VERSION_SWEEP.md`](EXPERIMENT_061_VLLM_VERSION_SWEEP.md) |
| **Phase 15C-env** | vLLM container/CUDA-13 feasibility — [`EXPERIMENT_062_VLLM_CONTAINER_FEASIBILITY.md`](EXPERIMENT_062_VLLM_CONTAINER_FEASIBILITY.md) |
| **Phase 15C** | vLLM API surface reconnaissance — [`EXPERIMENT_063_VLLM_API_SURFACE_RECON.md`](EXPERIMENT_063_VLLM_API_SURFACE_RECON.md) |
| **Phase 15D** | vLLM KV/cache visibility probe — [`EXPERIMENT_064_VLLM_KV_VISIBILITY_PROBE.md`](EXPERIMENT_064_VLLM_KV_VISIBILITY_PROBE.md) |
| **Phase 15E** | Idle-GPU vLLM object KV probe — [`EXPERIMENT_065_IDLE_VLLM_OBJECT_KV_PROBE.md`](EXPERIMENT_065_IDLE_VLLM_OBJECT_KV_PROBE.md) |
| **Prototype runtime** (future) | Implements `rollback_fallback_path`; may advance status toward `PROTOTYPE_READY` |

---

## 13. Claims boundary

| Allowed | Forbidden |
|---|---|
| vLLM prototype contract metadata exists | vLLM integration exists |
| Gates and mapping documented | Speedup / latency / throughput improvement |
| Design readiness for future prototype | Active memory savings |
| Exactness gate cited | Production serving readiness |
| | VeriCache throughput reproduction |
