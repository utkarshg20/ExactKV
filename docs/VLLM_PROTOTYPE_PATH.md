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

## 9. How Stage 6+ build on this

| Stage | Connection |
|---|---|
| **Stage 6** — LMCache | Verifier-tier backing; separate contract after vLLM gates clear |
| **Stage 8** — Throughput harness | Required before speed/latency claims on any backend |
| **Prototype runtime** (future) | Implements `rollback_fallback_path`; may advance status toward `PROTOTYPE_READY` |

---

## 10. Claims boundary

| Allowed | Forbidden |
|---|---|
| vLLM prototype contract metadata exists | vLLM integration exists |
| Gates and mapping documented | Speedup / latency / throughput improvement |
| Design readiness for future prototype | Active memory savings |
| Exactness gate cited | Production serving readiness |
| | VeriCache throughput reproduction |
