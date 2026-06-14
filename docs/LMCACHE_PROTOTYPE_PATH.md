# LMCache Prototype Path (Phase 11G)

**Status:** LMCache prototype contract only — **LMCache is not imported or required by ExactKV.**

> This is an **LMCache prototype contract**, not an LMCache integration.  
> **LMCache is not imported or required by ExactKV.**  
> **ExactKV still does not implement remote prefix caching.**  
> **ExactKV still does not implement production serving.**  
> No speedup, latency improvement, throughput improvement, active memory savings, remote-prefix-cache, or production-serving claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput results.

Companion: [`VERICACHE_SYSTEMS_ROADMAP.md`](VERICACHE_SYSTEMS_ROADMAP.md) · [`VLLM_PROTOTYPE_PATH.md`](VLLM_PROTOTYPE_PATH.md) · [`FULL_KV_STORAGE_MANAGER.md`](FULL_KV_STORAGE_MANAGER.md) · `exactkv/integrations/lmcache_contract.py`

---

## 1. Why LMCache is relevant for VeriCache parity

VeriCache uses **LMCache** (or equivalent) as a **full-KV backing tier**: prefix reuse, serialization, restore, and optional remote tiers for practical inference. ExactKV today keeps authoritative full KV in-process on the HF harness with a **storage contract spike** (Phase 11C) but no external prefix-cache integration.

Stage 6 documents **what must be true** before ExactKV can honestly start LMCache / prefix-cache prototype work — without claiming integration exists today.

---

## 2. What is implemented now

| Layer | Phase | Status |
|---|---|---|
| Dual-cache roles (`DRAFT` / `VERIFIER`) | 11B | Contract only |
| Full-KV storage manager | 11C | In-memory + file backends |
| Materialized compressed-draft backend | 11D | Metadata + synthetic smoke |
| Verification scheduler (`FUTURE_LMCACHE` placeholder) | 11E | Policy metadata only |
| vLLM prototype gates | 11F | Contract only |
| **LMCache prototype gates** | **11G** | **`LMCachePrototypePlan` + validators** |

**Not implemented:** LMCache import, prefix-cache client, remote prefix runtime, async restore adapter, serving tests.

---

## 3. What is not implemented

- LMCache package dependency or import
- Local or remote prefix-cache runtime
- Async KV restore integrated with synchronous verify loop
- Multi-request serving or batching
- Throughput or active memory benefit claims
- Remote prefix caching (Stage 7 — separate from this contract)

Exp 007/017 concluded **no-go** for direct LMCache integration until authoritative full-KV restore semantics align with verify.

---

## 4. Storage contract mapping (future LMCache path)

| ExactKV contract (today) | Future LMCache target (design only) |
|---|---|
| `KVStorageBackend` | LMCache tier backing store |
| `StoredKVEntry` / `KVStorageMetadata` | Serialized prefix blob + metadata |
| `CacheRole.VERIFIER` | Authoritative full-KV tier for verify steps |
| `InMemoryKVStorageBackend` | Local prefix-cache semantics reference |
| `FileKVStorageBackend` | Disk-tier persistence reference |
| `VerificationExecutionMode.FUTURE_LMCACHE` (11E) | Scheduler policy when verify reads from LMCache |

This mapping is **documentation and gate metadata** — not a working adapter.

---

## 5. Interaction with the future vLLM path

| Concern | Phase 11G stance |
|---|---|
| vLLM worker owns paged decode | LMCache backs **verifier-tier** full KV — contract-only per Phase 11F |
| vLLM status | `VLLMPrototypePlan` is `CONTRACT_ONLY`; `rollback_fallback_path` unsatisfied |
| LMCache + vLLM ordering | Gates require vLLM contract interaction documented; **no active vLLM integration** assumed |
| Remote prefix | **Not active** — `remote_prefix_cache_active` must remain `False` |

LMCache prototype work must not proceed as if vLLM integration were already active.

---

## 6. Why this phase does not integrate LMCache

Phase 11G answers: *What must be true before prototype code?* It does **not** write prototype code because:

1. Exp 017 reaffirmed **no-go** for unsafe async restore vs synchronous verify.
2. Remote prefix caching is Stage 7 — explicitly not active here.
3. vLLM prototype (11F) remains contract-only; LMCache tiers depend on that boundary.
4. Claim firewall requires gates before any integration or performance language.

---

## 7. Integration gates

| Gate | Required | Default (11G) | Purpose |
|---|---|---|---|
| `optional_dependency_isolation` | yes | satisfied | LMCache stays optional extra, not core dep |
| `no_required_lmcache_import` | yes | satisfied | Default install never imports LMCache |
| `local_prefix_cache_semantics_identified` | yes | satisfied | Local prefix ↔ storage manager mapping |
| `remote_prefix_cache_semantics_identified` | yes | satisfied (blocked) | Stage 7 target — not active |
| `full_kv_serialization_mapping` | yes | satisfied | Storage serialize path documented |
| `full_kv_restore_mapping` | yes | satisfied (blocked) | Restore vs sync verify blocker noted |
| `verifier_cache_correctness_gate` | yes | satisfied | Verifier role integrity |
| `async_load_blocking_semantics_documented` | yes | satisfied | Async load documented; HF blocking default |
| `eviction_invalidation_semantics_documented` | yes | satisfied | Eviction policy documented only |
| `vllm_contract_interaction_identified` | yes | satisfied | 11F contract-only reference |
| `exactness_test_plan` | yes | satisfied | `exactkv_failures == 0` panel gate |
| `rollback_fallback_path` | yes | **unsatisfied** | HF path remains fallback |
| `no_speed_claim_before_benchmark` | yes | satisfied | Stage 8 harness gate |
| `no_memory_claim_before_measurement` | yes | satisfied | Active measurement gate |
| `no_production_claim_before_serving_tests` | yes | satisfied | Serving test gate |
| `remote_prefix_gate_before_remote_claim` | yes | satisfied | Stage 7 required before remote claims |

`PROTOTYPE_READY` status is **blocked** until all required gates pass validation.

---

## 8. Claim gates

| Gate type | Rule |
|---|---|
| **Exactness** | Bounded panel with `exactkv_failures == 0` before any performance comparison |
| **Remote prefix** | Stage 7 remote-prefix prototype before any remote-prefix-cache claim |
| **Memory** | Active memory measurement before any savings claim |
| **Serving** | Multi-request serving tests before production-serving claim |

---

## 9. JSON schema (plan)

```json
{
  "status": "CONTRACT_ONLY",
  "capabilities_required": ["LOCAL_PREFIX_CACHE", "KV_SERIALIZATION", "KV_RESTORE"],
  "gates": [
    {
      "gate_name": "rollback_fallback_path",
      "required": true,
      "satisfied": false,
      "blocker": "No prototype runtime"
    }
  ],
  "allowed_claims": ["LMCache prototype contract metadata exists"],
  "forbidden_claims": ["speedup", "remote prefix caching"],
  "claim_note": "...",
  "dependency_import_attempted": false,
  "remote_prefix_cache_active": false,
  "vllm_contract_status": "CONTRACT_ONLY"
}
```

---

## 10. How Stage 7+ build on this

| Stage | Connection |
|---|---|
| **Stage 7** — Remote prefix | Extends `REMOTE_PREFIX_CACHE` capability; separate runtime experiment |
| **Stage 8** — Throughput harness | Required before speed/latency claims on any backend |
| **Prototype runtime** (future) | Implements `rollback_fallback_path`; may advance status toward `PROTOTYPE_READY` |

---

## 11. Claims boundary

| Allowed | Forbidden |
|---|---|
| LMCache prototype contract metadata exists | LMCache integration exists |
| Gates and mapping documented | Remote prefix caching implemented |
| Design readiness for future prototype | Speedup / latency / throughput improvement |
| vLLM interaction documented as contract-only | Active memory savings |
| Exactness + remote-prefix gates cited | Production serving readiness |
| | VeriCache throughput reproduction |
