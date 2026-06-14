# VeriCache Systems Roadmap (Phase 11A)

**Status:** Planning document only — **no implementation in Phase 11A.**

> Future VeriCache parity work is a **systems roadmap**, not current status.  
> ExactKV today reproduces **algorithmic semantics**, not the **serving system**.

Companion: [`VERICACHE_PARITY_AUDIT.md`](VERICACHE_PARITY_AUDIT.md) · [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md) · [`ROADMAP.md`](ROADMAP.md)

---

## 1. North star

ExactKV should eventually implement **VeriCache-equivalent functionality**:

- lossy compressed KV drafts tokens
- full KV verifies and corrects drift
- **final greedy output matches full KV**
- **plus** the systems pieces VeriCache uses for practical inference: dual-cache residency, extended verification, serving integration, and measured throughput/memory panels

**Today:** stages 0–1 of this roadmap are partially satisfied by the V13 correctness harness. Stages 2–10 are **not implemented**.

---

## 2. Staged implementation

### Stage 0 — Parity audit and claim firewall ✅ (Phase 11A)

| Field | Detail |
|---|---|
| **Goal** | Document what VeriCache covers vs ExactKV; freeze forbidden claims |
| **Files likely touched** | `docs/VERICACHE_PARITY_AUDIT.md`, `docs/CLAIMS_AUDIT.md`, tests |
| **Implementation risk** | Low — documentation only |
| **Test gate** | `tests/test_vericache_parity_docs.py`; `audit_public_claims.py` |
| **Claims unlocked** | Scoped algorithm-semantics wording only; **no** system parity |

---

### Stage 1 — Dual-cache abstraction ✅ (Phase 11B)

| Field | Detail |
|---|---|
| **Goal** | Formalize draft/verifier cache roles, residency, materialization, and claim invariants |
| **Files likely touched** | `exactkv/cache/dual_cache.py`, [`DUAL_CACHE_ABSTRACTION.md`](DUAL_CACHE_ABSTRACTION.md) |
| **Implementation risk** | Low — contract only; no generator wiring |
| **Test gate** | `tests/test_dual_cache_abstraction.py` |
| **Claims unlocked** | “Dual-cache contract exists” — **not** memory savings or serving |

**Current:** `CacheView` / `DualCacheState` + validators; **not** wired into `ExactKVGenerator`.

---

### Stage 2 — Full-KV storage manager ✅ (Phase 11C design spike)

| Field | Detail |
|---|---|
| **Goal** | Pluggable full-KV backing: serialize, store, reload tiny verifier payloads |
| **Files likely touched** | `exactkv/cache/storage.py`, [`FULL_KV_STORAGE_MANAGER.md`](FULL_KV_STORAGE_MANAGER.md) |
| **Implementation risk** | Medium — production format/eviction deferred |
| **Test gate** | `tests/test_full_kv_storage_manager.py` |
| **Claims unlocked** | “Storage contract round-trips on tiny payloads” — **not** offload or savings |

**Current:** in-memory + file backends; **not** wired into `ExactKVGenerator`. Production GPU/host tiers remain future work.

---

### Stage 3 — Materialized compressed-draft backend ✅ (Phase 11D design spike)

| Field | Detail |
|---|---|
| **Goal** | Describe draft/verifier split when draft path materializes compressed KV |
| **Files likely touched** | `exactkv/cache/materialized_backend.py`, [`MATERIALIZED_COMPRESSED_DRAFT_BACKEND.md`](MATERIALIZED_COMPRESSED_DRAFT_BACKEND.md) |
| **Implementation risk** | Medium — must not imply hot compressed attention |
| **Test gate** | `tests/test_materialized_compressed_draft_backend.py` |
| **Claims unlocked** | Valid `DualCacheState` from materialized draft + stored verifier metadata — **not** savings |

**Current:** synthetic tensor smoke; identity / simulated / external-adapter kinds; **not** wired to generator.

---

### Stage 4 — Extended verification scheduler ✅ (Phase 11E contract spike)

| Field | Detail |
|---|---|
| **Goal** | Policy metadata for sequential, span, bonus-disabled, serving placeholder schedules |
| **Files likely touched** | `exactkv/verify/scheduler.py`, [`EXTENDED_VERIFICATION_SCHEDULER.md`](EXTENDED_VERIFICATION_SCHEDULER.md) |
| **Implementation risk** | Low — metadata only; runtime unchanged |
| **Test gate** | `tests/test_extended_verification_scheduler.py` |
| **Claims unlocked** | Scheduler policy contracts exist — **not** throughput or parallel runtime |

**Current:** factories + validators; bonus-token disabled; vLLM/LMCache placeholders; **not** wired to generator.

---

### Stage 5 — vLLM prototype path

| Field | Detail |
|---|---|
| **Goal** | Prototype authoritative full-KV export / verify hook — **not** production merge |
| **Files likely touched** | New `exactkv/serving/vllm/`; revisit Exp 017 conclusions |
| **Implementation risk** | Very high — paged KV ≠ HF `FullKVState`; fork risk |
| **Test gate** | Prototype panel: `exactkv_failures == 0` on bounded prompt set |
| **Claims unlocked** | “vLLM prototype path evaluated” — **not** “vLLM integrated” until Stage 10 |

**Current:** Exp 007/017 **no-go** for direct integration.

---

### Stage 6 — LMCache / prefix cache integration

| Field | Detail |
|---|---|
| **Goal** | LMCache (or equivalent) as full-KV backing tier for verify steps |
| **Files likely touched** | `exactkv/storage/` + LMCache client adapter |
| **Implementation risk** | High — async restore vs synchronous verify |
| **Test gate** | Restore correctness tests; ownership invariant tests |
| **Claims unlocked** | “LMCache-backed verify path tested on panel” — **not** production tiering |

---

### Stage 7 — Remote prefix caching experiment

| Field | Detail |
|---|---|
| **Goal** | Simulated remote drafter + near-storage verifier (per `FUTURE_RESEARCH.md`) |
| **Files likely touched** | New `exactkv/distributed/` or experiment scripts |
| **Implementation risk** | High — distributed correctness |
| **Test gate** | Multi-process replay tests; `exactkv_failures == 0` on sim panel |
| **Claims unlocked** | “Remote prefix prototype on sim panel” only |

---

### Stage 8 — Throughput benchmark harness

| Field | Detail |
|---|---|
| **Goal** | Reproducible tokens/sec / latency methodology **including verify overhead** |
| **Files likely touched** | `exactkv/metrics/timing.py`, benchmark scripts, reporting |
| **Implementation risk** | Medium — honest comparison vs full greedy |
| **Test gate** | Harness CI smoke; methodology doc sign-off |
| **Claims unlocked** | **Diagnostic** throughput numbers with verify overhead stated — **not** speedup claims until Stage 9–10 |

**Current:** Exp 030 diagnostic only — ExactKV slower on tested panel.

---

### Stage 9 — Paper-like reproduction panel

| Field | Detail |
|---|---|
| **Goal** | Fixed compressor × model × benchmark panel comparable to VeriCache paper **methodology** |
| **Files likely touched** | Experiment scripts, `reports/` (gitignored), leaderboard tier |
| **Implementation risk** | Medium — scope creep into fake paper numbers |
| **Test gate** | Panel tests; claims audit pass; explicit panel citation |
| **Claims unlocked** | “On panel P, ExactKV measured X” — panel-bound only |

**Current:** V10 suites + restricted probes — **not** paper panel.

---

### Stage 10 — Release candidate for VeriCache-parity claim

| Field | Detail |
|---|---|
| **Goal** | Independent review: algorithm + systems + measured panels satisfy pre-defined parity checklist |
| **Files likely touched** | Release notes, parity audit update, launch validation |
| **Implementation risk** | Process — overclaim if gates skipped |
| **Test gate** | Full pytest; clean-clone smoke; parity audit all **done** or explicitly deferred; no forbidden claims |
| **Claims unlocked** | **Only after checklist:** “ExactKV implements VeriCache-equivalent functionality on documented panel” — still **not** automatic production/speed claim |

---

## 3. Dependency graph (simplified)

```text
Stage 0 (audit)
  → Stage 1 (dual-cache API)
    → Stage 2 (full-KV storage)
      → Stage 3 (hot compressed draft)
      → Stage 4 (extended verify)
        → Stage 8 (throughput harness)
          → Stage 9 (paper panel)
            → Stage 10 (parity RC)
    → Stage 5 (vLLM prototype) ─┐
    → Stage 6 (LMCache)        ├→ Stage 7 (remote prefix)
```

Stages 5–7 can proceed in parallel after Stage 2 but **must not** skip exactness gates.

---

## 4. What remains forbidden until Stage 10

- Full VeriCache reproduction claim.
- VeriCache throughput or memory benefit claims.
- Speedup, active GPU memory savings, production serving.
- vLLM / LMCache integration claims (prototype ≠ integrated).
- Paper numbers cited as ExactKV results.

---

## 5. Recommended next phase (after 11A)

**Stage 1 hardening** or **Stage 2 spike design doc** — pick one storage-backed full-KV experiment on a **tiny panel** before any serving integration. Do **not** jump to vLLM until dual-cache storage contracts are stable.

See [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md) for deferred IDs (D11 vLLM, D12 LMCache, D21 extended verify).
