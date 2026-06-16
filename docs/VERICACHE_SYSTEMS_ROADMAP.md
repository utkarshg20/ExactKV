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

**Phase 12A–12H + 13A–13B + 14A–14C (restore + runner + experimental opt-in + CLI + CUDA runtime gate + GPU memory diagnostics):** [`EXPERIMENT_046_FULL_KV_RESTORE_SMOKE.md`](EXPERIMENT_046_FULL_KV_RESTORE_SMOKE.md) through [`EXPERIMENT_058_EXPANDED_GPU_MEMORY_PANEL.md`](EXPERIMENT_058_EXPANDED_GPU_MEMORY_PANEL.md) — **not** serving, memory savings, or throughput.

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

### Stage 5 — vLLM prototype path ✅ (Phase 11F contract spike)

| Field | Detail |
|---|---|
| **Goal** | vLLM prototype integration gates and cache-mapping metadata — **not** runtime |
| **Files likely touched** | `exactkv/integrations/vllm_contract.py`, [`VLLM_PROTOTYPE_PATH.md`](VLLM_PROTOTYPE_PATH.md) |
| **Implementation risk** | Low — metadata only; vLLM not imported |
| **Test gate** | `tests/test_vllm_prototype_contract.py` |
| **Claims unlocked** | “vLLM prototype contract metadata exists” — **not** “vLLM integrated” |

**Current:** `VLLMPrototypePlan` + gates; `rollback_fallback_path` unsatisfied; Exp 007/017 no-go unchanged; **contract-only, not integrated**.

---

### Stage 6 — LMCache / prefix cache integration ✅ (Phase 11G contract spike)

| Field | Detail |
|---|---|
| **Goal** | LMCache prototype integration gates and storage-mapping metadata — **not** runtime |
| **Files likely touched** | `exactkv/integrations/lmcache_contract.py`, [`LMCACHE_PROTOTYPE_PATH.md`](LMCACHE_PROTOTYPE_PATH.md) |
| **Implementation risk** | Low — metadata only; LMCache not imported |
| **Test gate** | `tests/test_lmcache_prototype_contract.py` |
| **Claims unlocked** | “LMCache prototype contract metadata exists” — **not** “LMCache integrated” |

**Current:** `LMCachePrototypePlan` + gates; `rollback_fallback_path` unsatisfied; remote prefix **not active**; vLLM contract **contract-only**; **not integrated**.

---

### Stage 7 — Remote prefix caching experiment ✅ (Phase 11H semantics spike)

| Field | Detail |
|---|---|
| **Goal** | Prefix identity, compatibility, loopback mock via storage backends — **not** network runtime |
| **Files likely touched** | `exactkv/cache/remote_prefix.py`, [`REMOTE_PREFIX_CACHE_SEMANTICS.md`](REMOTE_PREFIX_CACHE_SEMANTICS.md) |
| **Implementation risk** | Low — loopback only; no generator wiring |
| **Test gate** | `tests/test_remote_prefix_cache_semantics.py` |
| **Claims unlocked** | “Prefix identity + loopback round-trip on tiny tensors” — **not** remote prefix runtime |

**Current:** `LoopbackPrefixCache` + `PrefixRestorePlan`; no network I/O; remote placeholder blocked; **not** wired to generator.

---

### Stage 8 — Throughput benchmark harness ✅ (Phase 11I contract spike)

| Field | Detail |
|---|---|
| **Goal** | Throughput/latency methodology contracts + diagnostic schema — **not** speedup claim |
| **Files likely touched** | `exactkv/benchmarks/throughput_contract.py`, [`THROUGHPUT_BENCHMARK_HARNESS.md`](THROUGHPUT_BENCHMARK_HARNESS.md) |
| **Implementation risk** | Low — metadata only; Exp 030 diagnostic cited |
| **Test gate** | `tests/test_throughput_benchmark_contract.py` |
| **Claims unlocked** | Panel-bound **diagnostic** timing with exactness gate — **not** speedup until `CLAIM_ALLOWED` gates pass |

**Current:** `ThroughputBenchmarkPlan` + validators; Exp 030 shows ExactKV **slower** than full greedy on tested panel; **not** wired to generator.

---

### Stage 9 — Paper-like reproduction panel ✅ (Phase 11J contract spike)

| Field | Detail |
|---|---|
| **Goal** | Paper-like panel metadata: models, compressors, workloads, metrics, gates — **not** a run |
| **Files likely touched** | `exactkv/benchmarks/paper_panel_contract.py`, [`PAPER_LIKE_REPRODUCTION_PANEL.md`](PAPER_LIKE_REPRODUCTION_PANEL.md) |
| **Implementation risk** | Low — metadata only; no new benchmark runs |
| **Test gate** | `tests/test_paper_panel_contract.py` |
| **Claims unlocked** | Panel contract + gate requirements — **not** paper reproduction |

**Current:** `PaperLikeReproductionPanel` defaults `CONTRACT_ONLY`; `claim_eligible=False`; external paper numbers **not** ExactKV results.

---

### Stage 10 — Release candidate for VeriCache-parity claim ✅ (Phase 11K claim gate)

| Field | Detail |
|---|---|
| **Goal** | Conservative claim gate: allowed / forbidden / blocked categories — **not** RC certification |
| **Files likely touched** | `exactkv/claims/vericache_parity_gate.py`, [`VERICACHE_PARITY_CLAIM_GATE.md`](VERICACHE_PARITY_CLAIM_GATE.md) |
| **Implementation risk** | Low — classification metadata; human review required for any unlock |
| **Test gate** | `tests/test_vericache_parity_claim_gate.py` |
| **Claims unlocked** | Explicit allowed-with-scope list — **not** full parity; `full_parity_claim_allowed=False` |

**Current:** default gate classifies algorithmic/correctness as allowed-with-scope; throughput/memory/serving/full parity **forbidden**; vLLM/LMCache **contract-only**.

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

**Phase 15A (vLLM feasibility probe):** [`EXPERIMENT_059_VLLM_FEASIBILITY_PROBE.md`](EXPERIMENT_059_VLLM_FEASIBILITY_PROBE.md) — install-safe import probe; **not** vLLM integration or throughput.

**Phase 15B (isolated vLLM venv feasibility):** [`EXPERIMENT_060_VLLM_VENV_FEASIBILITY.md`](EXPERIMENT_060_VLLM_VENV_FEASIBILITY.md) — vLLM installed only in `.venv-vllm`; import + tiny generation smoke; **not** ExactKV integration or throughput.

**Phase 15B-unblock (vLLM version sweep):** [`EXPERIMENT_061_VLLM_VERSION_SWEEP.md`](EXPERIMENT_061_VLLM_VERSION_SWEEP.md) — all five pip candidates failed `libcudart.so.13` on cu128 pod.

**Phase 15C-env (vLLM container feasibility):** [`EXPERIMENT_062_VLLM_CONTAINER_FEASIBILITY.md`](EXPERIMENT_062_VLLM_CONTAINER_FEASIBILITY.md) — CUDA 13 template import pass; generation smoke OOM when server running.

**Phase 15C (vLLM API surface recon):** [`EXPERIMENT_063_VLLM_API_SURFACE_RECON.md`](EXPERIMENT_063_VLLM_API_SURFACE_RECON.md) — import/object visibility; **not** ExactKV integration.

**Phase 15D (vLLM KV/cache visibility probe):** [`EXPERIMENT_064_VLLM_KV_VISIBILITY_PROBE.md`](EXPERIMENT_064_VLLM_KV_VISIBILITY_PROBE.md) — metadata-only object inspection on idle GPU; **not** ExactKV integration.

**Phase 15E (idle-GPU object KV probe):** [`EXPERIMENT_065_IDLE_VLLM_OBJECT_KV_PROBE.md`](EXPERIMENT_065_IDLE_VLLM_OBJECT_KV_PROBE.md) — **deferred** on auto-serving RunPod vLLM template; idle pod required; **not** ExactKV integration.

**Phase 16A (streaming quantized-KV attention feasibility):** [`EXPERIMENT_066_STREAMING_QUANT_ATTENTION_FEASIBILITY.md`](EXPERIMENT_066_STREAMING_QUANT_ATTENTION_FEASIBILITY.md) — tensor-level reference; chunked dequantized attention; **not** inference integration or throughput.

**Phase 16B (HF single-layer attention-drift probe):** [`EXPERIMENT_067_HF_SINGLE_LAYER_ATTENTION_DRIFT.md`](EXPERIMENT_067_HF_SINGLE_LAYER_ATTENTION_DRIFT.md) — offline HF Q/K/V drift; **not** generation integration or throughput.

**Phase 16C (Qwen RoPE/GQA long-context probe):** [`EXPERIMENT_068_QWEN_ROPE_LONG_CONTEXT_ATTENTION_PROBE.md`](EXPERIMENT_068_QWEN_ROPE_LONG_CONTEXT_ATTENTION_PROBE.md) — RoPE/GQA extraction fix + long-context chunking; **not** generation integration or throughput.

**Phase 16D (Multi-layer drift accumulation):** [`EXPERIMENT_069_MULTILAYER_ATTENTION_DRIFT_ACCUMULATION.md`](EXPERIMENT_069_MULTILAYER_ATTENTION_DRIFT_ACCUMULATION.md) — offline multi-layer compressed-path replay; **not** generation integration or throughput.

**Phase 16E (Streaming numerics audit):** [`EXPERIMENT_070_STREAMING_MULTILAYER_NUMERICS_AUDIT.md`](EXPERIMENT_070_STREAMING_MULTILAYER_NUMERICS_AUDIT.md) — Phase 16D boundary failure diagnosis; **not** generation integration or throughput.

**Post-11K / Post-13B / Post-14A / Post-14B / Post-14C / Post-15A / Post-15B / Post-15B-unblock / Post-15C-env / Post-15C / Post-15D / Post-15E / Post-16A / Post-16B / Post-16C / Post-16D / Post-16E:** vLLM environment feasibility track deferred at 15E; Phase 16A–16E compressed-attention research; default CLI/runtime unchanged.

See [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md) for deferred IDs (D11 vLLM, D12 LMCache, D21 extended verify).
