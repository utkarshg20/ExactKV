# Materialized Compressed-Draft Backend (Phase 11D)

**Status:** Isolated contract/design spike — **not** runtime integration.

> This is an **isolated contract/design spike**, not a serving runtime.  
> **Current generation and verification behavior is unchanged.**  
> The backend materializes decompressed K/V for use and therefore does **not** prove active GPU memory savings.  
> ExactKV still does **not** implement vLLM or LMCache integration.  
> ExactKV still does **not** reproduce VeriCache throughput results.

Companion: [`DUAL_CACHE_ABSTRACTION.md`](DUAL_CACHE_ABSTRACTION.md) · [`FULL_KV_STORAGE_MANAGER.md`](FULL_KV_STORAGE_MANAGER.md) · [`VERICACHE_SYSTEMS_ROADMAP.md`](VERICACHE_SYSTEMS_ROADMAP.md)

---

## 1. Why this exists (VeriCache Stage 3)

VeriCache drafts from **compressed KV** while keeping **full KV** for verification. Many ExactKV compressors today **materialize** dequantized tensors for attention — stored bytes may shrink, but the **working copy** during draft often equals full precision.

Stage 3 asks: can ExactKV **describe and validate** that split using `DualCacheState` + storage manager metadata — before any hot compressed-attention kernel work?

Phase 11D answers: **yes, on tiny synthetic payloads**.

---

## 2. What a materialized compressed-draft backend means

| Term | Meaning |
|---|---|
| **Compressed storage** | Quantised or simulated smaller stored representation |
| **Materialized working KV** | Full-precision tensors used during attention/draft |
| **Materialized backend** | Compressor path that decompresses/materialises for draft |

This spike **describes** both byte buckets — it does **not** implement fused decompress kernels or hot compressed attention.

---

## 3. How this differs from real hot compressed attention

| Materialized spike (11D) | Hot compressed attention (future) |
|---|---|
| Metadata + synthetic tensors | Fused kernels / paged blocks |
| `materialized_bytes ≈ full KV` | May avoid full materialization |
| No model load | Requires model + CUDA path |
| Claim firewall enforced | Requires measured memory panel |

---

## 4. Relation to SpectralQuant materializing adapter

SpectralQuant Exp 045 is a **restricted external adapter** that compresses K/V then **materialises** dequant tensors for draft — `EXTERNAL_ADAPTER_MATERIALIZED` kind in this spike mirrors that pattern at metadata level only.

- Not integrated into default registry
- Not a VeriCache system parity claim
- No speed/memory/serving claim

---

## 5. What is implemented now

| Component | Location |
|---|---|
| `DraftBackendKind` | `exactkv/cache/materialized_backend.py` |
| `MaterializedDraftMetadata` | same |
| `MaterializedDraftBackend` / `SyntheticMaterializedDraftBackend` | same |
| `build_draft_cache_view`, `dual_cache_with_materialized_draft_and_verifier` | same |
| `compute_dual_cache_footprint`, `validate_materialized_dual_cache` | same |
| `smoke_materialized_dual_cache` | tiny tensor smoke with storage backends |

Kinds exercised in smoke:

- `IDENTITY`
- `SIMULATED_COMPRESSED`
- `EXTERNAL_ADAPTER_MATERIALIZED` (and `REAL_COMPRESSED_MATERIALIZED` via helpers)

---

## 6. What is not implemented

- Runtime wiring into `ExactKVGenerator`
- vLLM / LMCache / remote prefix / batching / serving
- CPU or disk offload as runtime features
- CUDA/Triton hot draft kernels
- Model-scale experiments or new benchmark panels
- Active GPU memory savings proof

---

## 7. Why this does not prove memory savings

`stored_bytes` may be smaller than full KV, but `materialized_bytes` reflects the **working copy** used during draft — typically full precision. Exp 031 and V5 accounting document that peak footprint is dominated by materialization, not stored bytes alone.

---

## 8. Why this does not prove speedup

No timing harness. No scheduler. Diagnostic-only path.

---

## 9. Why this does not prove serving readiness

Isolated metadata composition only — not multi-request, not paged serving.

---

## 10. Prepares Stage 4 and later runtime work

| Next | How 11D helps |
|---|---|
| **Stage 4** — extended verification scheduler | Scheduler can reference draft/verifier `CacheView` contracts |
| **Runtime integration** (future) | `MaterializedDraftMetadata` + `KVStorageMetadata` compose into `DualCacheState` before generator wiring |

---

## 11. Claims boundary

| Allowed | Forbidden |
|---|---|
| Materialized draft metadata + stored verifier compose valid `DualCacheState` | Active GPU memory savings |
| Describes SpectralQuant-style materializing pattern at metadata level | Speedup / throughput improvement |
| Tiny synthetic tensor smoke | Production serving readiness |
| | vLLM / LMCache integration exists |
| | Full VeriCache reproduction |
