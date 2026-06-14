# Dual-Cache Abstraction (Phase 11B)

**Status:** Contract layer only — **generation and verification behavior unchanged.**

> The dual-cache abstraction is a **contract layer**, not a serving runtime.  
> ExactKV still does **not** implement vLLM or LMCache integration.  
> ExactKV still does **not** implement active GPU memory savings.  
> ExactKV still does **not** reproduce VeriCache throughput results.  
> **Current generation and verification behavior is unchanged.**

Companion: [`VERICACHE_PARITY_AUDIT.md`](VERICACHE_PARITY_AUDIT.md) · [`VERICACHE_SYSTEMS_ROADMAP.md`](VERICACHE_SYSTEMS_ROADMAP.md) · `exactkv/cache/dual_cache.py`

---

## 1. Why ExactKV needs dual-cache contracts

VeriCache separates **lossy compressed KV** (draft path) from **full KV** (verify path). ExactKV already implements this **algorithmically** via `CompressedKVState` and `FullKVState`, but lacked a **portable, serializable contract** for:

- cache roles (draft vs verifier)
- residency and materialization metadata
- byte accounting with claim guardrails
- future Stage 2 storage manager and Stage 3 hot-draft backends

Phase 11B adds that contract without changing the generation loop.

---

## 2. Draft cache vs verifier cache

| Side | Role | ExactKV today | Contract type |
|---|---|---|---|
| **Draft** | Proposes tokens from compressed/lossy KV | `CompressedKVState` | `CacheView(role=DRAFT)` |
| **Verifier** | Authoritative full-KV greedy reference | `FullKVState` | `CacheView(role=VERIFIER)` |

The verifier cache is **authoritative**. The draft cache is **provisional** until verified and committed.

---

## 3. What is implemented now (Phase 11B)

| Component | Location |
|---|---|
| `CacheRole`, `CacheResidency`, `CacheMaterialization` | `exactkv/cache/dual_cache.py` |
| `CacheView` — per-side metadata + accounting | same |
| `DualCacheState` — draft + verifier pair | same |
| `validate_cache_view`, `validate_dual_cache_state` | same |
| `to_dict` / `from_dict` JSON schema | same |
| `build_identity_dual_cache` test helper | same |

**Not wired** into `ExactKVGenerator` or `VerificationEngine` in this phase.

---

## 4. What is explicitly not implemented

- vLLM or LMCache integration
- CPU offload, disk-backed KV, remote prefix cache
- Batching / serving scheduler
- Active GPU memory savings proof
- VeriCache throughput reproduction
- Automatic population from live generation (future adapter)

---

## 5. Invariants (claim firewall)

1. Draft role = `DRAFT`; verifier role = `VERIFIER` (distinct).
2. Verifier materialization = `FULL` or `UNKNOWN`.
3. All byte fields non-negative.
4. `supports_real_bytes_claim=False` requires non-empty `claim_note`.
5. `MATERIALIZED` / `SIMULATED` draft views cannot claim real-byte GPU savings.
6. Off-GPU residency with `supports_real_bytes_claim=True` requires explicit note.

These invariants prevent accidental memory-savings headlines from metadata alone.

---

## 6. Why this does not prove memory savings

`kv_bytes` and related fields are **accounting estimates** (see `exactkv/metrics/memory.py`). The dual-cache contract **describes** storage; it does **not** measure peak device residency or prove VRAM reduction. Exp 031 found no active GPU savings at tested scale.

---

## 7. Why this does not prove serving readiness

Residency enums (`CPU`, `DISK`, `REMOTE`) are **placeholders** for future Stage 2+ work. No storage manager, no async transfer, no vLLM block tables.

---

## 8. Stage 2 — full-KV storage manager (next)

Stage 2 will implement pluggable backing for verifier `CacheView` residency (GPU → host → mmap) while preserving:

- `DualCacheState.validate()` gates
- `exactkv_failures == 0` exactness requirement
- claim firewall from this document

See [`VERICACHE_SYSTEMS_ROADMAP.md`](VERICACHE_SYSTEMS_ROADMAP.md) Stage 2.

---

## 9. JSON schema (v1.0)

```json
{
  "schema_version": "1.0",
  "notes": "optional human note",
  "draft": {
    "role": "DRAFT",
    "backend_name": "int8",
    "residency": "GPU",
    "materialization": "COMPRESSED",
    "kv_bytes": 12345,
    "metadata_bytes": 128,
    "temporary_workspace_bytes": 0,
    "supports_real_bytes_claim": true,
    "claim_note": "..."
  },
  "verifier": {
    "role": "VERIFIER",
    "backend_name": "full_kv",
    "residency": "GPU",
    "materialization": "FULL",
    "kv_bytes": 45678,
    "metadata_bytes": 0,
    "temporary_workspace_bytes": 0,
    "supports_real_bytes_claim": true,
    "claim_note": "..."
  }
}
```

Missing optional fields default safely in `from_dict()`.

---

## 10. Claims boundary

| Allowed | Forbidden |
|---|---|
| Dual-cache **contract** exists for future systems work | Full VeriCache reproduction |
| Describes draft vs verifier roles | Active GPU memory savings |
| Serializable metadata for reports (future) | Production serving readiness |
| | vLLM / LMCache integration exists |
| | VeriCache throughput benefits |
