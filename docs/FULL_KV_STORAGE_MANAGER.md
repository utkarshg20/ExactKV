# Full-KV Storage Manager (Phase 11C)

**Status:** Storage contract + tiny payload smoke — **not** runtime integration.

> This is a **storage contract and tiny payload smoke**, not a serving runtime.  
> ExactKV still does **not** implement vLLM or LMCache integration.  
> ExactKV still does **not** implement active GPU memory savings.  
> ExactKV still does **not** reproduce VeriCache throughput results.  
> **Current generation and verification behavior is unchanged.**

Companion: [`DUAL_CACHE_ABSTRACTION.md`](DUAL_CACHE_ABSTRACTION.md) · [`VERICACHE_SYSTEMS_ROADMAP.md`](VERICACHE_SYSTEMS_ROADMAP.md) · `exactkv/cache/storage.py`

---

## 1. Why a full-KV storage manager

VeriCache keeps **authoritative full KV** available for verification while compressed KV drafts on a hotter path. ExactKV’s `FullKVState` today lives in-process only. Stage 2 asks whether ExactKV can **serialize, store, reload, and validate** verifier KV through a pluggable contract — prerequisite for CPU/disk tiers and later serving integration.

Phase 11C answers: **yes, for tiny tensor payloads** via in-memory and file backends.

---

## 2. What is implemented now

| Component | Description |
|---|---|
| `KVStorageBackend` | Abstract `put` / `get` / `exists` / `delete` / `metadata` |
| `InMemoryKVStorageBackend` | Dict-backed; residency metadata `CPU` |
| `FileKVStorageBackend` | `torch.save` + JSON sidecar; residency metadata `DISK` |
| `KVStorageHandle` | `namespace`, `key`, `version` |
| `KVStorageMetadata` | Tensor count, bytes, dtype/shape summaries, claim note |
| `build_verifier_storage_metadata` | Derive metadata from payload |
| `cache_view_from_storage_metadata` | Build conservative verifier `CacheView` |
| `dual_cache_with_stored_verifier` | Pair draft view + stored verifier view |
| `smoke_store_verifier_payload` | Put/get round-trip helper for tests |

**Not wired** into `ExactKVGenerator`, `VerificationEngine`, or benchmarks.

---

## 3. In-memory backend

- Stores arbitrary nested tensor payloads in a process-local dict.
- Metadata residency: **`CPU`** (logical host RAM — not an offload claim).
- Round-trip: `put` → `get` preserves tensor values and metadata.

---

## 4. File backend

- Writes `{stem}.pt` (payload via `torch.save`) and `{stem}.meta.json` (metadata).
- Metadata residency: **`DISK`** (local directory — design spike only).
- Uses `tmp_path` in tests; not a production on-disk format.

---

## 5. Tiny payload smoke only

Tests use small random tensors (e.g. `torch.randn(2, 4)`), not model-scale KV from live generation. This validates the **contract**, not production capacity or performance.

---

## 6. What is explicitly not implemented

| Item | Status |
|---|---|
| Runtime integration | **No** |
| CPU offload claim | **No** — metadata label only |
| Disk offload claim | **No** — local test files only |
| Active GPU memory savings | **No** — `supports_real_bytes_claim=False` on stored verifier views |
| vLLM / LMCache | **No** |
| Serving / batching | **No** |
| VeriCache throughput reproduction | **No** |

---

## 7. Dual-cache integration

Stored verifier metadata maps to:

- `CacheRole.VERIFIER`
- `CacheMaterialization.FULL`
- `CacheResidency.CPU` or `DISK` (per backend)
- `supports_real_bytes_claim=False` with mandatory `claim_note`

`dual_cache_with_stored_verifier(draft, metadata)` produces a `DualCacheState` that must pass `validate_dual_cache_state()`.

---

## 8. Prepares Stage 3 and Stage 4

| Next stage | How 11C helps |
|---|---|
| **Stage 3** — materialized compressed-draft backend | Storage manager can hold verifier KV while draft path experiments with hotter decompress |
| **Stage 4** — extended verification scheduler | Scheduler can request verifier reload from storage without changing commit semantics |

---

## 9. Claims boundary

| Allowed | Forbidden |
|---|---|
| Storage contract + tiny tensor round-trip on tests | Full VeriCache reproduction |
| Verifier metadata derives valid `CacheView` | Active GPU memory savings |
| File/in-memory smoke passes | Production serving readiness |
| | vLLM / LMCache integration exists |
| | CPU/disk offload benefits |
