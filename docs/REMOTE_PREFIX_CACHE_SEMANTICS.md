# Remote Prefix Cache Semantics (Phase 11H)

**Status:** Remote-prefix semantics spike with loopback mock — **not** a remote prefix cache runtime.

> This is a **remote-prefix-cache semantics spike**, not a remote prefix cache runtime.  
> The **loopback mock does not perform network I/O**.  
> **LMCache is not imported or required.**  
> **vLLM is not imported or required.**  
> **ExactKV still does not implement production serving.**  
> No speedup, latency improvement, throughput improvement, active memory savings, remote-prefix-cache runtime, or production-serving claim is made.  
> **Current generation and verification behavior is unchanged.**

Companion: [`VERICACHE_SYSTEMS_ROADMAP.md`](VERICACHE_SYSTEMS_ROADMAP.md) · [`LMCACHE_PROTOTYPE_PATH.md`](LMCACHE_PROTOTYPE_PATH.md) · [`FULL_KV_STORAGE_MANAGER.md`](FULL_KV_STORAGE_MANAGER.md) · `exactkv/cache/remote_prefix.py`

---

## 1. Why remote prefix caching matters for VeriCache parity

VeriCache discusses **remote prefix reuse**: a drafter may consume a prefix KV tier while a verifier holds authoritative full KV near storage. ExactKV Phase 11G documented LMCache gates; Stage 7 adds **executable semantics** for prefix identity, compatibility, and restore planning — still without real network I/O.

---

## 2. What is implemented now

| Component | Purpose |
|---|---|
| `PrefixIdentity` | Stable model/tokenizer/prompt/token hash metadata |
| `PrefixCacheEntry` | Links identity to `KVStorageHandle` + metadata |
| `PrefixRestorePlan` | Compatibility decision + fallback flag |
| `check_prefix_compatibility` | model/tokenizer/length/version/dtype/shape checks |
| `LoopbackPrefixCache` | Store/retrieve via `InMemoryKVStorageBackend` or `FileKVStorageBackend` |
| `build_remote_placeholder_entry` | Metadata-only remote placeholder (blocked) |

**Not wired** into `ExactKVGenerator` or `VerificationEngine`.

---

## 3. Loopback / mock semantics

| Property | Behavior |
|---|---|
| Transport | **None** — uses existing local storage backends |
| Mode | `PrefixCacheMode.LOCAL_LOOPBACK` |
| Status | `PrefixCacheStatus.LOOPBACK_MOCK` |
| Payload | Tiny synthetic `torch` tensors only (smoke tests) |
| Restore | `PrefixRestorePlan.restore_allowed` only when compatibility passes |

The loopback mock answers: *Can ExactKV represent prefix identity and round-trip a handle safely?* — not *Does remote prefix caching work in production?*

---

## 4. What is not implemented

- Real network RPC or remote LMCache client
- vLLM paged-KV prefix export
- Multi-process distributed drafter/verifier
- Prefix cache wired into generation loop
- Throughput, latency, or memory benefit claims
- Production serving or batching

---

## 5. Why this is not LMCache integration

Phase 11G defined **LMCache prototype gates** (`LMCACHE_FUTURE` mode is metadata only). Phase 11H reuses **storage manager** backends locally — it does **not** import LMCache or call LMCache APIs.

---

## 6. Why this is not real remote prefix caching

`REMOTE_PLACEHOLDER` entries are **blocked** (`remote_placeholder_active` must stay `False`). No socket, HTTP, or LMCache remote tier is exercised. Claims about remote prefix **runtime** remain forbidden.

---

## 7. What this does not prove

| Claim | Status |
|---|---|
| Speedup / latency / throughput | **Not claimed** |
| Active memory savings | **Not claimed** |
| Remote prefix cache runtime | **Not claimed** |
| Production serving readiness | **Not claimed** |
| VeriCache throughput reproduction | **Not claimed** |

---

## 8. Compatibility and restore rules

| Check | On mismatch |
|---|---|
| `model_id` | `restore_allowed=False`, `fallback_required=True` |
| `tokenizer_id` | same |
| `prefix_token_count` | same |
| `cache_version` | same |
| `prompt_hash` / `token_ids_hash` | same |
| `dtype_summary` / `shape_summary` (when present) | same |

`EXPERIMENTAL_ACTIVE` status is **forbidden** in Phase 11H.

---

## 9. How Stage 8+ build on this

| Stage | Connection |
|---|---|
| **Stage 8** — Throughput harness | Diagnostic timing only; no speedup claim without harness |
| **LMCache prototype runtime** (future) | Would replace loopback with real client behind same identity contract |
| **Generator wiring** (future phase) | Would consume `PrefixRestorePlan` only after exactness gates |

---

## 10. JSON schema (identity)

```json
{
  "model_id": "qwen-0.5b",
  "tokenizer_id": "qwen-tokenizer",
  "prompt_hash": "abc123",
  "token_ids_hash": "def456",
  "prefix_token_count": 128,
  "dtype_summary": "torch.float32",
  "shape_summary": "[(1, 2, 4)]",
  "cache_version": "1"
}
```

---

## 11. Claims boundary

| Allowed | Forbidden |
|---|---|
| Prefix identity + loopback round-trip on tiny tensors | Remote prefix tier is live |
| Compatibility / restore plan metadata | LMCache or vLLM integrated |
| Remote placeholder semantics documented | Speed / latency / throughput improvement |
| Fallback required on mismatch | Active memory savings |
| | Production serving readiness |
