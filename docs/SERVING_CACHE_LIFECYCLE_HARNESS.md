# Serving Cache Lifecycle Harness (V8 Phase B)

**Status:** Phase B complete — local compatibility harness; no vLLM/LMCache integration.
**Builds on:** [`docs/SERVING_CONTEXT_FEASIBILITY.md`](SERVING_CONTEXT_FEASIBILITY.md) (Phase A
no-go for direct stack integration); ExactKV `v0.7.0`.
**Code:** `exactkv/serving/cache_lifecycle.py`

> This is a **local compatibility harness**, not vLLM or LMCache integration.
> It does **not** claim production serving behavior.
> It does **not** measure throughput, latency, speedup, runtime, or tokens/sec.
> It does **not** report active GPU memory.
> It preserves ExactKV's exactness gate (`exactkv_failures == 0`).

---

## 1. Purpose

Provide a **restricted local serving-context harness** that models how production
serving systems think about KV cache ownership and lifecycle — while wrapping
ExactKV's existing `FullKVState` and `CompressedKVState` without changing
`ExactKVGenerator` or `VerificationEngine`.

Phase B answers: *can we represent serving-style cache semantics alongside
ExactKV's draft-verify-commit loop without breaking alignment or exactness?*

---

## 2. Why a local harness after Phase A no-go

Phase A concluded **no-go** for vLLM and LMCache Phase C integration:

- vLLM does not expose authoritative full-precision KV in the shape
  `VerificationEngine` requires.
- LMCache tiering/async semantics worsen ownership clarity.
- Direct stack integration risks hook safety, block-mapping drift, and
  exactness regressions.

Phase B still delivers serving-context **evaluation value** by simulating
PagedAttention-*style* concepts (logical vs physical length, block tables,
separate stores) in a deterministic local package with zero external serving
dependencies.

---

## 3. What the harness models

| Concept | Harness representation |
|---|---|
| Authoritative full KV | `authoritative_full` entry from `FullKVState` |
| Compressed/draft KV | `compressed_draft` entry from `CompressedKVState` |
| Logical sequence length | `logical_seq_len` per entry |
| Physical sequence length | `physical_seq_len` per entry |
| Block/page mapping | `CacheBlock` table (`logical_*`, `physical_*` ranges) |
| Append after commit | `append_committed_tokens(count)` |
| Lifecycle invariants | `validate_invariants()` |
| Memory honesty | V5 fields in `summarize()` per entry |

---

## 4. What it does not model

- vLLM scheduler, LMCache tiers, or real PagedAttention kernels
- Multi-request batching, continuous batching, or request preemption
- GPU residency, CUDA kernels, CPU offload
- Throughput, latency, speedup, `runtime_seconds`, or tokens/sec
- Active GPU memory measurement (`active_gpu_kv_bytes`)
- Changes to generation, verification, compressor registry, or report schemas
- Production-serving or production-readiness claims

---

## 5. Cache ownership model

Three owner constants:

| Owner | Role |
|---|---|
| `authoritative_full` | Ground-truth KV for verification and commit |
| `compressed_draft` | Lossy/materialised draft path only |
| `serving_harness` | Metadata owner for the harness summary itself |

**Invariants enforced:**

- Authoritative full and compressed draft are **separate entries**.
- Verification conceptually uses `authoritative_full` only.
- `compressed_draft` **cannot replace** an `authoritative_full` entry.
- At round boundaries, both entries share the same `logical_seq_len` when both
  are registered.

---

## 6. Logical vs physical sequence length

| Case | Behaviour |
|---|---|
| `physical_seq_len == logical_seq_len` | Identity mapping; blocks cover matching ranges |
| `physical_seq_len < logical_seq_len` | **Requires** explicit `retained_logical_positions` |
| Missing retained positions when pruned | **Fails loudly** — no fabricated mapping |

`retained_logical_positions` is a strictly increasing list of length
`physical_seq_len`, mapping each physical slot to a logical prompt position.
This mirrors pruned-cache backends (e.g. restricted kvpress KnormPress) where
logical bookkeeping continues while physical KV is shorter.

---

## 7. Block/page mapping

`CacheBlock` records:

- `block_id` — sequential block index
- `logical_start`, `logical_end` — half-open logical range
- `physical_start`, `physical_end` — half-open physical range

Default `block_size` is 16 (configurable on harness construction). Blocks
partition the physical sequence; logical ranges are derived from identity or
retained-position mapping.

`validate_invariants()` rejects negative or reversed block ranges.

---

## 8. Memory-honesty fields

`summarize()` and per-entry metadata include when available:

| Field | Meaning |
|---|---|
| `stored_kv_bytes` | Persistent compressed/quantised representation |
| `materialized_working_kv_bytes` | Working copy during attention |
| `total_kv_footprint_bytes` | Conservative accounting sum |
| `supports_real_bytes_claim` | From compressor capabilities |
| `is_simulated` | From compressor capabilities |
| `note` | Harness + simulated-storage disclaimers |

`total_kv_footprint_bytes` is a **conservative accounting sum**, not measured
peak GPU memory. Simulated (`_sim`) compressors use int8 containers — not real
packed-bit storage.

---

## 9. How it relates to vLLM/LMCache/PagedAttention context

| External concept | Harness analogue |
|---|---|
| PagedAttention block table | `CacheBlock` list |
| Logical token positions | `logical_seq_len` + retained mapping |
| Physical block slots | `physical_seq_len` + block ranges |
| Separate draft vs verify stores | `compressed_draft` vs `authoritative_full` |

Naming vLLM, LMCache, or PagedAttention here is **context only**. ExactKV does
**not** implement those systems in Phase B.

---

## 10. Tests and gates

**Test file:** `tests/test_serving_cache_lifecycle.py`

Covers:

- Identity and pruned logical/physical mapping
- Ownership separation and replacement guards
- `append_committed_tokens` alignment
- Block-range invariant validation
- State non-mutation on registration
- `k8_v4_sim` and `backend_passthrough` integration
- Smoke ExactKV run with harness around prefill/compress lifecycle
- Forbidden performance field audit

**Hard gate:** smoke ExactKV run with harness must satisfy
`exactkv_output_ids == full_output_ids` (equivalent to `exactkv_failures == 0`
for that cell).

Run:

```bash
pytest tests/test_serving_cache_lifecycle.py -v
```

---

## 11. Limitations

- Harness tracks **metadata** about lifecycle; it does not allocate GPU blocks.
- After commit, pruned physical length is not auto-refreshed — re-register
  compressed state after `update_after_commit` in full workflows.
- No multi-request or cross-request cache sharing.
- No Experiment 007 sweep in Phase B — that is Phase D.

---

## 12. How it will feed Experiment 007

Experiment 007 **Mode B** (default per Phase A) will:

1. Run the core prompt suite through ExactKV with the harness wrapping each
   prefill/compress/commit lifecycle.
2. Record per-cell lifecycle summaries alongside existing acceptance/exactness
   metrics.
3. Label results `experiment_class: harness_sim`.
4. Require `exactkv_failures == 0` and honest workspace-memory tables.

Phase C (vLLM/LMCache PoC) remains **deferred/no-go** unless separately
re-approved.

---

## API sketch

```python
from exactkv.serving import ServingCacheLifecycleHarness

harness = ServingCacheLifecycleHarness(block_size=16)
harness.register_authoritative_full(full_state)
harness.register_compressed_cache(compressed, compressor=compressor)
harness.validate_invariants()

# After each commit round:
harness.append_committed_tokens(num_committed)
harness.validate_invariants()

summary = harness.summarize()  # JSON-serialisable; no performance fields
```

---

## Related documents

| Document | Relevance |
|---|---|
| [`V8_SCOPE_STATEMENT.md`](V8_SCOPE_STATEMENT.md) | V8 phases and exit criteria |
| [`SERVING_CONTEXT_FEASIBILITY.md`](SERVING_CONTEXT_FEASIBILITY.md) | Phase A no-go rationale |
| [`BACKEND_ADAPTER_INTERFACE.md`](BACKEND_ADAPTER_INTERFACE.md) | Adapter boundary |
| [`EXPERIMENT_005_KVPRESS_KNORM.md`](EXPERIMENT_005_KVPRESS_KNORM.md) | Pruned physical < logical precedent |
