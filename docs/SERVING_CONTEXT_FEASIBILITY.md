# Serving-Context Feasibility Research (V8 Phase A)

**Status:** Phase A complete — research document only; no code, no experiments.
**Builds on:** [`docs/V8_SCOPE_STATEMENT.md`](V8_SCOPE_STATEMENT.md) Phase 0;
ExactKV `v0.7.0` (HF-centric runtime, `exactkv_failures == 0` on Experiments
005–006C).
**Purpose:** Decide whether vLLM, LMCache, or PagedAttention-style cache management
can safely support ExactKV's verified compressed-KV workflow, and recommend a
Phase B path.

> ExactKV does **not** implement vLLM, LMCache, PagedAttention, CUDA/Triton
> kernels, KIVI, KVQuant, TurboQuant+, or Sparse V in Phase A.
> External throughput/latency figures cited below are **attributed to those
> projects' papers** — they are **not** ExactKV results.
> This document makes **no** speedup, throughput, latency, runtime, or production
> readiness claim.

---

## 1. Purpose

Answer the V8 §7 serving-context questions for each candidate stack and determine:

1. Whether **full authoritative KV** remains available for verification.
2. Whether **compressed KV** can live separately from authoritative full KV on the
   draft path.
3. Whether **logical vs physical sequence length** can be preserved and mapped.
4. Whether direct stack integration is **go** or **no-go** for Phase C.
5. What **Phase B** should build if direct integration is deferred.

Phase A is **feasibility research only**. It does not prove compatibility; Experiment
007 (Phase D) would validate any harness or PoC.

---

## 2. ExactKV serving-context requirements

Derived from [`exactkv/runtime/exactkv_generator.py`](../exactkv/runtime/exactkv_generator.py),
[`exactkv/verification/engine.py`](../exactkv/verification/engine.py),
[`exactkv/cache/full_state.py`](../exactkv/cache/full_state.py),
[`exactkv/cache/compressed_state.py`](../exactkv/cache/compressed_state.py), and
[`docs/BACKEND_ADAPTER_INTERFACE.md`](BACKEND_ADAPTER_INTERFACE.md).

### Hard invariants

| # | Requirement | Source |
|---|---|---|
| R1 | `exactkv_output_ids == full_output_ids` under greedy decoding | Global exactness gate |
| R2 | `exactkv_failures == 0` on every experiment cell | Global gate |
| R3 | **Verification uses authoritative full-precision KV only** | `VerificationEngine`; verify path never compressor-aware |
| R4 | **Authoritative `FullKVState` is not mutated by draft or verify** | `verify_sequential` deep-copies cache; `_draft` deep-copies materialized KV |
| R5 | **Cache alignment:** `full_state.seq_len == compressed.logical_seq_len` at round boundaries | `_assert_alignment` every round |
| R6 | **Commit advances authoritative full KV**; compressed state refreshed from new full state | `_commit` → `update_after_commit` |
| R7 | **Draft uses materialized compressed KV only** | `_draft` → `materialize_for_draft` |
| R8 | **HF-compatible `past_key_values`** for draft/verify forward (tuple, DynamicCache v4/v5) | `cache/utils.py` |
| R9 | **V5 workspace fields** honest per compressor; simulated labelling preserved | V5/V7 policy |
| R10 | **No verification-engine or generation-logic changes** without re-validation | V8 scope |

### Soft requirements (serving context)

| # | Requirement |
|---|---|
| S1 | Identify **cache owner** for authoritative vs draft stores |
| S2 | Support **logical sequence length** for acceptance bookkeeping |
| S3 | If physical KV length differs (pruning, paging), **map blocks to logical positions** |
| S4 | **Hook safety** on verification model (V6 kvpress lesson: 0 hooks during verify) |
| S5 | **Deterministic materialize** for draft path |

**Rejection rule (V8 §7):** If full authoritative KV cannot be accessed safely for
verification, direct integration must be **rejected or deferred**.

---

## 3. vLLM feasibility analysis

### Architecture (research summary)

vLLM ([Kwon et al., SOSP 2023](https://arxiv.org/abs/2309.06180)) manages KV cache
as **fixed-size blocks** allocated from a GPU pool. PagedAttention maps each
sequence's **logical tokens** to **physical block slots** via a **block table**.
The engine owns block allocation, prefix caching, and in-place decode appends.
Serving is **multi-request**, **GPU-resident**, and **not** organized as a single
Hugging Face `DynamicCache` per ExactKV round.

External papers report throughput/latency improvements for **their** serving
workloads. ExactKV does not reproduce or claim those numbers.

### Cache ownership

| Store | Owner in vLLM | ExactKV need |
|---|---|---|
| Paged KV blocks | vLLM scheduler / `CacheEngine` | Draft materialization only (if used) |
| Block table | Per-sequence metadata in vLLM | Logical→physical mapping |
| Authoritative full HF KV | **Not a first-class export** | **Required** for verify (R3) |

vLLM optimizes for **block reuse and fragmentation avoidance**, not for exporting
a contiguous full-precision KV tensor per token position for an external verifier.

### Access to blocks, block tables, logical positions

vLLM internals expose block tables and slot mappings to kernels and the worker
loop. **Public stable APIs for exporting per-layer K/V tensors at full precision
for an arbitrary external verification loop are not designed for this use case.**

Possible integration patterns considered:

| Pattern | Feasibility | Risk |
|---|---|---|
| A. Run entire ExactKV loop inside vLLM worker | **Low** | Would require forking vLLM generation; violates R10; breaks HF `ExactKVGenerator` |
| B. Export paged blocks → assemble HF `DynamicCache` each round | **Low–Medium** | Assembly cost; block layout ≠ HF layout; version fragility; GPU sync |
| C. **Dual runtime:** vLLM for draft cache only; parallel HF `ModelRuntime` for verify/commit | **Medium** | Two models in memory; drift if weights/config diverge; not true "serving integration" |
| D. vLLM produces cache snapshot once; ExactKV runs offline on snapshot | **Medium** | Point-in-time only; no lifecycle; limited serving-context value |

### Full authoritative KV for verification

**Finding:** vLLM does **not** natively expose an authoritative full-precision KV
state in the Hugging Face `FullKVState` shape that `VerificationEngine` expects,
updated token-by-token through ExactKV's `_commit` loop, without either:

- reconstructing full KV from paged blocks (non-trivial, model-specific), or
- maintaining a **separate HF authoritative model** (Pattern C — not vLLM-owned KV).

Pattern C preserves R3 but **does not evaluate vLLM-owned authoritative KV** —
it evaluates compressors against a parallel HF baseline while vLLM might only
supply draft-side bytes. That weakens the serving-context claim.

### Cache mutation during generation

vLLM **mutates** paged KV during decode (append tokens, prefix cache updates,
potential eviction). ExactKV **also** mutates authoritative KV on commit and
deep-copies for verify/draft.

If a single shared model/cache were used, **draft forward passes and vLLM decode
would race on the same block table** unless strictly isolated (separate sequences,
separate engines, or separate model copies — as in V6 kvpress `deepcopy` isolation).

Experiment 005 showed that sharing one model between compression hooks and
verification **fails hook-safety** unless `isolate_compression_model=True`.
vLLM's integrated worker loop is **higher risk** than kvpress's scoped context
manager.

### vLLM summary

| Criterion | Assessment |
|---|---|
| R3 full authoritative KV from vLLM | **Not safely available** without reconstruction or dual runtime |
| R4/R5 alignment with paged physical ≠ logical | **Non-trivial**; prefix caching adds sharing |
| R8 HF `past_key_values` from vLLM blocks | **Requires adapter layer** not present today |
| Hook / mutation safety | **High risk** on shared worker |
| Dependency / version fork | **High** (vLLM pins torch/CUDA/transformers differently from ExactKV) |

---

## 4. LMCache feasibility analysis

### Architecture (research summary)

LMCache ([Liu et al., 2025](https://arxiv.org/abs/2510.09665)) is an **external
KV-cache tier** that offloads, reuses, and shares KV across queries and engines
(CPU, disk, remote). It typically sits **beside** a serving engine (often vLLM)
via a connector API — disaggregated prefill/decode, cache lookup/store, and
tiered retention.

External papers report throughput gains when combined with vLLM in **their**
workloads. ExactKV does not claim those results.

### Cache ownership

| Layer | Owner | Implication |
|---|---|---|
| GPU paged KV | Serving engine (vLLM) | Same issues as §3 |
| LMCache tier | LMCache controller + storage backend | Async offload/restore |
| Authoritative full KV for verify | **Split across tiers** | R3 harder than HF-only |

LMCache introduces **additional ownership**: KV may be **absent from GPU** (offloaded),
**shared across requests** (prefix hit), or **stale relative to logical sequence**
if restore timing differs from ExactKV commit boundaries.

### Full authoritative KV availability

**Finding:** LMCache optimizes for **reuse and disaggregation**, not for guaranteeing
a synchronous, full-precision, per-token authoritative KV image at each ExactKV
verification step.

Risks:

- **Async restore** — verify may see partial or tier-mismatched KV.
- **Shared cache entries** — logical sequence may map to shared physical prefix;
  commit append may invalidate shared views.
- **Compression at rest** — LMCache may store KV in engine-specific layouts; not
  necessarily full fp16/fp32 tensors per layer.

LMCache **depends on vLLM** (or similar) for typical integration — it inherits §3
block-model constraints and adds tiering complexity.

### Compressed KV store feasibility

A separate ExactKV `CompressedKVState` is **conceptually compatible** if LMCache
only holds **engine-native** KV and ExactKV compressed store is **application-side**
(as today). But LMCache may **also** cache compressed or quantized KV in future
connector modes — collision risk with ExactKV honesty labelling unless explicitly
separated.

### LMCache summary

| Criterion | Assessment |
|---|---|
| R3 authoritative full KV | **Worse than vLLM alone** due to tiering/async |
| Logical/physical mapping | **Engine + LMCache metadata**; not ExactKV-controlled |
| Verification safety | **High risk** without strict GPU-resident authoritative copy |
| Integration dependency | Requires vLLM (or equivalent) + LMCache pins + connector API stability |

---

## 5. PagedAttention context analysis

PagedAttention is the **block-mapping model** underlying vLLM — not a standalone
library ExactKV would import in Phase B.

### Concepts relevant to ExactKV

| Concept | Meaning | ExactKV precedent |
|---|---|---|
| **Logical sequence length** | Tokens in prompt + committed generation | `FullKVState.seq_len`, `CompressedKVState.logical_seq_len` |
| **Physical KV length** | Slots/blocks actually stored | kvpress KnormPress: physical `<` logical (Experiment 005) |
| **Block table** | Maps logical token ranges → physical block IDs | Not in ExactKV today; needed for serving lifecycle |
| **Prefix sharing** | Multiple logical sequences share physical prefix blocks | Out of V8 single-request scope; complicates ownership |
| **Append on decode** | New token → new slot or block extension | Maps to `_commit` forward passes |

### Can PagedAttention preserve logical vs physical length?

**Yes, as a data model** — if ExactKV maintains:

- `logical_seq_len` = token count (alignment invariant R5)
- `physical_kv_len` = sum of resident block slots (may be `<` logical under pruning)
- `block_table: list[int]` mapping logical positions → block slots

Experiment 005 already proved **logical alignment can hold while physical length
shrinks** (kvpress: logical 5, physical 2). PagedAttention generalizes this to
fixed-size pages instead of token-dropping.

**Finding:** PagedAttention **concepts are feasible to simulate locally** without
vLLM. A Phase B harness can implement block tables and physical/logical split
using existing HF `DynamicCache` or tuple caches underneath — modelling serving
lifecycle without claiming production serving.

---

## 6. Restricted local harness option

### Design sketch (Phase B — not implemented in Phase A)

A **ServingContextHarness** (name illustrative) would:

1. **Wrap** existing `ExactKVGenerator` — no verification/generation logic change.
2. **Model** block/page allocation over HF KV tensors (or side metadata only).
3. **Track** `logical_seq_len`, `physical_kv_len`, `block_table`, `cache_owner`.
4. **Store** compressed KV in ExactKV `CompressedKVState` separately from
   authoritative `FullKVState` (already true).
5. **Simulate** lifecycle events: prefill allocate, decode append, optional
   page eviction / prune (kvpress-like), optional tier-offload flag (LMCache-like
   **label only**, no real remote I/O).
6. **Assert** R5 alignment and R3 verify path unchanged.

### Why harness-first

| Advantage | Detail |
|---|---|
| Preserves exactness gate | Same `VerificationEngine`, same `_commit` |
| No vLLM/LMCache dependency | Default install unchanged |
| Models Experiment 005 lesson | physical ≠ logical with alignment |
| Extends V5 accounting | Add `cache_owner`, `physical_kv_len` as additive report labels |
| Reversible | If Phase C ever approved, harness interfaces could wrap real block export |

### What harness is not

- Not production serving
- Not vLLM or LMCache
- Not a performance benchmark
- Not packed-bit or GPU profiling (unless separately approved)

---

## 7. Cache ownership comparison

| Context | Authoritative full KV owner | Draft / compressed KV owner | Verify path owner |
|---|---|---|---|
| **ExactKV today (HF)** | `FullKVState` in `ExactKVGenerator` | `CompressedKVState` via compressor | `VerificationEngine` on `FullKVState` |
| **vLLM** | Paged block pool (engine) | Same pool if shared — **conflict** | Would need HF copy or block assembly |
| **LMCache** | Split: GPU blocks + external tier | Connector + engine | Same as vLLM + restore timing risk |
| **Phase B harness** | `FullKVState` (unchanged) | `CompressedKVState` + harness metadata | `VerificationEngine` (unchanged) |

**Conclusion:** Only the HF baseline and the proposed harness keep authoritative
full KV under ExactKV control end-to-end.

---

## 8. Full authoritative KV availability

| Candidate | Available safely for verify? | Notes |
|---|---|---|
| HF `ModelRuntime` (baseline) | **Yes** | All V1–V7 experiments |
| vLLM paged cache | **No** (without reconstruction or dual runtime) | Not designed for external sequential verify |
| LMCache tiers | **No** | Async + shared + offloaded |
| Dual HF model + vLLM draft | **Partial** | Verify safe on HF copy; **not** vLLM authoritative KV |
| Phase B harness | **Yes** | Authoritative path unchanged |

---

## 9. Compressed KV store feasibility

| Candidate | Separate compressed store? | Notes |
|---|---|---|
| HF + `KVCompressor` | **Yes** | Current design |
| vLLM | **Only if** ExactKV compressor wraps exported/assembled blocks | High adapter cost |
| LMCache | **Theoretically** if LMCache stores engine KV and ExactKV stores app-side compressed | Tier collision risk |
| Harness | **Yes** | `CompressedKVState` unchanged; harness adds lifecycle metadata |

ExactKV's separation of **compress → materialize_for_draft** vs **verify on full**
maps cleanly to serving **if** authoritative full KV remains HF-controlled.

---

## 10. Verification safety risks

| Risk | vLLM | LMCache | Harness |
|---|---|---|---|
| Shared cache mutation during verify | **High** | **High** | **Low** (deep-copy unchanged) |
| Hook contamination on verify model | **High** (integrated worker) | **High** (via engine) | **Low** |
| Non-deterministic materialize | Medium (kernel paths) | **High** (async tier) | Low (controlled sim) |
| `logical_seq_len` drift | Medium (block accounting bugs) | **High** | Low (explicit asserts) |
| Version/API breakage | **High** | **High** | Low (internal) |

V6 kvpress mitigation (`isolate_compression_model`, `verification_mode`) is
**necessary but not sufficient** for vLLM/LMCache — the serving loop owns scheduling
and cache mutation outside ExactKV's control.

---

## 11. Logical vs physical sequence mapping

| System | Logical length | Physical length | Mapping |
|---|---|---|---|
| ExactKV HF | `full_state.seq_len` | `kv_seq_len(past_key_values)` — usually equal | 1:1 |
| kvpress (Exp 005) | `logical_seq_len` | physical `<` logical | Pruned `DynamicCache` |
| PagedAttention | Token count in sequence | Occupied blocks × block_size (≥ logical slots) | Block table |
| Harness (proposed) | `logical_seq_len` (R5) | `physical_kv_len` + `block_table` | Explicit metadata |

**Finding:** A harness can generalize the Experiment 005 physical/logical split
to page-oriented serving semantics **without** vLLM.

---

## 12. Workspace-memory implications

| Context | `stored_kv_bytes` | `materialized_working_kv_bytes` | Honesty |
|---|---|---|---|
| HF baseline | Compressor stats | Usually `full_kv_bytes` | Established V5 |
| vLLM blocks | Would need block-pool byte accounting | Assembly to full for attention — likely still full-sized working set | New adapter; easy to overclaim |
| LMCache | GPU + offloaded tiers | Restore → materialize | Tier bytes ≠ ExactKV compressor bytes |
| Harness | Existing compressor stats + optional `metadata_bytes` for block table | Unchanged unless harness simulates tier | Add `cache_owner` label |

**Rules unchanged:**

- `total_kv_footprint_bytes` = conservative accounting sum, **not** measured peak GPU.
- Simulated compressors: int8 containers; no packed-bit claim.
- Do not cite vLLM/LMCache external memory figures as ExactKV results.

---

## 13. Active GPU memory profiling feasibility

Phase A assessment:

| Prerequisite | Status |
|---|---|
| Real serving stack integrated | **Not feasible** in Phase C per this research |
| Stable GPU resident authoritative KV | HF dual-runtime only — optional, not serving-owned |
| Isolated measurement checkpoint | Easier on HF baseline or harness than vLLM worker |

**Recommendation:** Defer active GPU profiling to **optional post-harness** sub-phase
(Phase D or later) on HF/harness path. Not a blocker for Phase B.

If pursued later: follow V8 §10 requirements (`torch.cuda.memory_reserved`, documented
hardware, separate from `total_kv_footprint_bytes`).

---

## 14. Environment and dependency risks

| Risk | vLLM | LMCache | Harness |
|---|---|---|---|
| transformers version fork | **High** (vLLM pins differ from ExactKV 5.8.x) | Inherits vLLM | **None** |
| CUDA / torch pins | **High** | **High** | Default ExactKV env |
| Optional isolated venv | Required (like `.venv-kvpress`) | Required | Not required |
| API stability | Worker internals change frequently | Young connector API | Internal only |
| CI | GPU + heavy deps | Worse | CPU smoke viable |

ExactKV default install must remain **free of vLLM/LMCache imports** (same principle
as kvpress isolation).

---

## 15. Go/no-go recommendation for vLLM

### Phase C vLLM PoC: **NO-GO** (deferred)

**Rationale:**

1. Full authoritative KV for `VerificationEngine` is **not safely available** from
   vLLM-owned paged cache without non-trivial reconstruction or a parallel HF
   authoritative model (which does not validate vLLM-owned verify path).
2. Shared worker mutation and hook risk **exceed** V6 kvpress isolation lessons.
3. Dependency/version fork is **high** for marginal serving-context evidence.
4. Pattern C (dual HF runtime) is **not recommended** as a vLLM Phase C deliverable —
   it evaluates compressors on HF, not vLLM cache lifecycle.

**Deferred path (explicit re-approval required):** Read-only export of vLLM block
layout for **metadata-only** Experiment 007 rows (`experiment_class: vllm_layout_observed`)
with ExactKV verify still on HF authoritative path. **Not approved in Phase A.**

---

## 16. Go/no-go recommendation for LMCache

### Phase C LMCache PoC: **NO-GO** (deferred)

**Rationale:**

1. Inherits all vLLM block-model issues (§15).
2. Adds **tiering, async offload, and shared-cache** semantics that conflict with
   synchronous authoritative KV required at each verify step (R3, R4).
3. Connector API and storage layout are **not aligned** with ExactKV `FullKVState`.
4. Integration complexity **exceeds** value for V8 compatibility goals.

**Deferred:** Document LMCache as **incompatible** with current ExactKV verify semantics
unless a future scope approves HF-authoritative sidecar with LMCache as non-authoritative
tier only.

---

## 17. Recommendation for Phase B

### Primary path: **Restricted local serving harness** (V8 Phase B)

Implement Phase B as specified in V8 §6:

| Deliverable | Description |
|---|---|
| Cache-lifecycle simulator | Block table, logical/physical lengths, append/evict simulation |
| Exactness smoke gate | 2+ prompts, 2 draft lengths, `exactkv_failures == 0` |
| Additive metadata | `cache_owner`, `physical_kv_len`, optional `block_table` summary in reports |
| Experiment 007 Mode B | Harness-based serving-context evaluation (default) |

### Phase C

**Skip vLLM and LMCache PoC** unless V8 scope is revised after Phase B harness results.
Default Experiment 007 = **Mode B (harness-only)**.

### Phase B must not

- Import vLLM or LMCache
- Change `VerificationEngine` or `_commit` semantics
- Claim production serving or performance
- Imply packed-bit savings for `_sim` compressors

### Phase B should

- Reuse `BackendAdapter` / `KVCompressor` boundary for draft path only
- Generalize Experiment 005 physical `<` logical invariant to page semantics
- Document incompatibility with direct vLLM/LMCache in Experiment 007 report

---

## 18. What Phase A does not prove

- **No compatibility proof** — research conclusions only; Experiment 007 validates.
- **No vLLM or LMCache integration** — not implemented.
- **No performance measurements** — external paper numbers are cited for context only.
- **No GPU profiling** — deferred.
- **No change** to `exactkv_failures == 0` on existing experiments.
- **No claim** that harness simulation equals production vLLM behaviour.
- **No claim** that Phase B replaces real serving stacks — it models lifecycle concepts.

---

## Summary table

| Candidate | Phase C go/no-go | Authoritative full KV | Phase B role |
|---|---|---|---|
| vLLM | **NO-GO** | Not safely available | Concepts inform harness block model |
| LMCache | **NO-GO** | Worse (tiering) | Tier-offload **simulation flag** only |
| PagedAttention context | N/A (not imported) | Via harness metadata | **Core harness model** |
| HF baseline | N/A (control) | **Yes** | Unchanged reference |
| Local harness | **GO** (Phase B) | **Yes** | **Primary deliverable** |

---

## Related documents

| Document | Role |
|---|---|
| [`V8_SCOPE_STATEMENT.md`](V8_SCOPE_STATEMENT.md) | Phase definitions and §7 questions |
| [`EXPERIMENT_005_KVPRESS_KNORM.md`](EXPERIMENT_005_KVPRESS_KNORM.md) | physical/logical split precedent |
| [`BACKEND_ADAPTER_INTERFACE.md`](BACKEND_ADAPTER_INTERFACE.md) | Draft-only adapter boundary |
| [`RELATED_WORK_KV_CACHE_COMPRESSION.md`](RELATED_WORK_KV_CACHE_COMPRESSION.md) | vLLM/LMCache survey (external claims attributed) |

## Attribution

- vLLM / PagedAttention: Kwon et al., arXiv:2309.06180, 2023
- LMCache: Liu et al., arXiv:2510.09665, 2025
- VeriCache / ExactKV algorithm: Yao et al., arXiv:2605.17613, 2026

External throughput and latency figures belong to those systems, **not** ExactKV.
