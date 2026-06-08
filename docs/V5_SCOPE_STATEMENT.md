# V5 Scope Statement: Workspace-Aware Memory Accounting

**Status:** Approved scope for V5 implementation.
**Supersedes:** `docs/V5_SCOPE_DRAFT.md` (draft only, not approved).
**Builds on:** `v0.4.0` — asymmetric K/V compressor experiments, Experiment 003.

> No V5 code is included in this document. This is a scope statement only.
> No real compressor backends are implemented in V5.
> No performance, throughput, latency, speedup, or production-readiness claims.

---

## 1. V5 goal

Add **workspace-aware memory accounting** to ExactKV so that reported memory
figures honestly distinguish *stored compressed bytes*, *materialized working
bytes*, *compression metadata*, and *temporary scratch*, rather than collapsing
everything into a single `compressed_kv_bytes` number that can mislead.

Optionally, lay the groundwork for a **real backend adapter interface** (design
and documentation only — no real backend implementation without separate
approval).

---

## 2. Why V5 matters after V4

V4 proved that key compression is far more damaging to ExactKV acceptance than
value compression. The next natural question is: **what does this mean for memory
usage?** V4's memory accounting has two limitations that V5 fixes:

**Limitation 1 — `compressed_kv_bytes` hides materialization cost.**
A compressor that stores 4-bit values in `int8` containers reports a certain
number of stored bytes. But when those values are dequantized during attention,
a full-precision (float32 or bfloat16) working copy must be materialised
transiently. The stored byte count understates the true peak memory footprint.

**Limitation 2 — simulated compressors claim int8 storage, not packed storage.**
V4's `_sim` compressors (e.g. `k8_v4_sim`) correctly report `is_simulated=True`
and `supports_real_bytes_claim=False`, but the single `compressed_kv_bytes` field
does not explain *why* the claim is limited. V5 makes this explicit: the field
names themselves distinguish stored from materialized.

**V4 opened the question. V5 answers it honestly.**

### Related-work motivation

The need for workspace-aware accounting is reinforced by how real KV-cache
backends actually use memory (see
[`docs/RELATED_WORK_KV_CACHE_COMPRESSION.md`](RELATED_WORK_KV_CACHE_COMPRESSION.md)):

* **KIVI** keeps a full-precision residual alongside its 2-bit grouped cache, so
  stored bytes understate the working footprint.
* **KVQuant** carries per-channel scales, non-uniform datatypes, and a sparse
  outlier side-channel — all of which are metadata beyond the quantised tensors.
* **Palu** and **KVTC** reconstruct a dense working cache on the fly from a
  low-rank / transform-coded stored form, plus projection matrices, codebooks, or
  entropy-coder state as metadata.
* **TurboQuant / TurboQuant+** dequantize (and rotate) during decode, and its
  "Sparse V dequant" work exists precisely because decode-time dequantization
  cost is real.
* **PagedAttention (vLLM)** and **LMCache** define the realistic memory hierarchy
  (GPU pages, CPU offload, remote storage) a real backend would live in.

The shared lesson: **stored compressed KV bytes alone are incomplete** when
decode requires:

* materialized working KV (dequantized/reconstructed form used for attention),
* metadata (scales, zero-points, codebooks, projection matrices),
* temporary dequantization workspace, and
* dense scratch buffers.

ExactKV does **not** implement TurboQuant+, KVTC, KIVI, KVQuant, LMCache, or
PagedAttention. They are cited as **motivation** for honest memory accounting.
**V5 stays focused on memory accounting, not backend implementation.**

---

## 3. What V5 adds

### 3.1 Extended `MemorySummary` dataclass

Replace the current flat set of byte fields with a structured, named set:

| New field | Replaces / extends | Meaning |
|---|---|---|
| `stored_kv_bytes` | `compressed_kv_bytes` | Bytes of the KV cache as held in memory between decode steps. For `_sim` compressors this is the int8-container reality. |
| `full_kv_bytes` | `full_kv_bytes` (kept) | Bytes of the full-precision reference cache (unchanged). |
| `materialized_working_kv_bytes` | new | Bytes when the stored cache is dequantized/materialized for use during an attention step. Equals `full_kv_bytes` for all current compressors since they all dequantize to full precision. |
| `metadata_bytes` | new | Bytes of per-tensor compression metadata (scales, zero-points, indices). For symmetric quantization: one float32 scale per tensor. |
| `temporary_workspace_bytes` | new | Estimated transient allocation during `compress()` or `materialize_for_draft()`. For current PyTorch compressors: approximately equals the tensor being processed (one extra copy). |
| `total_kv_footprint_bytes` | new | Conservative peak estimate: `stored_kv_bytes + metadata_bytes + max(materialized_working_kv_bytes, temporary_workspace_bytes)`. |
| `supports_real_bytes_claim` | kept | `True` only when all figures reflect actual storage, not simulation. |
| `memory_claim_note` | kept | Human-readable explanation of any limitation. |

> **Backward compatibility:** The existing `compressed_kv_bytes` field is kept
> as an alias for `stored_kv_bytes` during a transition window so that existing
> consumers (analysis layer, CSV schema, tests) do not break.

### 3.2 Per-compressor `stats()` updated

Each compressor's `stats()` method is extended to populate the new fields.
The interface change is additive: old fields are not removed.

### 3.3 Report schema updated (additive)

New memory fields added to JSON enrichment and CSV flattening. Existing columns
unchanged. Old reports without the new fields load safely (backward compatible).

### 3.4 CLI and Markdown updated

`list-compressors` shows the new memory field names.
Markdown reports include the new memory schema in the "Memory Honesty Notes"
section.

### 3.5 Experiment 004 (see §11)

---

## 4. What V5 explicitly does not add

* ❌ **No real compressor backends.** KIVI, kvpress, TurboQuant, KVQuant, SnapKV
  and similar remain V6 candidates.
* ❌ **No real INT4 or INT2 bit-packing.** All `_sim` compressors continue to
  store sub-INT8 values in `int8` containers.
* ❌ **No vLLM, LMCache, Triton, CUDA kernels, CPU offload.**
* ❌ **No batching, sampling, parallel verification, bonus-token acceptance.**
* ❌ **No timing, latency, throughput, tokens-per-second, speedup, or
  production-readiness claims.**
* ❌ **No workspace-aware GPU profiling** (measuring actual peak device memory is
  a V6 concern; V5 reports *estimated* figures based on tensor sizes).
* ❌ **No new compressors.**
* ❌ **No changes to generation logic or the draft-verify-commit loop.**

---

## 5. Workspace-aware memory schema: field definitions

### 5.1 `stored_kv_bytes`

The number of bytes occupied by the KV cache *as it persists between decode
steps*.

- **`noop`**: full-precision tensors → `stored_kv_bytes == full_kv_bytes`.
- **`int8`**: int8 tensors + float32 scales → smaller than `full_kv_bytes`.
- **`int4_sim`**: int8-container tensors + scales → same storage as `int8`
  (no real packing); `supports_real_bytes_claim=False`.
- **`k8_v4_sim`**, etc.: int8-container for both K and V, with separate K and V
  scales → same total container storage as `int8`; `supports_real_bytes_claim=False`.
- **`k8_v_full`**: int8 K + full-precision V → `stored_kv_bytes` between int8
  and full; `supports_real_bytes_claim=True` (both sides are real).

### 5.2 `full_kv_bytes`

Bytes of the uncompressed full-precision KV cache used as the reference. This is
the same `full_kv_bytes` as in V4. Not changed.

### 5.3 `materialized_working_kv_bytes`

Bytes of the KV cache after dequantization, as needed to perform attention.

For **all current ExactKV compressors**, `materialize_for_draft()` dequantizes
to the original tensor dtype (float32 in CPU tests). Therefore:

```
materialized_working_kv_bytes == full_kv_bytes
```

for every current compressor. This means the *stored* savings do not reduce the
*working* footprint during attention — an important honesty point.

For a hypothetical real backend with a true fused dequantize-and-attend kernel,
this would be smaller. V5 does not implement such a backend; it just names the
field so a future backend can populate it honestly.

### 5.4 `metadata_bytes`

Bytes of per-tensor quantization metadata (scales, zero-points).

For current per-tensor symmetric quantization (one float32 scale per KV tensor
per layer per head):

```
metadata_bytes ≈ num_kv_tensors × sizeof(float32)
```

For `noop` and `debug_noise` (no quantization): `metadata_bytes = 0`.

### 5.5 `temporary_workspace_bytes`

Estimated peak transient allocation during `compress()` or
`materialize_for_draft()`. In PyTorch, an in-place quantize operation that
creates a new tensor allocates approximately one tensor-equivalent of scratch.

For **all current compressors**:

```
temporary_workspace_bytes ≈ stored_kv_bytes
```

(One scratch copy of the tensor during quantize/dequantize. In practice PyTorch
may use more, but this is the conservative minimum estimate.)

### 5.6 `total_kv_footprint_bytes`

A conservative peak estimate:

```
total_kv_footprint_bytes =
    stored_kv_bytes
    + metadata_bytes
    + max(materialized_working_kv_bytes, temporary_workspace_bytes)
```

The `max(...)` term reflects that the working copy and the scratch buffer are not
simultaneously resident; only one dominates at any moment. For current
compressors both equal `full_kv_bytes`, so the estimate is:

```
total_kv_footprint_bytes = stored_kv_bytes + metadata_bytes + full_kv_bytes
```

This is intentionally conservative. A real profiler would give a tighter number.

> **Important (Phase B clarification):** `total_kv_footprint_bytes` is an
> **accounting sum computed from tensor shapes and dtype widths**.  It is
> **NOT a measured peak GPU memory value**.  Actual peak GPU memory depends on
> PyTorch allocator behaviour, attention intermediate buffers, activation
> memory, and other framework overhead that ExactKV does not instrument.
> Active GPU memory measurement is deferred to a later CUDA-specific
> validation phase (post-V5).

---

## 6. How each field is interpreted per compressor type

| Compressor | `stored_kv_bytes` | `materialized_working_kv_bytes` | `metadata_bytes` | `temporary_workspace_bytes` | `supports_real_bytes_claim` |
|---|---|---|---|---|---|
| `noop` | == full | == full | 0 | ≈ 0 | yes |
| `int8` | < full (int8 vs fp32) | == full | one scale/tensor | ≈ stored | yes |
| `int4_sim` ⚠️ | == int8 (no packing) | == full | one scale/tensor | ≈ stored | **no** |
| `k8_v4_sim` ⚠️ | == int8 (no packing) | == full | K scale + V scale | ≈ stored | **no** |
| `k8_v_full` | < full (K int8, V full) | == full | one K scale/tensor | ≈ stored | yes |
| `k_full_v8` | < full (K full, V int8) | == full | one V scale/tensor | ≈ stored | yes |
| `debug_noise` | == full | == full | 0 | ≈ 0 | yes |

> ⚠️ `stored_kv_bytes` for `_sim` compressors reflects int8-container reality,
> not hypothetical packed bytes. Do not compare `_sim` stored bytes against a
> real INT4 backend claiming true packed storage.

---

## 7. How to avoid misleading simulated-memory claims

V5 enforces the following invariants throughout the codebase:

1. **`supports_real_bytes_claim=False` for all `_sim` compressors** — unchanged
   from V4. Never relaxed without a real bit-packing implementation.

2. **`stored_kv_bytes` ≠ hypothetical packed bytes** — `int4_sim` at 4-bit
   would be `full_kv_bytes / 8` if truly packed. V5 must not report that figure.
   It reports int8-container bytes (same as `int8`).

3. **`materialized_working_kv_bytes` is always honest** — for all current
   compressors this equals `full_kv_bytes`. V5 makes this visible rather than
   hiding it behind a single `compression_ratio` number.

4. **`total_kv_footprint_bytes` is conservative** — the formula is documented
   and reproducible. No field may be presented as a memory saving unless
   `supports_real_bytes_claim=True` *and* `materialized_working_kv_bytes` is
   also smaller than `full_kv_bytes`.

5. **Markdown and CLI output** must include a note whenever any memory field
   has `supports_real_bytes_claim=False`, and must not present
   `compression_ratio` as evidence of real memory savings for `_sim`
   compressors.

6. **Average effective bit width remains a comparison aid only** — unchanged
   from V4. It is not a memory estimate and must not be framed as one.

---

## 8. Implementation phases

### Phase 0 — Scope documentation (this document)

Finalise and commit `docs/V5_SCOPE_STATEMENT.md`. No code.

**Exit gate:** This document committed and reviewed. All prior tests green.

---

### Phase A — Extend `MemorySummary` and compressor `stats()`

**Files:**
- `exactkv/benchmarks/runner.py` (or a new `exactkv/benchmarks/memory.py`) —
  add `MemorySummary` dataclass with all new fields.
- `exactkv/compressors/noop.py`, `int8.py`, `int4_sim.py`, `debug_noise.py`,
  `asymmetric_sim.py` — update each compressor's `stats()` to populate new
  fields.

**Design rule:** All new fields are optional with `None` as default; old callers
that ignore the new fields continue to work.

**Exit gates:**
- All compressors populate `stored_kv_bytes`, `materialized_working_kv_bytes`,
  `metadata_bytes`, `temporary_workspace_bytes`, `total_kv_footprint_bytes`.
- `noop`: `stored_kv_bytes == full_kv_bytes`; `metadata_bytes == 0`.
- `int8`: `stored_kv_bytes < full_kv_bytes`; `materialized_working_kv_bytes == full_kv_bytes`.
- All `_sim` compressors: `stored_kv_bytes == int8_container_bytes` (not hypothetical packed); `supports_real_bytes_claim=False`.
- `total_kv_footprint_bytes` formula is unit-tested.
- No forbidden performance fields (`tokens_per_second`, `throughput`, `latency`,
  `speedup`, `runtime_seconds`) appear in any output.

---

### Phase B — Report schema and CSV updates (additive)

**Files:**
- `exactkv/benchmarks/reports.py` — add new memory fields to JSON enrichment
  and CSV flattening.
- Keep `compressed_kv_bytes` as an alias for `stored_kv_bytes` for backward
  compatibility (populated as before, plus the alias).

**Exit gates:**
- JSON round-trip includes all new memory fields.
- CSV has new memory columns; existing columns unchanged.
- Legacy reports missing the new fields load safely (no `KeyError`).
- No forbidden performance fields in schema.

---

### Phase C — CLI and Markdown updates

**Files:**
- `exactkv/cli.py` — `list-compressors` shows new memory field names.
- `exactkv/reporting/markdown.py` — "Memory Honesty Notes" section updated to
  explain stored vs working vs total.
- `exactkv/reporting/leaderboard.py` — optional column for
  `total_kv_footprint_bytes`.

**Exit gates:**
- `list-compressors` output shows new field names.
- Markdown report includes stored vs materialized note.
- No forbidden performance fields in any output.

---

### Phase D — Experiment 004

Run a core-suite sweep and generate
`docs/EXPERIMENT_004_MEMORY_ACCOUNTING.md` using `python -m exactkv report`,
documenting the new memory schema across all current compressors.

**Exit gates:**
- `exactkv_failures == 0` across all runs.
- Report contains `stored_kv_bytes`, `materialized_working_kv_bytes`, and
  `total_kv_footprint_bytes` for every compressor.
- Report includes the "stored ≠ working" honesty note.
- No real-memory-savings claim for `_sim` compressors.
- No forbidden performance fields.

---

### Phase E (optional, separate approval) — Real backend interface design

Design only: document a `BackendAdapter` interface spec describing how a real
quantisation backend (e.g. KIVI or kvpress) would register with the existing
`KVCompressor` protocol, declare `CompressorCapabilities`, and honestly populate
the V5 memory fields.

**No code written in this phase without separate explicit approval.**

---

## 9. Tests and gates per phase

### Global gates (every phase)

- All prior tests (995 at v0.4.0) remain green.
- Primary correctness criterion unchanged:
  `exactkv_output_ids == full_output_ids` under greedy decoding.
- No forbidden performance fields in any code, test, or doc:
  `tokens_per_second`, `throughput`, `latency`, `speedup`, `runtime_seconds`.

### Phase A gates

| Test | Assertion |
|---|---|
| `noop` `stats()` has new fields | `stored_kv_bytes == full_kv_bytes` |
| `int8` materialized bytes | `materialized_working_kv_bytes == full_kv_bytes` |
| `int4_sim` stored bytes | `stored_kv_bytes == int8_container_bytes` (not `/ 4`) |
| All `_sim` compressors | `supports_real_bytes_claim=False` |
| `total_kv_footprint_bytes` formula | `stored + metadata + max(working, scratch)` |
| No hypothetical packed-byte figures | `stored_kv_bytes` never reports `full_kv_bytes / bit_width / 8` |
| No forbidden performance fields | `_assert_no_forbidden_fields` passes on `stats()` output |

### Phase B gates

| Test | Assertion |
|---|---|
| JSON round-trip includes new fields | `key in loaded_report["results"][0]["memory"]` |
| CSV has new columns | `"stored_kv_bytes" in csv_headers` |
| Legacy report loads safely | No `KeyError` when fields absent |
| Backward compat alias | `"compressed_kv_bytes" in csv_row` |

### Phase C gates

| Test | Assertion |
|---|---|
| `list-compressors` shows new fields | Output contains `stored_kv_bytes` |
| Markdown memory section updated | Section contains `materialized_working_kv_bytes` |
| No forbidden performance output | Pattern audit passes on rendered Markdown |

### Phase D gates

| Test | Assertion |
|---|---|
| Experiment 004 report exists | `docs/EXPERIMENT_004_MEMORY_ACCOUNTING.md` present |
| `exactkv_failures == 0` | Hard correctness gate |
| Report contains new fields | Section headings and table columns verified |
| No real-memory claim for `_sim` | Audit passes |

---

## 10. Experiment 004 plan

**Name:** Workspace-Aware Memory Accounting Sweep

**Goal:** Re-run the core-suite sweep with all V4 compressors and render a
Markdown report that surfaces the full V5 memory schema for each compressor,
clearly illustrating stored vs materialized vs total.

**Sweep configuration:**

```bash
python -m exactkv sweep \
  --model Qwen/Qwen2.5-0.5B \
  --suite core \
  --compressors noop,int8,int4_sim,k8_v4_sim,k8_v_full,k_full_v8 \
  --draft-lengths 4,8 \
  --max-new-tokens 24 \
  --json-out reports/experiment_004_memory_accounting.json \
  --csv-out  reports/experiment_004_memory_accounting.csv

python -m exactkv report \
  --report reports/experiment_004_memory_accounting.json \
  --markdown-out docs/EXPERIMENT_004_MEMORY_ACCOUNTING.md \
  --title "Experiment 004: Workspace-Aware Memory Accounting"
```

**What the report should include beyond Experiment 003 format:**

- `stored_kv_bytes` per compressor
- `materialized_working_kv_bytes` per compressor (expected: always equals
  `full_kv_bytes` for current compressors)
- `metadata_bytes` per compressor
- `total_kv_footprint_bytes` per compressor
- A table comparing `stored_kv_bytes` vs `full_kv_bytes` and noting that
  `materialized_working_kv_bytes == full_kv_bytes` for all current compressors

**Required wording in the report:**

- "Simulated compressors store sub-INT8 values in int8 containers. `stored_kv_bytes`
  reflects int8 storage reality, not hypothetical packed-bit savings."
- "`materialized_working_kv_bytes` equals `full_kv_bytes` for all current
  compressors, because dequantization produces a full-precision working copy."
- "Average effective bit width is a comparison aid, not a memory measurement."
- "This report does not claim speedup, throughput, latency, or production readiness."
- VeriCache attribution.

**Not reported:** tokens/second, throughput, latency, speedup, wall-clock runtime.

---

## 11. What remains deferred to V6

| Item | Reason for deferral |
|---|---|
| Real backend implementations (KIVI, kvpress, etc.) | Requires careful design of `BackendAdapter` interface and honest memory comparison framework. Not V5 unless Phase E is separately approved and completed. |
| True GPU memory profiling (`torch.cuda.memory_allocated` etc.) | Tied to real hardware and a real backend. Estimates in V5 are conservative formulas, not measured peaks. |
| Workspace-aware GPU memory fields (`active_gpu_kv_bytes`) | No device-resident compressors exist yet. Field reserved in schema; not populated. |
| vLLM / LMCache integration | Out of scope for the correctness-first framework. |
| Throughput / latency benchmarks | Requires careful methodology, real bit-packing, and hardware disclaimers. Never default; only after a real backend exists. |
| Parallel (single-pass) speculative verification | Separate correctness project. |
| Attention-aware divergence analysis | Future research; not V5. |

---

## 12. No-performance-claim policy (unchanged from V1)

The following fields may **never** appear as data fields, table columns, or key-value
pairs in any ExactKV output — in code, tests, CLI output, or Markdown reports:

```
tokens_per_second
throughput
latency
speedup
runtime_seconds
```

These terms may appear **only in explicit negation prose**, such as:
- "This report does not claim speedup…"
- "No throughput or latency metric is produced."
- "ExactKV does not measure tokens/second."

The no-performance-field audit (`_assert_no_forbidden_fields` and the pattern
regex audit in CI) must continue to pass across all V5 phases.

Memory fields added in V5 (`stored_kv_bytes`, `materialized_working_kv_bytes`,
etc.) are honesty fields, not performance claims. They describe the memory
footprint of correctness-preserving operations under the existing sequential
draft-verify-commit loop.

---

## 13. V5 exit criteria

V5 is complete when all of the following hold:

- [ ] `MemorySummary` carries all new workspace-aware fields.
- [ ] Every existing compressor populates the new fields in `stats()`.
- [ ] `int4_sim` and all `_sim` compressors: `stored_kv_bytes` reflects
      int8-container reality; `supports_real_bytes_claim=False` preserved.
- [ ] `materialized_working_kv_bytes == full_kv_bytes` for all current compressors
      and is surfaced in reports.
- [ ] Report JSON/CSV schema updated additively; backward compatibility preserved.
- [ ] Markdown reports include the stored vs working vs total memory section.
- [ ] CLI `list-compressors` shows new memory field names.
- [ ] `docs/EXPERIMENT_004_MEMORY_ACCOUNTING.md` generated with `exactkv_failures == 0`.
- [ ] No forbidden performance data fields anywhere.
- [ ] Full prior test suite remains green.

---

## Attribution

The draft-then-verify compressed-KV algorithm is from:

> **VeriCache: Turning Lossy KV Cache into Lossless LLM Inference.**
> Yao et al., arXiv:2605.17613, 2026.

ExactKV does not claim to have invented this algorithm. V5 extends the framework
with honest memory accounting for the compressors already implemented in V1–V4.
