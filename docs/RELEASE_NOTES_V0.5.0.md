# ExactKV v0.5.0 Release Notes

**Status:** V5 implementation complete (Phases A–D).
**Base:** Builds on `v0.4.0` (asymmetric K/V compressor experiments, Experiment 003).

> **No real compressor backends are implemented in V5.**
> No performance, throughput, latency, speedup, or production-readiness claims.

---

## What V5 Adds

### Workspace-Aware Memory Accounting

V5 introduces honest, field-level memory accounting that distinguishes:

| Field | Meaning |
|---|---|
| `stored_kv_bytes` | Bytes of the compressed/quantised tensor representation |
| `materialized_working_kv_bytes` | Full-precision working copy needed for attention |
| `metadata_bytes` | Per-tensor scales, zero-points, or similar overhead |
| `temporary_workspace_bytes` | Conservative transient scratch estimate |
| `total_kv_footprint_bytes` | Accounting sum of all four fields |

**Why this matters:** Stored bytes alone understate the true peak KV memory footprint for
materializing compressors. All current ExactKV compressors dequantise the stored cache to full
precision for each attention call, so `materialized_working_kv_bytes == full_kv_bytes`.
This means `total_kv_footprint_bytes > full_kv_bytes` for most compressors — an honest
accounting choice motivated by how real backends (KIVI, KVQuant, Palu, etc.) actually use memory.

### Simulated-Compressor Honesty

For `_sim` compressors (e.g. `int4_sim`, `k8_v4_sim`), `stored_kv_bytes` reflects
**int8 container storage**, not hypothetical packed 4-bit or 2-bit storage.
`supports_real_bytes_claim=False` is enforced for all simulated compressors.

### JSON Report Enrichment (Phase B)

All benchmark JSON reports now include `stored_kv_bytes`, `materialized_working_kv_bytes`,
`metadata_bytes`, `temporary_workspace_bytes`, and `total_kv_footprint_bytes` in every
result's `memory` sub-dict. CSV exports include corresponding columns.

Old V1–V4 reports remain valid; the new fields default to 0 for legacy results.

### Markdown Report Workspace Section (Phase C)

Markdown reports now include a **Workspace-Aware Memory Accounting** section with:
- A per-compressor table showing all five workspace fields in human-readable units.
- An explicit note that `total_kv_footprint_bytes` is a **conservative accounting sum,
  not a measured peak GPU memory value.**
- A note that active GPU measurement is **deferred to a later CUDA-specific phase.**
- Clear labelling of simulated compressors and their int8 container reality.

The Memory Honesty Notes section is now a compact table (one row per compressor) with
a shared footer, replacing the previous verbose per-compressor paragraphs.

### CLI Summary Lines (Phase C)

- `bench` and `sweep`: print a workspace memory note when V5 fields are populated.
- `report`: prints "Workspace memory: included in Markdown report".

### Experiment 004 (Phase D)

Experiment 004 runs the core prompt suite (34 prompts) across 10 compressors with
draft_len=4 and max_new_tokens=16 on CPU (Qwen/Qwen2.5-0.5B):

| Metric | Value |
|---|---|
| Total runs | 340 |
| ExactKV failures | **0** (PASS) |
| Lossy divergences | 175 |
| Mean acceptance rate | 0.800 |
| Compressors | noop, int8, int4_sim, k8_v4_sim, k8_v2_sim, k4_v8_sim, k_full_v4_sim, k4_v_full_sim, k8_v_full, k_full_v8 |

**Memory accounting highlights (Qwen2.5-0.5B, core suite, first prompt representative):**

| Compressor | Stored KV | Materialized KV | Total footprint |
|---|---|---|---|
| `noop` | 120.0 KiB | 120.0 KiB | 240.0 KiB |
| `int8` | 30.0 KiB | 120.0 KiB | 150.4 KiB |
| `int4_sim` ⚠️ | 30.0 KiB (int8) | 120.0 KiB | 150.4 KiB |
| `k8_v4_sim` ⚠️ | 30.0 KiB (int8) | 120.0 KiB | 150.4 KiB |
| `k8_v2_sim` ⚠️ | 30.0 KiB (int8) | 120.0 KiB | 150.4 KiB |
| `k8_v_full` | 75.0 KiB | 120.0 KiB | 195.2 KiB |
| `k_full_v8` | 75.0 KiB | 120.0 KiB | 195.2 KiB |

⚠️ Simulated: `stored_kv_bytes` is int8 container size, not packed sub-INT8 bytes.

All values are accounting estimates derived from tensor shapes and dtype widths.
None are measured peak GPU memory values.

Full report: [`docs/EXPERIMENT_004_WORKSPACE_MEMORY.md`](EXPERIMENT_004_WORKSPACE_MEMORY.md).

---

## What Remains Out of Scope

### Active GPU Memory Measurement

`total_kv_footprint_bytes` is a conservative accounting sum only.
Actual peak GPU memory (via `torch.cuda.memory_reserved` or device profiler) is
**not measured in V5** and is deferred to a later CUDA-specific validation phase.
This phase requires:
- A CUDA-enabled device
- Torch memory profiler instrumentation
- Careful isolation of KV-cache vs attention vs activation memory

### Real Backend Adapters

V5 does not implement any real quantisation backend. The following are explicitly deferred:

- **KIVI** — 2-bit KV quantisation with full-precision residual
- **KVQuant** — sub-4-bit with per-channel / pre-RoPE K and dense-sparse V
- **TurboQuant / TurboQuant+** — data-oblivious rotation + quantisation; "Sparse V" decode
- **KVTC** — PCA + adaptive quantisation + entropy coding
- **Palu** — low-rank projection for KV cache
- **LMCache** — KV cache offload and reuse layer
- **vLLM / PagedAttention** — OS-style paging and prefix sharing

The roadmap for real backends is V6+, requiring separate scope approval.

### Real INT4 / INT2 Bit-Packing

All `_sim` compressors in V5 store sub-INT8 numeric values in int8 containers.
No real packed 4-bit or 2-bit storage is implemented. Any `stored_kv_bytes` figure for
`_sim` compressors reflects int8 container reality, not theoretical packed-bit savings.

---

## No Performance Claims

ExactKV V5 does not measure, report, or claim:

- Wall-clock time, latency, or tokens per second
- Throughput or speedup vs any baseline
- Production-readiness or production serving performance
- Peak GPU memory utilisation

V5 documents **correctness** (ExactKV output == full-KV output under greedy decoding)
and **memory accounting** (conservative estimates of KV storage and working-copy sizes).

---

## Changelog Summary

| Phase | What was done |
|---|---|
| V5 Phase 0 | Finalised V5 scope statement; research/docs consolidation |
| V5 Phase A | Added workspace-aware fields to `CompressionStats` and `MemorySummary`; per-compressor `stats()` updates |
| V5 Phase B | JSON/CSV report schema enrichment; backward-compat with V1–V4 reports |
| V5 Phase C | Markdown workspace table; compact Memory Honesty Notes; CLI workspace lines |
| V5 Phase D | Experiment 004 sweep; `docs/EXPERIMENT_004_WORKSPACE_MEMORY.md` |

---

## Test Count

All 1263 tests pass (V5 Phase D baseline).
