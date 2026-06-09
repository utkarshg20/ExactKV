# ExactKV v0.8.0 Release Notes

**Status:** V8 complete (Phases 0, A, B, D, E). Phase C deferred (no-go).
**Base:** Builds on `v0.7.0` (layer-aware V policies, Experiments 006/006C).
**Tag:** `v0.8.0` — **research milestone, not public launch.**

> **V8 is a serving-context evaluation release, not a performance or production release.**
> V8 does **not** implement vLLM, LMCache, or PagedAttention integration.
> Experiment 007 used a **restricted local harness only** (Mode B).
> `total_kv_footprint_bytes` is a conservative accounting sum, not measured peak
> GPU memory. Active GPU memory is **not** reported.
> ExactKV does **not** claim speedup, throughput, latency, runtime, tokens/sec, or
> production readiness.
> TurboQuant, TurboQuant+, KIVI, and KVQuant are **not** implemented in v0.8.0.

---

## 1. V8 summary

V8 evaluates whether ExactKV's verified compressed-KV workflow remains correct and
measurable in a **serving-style cache context** — ownership separation, logical vs
physical sequence mapping, block/page tables, and append-after-commit lifecycle —
without integrating external serving stacks.

V8 delivers:

- **Phase A:** Serving-context feasibility research; vLLM/LMCache Phase C **no-go**.
- **Phase B:** `ServingCacheLifecycleHarness` in `exactkv/serving/`.
- **Phase D:** Experiment 007 — 238-cell harness evaluation (`exactkv_failures == 0`).
- **Phase E:** Release notes, experiment index, project status, deferred-work register.

Phase C (vLLM/LMCache PoC) remains **deferred** per Phase A findings.

---

## 2. What V8 adds

| Deliverable | Location |
|---|---|
| V8 scope statement | [`docs/V8_SCOPE_STATEMENT.md`](V8_SCOPE_STATEMENT.md) |
| Serving feasibility research (Phase A) | [`docs/SERVING_CONTEXT_FEASIBILITY.md`](SERVING_CONTEXT_FEASIBILITY.md) |
| Cache-lifecycle harness (Phase B) | [`docs/SERVING_CACHE_LIFECYCLE_HARNESS.md`](SERVING_CACHE_LIFECYCLE_HARNESS.md), `exactkv/serving/` |
| Experiment 007 report (Phase D) | [`docs/EXPERIMENT_007_SERVING_CONTEXT.md`](EXPERIMENT_007_SERVING_CONTEXT.md) |
| Experiment runner | `scripts/run_experiment_007_serving_context.py` |
| Harness tests | `tests/test_serving_cache_lifecycle.py` |
| v0.8.0 documentation package (Phase E) | This file, [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md), [`PROJECT_STATUS_V0.8.0.md`](PROJECT_STATUS_V0.8.0.md), [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md) |

**Unchanged:** generation logic, verification logic, compressor registry (15 built-in),
report schema (additive `serving_harness` metadata in Experiment 007 JSON only).

---

## 3. Phase A summary

Full report: [`docs/SERVING_CONTEXT_FEASIBILITY.md`](SERVING_CONTEXT_FEASIBILITY.md).

- **Serving feasibility research** comparing vLLM, LMCache, PagedAttention concepts,
  and ExactKV's HF `FullKVState` / `CompressedKVState` model.
- **vLLM direct integration: no-go/deferred** — authoritative full-precision KV is
  not safely exportable for ExactKV verification in the vLLM worker loop.
- **LMCache direct integration: no-go/deferred** — tiering/async semantics worsen
  ownership clarity; not compatible with ExactKV's verify path without isolation.
- **PagedAttention** naming is **context only**; Phase B models block tables and
  logical/physical mapping **locally** without kernel integration.

---

## 4. Phase B summary

Full report: [`docs/SERVING_CACHE_LIFECYCLE_HARNESS.md`](SERVING_CACHE_LIFECYCLE_HARNESS.md).

- **Local serving/cache-lifecycle harness** — `exactkv/serving/cache_lifecycle.py`.
- **Types:** `CacheOwner`, `CacheBlock`, `ServingCacheEntry`,
  `ServingCacheLifecycleHarness`.
- **Ownership separation:** `authoritative_full` vs `compressed_draft`; verification
  conceptually uses authoritative full KV only.
- **Logical vs physical mapping:** identity when equal; explicit
  `retained_logical_positions` when physical &lt; logical (no fabricated mapping).
- **Append lifecycle:** `append_committed_tokens` after each commit round.
- **Memory-honesty metadata:** `stored_kv_bytes`, `materialized_working_kv_bytes`,
  `total_kv_footprint_bytes`, `supports_real_bytes_claim`, `is_simulated`.

---

## 5. Phase D summary — Experiment 007

Full report: [`docs/EXPERIMENT_007_SERVING_CONTEXT.md`](EXPERIMENT_007_SERVING_CONTEXT.md).

| Metric | Value |
|---|---|
| Mode | **harness_sim** (Mode B harness-only) |
| Total cells | **238** |
| Core prompts | **34** |
| Compressors | **7** |
| Model | `Qwen/Qwen2.5-0.5B`, float32, CPU-first |
| `draft_len` | 4 |
| `max_new_tokens` | 16 |
| **ExactKV failures** | **0** |
| Harness invariants | **all pass** (238 / 238) |
| Commit rounds tracked | **907** |

Every cell: separate authoritative/compressed owners, `verification_uses ==
authoritative_full`, valid block tables, identity physical/logical mapping in this
panel.

---

## 6. Acceptance by compressor (Experiment 007)

| Compressor | Accept rate |
|---|---|
| `noop` | **1.000** |
| `backend_passthrough` | **1.000** |
| `k_full_v8` | **0.990** |
| `k8_v_full` | **0.963** |
| `int8` | **0.961** |
| `k8_v4_boundary4_v8_sim` | **0.954** |
| `k8_v4_sim` | **0.891** |

Lossy divergence cells: 51 / 238 (expected; ExactKV corrected all).

---

## 7. What V8 proves

- ExactKV's exactness gate (`exactkv_output_ids == full_output_ids`) holds while a
  serving-style lifecycle harness tracks cache ownership and invariants on every cell.
- Authoritative full KV and compressed draft KV can be **modelled as separate stores**
  alongside the existing HF runtime without changing generation or verification logic.
- Logical sequence length, block/page mapping, and append-after-commit can be
  represented deterministically for compatibility evaluation.
- V5 workspace-memory honesty remains correct in harness summaries.

---

## 8. What V8 does not prove

- Compatibility with vLLM, LMCache, or production PagedAttention serving.
- Throughput, latency, speedup, or production-serving behaviour.
- Multi-request batching, GPU block pools, or cross-request cache sharing.
- That simulated `_sim` compressors achieve real packed-bit memory savings.
- That external serving-stack paper results are reproduced by ExactKV.

---

## 9. Known limitations

- **Not vLLM integration** — Phase C no-go/deferred.
- **Not LMCache integration** — Phase C no-go/deferred.
- **Not PagedAttention integration** — concepts modelled locally only.
- **Identity physical/logical mapping** in the Experiment 007 panel (no pruned kvpress
  cells in this sweep).
- **No active GPU memory** measurement or reporting.
- **No production-serving claim** — evaluation framework only.
- **Single small model** (`Qwen/Qwen2.5-0.5B`) on CPU-first sweeps.

---

## 10. Deferred work

See [`docs/DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md) for the full register.

| Item | Target |
|---|---|
| TurboQuant / TurboQuant+ full integration | V9 |
| KIVI adapter | V9 |
| KVQuant-style adapter | V9 |
| Sparse V dequantization | V10 |
| True attention logging | V10 |
| Per-layer/head/token divergence forensics | V10 |
| Direct or sidecar vLLM/LMCache probe | V11 |
| Active GPU memory profiling | V11 |
| Larger-model RunPod validation | V11 |
| Raw report bundle for final release | V12 / v1.0.0 |
| Final public launch narrative | v1.0.0 |

---

## 11. Upgrade / reproduction notes

**From v0.7.0:** No breaking API changes. Harness package is additive.

**Reproduce Experiment 007:**

```bash
TRANSFORMERS_OFFLINE=1 python3 scripts/run_experiment_007_serving_context.py
```

**Artifacts (gitignored):**

- `reports/experiment_007_serving_context.json`
- `reports/experiment_007_serving_context.csv`

**Harness smoke tests:**

```bash
pytest tests/test_serving_cache_lifecycle.py -v
```

---

## 12. v0.8.0 tag readiness

| Criterion | Status |
|---|---|
| Phase A feasibility doc | ✅ |
| Phase B harness + tests | ✅ |
| Experiment 007 (`exactkv_failures == 0`) | ✅ |
| Experiment 007 report | ✅ |
| Release notes (this file) | ✅ |
| Experiment index | ✅ |
| Project status | ✅ |
| Deferred-work register | ✅ |
| No forbidden performance fields in docs | ✅ (audited) |
| Public launch narrative | ❌ **Deferred** |

**Ready to tag `v0.8.0`.**

---

## 13. Why v1.0.0 is not next yet

v0.8.0 closes the **serving-context evaluation chapter** but the project is **not
public-launch final**. Substantial deferred work remains before a credible v1.0.0:

- Real backend integration gauntlet (TurboQuant+, KIVI, KVQuant-style adapters).
- Deeper compression research (Sparse V, attention logging, divergence forensics).
- Scale and serving probes (larger models, GPU memory profiling, sidecar stack evaluation).
- Curated raw report bundle and reviewed public launch narrative.

The roadmap continues through **V9 → V10 → V11 → v1.0.0**. See
[`docs/ROADMAP.md`](ROADMAP.md) and [`docs/DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md).

---

## Attribution

**VeriCache** (draft-then-verify algorithm): Yao et al., arXiv:2605.17613, 2026.

**Serving-stack references** (evaluation context only; not implemented by ExactKV):

- vLLM / PagedAttention: Kwon et al., SOSP 2023, arXiv:2309.06180
- LMCache: Liu et al., 2025, arXiv:2510.09665

ExactKV does not reproduce or claim external-system performance or accuracy results.
