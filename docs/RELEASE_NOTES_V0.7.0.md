# ExactKV v0.7.0 Release Notes

**Status:** V7 implementation complete (Phases 0–E).
**Base:** Builds on `v0.6.0` (`BackendAdapter`, restricted `KVPressKnormAdapter`,
Experiment 005).

> **V7 is a research and evaluation release, not a performance release.**
> V7 compressors are **simulated policies** unless explicitly stated otherwise.
> Boundary V policies use **int8 containers**, not packed INT4 storage.
> Boundary V policies do **not** use true attention weights.
> ExactKV does **not** implement Sparse V dequantization, TurboQuant+, KVQuant, or
> KIVI in V7. External-paper claims are **not** ExactKV results.
> `total_kv_footprint_bytes` is a conservative accounting sum, not measured peak
> GPU memory. Active GPU memory is **not** reported.
> ExactKV does **not** claim speedup, throughput, latency, runtime, or production
> readiness.

---

## 1. V7 summary

V7 evaluates **attention-aware and V-specific compression ideas** through ExactKV's
existing draft-verify-commit framework — exactness, acceptance, divergence/rejection/
correction counts, and V5 workspace-memory accounting — without changing generation or
verification logic.

V7 delivers:

- **Phase A:** Proxy divergence analysis on existing Experiments 003–005 (no new runs).
- **Phase B:** Simulated layer-aware V policy (`k8_v4_boundary_v8_sim`).
- **Phase D:** Experiment 006 — 374-run core-suite sweep comparing layer-aware and
  asymmetric baselines (`exactkv_failures == 0`).
- **Phase C:** Boundary-depth ablation after Experiment 006 — `k8_v4_boundary2_v8_sim`
  and `k8_v4_boundary4_v8_sim`; Experiment 006C (170 runs,
  `exactkv_failures == 0`).
- **Reporting honesty cleanup:** Mixed per-layer V precision renders as `mixed 8/4-sim`;
  avg effective bits `n/a` for boundary compressors (no longer shown as full-V / 20.0).

V7 evaluates policies by **acceptance behaviour and memory honesty only** — never by
speed or serving performance.

---

## 2. What V7 adds

| Deliverable | Location |
|---|---|
| V7 scope statement | [`docs/V7_SCOPE_STATEMENT.md`](V7_SCOPE_STATEMENT.md) |
| Proxy divergence analysis module | `exactkv/analysis/attention_weighted.py` |
| Experiment 006A report (Phase A) | [`docs/EXPERIMENT_006A_DIVERGENCE_ANALYSIS.md`](EXPERIMENT_006A_DIVERGENCE_ANALYSIS.md) |
| Layer-aware sim compressor | `exactkv/compressors/layer_aware_sim.py` |
| `k8_v4_boundary_v8_sim` (N=1) | Registry |
| `k8_v4_boundary2_v8_sim` (N=2) | Registry (Phase C) |
| `k8_v4_boundary4_v8_sim` (N=4) | Registry (Phase C) |
| Experiment 006 report (Phase D) | [`docs/EXPERIMENT_006_LAYER_AWARE_V.md`](EXPERIMENT_006_LAYER_AWARE_V.md) |
| Experiment 006C report (Phase C) | [`docs/EXPERIMENT_006C_BOUNDARY_DEPTH_ABLATION.md`](EXPERIMENT_006C_BOUNDARY_DEPTH_ABLATION.md) |
| Mixed-V reporting labels | `value_bit_width_label` on `CompressorCapabilities`; renderer updates |

**Unchanged:** generation logic, verification logic, draft-verify-commit loop, and
the no-performance-claim policy. Report schema unchanged except additive optional
`key_bit_width_label` / `value_bit_width_label` on capabilities (backward-compatible).

**Registry count:** 15 built-in compressors (4 V1–V3 + 7 V4 + 3 V7 layer-aware + 1 V6
`backend_passthrough`).

---

## 3. Phase A summary

Full report: [`docs/EXPERIMENT_006A_DIVERGENCE_ANALYSIS.md`](EXPERIMENT_006A_DIVERGENCE_ANALYSIS.md).

- **Proxy divergence analysis** on Experiments 003, 004, and 005 — reuses existing
  report fields only; **no new generation runs**.
- **No true attention weights available** in source reports (`has_attention_weights ==
  False` on all three).
- **No attention weights fabricated.**
- **Aggressive compressors diverge early:** e.g. `k8_v2_sim` and `int4_sim` show ~97%
  divergence rate with mean first-divergence index ~1.9–2.5; conservative policies
  (`k_full_v8`) diverge rarely.
- Lower acceptance co-occurs with higher divergence rate and earlier first-divergence
  index — **proxy correlation only**, not a causal attention claim.

---

## 4. Phase B summary

- New **simulated layer-aware V compressor:** `k8_v4_boundary_v8_sim` in
  `exactkv/compressors/layer_aware_sim.py`.
- **Policy:** K = INT8 simulated quantization on **all** layers; V = INT8 on first/last
  **1** layer (`boundary_layers=1`); V = INT4-range simulation on **interior** layers.
- **`is_simulated=True`**, **`supports_real_bytes_claim=False`**.
- Values stored in **int8 containers** — no real packed-bit memory savings claim.
- Does **not** use true attention weights, Sparse V dequantization, TurboQuant+, KVQuant,
  or KIVI.

---

## 5. Phase D summary — Experiment 006

Full report: [`docs/EXPERIMENT_006_LAYER_AWARE_V.md`](EXPERIMENT_006_LAYER_AWARE_V.md).

| Metric | Value |
|---|---|
| Total cells | **374** |
| Core prompts | **34** |
| Compressors | **11** |
| Model | `Qwen/Qwen2.5-0.5B`, float32, CPU |
| `draft_len` | 4 |
| `max_new_tokens` | 16 |
| **ExactKV failures** | **0** |

**Key result — `k8_v4_boundary_v8_sim`:**

| Comparison | Accept rate |
|---|---|
| `k8_v4_boundary_v8_sim` | **0.904** |
| `k8_v4_sim` (uniform K8/V4) | 0.891 |
| Δ vs `k8_v4_sim` | **+0.013** (modest gain) |
| `k_full_v4_sim` | 0.909 |
| Δ vs `k_full_v4_sim` | −0.005 (did not beat full-K reference) |

Lossy divergences (192 / 374 cells) are **expected**; final ExactKV output remains
exact because verification uses authoritative full KV.

Artifacts (gitignored): `reports/experiment_006_layer_aware_v.{json,csv}`.

---

## 6. Phase C summary — boundary-depth ablation

**Note:** Phase C was **completed after Experiment 006** as a **boundary-depth ablation**
— not a real-backend adapter phase (KIVI/KVQuant/TurboQuant remain deferred).

Full report: [`docs/EXPERIMENT_006C_BOUNDARY_DEPTH_ABLATION.md`](EXPERIMENT_006C_BOUNDARY_DEPTH_ABLATION.md).

| Metric | Value |
|---|---|
| Total cells | **170** |
| Compressors | **5** (`k8_v4_sim`, boundary N=1/2/4, `k_full_v4_sim`) |
| **ExactKV failures** | **0** |

**New variants:** `k8_v4_boundary2_v8_sim` (N=2), `k8_v4_boundary4_v8_sim` (N=4).

**Acceptance rose monotonically with boundary depth:**

| Compressor | Boundary N | Accept rate |
|---|---:|---:|
| `k8_v4_sim` | — | 0.891 |
| `k8_v4_boundary_v8_sim` | 1 | 0.904 |
| `k8_v4_boundary2_v8_sim` | 2 | 0.906 |
| **`k8_v4_boundary4_v8_sim`** | **4** | **0.954** |
| `k_full_v4_sim` | — | 0.909 |

**`k8_v4_boundary4_v8_sim` highlights:**

- Beat `k8_v4_sim` by **+0.063** acceptance.
- Beat `k_full_v4_sim` by **+0.045** acceptance on this suite.
- Fewer divergence cells (11 / 34) and lower rejection/correction burden than
  shallower boundary depths.

Artifacts (gitignored): `reports/experiment_006c_boundary_depth_ablation.{json,csv}`.

---

## 7. Reporting honesty cleanup

Mixed per-layer V precision on boundary compressors was previously rendered as **V bits =
full** and **avg eff bits = 20.0** because `value_bit_width=None` was interpreted as
full precision.

**Fix (backward-compatible):**

- Additive `value_bit_width_label="mixed 8/4-sim"` on layer-aware compressors.
- Report renderers prefer labels; **avg effective bits = `n/a`** when mixed.
- `enrich_caps_from_registry()` overlays labels when regenerating Markdown from old JSON
  without rerunning sweeps.

Boundary compressors **no longer appear as full-V compressors** in K/V metadata tables.

---

## 8. What V7 proves

1. **Exactness gate holds** for all V7 experiments: `exactkv_failures == 0` on 006,
   006C, and all layer-aware compressor tests.
2. **Proxy divergence analysis** separates aggressive vs conservative compressors by
   early-divergence timing without fabricating attention weights.
3. **Simulated boundary V protection** improves draft acceptance vs uniform `k8_v4_sim`;
   deeper boundary depth (N=4) substantially improves acceptance on the core suite.
4. **`k8_v4_boundary4_v8_sim`** achieves the highest acceptance among tested simulated
   layer-aware policies in V7 (0.954 on Experiment 006C panel).
5. **Reporting honesty** for mixed V precision is enforceable via additive capability
   labels without changing experiment results.

---

## 9. What V7 does not prove

ExactKV V7 does **not** measure, report, or claim:

- Speedup, throughput, tokens per second, latency, or wall-clock runtime
- Production-readiness or production serving performance
- True attention-weighted or attention-gated V selection
- Real packed-bit memory savings for `_sim` or layer-aware compressors
- Reproduction of TurboQuant+, KVQuant, KIVI, PyramidKV, or other external-paper
  accuracy or speed numbers as ExactKV results
- Generalization beyond `Qwen/Qwen2.5-0.5B`, `draft_len=4`, `max_new_tokens=16` on
  the core suite without further sweeps

V7 documents **correctness**, **acceptance behaviour**, **divergence/rejection/correction
counts**, and **conservative workspace-memory accounting** only.

---

## 10. Known limitations

1. **All V7 layer-aware compressors are simulated** — int8 containers; no packed INT4
   storage; `supports_real_bytes_claim=False`.
2. **Boundary depth is structural** — not attention-gated; no attention weights logged.
3. **V5 `stored_kv_bytes` unchanged** across boundary depths — int8-container accounting
   hides per-layer precision differences in stored-byte totals.
4. **Single model** in Experiments 006 and 006C — `Qwen/Qwen2.5-0.5B` only.
5. **CPU evaluation** — GPU behaviour not characterised.
6. **`total_kv_footprint_bytes` is an accounting sum**, not measured peak GPU memory.
   Active GPU memory is not reported.
7. **N=4 results on one suite** do not guarantee N=4 is optimal on all models or tasks.
8. **kvpress scope unchanged** — V6 restricted KnormPress only; not broadened in V7.

---

## 11. Deferred work

| Item | Notes |
|---|---|
| True attention logging | Optional future sub-phase; no fabricated weights |
| Sparse V dequantization | Not implemented in V7 |
| Pre-RoPE key quantization | KVQuant-motivated; deferred |
| KVQuant-style adapter | Real backend; separate approval |
| TurboQuant-style adapter | Real backend; separate approval |
| KIVI | Real asymmetric backend; separate approval |
| Serving-stack integration (vLLM, LMCache, PagedAttention) | **V8** evaluation context only |
| Active GPU memory profiling | **V8** at earliest |
| CUDA/Triton kernels, CPU offload, batching, sampling | Out of scope |
| Parallel verification, bonus-token acceptance | Out of scope |
| Experiment 006b (real vs simulated asymmetric) | When an approved real adapter exists |

---

## 12. Upgrade notes

### Default install (unchanged)

```bash
pip install -e .
pytest
```

### New compressors (default registry)

```python
from exactkv.compressors import get_compressor

get_compressor("k8_v4_boundary_v8_sim")   # N=1 boundary
get_compressor("k8_v4_boundary2_v8_sim") # N=2 boundary
get_compressor("k8_v4_boundary4_v8_sim")  # N=4 boundary
```

All three: `is_simulated=True`, `supports_real_bytes_claim=False`,
`value_bit_width_label="mixed 8/4-sim"`.

### Reproduce experiment reports (artifacts gitignored)

```bash
# Experiment 006 (374 cells) — already run for v0.7.0
python3 -m exactkv sweep \
  --model Qwen/Qwen2.5-0.5B --suite core \
  --compressors noop,int8,int4_sim,k8_v4_sim,k4_v8_sim,k_full_v4_sim,k4_v_full_sim,k8_v2_sim,k_full_v8,k8_v_full,k8_v4_boundary_v8_sim \
  --draft-lengths 4 --max-new-tokens 16 \
  --json-out reports/experiment_006_layer_aware_v.json \
  --csv-out reports/experiment_006_layer_aware_v.csv

# Experiment 006C (170 cells)
python3 -m exactkv sweep \
  --model Qwen/Qwen2.5-0.5B --suite core \
  --compressors k8_v4_sim,k8_v4_boundary_v8_sim,k8_v4_boundary2_v8_sim,k8_v4_boundary4_v8_sim,k_full_v4_sim \
  --draft-lengths 4 --max-new-tokens 16 \
  --json-out reports/experiment_006c_boundary_depth_ablation.json \
  --csv-out reports/experiment_006c_boundary_depth_ablation.csv
```

### Report compatibility

V1–V6 reports remain valid. V7 adds optional `key_bit_width_label` /
`value_bit_width_label` on `CompressorCapabilities`. Regenerating Markdown from old
JSON enriches labels from the live registry when absent.

---

## 13. Next version: V8

**V8 — serving-stack integration (evaluation context only).** With V7 establishing
simulated layer-aware V policies and boundary-depth evidence, V8 may evaluate caches
managed by a serving stack (vLLM/PagedAttention, LMCache) — **never** as a source of
performance claims.

Still **no throughput/latency/speedup claims**. Active GPU memory profiling earliest
in V8. See [`docs/FUTURE_ROADMAP_V6_V8.md`](FUTURE_ROADMAP_V6_V8.md) and
[`docs/ROADMAP.md`](ROADMAP.md).

---

## Changelog summary

| Phase | What was done |
|---|---|
| V7 Phase 0 | Scope statement (`docs/V7_SCOPE_STATEMENT.md`) |
| V7 Phase A | `exactkv/analysis/attention_weighted.py`; Experiment 006A report |
| V7 Phase B | `k8_v4_boundary_v8_sim`; `layer_aware_sim.py` + tests |
| V7 Phase D | Experiment 006 (374 cells); `docs/EXPERIMENT_006_LAYER_AWARE_V.md` |
| V7 Phase C | `k8_v4_boundary2_v8_sim`, `k8_v4_boundary4_v8_sim`; Experiment 006C (170 cells) |
| V7 reporting | Mixed-V metadata labels; renderer honesty cleanup |
| V7 Phase E | This release; README/ROADMAP updates; audit |

---

## Attribution

**VeriCache** (draft-then-verify algorithm):

> Yao et al., *VeriCache: Turning Lossy KV Cache into Lossless LLM Inference*, arXiv:2605.17613, 2026.

**Related work** (TurboQuant+, KVQuant, KIVI, PyramidKV, etc.):

> Named for research motivation only. ExactKV V7 does **not** implement these backends
> and does **not** cite their benchmark numbers as ExactKV results.
> See [`docs/RELATED_WORK_KV_CACHE_COMPRESSION.md`](RELATED_WORK_KV_CACHE_COMPRESSION.md).

---

## Test count

Default env: targeted layer-aware, reporting, and analysis tests pass at Phase E gate.
Full pytest not required for Markdown-only Phase E deliverables; code phases gated
separately during Phases B/C and reporting cleanup.
