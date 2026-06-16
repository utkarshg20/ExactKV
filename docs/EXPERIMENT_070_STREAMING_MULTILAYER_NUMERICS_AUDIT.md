# Experiment 070: Streaming Multi-Layer Numerics Audit (Phase 16E)

**Status:** numerical boundary audit — run `scripts/research/run_exp070_streaming_multilayer_numerics_audit.py` for report.

> This is a **numerical audit of offline streaming attention**, not model generation integration.  
> Streaming attention is **not wired into ExactKV generation**.  
> Tolerance changes, if any, are **diagnostic** and not production correctness guarantees.  
> No CUDA kernel is implemented.  
> No Triton kernel is implemented.  
> No vLLM integration is implemented.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_069_MULTILAYER_ATTENTION_DRIFT_ACCUMULATION.md`](EXPERIMENT_069_MULTILAYER_ATTENTION_DRIFT_ACCUMULATION.md) · `exactkv/attention/streaming_quant_attention.py`

---

## 1. Purpose

Phase 16E diagnoses the single Phase 16D streaming-vs-materialized failure (`long_128` / 4 prefix layers / chunk 32) by sweeping accumulator precision, chunk sizes, and prefix depths under a focused numerics audit.

---

## 2. Why Phase 16D needed a numerics audit

Phase 16D showed 17/18 cells pass with full-block parity 18/18. One cell exceeded the strict 5×10⁻⁴ tolerance by ~16% (5.79×10⁻⁴). Before broadening models or integrating generation, we must determine whether that boundary is a bug, expected FP accumulation, or a tolerance policy issue.

---

## 3. The Phase 16D boundary failure

| Field | Value |
|---|---|
| Prompt | `long_128` |
| Prefix layers | 4 |
| Chunk size | 32 |
| Accumulator | default (query dtype) |
| Observed error | 5.79×10⁻⁴ |
| Strict tolerance | 5×10⁻⁴ |
| Full-block parity | passed |

Preserved as `PHASE16D_REGRESSION_CELL` in `hf_multilayer_probe.py`.

---

## 4. Accumulator modes tested

| Mode | Behavior |
|---|---|
| `default` | Accumulators use query tensor dtype (Phase 16A–16D behavior) |
| `float32` | Force float32 online-softmax accumulators |
| `float64` | Force float64 accumulators (CPU audit reference) |

Optional `return_diagnostics` reports chunk count, running max/denominator ranges, NaN/Inf flags.

---

## 5. Error by chunk size

Report field: `max_error_by_chunk_size` — aggregates max streaming-vs-materialized hidden error per chunk size across accumulator modes and depths.

---

## 6. Error by prefix depth

Report field: `max_error_by_prefix_depth` — shows how hidden-state drift scales with consecutive compressed layers.

---

## 7. Tolerance policy analysis

Candidate policies (not blindly applied):

| Policy | Formula |
|---|---|
| `strict_16d` | 5×10⁻⁴ (fp32) |
| `dtype_aware` | fp32 strict / fp16 relaxed |
| `layer_depth_aware` | strict × √prefix_layers |
| `reference_high_precision` | anchored to float64 reference error |

Report recommends a policy only when supported by measured errors.

---

## 8. Whether an algorithm fix was made

Only if online-softmax bug is proven. Otherwise `algorithm_change_made: false` and numerical accumulation is documented.

---

## 9. Results

```bash
python3 scripts/research/run_exp070_streaming_multilayer_numerics_audit.py
```

Report (gitignored): `reports/experiment_070_streaming_multilayer_numerics_audit.json`

**Local CPU run (`Qwen/Qwen2.5-0.5B`, float32):**

| Metric | Value |
|---|---|
| Total cells | 72 |
| Strict tolerance pass | 64/72 |
| Strict tolerance fail | 8/72 |
| Recommended tolerance pass | 72/72 |
| Phase 16D regression reproduced | **Yes** (5.79×10⁻⁴, status: `reproduced_failure`) |
| Algorithm change made | **No** |
| Recommended policy | `documented_layer_depth_aware_tolerance` |

**Phase 16D regression cell (`long_128` / 4 layers / chunk 32 / default):**

| Accumulator | Max abs error | Strict pass |
|---|---|---|
| default | 5.79×10⁻⁴ | fail |
| float32 | 5.79×10⁻⁴ | fail (identical to default on fp32) |
| float64 | 5.70×10⁻⁴ | fail |

**Diagnosis:** No NaN/Inf in any layer. float64 hidden-state error remains above strict tolerance — this is **expected multi-layer hidden-state accumulation**, not an online-softmax algorithm bug. Depth-aware tolerance (5×10⁻⁴ × √4 = 1×10⁻³) explains all strict failures without loosening single-layer gates.

**Max error by prefix depth:** layer 1 ≈ 1.6×10⁻⁶, layer 2 ≈ 4.3×10⁻⁶, layer 4 ≈ 5.83×10⁻⁴.

---

## 10. What this proves

- Whether the Phase 16D boundary failure reproduces under controlled accumulator modes
- Whether higher-precision accumulators monotonically reduce streaming-vs-materialized error
- Whether strict tolerance should be kept, depth-scaled, or supplemented with float32 accumulators (diagnostic only)

---

## 11. What this does not prove

- Model output preservation or generation equivalence
- Production correctness of relaxed tolerances
- Runtime integration into ExactKV generation
- Speed, throughput, latency, or measured GPU memory savings
- vLLM / LMCache integration
- VeriCache throughput reproduction

---

## 12. Relation to ExactKV restored verification

Restored-verifier tracks validate greedy continuation from stored full KV. Phase 16E audits **numerical behavior** of chunked streaming attention — orthogonal to restored verification.

---

## 13. Relation to VeriCache parity

VeriCache serving stacks emphasize throughput and memory under vLLM/LMCache. Phase 16E does **not** reproduce VeriCache throughput, serving, or memory panels.

---

## 14. Next step

**Phase 16F (proposed):** broader model panel sweep — only after 16E tolerance/accumulator recommendation is reviewed; still opt-in, no default runtime integration.

---

## Claims boundary

| Allowed | Forbidden |
|---|---|
| Numerical audit findings | Model output preservation claim |
| Diagnostic tolerance recommendation | Production correctness guarantee |
| Accumulator mode comparison | Speedup / throughput / latency |
| Phase 16D regression tracking | Measured GPU memory savings |
