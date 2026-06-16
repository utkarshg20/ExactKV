# Experiment 071: Full-Prefix Logit Drift Smoke (Phase 16F)

**Status:** offline logit drift smoke — run `scripts/research/run_exp071_full_prefix_logit_drift_smoke.py` for report.

> This is an **offline full-prefix logit drift smoke**, not model generation integration.  
> Streaming attention is **not wired into ExactKV generation**.  
> Computing logits for fixed prompts is **not the same as generating tokens**.  
> Full-model parity must be reported before interpreting compression drift.  
> No CUDA kernel is implemented.  
> No Triton kernel is implemented.  
> No vLLM integration is implemented.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_070_STREAMING_MULTILAYER_NUMERICS_AUDIT.md`](EXPERIMENT_070_STREAMING_MULTILAYER_NUMERICS_AUDIT.md) · `exactkv/attention/hf_full_replay_probe.py`

---

## 1. Purpose

Phase 16F runs a generation-adjacent offline smoke: replay the **full** Qwen2.5-0.5B decoder stack with full, materialized-compressed, and streaming-compressed attention, then compare final hidden states and **next-token logits** for fixed prompts.

---

## 2. Relation to Phase 16E

Phase 16E diagnosed the Phase 16D boundary failure as expected multi-layer FP accumulation and recommended depth-aware tolerance (`5×10⁻⁴ × √layers`). Phase 16F applies that policy across the **entire** decoder stack and extends comparison to logits and top-k overlap.

---

## 3. Why this is still offline and opt-in

- No changes to `ExactKVGenerator` or default runtime
- No token generation, sampling, or serving
- Fixed-prompt logit computation only
- CPU-safe defaults

---

## 4. Full decoder-stack replay method

For each path, replay:

1. `embed_tokens`
2. All decoder layers (attention path substituted per layer)
3. Final `norm`
4. `lm_head` → next-token logits at last position

Paths: `hf_reference`, `manual_full`, `materialized_compressed`, `streaming_compressed`.

---

## 5. Full-model parity smoke

Before interpreting compression drift:

- Compare `manual_full` logits vs HF reference logits
- Compare final hidden states
- Metrics: max/mean abs, relative L2, cosine similarity, top-1 agreement, top-5 overlap

| `full_model_parity_status` | Meaning |
|---|---|
| `passed` | Manual replay within tolerance |
| `failed` | Drift beyond tolerance — reported, not hidden |
| `blocked` | Replay unavailable |

---

## 6. Full vs materialized compressed vs streaming compressed paths

**Primary pass:** streaming compressed logits ≈ materialized compressed logits (depth-aware tolerance).

**Secondary:** full vs streaming and full vs materialized logit drift — measured, not required to pass.

---

## 7. Logit drift metrics

Per comparison: max/mean abs error, relative L2, cosine similarity, top-1 agreement, top-5/10 overlap, top-1 probabilities (when softmax is safe), logit margin top-1 vs top-2.

---

## 8. Top-k overlap / top-1 change metrics

- `top1_changed_full_vs_streaming` — compression drift measurement, **not** a streaming-vs-materialized failure criterion
- `top5_overlap_streaming_vs_materialized` — primary path agreement
- `compressed_top1_changed_cells` — aggregate count in report

---

## 9. Memory accounting

Theoretical aggregate across all layers:

- `aggregate_full_kv_bytes`
- `aggregate_stored_quantized_kv_bytes`
- `aggregate_materialized_working_kv_bytes`
- `aggregate_streaming_peak_working_kv_bytes_conservative`
- `metadata_bytes`, `chunk_size`, `num_layers`, `context_length`
- `best_theoretical_streaming_reduction`

---

## 10. Results

```bash
python3 scripts/research/run_exp071_full_prefix_logit_drift_smoke.py
```

Report (gitignored): `reports/experiment_071_full_prefix_logit_drift_smoke.json`

Default sweep: 2 prompts × 3 chunk sizes = **6 cells** (when model loads).

**Local CPU run (`Qwen/Qwen2.5-0.5B`, float32, accumulator float32):**

| Metric | Value |
|---|---|
| Decoder layers replayed | **24** |
| Total cells | 6 |
| Full-model parity pass | **6/6** |
| Streaming vs materialized pass (depth-aware logit tol) | **0/6** |
| Top-1 changed (full vs streaming) | **0/6** |
| Max streaming-vs-materialized logit error | **0.134** |
| Max streaming-vs-materialized hidden error | **0.501** |
| Streaming vs materialized top-5 overlap | **5.0** (mean) |
| Streaming vs materialized top-10 overlap | **10.0** (mean) |
| Best theoretical streaming reduction | **98.96%** |

**Note:** All cells fail strict depth-aware logit tolerance at 24 layers (`~2.45×10⁻³`), but **streaming vs materialized top-1 agreement is 6/6** with top-10 overlap 10/10. Full-model manual replay parity passes (logits max abs ~3.6×10⁻⁵).

---

## 11. What this proves

- Streaming compressed can match materialized compressed through **full decoder depth** to final logits offline
- Full-vs-compressed logit drift and top-1/top-k changes can be measured for fixed prompts
- Full-model manual replay parity against HF can be validated before interpreting drift

---

## 12. What this does not prove

- Model output preservation or exact generation equivalence
- Token generation with streaming attention
- Runtime integration into ExactKV generation
- Speed, throughput, latency, or measured GPU memory savings
- vLLM / LMCache integration
- VeriCache throughput reproduction

---

## 13. Relation to ExactKV restored verification

Restored-verifier tracks validate greedy continuation from stored full KV. Phase 16F measures **logit drift** when compressed streaming attention substitutes through the full stack — a separate research question.

---

## 14. Relation to VeriCache parity

VeriCache serving stacks emphasize throughput and memory under vLLM/LMCache. Phase 16F does **not** reproduce VeriCache throughput, serving, or memory panels.

---

## 15. Next step

**Phase 16G (proposed):** broader model panel sweep — only after 16F report review; still opt-in, no default runtime integration.

---

## Claims boundary

| Allowed | Forbidden |
|---|---|
| Offline full-prefix logit drift measurement | Model output preservation claim |
| Full-model parity reporting | Exact generation equivalence claim |
| Top-1 change as drift metric | Speedup / throughput / latency |
| Theoretical memory accounting | Measured GPU memory savings |
