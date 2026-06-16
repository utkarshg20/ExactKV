# Experiment 069: Multi-Layer Attention Drift Accumulation (Phase 16D)

**Status:** offline multi-layer probe — run `scripts/research/run_exp069_multilayer_attention_drift_accumulation.py` for report.

> This is an **offline multi-layer drift accumulation probe**, not model generation integration.  
> Streaming attention is **not wired into ExactKV generation**.  
> Full-block parity is measured and must be reported before interpreting drift.  
> No CUDA kernel is implemented.  
> No Triton kernel is implemented.  
> No vLLM integration is implemented.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_068_QWEN_ROPE_LONG_CONTEXT_ATTENTION_PROBE.md`](EXPERIMENT_068_QWEN_ROPE_LONG_CONTEXT_ATTENTION_PROBE.md) · `exactkv/attention/hf_multilayer_probe.py`

---

## 1. Purpose

Phase 16D moves from single-layer attention drift (16B/16C) to **offline multi-layer drift accumulation**. It replays consecutive Qwen2.5 decoder blocks with full, materialized-compressed, and streaming-compressed attention substituted per layer.

It answers:

- Does streaming compressed still match materialized compressed across multiple layers?
- How does full-vs-compressed hidden-state drift accumulate with prefix depth?

---

## 2. Relation to Phase 16C

Phase 16C fixed RoPE/GQA extraction (`exact_qwen2_like`) and exercised long-context chunking on a **single layer**. Phase 16D reuses those helpers inside full decoder block replay (layernorm → attention → residual → MLP → residual) across **N prefix layers**.

---

## 3. Why this is still offline and opt-in

- No changes to `ExactKVGenerator` or default runtime
- No token generation, sampling, or serving
- CPU-safe defaults
- Research-only compressed attention substitution

---

## 4. Offline decoder block replay method

For each layer using the model’s real modules:

1. `input_layernorm`
2. `q_proj` / `k_proj` / `v_proj` + RoPE + GQA repeat (Phase 16C helpers)
3. Attention (`full` / `materialized_compressed` / `streaming_compressed`)
4. `o_proj` + residual
5. `post_attention_layernorm`
6. MLP (`gate_proj`, `up_proj`, `down_proj`, activation)
7. residual

Paths share identical weights; only the attention kernel differs per layer.

---

## 5. Full-block parity smoke

Before trusting drift numbers, offline `full_path` replay hidden state after N layers is compared to HF `hidden_states[N]` from `output_hidden_states=True`.

Metrics: max/mean abs error, relative L2, cosine similarity.

| `full_block_parity_status` | Meaning |
|---|---|
| `passed` | Replay within tolerance |
| `failed` | Drift beyond tolerance — reported, not hidden |
| `blocked` | Replay or reference unavailable |

Default: cells fail if parity fails unless `--allow-parity-fail`.

---

## 6. Full vs materialized compressed vs streaming compressed paths

| Path | Per-layer attention |
|---|---|
| `full_path` | Full-precision K/V |
| `materialized_compressed_path` | int8 reference KV, full dequant |
| `streaming_compressed_path` | int8 reference KV, chunked streaming |

**Primary pass:** final hidden state streaming ≈ materialized (within tolerance).

**Secondary:** full vs streaming and full vs materialized hidden-state drift — measured, not required to pass.

---

## 7. Multi-layer drift accumulation metrics

Per cell:

- `streaming_vs_materialized_hidden_metrics`
- `full_vs_streaming_hidden_metrics`
- `full_vs_materialized_hidden_metrics`

Each includes max/mean abs error, relative L2, cosine similarity.

---

## 8. Memory accounting

Per-layer and aggregate theoretical fields (not measured GPU VRAM):

- `full_kv_bytes_per_layer` (via `per_layer_memory_accounting`)
- `stored_quantized_kv_bytes_per_layer`
- `materialized_working_kv_bytes_per_layer`
- `streaming_peak_chunk_working_kv_bytes_per_layer`
- `aggregate_*` sums / conservative streaming peak
- `best_theoretical_streaming_reduction`

---

## 9. Results

```bash
python3 scripts/research/run_exp069_multilayer_attention_drift_accumulation.py
```

Report (gitignored): `reports/experiment_069_multilayer_attention_drift_accumulation.json`

**Local CPU run (`Qwen/Qwen2.5-0.5B`, float32):**

| Metric | Value |
|---|---|
| Model load | succeeded |
| Total cells | 18 |
| Successful cells | 18 |
| Blocked cells | 0 |
| Full-block parity pass | 18/18 |
| Streaming vs materialized pass | 17/18 |
| Max streaming-vs-materialized hidden error | 5.79e-4 |
| Longest context | 128 tokens |
| Max prefix layers | 4 |
| Max num chunks | 8 |
| Best theoretical streaming reduction | 96.875% (4 layers, chunk 16) |

**Failed cell:** `long_128` / prefix 4 / chunk 32 — streaming-vs-materialized max abs 5.79e-4 (tolerance 5e-4). Cosine similarity still ≈1.00018; full-block parity passed. Likely multi-layer numerical accumulation at this chunk boundary, not a parity blocker.

**Full-vs-compressed drift (secondary):** max abs ~1.09 (layer 1) to ~0.79 (4 layers); cosine remains high (≥0.98 single layer, ~1.0001 at 4 layers vs full).

Default sweep: 2 prompts × 3 prefix depths × 3 chunk sizes = **18 cells** (when model loads).

CLI highlights:

| Flag | Default |
|---|---|
| `--target-token-lengths` | `64,128` |
| `--prefix-layer-counts` | `1,2,4` |
| `--chunk-sizes` | `16,32,64` |
| `--max-prompts` | `2` |
| `--allow-parity-fail` | `false` |

---

## 10. What this proves

- Streaming compressed can match materialized compressed across **multiple consecutive layers** offline
- Full-vs-compressed hidden drift can be measured as prefix depth grows
- Full-block replay parity against HF hidden states can be validated before interpreting drift

---

## 11. What this does not prove

- Exact model output preservation or generation equivalence
- Full model fidelity without explicit parity validation
- Runtime integration into ExactKV generation
- Speed, throughput, latency, or serving improvement
- Measured active GPU memory savings
- vLLM / LMCache integration
- VeriCache throughput reproduction

---

## 12. Relation to ExactKV restored verification

Restored-verifier tracks validate greedy continuation from stored full KV. Phase 16D measures **hidden-state drift accumulation** when compressed streaming attention is substituted layer-by-layer — a separate research question.

---

## 13. Relation to VeriCache parity

VeriCache serving stacks emphasize throughput and memory under vLLM/LMCache. Phase 16D does **not** reproduce VeriCache throughput, serving, or memory panels.

---

## 14. Next step

**Phase 16E complete:** streaming multi-layer numerics audit — [`EXPERIMENT_070_STREAMING_MULTILAYER_NUMERICS_AUDIT.md`](EXPERIMENT_070_STREAMING_MULTILAYER_NUMERICS_AUDIT.md)

**Phase 16F (proposed):** broader model panel sweep — only after 16E recommendation review; still no default runtime integration.

---

## Claims boundary

| Allowed | Forbidden |
|---|---|
| Offline multi-layer drift measurement | Model output preservation claim |
| Full-block parity reporting | Exact full-model equivalence claim |
| Streaming≈materialized across layers | Speedup / throughput / latency |
| Theoretical memory accounting | Measured GPU memory savings |
