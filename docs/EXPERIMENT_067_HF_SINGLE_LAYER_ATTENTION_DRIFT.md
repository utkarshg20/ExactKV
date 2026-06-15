# Experiment 067: HF Single-Layer Attention-Drift Probe (Phase 16B)

**Status:** offline single-layer probe — run `scripts/research/run_exp067_hf_single_layer_attention_drift.py` for report.

> This is an **offline single-layer attention-drift probe**, not model generation integration.  
> Streaming attention is not wired into ExactKV generation.  
> Q/K/V extraction may be **projection-only** unless architecture-specific exactness is verified.  
> No CUDA kernel is implemented.  
> No Triton kernel is implemented.  
> No vLLM integration is implemented.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_066_STREAMING_QUANT_ATTENTION_FEASIBILITY.md`](EXPERIMENT_066_STREAMING_QUANT_ATTENTION_FEASIBILITY.md) · `exactkv/attention/hf_single_layer_probe.py`

---

## 1. Purpose

Phase 16B moves one step toward model relevance by applying Phase 16A attention methods to **Q/K/V tensors derived from a real Hugging Face transformer layer**.

It answers:

- How much **attention-output drift** does int8 reference KV quantization introduce on real prompt-derived hidden states?
- Does **streaming compressed attention** still match **materialized compressed attention** on those tensors?

---

## 2. Relation to Phase 16A

Phase 16A proved tensor-level streaming≈materialized on synthetic random Q/K/V (144/144 cells). Phase 16B reuses the same int8 reference quantizer and chunked attention implementation on **HF-derived** tensors.

---

## 3. Why this is still offline and opt-in

- No changes to `ExactKVGenerator` or default runtime
- No generation, sampling, or serving
- CPU-safe default (`--device cpu`)
- Single-layer slice only (no full block forward fidelity claim)

---

## 4. HF Q/K/V extraction method

Default model: `Qwen/Qwen2.5-0.5B`

For each prompt:

1. Tokenize deterministically
2. Forward with `output_hidden_states=True` (no cache, no generation)
3. For each selected layer, take **input hidden state** `hidden_states[layer_idx]`
4. Apply `q_proj`, `k_proj`, `v_proj` from `layer.self_attn`
5. Reshape to `[B, H, T, D]`
6. Repeat K/V heads for grouped-query attention when needed
7. Optionally apply RoPE via HF `apply_rotary_pos_emb` when rotary embeddings are accessible

---

## 5. Exact vs projection-only extraction caveat

| Mode | Meaning |
|---|---|
| `exact_qwen2_like` | RoPE applied successfully after projections |
| `projection_only` | Projections + GQA repeat only; **not** exact model-layer attention |
| `blocked` | Extraction failed; cell records blockers |

If RoPE is unavailable or fails and `--projection-only-ok` is true (default), results are labeled `projection_only`.

---

## 6. Attention comparison methods

Reuses Phase 16A:

| Method | Description |
|---|---|
| Full | Attention over full-precision K/V |
| Materialized compressed | Dequantize full int8 KV → attend |
| Streaming compressed | Chunked dequant + online softmax |

Default attention mode: **causal** (queries attend to prior keys; queries span full sequence).

**Pass criterion:** streaming ≈ materialized within tolerance (same as 16A).

**Drift measurement:** full vs materialized and full vs streaming — expected to differ under quantization.

---

## 7. Drift metrics

Per comparison pair:

- `max_abs_error`
- `mean_abs_error`
- `cosine_similarity`
- `relative_l2_error`
- `top_dim_max_abs` (optional summary)

Optional `o_proj` path applies output projection to attention context tensors only — **not** a full layer output (no residual, MLP, or layernorm).

---

## 8. Memory accounting

Reuses Phase 16A theoretical fields:

`full_kv_bytes`, `stored_quantized_kv_bytes`, `materialized_working_kv_bytes`, `streaming_peak_chunk_working_kv_bytes`, `metadata_bytes`, `chunk_size`, `num_chunks`, `theoretical_streaming_working_reduction_vs_materialized`

Theoretical tensor accounting only — not measured GPU VRAM.

---

## 9. Results

```bash
python3 scripts/research/run_exp067_hf_single_layer_attention_drift.py
```

Report (gitignored): `reports/experiment_067_hf_single_layer_attention_drift.json`

Default sweep: 4 prompts × 3 layers × 3 chunk sizes = **36 cells** (when model loads).

---

## 10. What this proves

- HF-derived Q/K/V can be fed through the 16A attention reference path
- Streaming compressed attention can still match materialized compressed attention on real hidden-state projections
- Quantization drift (full vs compressed) can be measured on prompt-derived tensors

---

## 11. What this does not prove

- Exact model output preservation
- Full-layer or full-model fidelity
- Runtime integration into ExactKV generation
- Speed, throughput, latency, or serving improvement
- Measured active GPU memory savings
- vLLM / LMCache integration

---

## 12. Relation to ExactKV restored verification

Restored-verifier tracks (Phase 12–14) validate greedy continuation from **stored full KV**. Phase 16B measures **attention-context drift** from **lossy int8 KV** on a single layer slice — a separate research question.

---

## 13. Relation to VeriCache parity

VeriCache serving stacks emphasize throughput and memory under vLLM/LMCache. Phase 16B does **not** reproduce VeriCache throughput, serving, or memory panels.

---

## 14. Next step

**Phase 16C (proposed):** multi-layer offline micro-benchmark comparing cumulative drift through several layers — still opt-in, no default runtime, no serving claims.

---

## Claims boundary

| Allowed | Forbidden |
|---|---|
| Offline single-layer drift measurement | Model output preservation claim |
| projection_only labeling | Exact full-layer attention claim |
| Streaming≈materialized on HF-derived Q/K/V | Speedup / throughput / latency |
| Theoretical memory accounting | Measured GPU memory savings |
