# Experiment 068: Qwen RoPE/GQA Long-Context Attention Probe (Phase 16C)

**Status:** offline single-layer long-context probe — run `scripts/research/run_exp068_qwen_rope_long_context_attention_probe.py` for report.

> This is an **offline single-layer long-context attention probe**, not model generation integration.  
> Streaming attention is **not wired into ExactKV generation**.  
> RoPE/GQA handling improves fidelity but does **not** by itself prove full model-layer equivalence unless parity is explicitly validated.  
> No CUDA kernel is implemented.  
> No Triton kernel is implemented.  
> No vLLM integration is implemented.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_067_HF_SINGLE_LAYER_ATTENTION_DRIFT.md`](EXPERIMENT_067_HF_SINGLE_LAYER_ATTENTION_DRIFT.md) · `exactkv/attention/hf_single_layer_probe.py`

---

## 1. Purpose

Phase 16C fixes Qwen2/Qwen2.5 RoPE and GQA extraction fidelity from Phase 16B and exercises **long-context** single-layer cells so chunked streaming attention uses **multiple chunks** with non-zero theoretical working-memory reduction.

---

## 2. Why Phase 16B needed follow-up

Phase 16B (Exp 067) showed streaming≈materialized on HF-derived Q/K/V, but:

- RoPE application failed (incorrect `apply_rotary_pos_emb` call) → **`projection_only`** extraction
- Short prompts (T≈5) → **`num_chunks=1`** → **0%** theoretical streaming working reduction
- GQA repeat worked but head counts were not surfaced in reports

Phase 16C addresses extraction fidelity and long-context chunking before any multi-layer accumulation work.

---

## 3. Qwen2/Qwen2.5 RoPE handling

Guarded path using HF utilities when available:

- `transformers.models.qwen2.modeling_qwen2.apply_rotary_pos_emb`
- Model `rotary_emb` module (`resolve_model_rotary_emb`)
- RoPE applied to **Q and K only** after projection reshape

| `rope_status` | Meaning |
|---|---|
| `applied` | RoPE succeeded → `exact_qwen2_like` |
| `failed: ...` | RoPE error; falls back to `projection_only` if allowed |
| `unsupported` | No rotary module or import failure |
| `skipped` | RoPE not attempted |

---

## 4. GQA handling

When `num_key_value_heads < num_heads`, K/V heads are repeated to match Q heads.

| `gqa_status` | Meaning |
|---|---|
| `not_needed` | MHA (equal head counts) |
| `repeated` | GQA repeat applied |
| `failed` | Head count mismatch |
| `unknown` | Unclassified |

Cells record `num_kv_heads_original` and `num_kv_heads_repeated`.

---

## 5. Long-context setup

Deterministic filler text extended until tokenized length reaches target; truncated to target.

| CLI flag | Default |
|---|---|
| `--target-token-lengths` | `64,128,256` |
| `--max-prompts` | `3` |
| `--chunk-sizes` | `16,32,64` |
| `--layers` | first / middle / last |
| `--model-id` | `Qwen/Qwen2.5-0.5B` |
| `--device` | `cpu` |
| `--dtype` | `float32` |

Optional `--target-token-lengths 512` when CPU runtime is acceptable.

---

## 6. Full vs materialized compressed vs streaming compressed comparison

Same Phase 16A reference path on HF-derived Q/K/V:

| Method | Description |
|---|---|
| Full | Full-precision K/V attention |
| Materialized compressed | Dequantize full int8 KV → attend |
| Streaming compressed | Chunked dequant + online softmax |

**Primary pass:** streaming ≈ materialized (within tolerance).  
**Secondary:** full vs compressed drift and output-projection drift — measured, not required to pass.

---

## 7. Drift metrics

Per comparison pair:

- `max_abs_error`
- `mean_abs_error`
- `relative_l2_error`
- `cosine_similarity`

Optional `o_proj` path applies output projection to attention context only — not full decoder layer output.

---

## 8. Memory accounting

Theoretical tensor accounting (not measured GPU VRAM):

- `full_kv_bytes`
- `stored_quantized_kv_bytes`
- `materialized_working_kv_bytes`
- `streaming_peak_chunk_working_kv_bytes`
- `metadata_bytes`
- `chunk_size`
- `num_chunks`
- `theoretical_streaming_working_reduction_vs_materialized`

Long contexts with `chunk_size < T` should show **`num_chunks > 1`** and **`streaming_peak_chunk_working_kv_bytes < materialized_working_kv_bytes`**.

---

## 9. Results

```bash
python3 scripts/research/run_exp068_qwen_rope_long_context_attention_probe.py
```

Report (gitignored): `reports/experiment_068_qwen_rope_long_context_attention_probe.json`

Default sweep: 3 target lengths × 3 layers × 3 chunk sizes = **27 cells** (when model loads).

Optional best-effort `layer_parity` smoke compares HF `self_attn` output vs manual attention+`o_proj` — non-blocking.

---

## 10. What this proves

- Qwen2/Qwen2.5 RoPE can be applied correctly on extracted Q/K when HF rotary module is available
- GQA repeat expands KV heads for attention comparisons
- Long-context cells exercise multi-chunk streaming with non-zero theoretical working-memory reduction
- Streaming compressed attention still matches materialized compressed on HF-derived tensors

---

## 11. What this does not prove

- Exact model output preservation
- Full-layer or full-model fidelity (unless parity smoke passes explicitly)
- Runtime integration into ExactKV generation
- Speed, throughput, latency, or serving improvement
- Measured active GPU memory savings
- vLLM / LMCache integration
- VeriCache throughput reproduction

---

## 12. Relation to ExactKV restored verification

Restored-verifier tracks validate greedy continuation from stored full KV. Phase 16C measures attention-context drift from lossy int8 KV on a **single layer slice** with improved extraction — separate research question.

---

## 13. Relation to VeriCache parity

VeriCache serving stacks emphasize throughput and memory under vLLM/LMCache. Phase 16C does **not** reproduce VeriCache throughput, serving, or memory panels.

---

## 14. Next step

**Phase 16D (proposed):** multi-layer offline accumulation micro-benchmark — still opt-in, no default runtime, no serving claims. Do **not** proceed until Phase 16C RoPE/long-context report is captured.

---

## Claims boundary

| Allowed | Forbidden |
|---|---|
| Offline RoPE/GQA long-context drift measurement | Model output preservation claim |
| `exact_qwen2_like` when RoPE applied | Exact full-layer claim without parity |
| Streaming≈materialized on HF-derived Q/K/V | Speedup / throughput / latency |
| Theoretical memory accounting with num_chunks>1 | Measured GPU memory savings |
