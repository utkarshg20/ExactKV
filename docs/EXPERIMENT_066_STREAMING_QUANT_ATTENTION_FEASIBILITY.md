# Experiment 066: Streaming Quantized-KV Attention Feasibility (Phase 16A)

**Status:** tensor-level reference probe — run `scripts/research/run_exp066_streaming_quant_attention_feasibility.py` for report.

> This is a **tensor-level feasibility probe**, not model inference integration.  
> Streaming compressed attention is not wired into ExactKV generation.  
> No CUDA kernel is implemented.  
> No Triton kernel is implemented.  
> **No vLLM integration** is implemented.  
> **No speed, throughput, latency, serving, active GPU memory, or production-memory claim** is made.  
> ExactKV still does **not** reproduce VeriCache throughput or serving results.

Companion module: `exactkv/attention/streaming_quant_attention.py`

---

## 1. Purpose

Phase 16A asks whether attention can be computed over **compressed KV** without **materializing the full decompressed K/V tensor at once**, using a small PyTorch reference.

Existing ExactKV compressed draft paths are **materializing**: they store compressed-ish KV but reconstruct a full working KV before attention/generation. GPU memory diagnostics showed **no active memory savings**. This experiment tests a **streaming/chunked dequantized attention** reference at tensor level only.

---

## 2. Why vLLM work is deferred

Phase 15E idle-GPU vLLM object-level probing is **deferred**. The RunPod vLLM CUDA-13 template auto-starts `vllm serve Qwen/Qwen3-8B` (or a misconfigured `vllm serve sleep infinity` entrypoint), which blocks idle object-level cache inspection. The vLLM import path works on CUDA 13, but further vLLM infra work is paused for this phase.

See [`EXPERIMENT_065_IDLE_VLLM_OBJECT_KV_PROBE.md`](EXPERIMENT_065_IDLE_VLLM_OBJECT_KV_PROBE.md) · [`VLLM_PROTOTYPE_PATH.md`](VLLM_PROTOTYPE_PATH.md)

---

## 3. Materialization bottleneck

| Path | Behavior |
|---|---|
| **Stored quantized KV** | Smaller int8 payload + scales |
| **Materialized compressed attention** | Dequantize **entire** K and V, then attend |
| **Problem** | Peak working memory still scales with full sequence length × head dim |

Phase 16A probes whether **chunked dequantization + online softmax** can match materialized dequantized attention numerically while keeping peak working KV proportional to **chunk size**, not full `T`.

---

## 4. Full vs materialized-compressed vs streaming-compressed attention

| Method | Description |
|---|---|
| **Full** | Attention over full-precision `k`, `v` (baseline) |
| **Materialized compressed** | Dequantize all int8 KV → full `k`, `v` → attention |
| **Streaming compressed** | For each chunk: dequantize chunk only → accumulate stable softmax statistics → **no full K/V tensor** |

Pass criterion: `streaming_compressed_output ≈ materialized_compressed_output` within dtype tolerance.

---

## 5. Online softmax / chunking method

For each KV chunk:

1. Dequantize `k_chunk`, `v_chunk` only.
2. Compute `scores = q @ k_chunk^T / sqrt(D)`.
3. Update running max logit, denominator, and weighted value sum using the **stable online softmax** recurrence (renormalize previous state when a new chunk max appears).
4. Final output: `weighted_sum / denominator`.

Causal mode (tested): queries occupy the **last Q positions** in a sequence of length `T`; keys after each query position are masked.

---

## 6. Memory accounting fields

Theoretical tensor accounting only (**not** measured active GPU memory):

| Field | Meaning |
|---|---|
| `full_kv_bytes` | Full-precision K+V storage |
| `stored_quantized_kv_bytes` | int8 K/V + scale metadata |
| `materialized_working_kv_bytes` | Peak if full K+V dequantized at once |
| `streaming_peak_chunk_working_kv_bytes` | Peak dequantized K+V for one chunk |
| `metadata_bytes` | Scale tensor bytes |
| `chunk_size` | Chunk length used |
| `num_chunks` | `ceil(T / chunk_size)` |
| `theoretical_streaming_working_reduction_vs_materialized` | `1 - streaming_peak / materialized_peak` |

---

## 7. Results

Generate on CPU (no CUDA/vLLM required):

```bash
python3 scripts/research/run_exp066_streaming_quant_attention_feasibility.py
```

Report (gitignored): `reports/experiment_066_streaming_quant_attention_feasibility.json`

Sweep: `B=1`, `H∈{2,4}`, `Q∈{1,4}`, `T∈{32,128,512}`, `D∈{32,64}`, `chunk_size∈{16,32,64}`, `dtype∈{float32,float16}` → **144 cells**.

---

## 8. What this proves

- Chunked dequantized attention can match **materialized dequantized attention** at tensor level within tolerance.
- Theoretical peak working KV for streaming can be **smaller than** full materialization when `chunk_size < T`.

---

## 9. What this does not prove

- Exact model output preservation
- Runtime integration into ExactKV generation
- Speed, throughput, or latency improvement
- **Measured** active GPU memory savings
- Production feasibility or serving support
- vLLM / LMCache integration

---

## 10. Relation to ExactKV restored verification

Restored-verifier and full-KV restore tracks (Phase 12–14) validate **exact greedy continuation** from stored full KV. Phase 16A is orthogonal: it asks whether **lossy compressed KV** could be attended **without full materialization** — a prerequisite research question before any compressed-active path could reduce working memory in inference.

---

## 11. Relation to VeriCache parity

VeriCache serving stacks emphasize throughput and memory behavior under vLLM/LMCache. Phase 16A does **not** reproduce VeriCache throughput, serving, or memory panels. It is a local tensor reference only.

---

## 12. Next step

**Phase 16B (complete):** HF single-layer offline attention-drift probe — [`EXPERIMENT_067_HF_SINGLE_LAYER_ATTENTION_DRIFT.md`](EXPERIMENT_067_HF_SINGLE_LAYER_ATTENTION_DRIFT.md)

**Phase 16C (proposed):** multi-layer offline drift accumulation — still opt-in, no default runtime.

---

## Claims boundary

| Allowed | Forbidden |
|---|---|
| Tensor-level streaming-vs-materialized numerical match | Speedup / throughput / latency improvement |
| Theoretical working-memory accounting | Measured active GPU memory savings |
| Reference int8 quantizer for probe | Production compressor claim |
| Deferred vLLM idle probe noted | vLLM integration exists |
