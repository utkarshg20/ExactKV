# Token-Level Drift in KV Cache Compression: A Cross-Model Evaluation of ExactKV

## Abstract

KV cache compression reduces memory footprint but introduces token-level drift between lossy draft generation and full-precision verification. We present ExactKV, a compressor-agnostic crash-test and leaderboard framework for LLM KV-cache compression exactness. It measures acceptance rate, first-divergence index, verifier agreement, and cross-model instability under compressed KV conditions. Across 336 Phase A benchmark cells (4 models, 7 compressors) and a Phase H+ scale panel of 1500 real-GPU cells on Llama-3.1-8B and Mistral-7B-Instruct-v0.3, INT8 achieves strong leaderboard scores with zero ExactKV failures in the reported panels. Simulated INT4, asymmetric K8/V4, and restricted external adapters exhibit elevated divergence rates and lower verifier agreement. ExactKV provides trace-only verification without runtime commit, enabling reproducible comparison of compression robustness prior to deployment.

## 1. Introduction

Large language model inference stores growing key-value (KV) caches. Compression methods—INT8 quantization, simulated INT4, asymmetric K/V schemes, and external adapters—trade memory for approximation error. When approximation error appears at the token level, greedy decoding diverges from the full-KV reference. ExactKV measures this drift through draft-verify acceptance without modifying the core generator at commit time.

## 2. System Overview

ExactKV evaluates three generation modes per cell: full-KV greedy (reference), lossy compressed-KV greedy, and ExactKV draft-verify loops. Phase A (`phaseA_scale_benchmark`) runs a unified panel; Phase B (`exactkv_leaderboard_platform`) normalizes scores across models. Optional Exp 116 extracts instability regimes; Exp 117 renders phase diagrams from stress-panel outputs.

## 3. Experimental Setup

| Parameter | Value |
|-----------|-------|
| Models | Qwen 0.5B, Qwen 0.5B-Instruct, Mistral-7B, Llama-3.1-8B |
| Compressors | noop, int8, int4_sim, k8_v4_sim, spectralquant, kvquant, shard |
| Prompts | 4 deterministic panel prompts |
| Lengths | [4, 8, 16] |
| Total cells | 336 |
| Deterministic mode | False |

Data sources: `reports/phaseA_benchmark.json`, `reports/leaderboard.json`.

## 4. Key Results

### 4.1 INT8 dominance

Global leaderboard mean score: **int8 = 0.916**, **noop = 0.995**. INT8 maintains zero reported divergence rate across all four models in Phase B aggregation while preserving high acceptance (mean 0.774).

### 4.2 Divergence across compressors

Simulated INT4 (`int4_sim`) and probe-only `shard` show the highest per-model divergence rates on Qwen 0.5B (0.333). Restricted mocks (`spectralquant`, `kvquant`) rank below built-in INT8/NOOP on composite score.

### 4.3 Model sensitivity

Qwen 0.5B-Instruct exhibits the widest compressor score spread (0.419) in leaderboard insights. Llama-3.1-8B shows elevated `k8_v4_sim` divergence (0.583) relative to INT8/NOOP baselines.

## 5. Canonical Demo Cases

- **structured_output_drift** (`int4_sim`, Qwen 0.5B): first divergence at token 1, acceptance 0.5
- **qa_partial_drift** (`shard`, Qwen 0.5B): first divergence at token None, acceptance 0.66
- **worst_case_compression** (`int4_sim`, Qwen 0.5B): first divergence at token 1, acceptance 0.3333333333333333
- **cross_model_disagreement** (`k8_v4_sim`, Qwen 0.5B-Instruct): first divergence at token None, acceptance 1.0
- **first_divergence_explosion** (`kvquant`, Qwen 0.5B): first divergence at token None, acceptance 1.0

## 6. Failure Taxonomy

| Class | Description | Evidence in panel |
|-------|-------------|-------------------|
| Early divergence | Mismatch at token 0–1 | `kvquant` / `p3_code_fn` first_divergence_index=0 |
| Structured output drift | JSON/tool-call prefix corruption | `p2_json_tool` + `int4_sim` |
| Partial QA drift | Prefix accepted, suffix diverges | `shard` probe, acceptance=0.25 |
| Cross-model split | Same cell, different models disagree | `k8_v4_sim` @ 16 tokens, 0.5B vs Instruct |
| Verifier rejection | Trace REJECT with mismatch_index | Exp 115 cells (when trace available) |

Exp 116 regime coverage (when available): stable=114, moderate_drift=16, high_divergence=14.

## 7. Discussion

Compression error concentrates at token boundaries: the earliest observed mismatch occurs at index 0, implying quantisation noise can alter the very first generated token. Asymmetric schemes (`k8_v4_sim`) increase divergence on larger models, consistent with key/value bit-width asymmetry fragility. Verifier-mediated ExactKV prevents silent failure by rejecting divergent drafts—acceptance rate drops even when final ExactKV output matches full KV.

## 8. Limitations

- Phase A deterministic runs do not log decoded output text; demos use token-index timelines and optional Exp 115 token IDs.
- ExactKV is **not a production serving system** and does **not reproduce VeriCache** serving throughput.
- Phase F kernel speedups (when cited) are **kernel microbenchmark** results only — **not end-to-end** inference speedups.
- Compression ratios are **stored tensor byte ratios** unless active GPU memory is explicitly measured; we do **not** claim active GPU memory savings.
- **SpectralQuant** uses **fallback/proxy** mode when the real dependency is unavailable; **Shard** is **probe-first** heuristic analysis, not a full Shard integration.
- No runtime kernel integration or unqualified serving claims.

## 9. Conclusion

ExactKV provides a reproducible, trace-only benchmark for KV compression robustness. INT8 remains the near-optimal baseline across models; aggressive compression and external probes increase divergence and reduce verifier agreement. The leaderboard platform enables canonical ranking without new inference.

## Visual References

- Phase C synthesis: `reports/visuals/phaseC/phaseC_visual_synthesis.json`
- Exp 117 atlas: `reports/visuals/exp117/phase_diagram.png, reports/visuals/exp117/model_comparison.png, reports/visuals/exp117/length_sensitivity.png, reports/visuals/exp117/interaction_heatmaps.png, reports/visuals/exp117/stability_surface.png, reports/visuals/exp117/boundary_overlay.png`
