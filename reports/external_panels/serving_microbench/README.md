# Serving microbench panel

**Status:** complete on RTX PRO 4000 Blackwell — **76 cells**, `exactkv_failures=0`.
Pack: `reports/systems/serving_microbench.{json,md}`.

See [`docs/SERVING_PATH_C.md`](../../../docs/SERVING_PATH_C.md).

## Design (strong panel · 72 cells + 4 extras)

| Axis | Values |
|------|--------|
| Models | Llama-3.1-8B, Mistral-7B-Instruct-v0.3 |
| Compressors | noop, int8, int4_sim |
| Context | 2048, 4096 |
| max_new | 64, 128 |
| Load | serial 1, 4, 8 requests (+ extras at 16) |
| Arms | full / lossy / ExactKV |

Metrics per arm: `ttft_like_ms`, `completed_requests_per_sec`,
`gpu_peak_allocated_bytes`, `peak_delta_vs_full_bytes`.

## Headline (strong-panel means)

| Metric | full | ExactKV |
|--------|-----:|--------:|
| Completed req/s | ~0.25 | ~0.13 (~1.9× lower) |
| TTFT-like (ms) | ~630 | ~940 (~1.5× higher) |

## Claim boundary

HF multi-request diagnostic harness — **not** vLLM, **not** continuous batching,
**not** unqualified production VRAM savings.

## Run

```bash
export HF_TOKEN=hf_...
bash scripts/run_serving_microbench_panel.sh 2>&1 | tee /workspace/serving_microbench.log
```
