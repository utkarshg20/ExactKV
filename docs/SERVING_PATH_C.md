# Serving path C — how ExactKV answers “real serving” metrics

**Status:** HF multi-request **serving microbench** complete on RTX PRO 4000
Blackwell (76 cells, `exactkv_failures=0`). Pack:
`reports/systems/serving_microbench.{json,md}`.
True **ExactKV-inside-vLLM** remains blocked until full-KV export exists.

## Friend / GPT metrics → where they live

| Asked for | Where ExactKV answers it | Caveat |
|-----------|--------------------------|--------|
| GPU memory | `systems_diagnostic` + `serving_microbench` peak CUDA | Weights+KV+temps; Δ vs full can be **positive** (ExactKV higher) |
| Runs faster? | Path wall-clock + TTFT-like + e2e | ExactKV is typically **slower** (verify cost) |
| Requests handled | `completed_requests_per_sec` under **serial** load | Not continuous batching / not vLLM RPS |

Artifacts:
- `reports/systems/systems_diagnostic.json` (96 cells, RTX PRO 4000)
- `reports/systems/serving_microbench.json` (76 cells, same GPU)

## Measured headline (strong panel means)

| Metric | full | ExactKV | Ratio |
|--------|-----:|--------:|------:|
| Completed req/s | ~0.25 | ~0.13 | ExactKV ~1.9× lower |
| TTFT-like (ms) | ~630 | ~940 | ExactKV ~1.5× higher |

Strong design: both models × `{noop,int8,int4_sim}` × ctx `{2048,4096}` × mnt
`{64,128}` × `n_requests` `{1,4,8}` (+ 4 `serial_16` extras).

## Why “C = ExactKV in vLLM” is still blocked

1. No stable public API to export per-step authoritative full-precision KV from vLLM’s paged cache (Exp 017 no-go).
2. Shared worker mutation races with ExactKV verify/commit.
3. RunPod vLLM templates auto-serve a model and leave no idle VRAM for probes (Exp 064/065).

Contracts exist (`exactkv/integrations/vllm_contract.py`); runtime does not.

## How we made “C” happen *practically*

**Deliverable:** ExactKV owns a **serving microbench harness** that publishes the
same *kinds* of numbers people ask for (TTFT-like, requests/sec, peak memory)
for full / lossy / ExactKV — claim-scoped as HF serial load, not production vLLM.

```bash
# On a torch GPU pod (NOT the auto-serve vLLM template):
export HF_TOKEN=hf_...
bash scripts/run_serving_microbench_panel.sh
```

## Unlock path for true vLLM ExactKV (future)

| Step | Action |
|------|--------|
| 1 | Idle CUDA pod **without** auto-serve template |
| 2 | Finish Exp 065 / private-API KV visibility spike |
| 3 | If export impossible → **Pattern C dual runtime**: vLLM (or any engine) for draft-only optional; ExactKV HF keeps authoritative full KV for verify |
| 4 | Only then flip `serving_claim_allowed` / throughput gates after exactness + measured panels |

Pattern C dual runtime is the honest way to get VeriCache-*style* serving semantics
without pretending vLLM exposes FullKVState today.
