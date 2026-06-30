# ExactKV — X / Twitter

> Claim-safe. Numbers from release artifacts. No "beats X", no speedup/VRAM claims.

---

## Main tweet

Benchmarks don't tell you when compressed LLM memory starts lying.

ExactKV does. We crash-test every token against full precision, before the whole output goes sideways.

We compressed 4× and checked every token:
6% wrong on code · 90% on reading · same model
8,132 GPU cells · Llama + Mistral · 106/106 tool calls still valid

---

## Thread — follow-up

Research eval only — not production serving, does not reproduce VeriCache, not throughput/VRAM claims. External panels measure drift, not official benchmark scores. Phase F = kernel microbenchmark only. Compression ratios = stored tensor byte ratios. SpectralQuant = fallback/proxy; Shard = probe-first.

Technical report + repro in repo:
https://github.com/utkarshg20/ExactKV/blob/main/paper/ExactKV_Technical_Report.md
https://github.com/utkarshg20/ExactKV/blob/main/site/index.html
