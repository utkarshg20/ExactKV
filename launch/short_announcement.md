# ExactKV — Short Announcement

**ExactKV: a crash-test for compressed KV caches.**

KV-cache compression looks fine until the first wrong token. ExactKV measures
exactly where that happens — the first-divergence index, draft acceptance rate,
verifier agreement, and exactness failures — across compressors and real 7B/8B
models, on a public leaderboard.

Headline (source: `reports/scale_7b/raw.json`): a **1,500-cell** real-GPU
benchmark on **Llama-3.1-8B** and **Mistral-7B-Instruct-v0.3** with
**exactkv_failures = 0** on the tested panel.

Research-grade evaluation framework — **not** a production serving system, and it
does **not** reproduce VeriCache. Kernel results are a microbenchmark (not
end-to-end speedup); compression ratios are stored tensor byte ratios (not active
GPU memory savings); SpectralQuant is fallback/proxy; Shard is probe-first.

→ Report: `paper/ExactKV_Technical_Report.md`
→ Leaderboard: `reports/public_release/leaderboard_final.json`
→ Reproduce: `python3 scripts/exactkv_repro.py --reports-only`
