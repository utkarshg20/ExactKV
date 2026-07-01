# ExactKV — Short Announcement

**ExactKV: a crash-test for compressed KV caches.**

KV-cache compression looks fine until the first wrong token. ExactKV measures
exactly where that happens — the first-divergence index, draft acceptance rate,
verifier agreement, and exactness failures — across compressors and real 7B/8B
models, on a public leaderboard.

**Headline evidence:** **8,132** completed GPU cells across external benchmark
panels with **exactkv_failures = 0** throughout. Core compressor rankings come
from the **1,500-cell** scale panel on **Llama-3.1-8B** and
**Mistral-7B-Instruct-v0.3** (`reports/scale_7b/raw.json`).

**Appendix (separate from headline):** **864-cell** faithful upstream adapter smoke
(KIVI r32 + SnapKV via kvpress + int8 baseline, both models). Confirms harness
integration; does not establish a strong faithful-compressor baseline (see report §6.17).

Research-grade evaluation framework — **not** a production serving system, and it
does **not** reproduce VeriCache. Kernel results are a microbenchmark (not
end-to-end speedup); compression ratios are stored tensor byte ratios (not active
GPU memory savings); SpectralQuant is fallback/proxy; Shard is probe-first.

→ GitHub: https://github.com/utkarshg20/ExactKV  
→ Report: https://github.com/utkarshg20/ExactKV/blob/main/paper/ExactKV_Technical_Report.md  
→ Landing page: https://github.com/utkarshg20/ExactKV/blob/main/site/index.html  
→ Leaderboard: https://github.com/utkarshg20/ExactKV/blob/main/reports/public_release/leaderboard_final.json  
→ Reproduce: `python3 scripts/exactkv_repro.py --reports-only`
