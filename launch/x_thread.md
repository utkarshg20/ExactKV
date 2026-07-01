# ExactKV — X / Twitter

> Claim-safe. Numbers from release artifacts. No "beats X", no speedup/VRAM claims.

---

## Main tweet

Benchmarks don't tell you when compressed LLM memory starts lying.

ExactKV does. We crash-test every token against full precision, before the whole output goes sideways.

We compressed 4× and checked every token:
6% token drift on code · 90% on reading · same model, same 4× compressor

8,132 GPU crash-test cells (Llama-3.1-8B + Mistral-7B) · exactness failures: 0
Even when half the tool-calling runs drifted, every valid JSON tool call still matched full precision (106/106 preserved)

Technical report + reproduction in repo:
https://github.com/utkarshg20/ExactKV/blob/main/paper/ExactKV_Technical_Report.md
https://github.com/utkarshg20/ExactKV/blob/main/site/index.html

---

## Thread — follow-up (post this)

Quick context on those numbers:

The 6% / 90% split is **one compressor** (`int4_sim`, 4× quant) — the case where drift is easy to see. Same models, different task type.

The report already has the wider picture — int8 near-clean on code, int6 + per-vector int4 in between, H2O-style eviction at 100% on reading. Different mechanisms, same crash test.

We also wired **real upstream adapters** (kvpress SnapKV, KIVI r32) into the same grid — **864 cells**, both models, separate from the 8,132 headline set.

Not to crown a winner: **int8** is still the only non-catastrophic real compressor (~8–9% drift). Faithful SnapKV: **~90–97%** drift. KIVI offline: **100%** (integration diagnostic, not production KIVI). **Exactness failures: 0** throughout.

KnormPress + TurboQuant wave-2 smoke (128 cells, Mistral only) interrupted when RunPod stopped. Checking whether the network volume still has MBPP 64/64 + partial BFCL; otherwise wave-2 rerun only (~1–2 hr). Early 20-cell MBPP checkpoint locally — not claim-ready.

Open source. Repro in the repo. More compressors + benchmarks welcome.

---

## Optional reply (only if someone misreads the main post — don't post by default)

Research eval only — not production serving, does not reproduce VeriCache. "Drift" = compressed KV picked a different token than full precision, not official LongBench/BFCL scores. No end-to-end speedup claims; compression ratios are stored tensor byte ratios; Phase F is kernel microbenchmark only. Core leaderboard: noop/int8/int4_sim; other registry slots are fallback/proxy or probe-first — not headline evidence. Full boundaries in the report.

---

## Internal (validators only — do not post)

Phase F microbenchmark. Stored tensor byte ratios. SpectralQuant fallback/proxy. Shard probe-first.
