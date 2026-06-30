# ExactKV Claim Decision Table (Release Synthesis — Part 3)

Authoritative per-claim decisions, derived from `docs/CLAIM_BOUNDARIES.md`, `docs/NOVELTY_AUDIT.md`, `reports/release_evidence_status.json`, and `reports/scale_7b/raw.json`. Machine-readable: [`claim_decision_table.json`](claim_decision_table.json).

| # | Claim | Decision | Evidence | Evidence artifact | Public-safe wording / forbidden wording |
|---|-------|----------|----------|-------------------|------------------------------------------|
| 1 | ExactKV is a KV-cache compression exactness benchmark | ✅ allowed | strong | `reports/scale_7b/raw.json; docs/CLAIM_BOUNDARIES.md` | ExactKV is a compressor-agnostic crash-test and leaderboard framework for LLM KV-cache compression exactness. |
| 2 | ExactKV measures token-level first divergence | ✅ allowed | strong | `docs/METRIC_DEFINITIONS.md (first_divergence_index); reports/phaseG_unified_truth.json` | ExactKV reports the first token index where compressed-KV greedy output diverges from the full-KV reference. |
| 3 | ExactKV measures acceptance rate / accepted span | ✅ allowed | strong | `reports/scale_7b/raw.json (acceptance_table); docs/METRIC_DEFINITIONS.md` | ExactKV reports draft acceptance rate and mean accepted span per verification round (panel-scoped, greedy decoding). |
| 4 | ExactKV has a public leaderboard | ⚠️ qualified | strong | `reports/public_release/leaderboard_final.json; reports/scale_7b/leaderboard.json` | ExactKV publishes a reproducible public leaderboard bundle (repo-local artifacts, not a hosted SaaS). |
| 5 | ExactKV ran real 7B/8B evaluation | ✅ allowed | strong | `reports/scale_7b/raw.json (models_evaluated, deterministic_mode=false)` | ExactKV evaluates real meta-llama/Llama-3.1-8B and mistralai/Mistral-7B-Instruct-v0.3 over a 1500-cell panel. |
| 6 | ExactKV had zero exactness failures | ⚠️ qualified | strong | `reports/scale_7b/raw.json (exactkv_failures=0); reports/scale_7b/scale_summary.json` | On the cited 1500-cell scale panel, exactkv_failures = 0. |
| 7 | ExactKV has a Triton kernel path | ⚠️ qualified | moderate | `reports/phaseF_kernel_benchmark.json (triton_available=true, device=cuda)` | ExactKV includes a CUDA/Triton KV-compression kernel path (tested shape/hardware; block_sparse uses the torch backend). |
| 8 | ExactKV shows kernel microbenchmark speedups | ⚠️ qualified | moderate | `reports/phaseF_kernel_benchmark.json (int8 1.63x, int4 1.54x)` | Phase F kernel microbenchmark shows int8 ~1.63x and int4 ~1.54x torch->Triton latency ratios on kv_shape=[1,8,512,64]. |
| 9 | ExactKV shows end-to-end speedups | ⛔ forbidden | missing | `none (no end-to-end inference timing measured)` | (do not claim) ExactKV makes no end-to-end speedup claim.  
**Forbidden:** end-to-end speedup / faster inference |
| 10 | ExactKV shows active GPU memory savings | ⛔ forbidden | missing | `none (compression ratios are stored tensor byte ratios)` | (do not claim) Compression ratios are stored tensor byte ratios, not active GPU memory savings.  
**Forbidden:** active GPU memory savings / VRAM savings |
| 11 | ExactKV reproduces VeriCache | ⛔ forbidden | strong | `paper/VeriCache.pdf; docs/VERICACHE_PARITY_CLAIM_GATE.md` | ExactKV does not reproduce VeriCache serving, scheduling, or throughput.
**Forbidden:** reproduces VeriCache / beats VeriCache |
| 11b | ExactKV invents draft + full-KV verify | ⛔ forbidden | strong | `paper/VeriCache.pdf` | VeriCache owns that pattern for lossless serving; ExactKV measures drift only.
**Forbidden:** invents draft+verify |
| 12 | ExactKV beats TurboQuant | ⛔ forbidden | missing | `none (no same-task/same-model/same-metric head-to-head with real TurboQuant)` | (do not claim) No same-task superiority comparison against TurboQuant exists.  
**Forbidden:** beats TurboQuant |
| 13 | ExactKV beats Shard | ⛔ forbidden | missing | `none (Shard slot is probe-first heuristic, not full integration)` | (do not claim) Shard appears only as a probe-first heuristic slot; no superiority comparison.  
**Forbidden:** beats Shard |
| 14 | ExactKV compares real SpectralQuant | ⛔ forbidden | missing | `reports/scale_7b/raw.json (spectralquant backend_tier=MOCK, delegate=int4_sim)` | SpectralQuant runs in fallback/proxy mode (delegates to int4_sim) when the real dependency is unavailable.  
**Forbidden:** real SpectralQuant comparison |
| 15 | ExactKV compares real Shard | ⛔ forbidden | missing | `reports/scale_7b/raw.json (shard backend_tier=PROBE_ONLY, probe_only=true)` | Shard is a probe-first heuristic slot, not a full Shard / ShardCache integration.  
**Forbidden:** real Shard comparison / real Shard integration |
| 16 | ExactKV is production ready | ⛔ forbidden | missing | `docs/CLAIM_BOUNDARIES.md (forbidden)` | ExactKV is a research-grade evaluation framework, not a production serving system.  
**Forbidden:** production ready / production serving system |
| 17 | ExactKV is unique / first ever | ⛔ forbidden | missing | `docs/NOVELTY_AUDIT.md (uniqueness not established; VeriCache closest prior art)` | (do not claim) Uniqueness vs all exactness benchmarks is not established.  
**Forbidden:** first ever / first and only / nothing like this exists |
| 18 | ExactKV is research-grade | ✅ allowed | strong | `docs/CLAIM_BOUNDARIES.md; docs/NOVELTY_AUDIT.md` | ExactKV is a research-grade evaluation framework. |
| 19 | ExactKV is release-ready | ⚠️ qualified | strong | `reports/release_evidence_status.json (Gate R0 PASS)` | ExactKV's public release evidence gate (R0) passes; full pytest, secret scan, and claim audit must pass before publishing. |

## Summary counts

- allowed: 5
- allowed_with_qualification: 5
- forbidden: 9
