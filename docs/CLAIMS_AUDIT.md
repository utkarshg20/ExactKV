# Claims Audit (V13 Phase 9A)

**Status:** Living document — review before any public-facing text, README hero, launch post, or release notes.

> ExactKV is a **correctness-first KV-cache compression crash-test lab**.
> This audit defines what may and may not be claimed based on published V10–V13 evidence.

Companion: [`LAUNCH_READINESS_GAP_AUDIT.md`](LAUNCH_READINESS_GAP_AUDIT.md) · [`EXPERIMENT_027_PERFORMANCE_MEMORY_TRUTH_BOUNDARY.md`](EXPERIMENT_027_PERFORMANCE_MEMORY_TRUTH_BOUNDARY.md)

---

## 1. Allowed claims (with scope)

Each claim must cite the **specific experiment or panel** it rests on. Do not generalize beyond tested scope.

| Claim | Scope / citation | Wording guidance |
|---|---|---|
| `exactkv_failures == 0` | Named experiment (e.g. Exp 012, 029, 033) | “On [panel], ExactKV preserved full-greedy output (`exactkv_failures == 0`).” |
| Compressed KV used as draft state | Core architecture | “Lossy compressed KV proposes draft tokens only.” |
| Full-KV verifier preserves exact greedy output | Tested panels only | “On cited panels, final ExactKV output matches full greedy exactly.” |
| Span verification passed exactness grid | Exp 029 | “600-cell span grid: `exactkv_failures == 0`; span ≡ sequential on exactness.” |
| Terminal demo shows verified semantic drift correction | Exp 034b / `pharm_001` | **Primary** replay: lossy `drop` rejected, verifier `pickup` committed |
| LongBench-style outcome-green drift demo (secondary) | Exp 037 / `lb_md_001` | LongBench-**style** only; transparent heuristic; not official LongBench score |
| Benchmark gap framing | Phase 10I | Outcome benchmarks vs ExactKV path equivalence — [`BENCHMARK_GAP_ANALYSIS.md`](BENCHMARK_GAP_ANALYSIS.md); complementary, not replacement |
| VeriCache parity (algorithm only) | Phase 11A | Draft/verify semantics on HF harness — [`VERICACHE_PARITY_AUDIT.md`](VERICACHE_PARITY_AUDIT.md); **not** full system reproduction |
| Dual-cache contract layer | Phase 11B | [`DUAL_CACHE_ABSTRACTION.md`](DUAL_CACHE_ABSTRACTION.md); metadata only — **not** storage manager or savings claim |
| Full-KV storage manager spike | Phase 11C | [`FULL_KV_STORAGE_MANAGER.md`](FULL_KV_STORAGE_MANAGER.md); in-memory + file backends; **not** runtime wired |
| Materialized draft backend spike | Phase 11D | [`MATERIALIZED_COMPRESSED_DRAFT_BACKEND.md`](MATERIALIZED_COMPRESSED_DRAFT_BACKEND.md); metadata only; **not** hot compressed attention |
| Extended verification scheduler | Phase 11E | [`EXTENDED_VERIFICATION_SCHEDULER.md`](EXTENDED_VERIFICATION_SCHEDULER.md); bonus-token disabled; **not** runtime wired |
| vLLM prototype path contract | Phase 11F | [`VLLM_PROTOTYPE_PATH.md`](VLLM_PROTOTYPE_PATH.md); gates + metadata only; **not** vLLM integrated |
| LMCache prototype path contract | Phase 11G | [`LMCACHE_PROTOTYPE_PATH.md`](LMCACHE_PROTOTYPE_PATH.md); gates + metadata only; **not** LMCache or remote prefix integrated |
| Remote prefix cache semantics | Phase 11H | [`REMOTE_PREFIX_CACHE_SEMANTICS.md`](REMOTE_PREFIX_CACHE_SEMANTICS.md); loopback mock only; **not** remote runtime |
| Throughput benchmark harness | Phase 11I | [`THROUGHPUT_BENCHMARK_HARNESS.md`](THROUGHPUT_BENCHMARK_HARNESS.md); methodology contract; **not** speedup claim |
| Paper-like reproduction panel | Phase 11J | [`PAPER_LIKE_REPRODUCTION_PANEL.md`](PAPER_LIKE_REPRODUCTION_PANEL.md); contract only; **not** paper reproduction |
| VeriCache parity RC claim gate | Phase 11K | [`VERICACHE_PARITY_CLAIM_GATE.md`](VERICACHE_PARITY_CLAIM_GATE.md); allowed/forbidden classification; **not** full parity |
| Full-KV restore smoke (real HF KV) | Phase 12A / Exp 046 | [`EXPERIMENT_046_FULL_KV_RESTORE_SMOKE.md`](EXPERIMENT_046_FULL_KV_RESTORE_SMOKE.md); tiny panel; in-memory + file; **not** runtime wired |
| Full-KV restore panel hardening | Phase 12B / Exp 047 | [`EXPERIMENT_047_FULL_KV_RESTORE_PANEL.md`](EXPERIMENT_047_FULL_KV_RESTORE_PANEL.md); 12-prompt panel; optional CUDA dtypes; **not** serving |
| Offline verifier restore smoke | Phase 12C / Exp 048 | [`EXPERIMENT_048_OFFLINE_VERIFIER_RESTORE_SMOKE.md`](EXPERIMENT_048_OFFLINE_VERIFIER_RESTORE_SMOKE.md); reloaded full-KV verifier; controlled draft; **not** runtime wired |
| Offline verifier lossy draft | Phase 12D / Exp 049 | [`EXPERIMENT_049_OFFLINE_VERIFIER_LOSSY_DRAFT.md`](EXPERIMENT_049_OFFLINE_VERIFIER_LOSSY_DRAFT.md); built-in lossy compressors; **not** compressor ranking |
| Offline restored-verifier drift stress | Phase 12E / Exp 050 | [`EXPERIMENT_050_OFFLINE_RESTORED_VERIFIER_DRIFT_STRESS.md`](EXPERIMENT_050_OFFLINE_RESTORED_VERIFIER_DRIFT_STRESS.md); drift-prone panel; **not** runtime wired |
| Offline verifier CUDA drift panel | Phase 12F / Exp 051 | [`EXPERIMENT_051_OFFLINE_VERIFIER_CUDA_DRIFT_PANEL.md`](EXPERIMENT_051_OFFLINE_VERIFIER_CUDA_DRIFT_PANEL.md); CUDA fp16/bf16 exactness; **not** runtime wired |
| Restored-verifier runner consolidation | Phase 12G / Exp 052 | [`RESTORED_VERIFIER_RUNNER.md`](RESTORED_VERIFIER_RUNNER.md); isolated runner API; **not** runtime wired |
| Runner-backed drift panel | Phase 12H / Exp 053 | [`EXPERIMENT_053_RESTORED_VERIFIER_RUNNER_PANEL.md`](EXPERIMENT_053_RESTORED_VERIFIER_RUNNER_PANEL.md); canonical runner path; **not** runtime wired |
| Experimental restored-verifier runtime | Phase 13A / Exp 054 | [`EXPERIMENTAL_RESTORED_VERIFIER_RUNTIME.md`](EXPERIMENTAL_RESTORED_VERIFIER_RUNTIME.md); explicit opt-in only; **not** default runtime |
| Explicit CLI experimental restored-verifier | Phase 13B / Exp 055 | [`EXPERIMENT_055_EXPERIMENTAL_RESTORED_VERIFIER_CLI.md`](EXPERIMENT_055_EXPERIMENTAL_RESTORED_VERIFIER_CLI.md); `--experimental-restored-verifier` only; **not** default CLI |
| CUDA experimental restored-verifier runtime gate | Phase 14A / Exp 056 | [`EXPERIMENT_056_CUDA_RESTORED_VERIFIER_RUNTIME_GATE.md`](EXPERIMENT_056_CUDA_RESTORED_VERIFIER_RUNTIME_GATE.md); `run_experimental_restored_verifier()` on CUDA; exactness only; **not** default runtime |
| GPU memory accounting diagnostic | Phase 14B / Exp 057 | [`EXPERIMENT_057_GPU_MEMORY_ACCOUNTING.md`](EXPERIMENT_057_GPU_MEMORY_ACCOUNTING.md); diagnostic CUDA memory only; **not** memory savings claim |
| Expanded GPU memory panel | Phase 14C / Exp 058 | [`EXPERIMENT_058_EXPANDED_GPU_MEMORY_PANEL.md`](EXPERIMENT_058_EXPANDED_GPU_MEMORY_PANEL.md); stability panel across prompts/draft/storage/dtype; **not** memory savings claim |
| vLLM feasibility probe (install-safe) | Phase 15A / Exp 059 | [`EXPERIMENT_059_VLLM_FEASIBILITY_PROBE.md`](EXPERIMENT_059_VLLM_FEASIBILITY_PROBE.md); import probe only; **not** vLLM integration |
| Isolated vLLM venv feasibility | Phase 15B / Exp 060 | [`EXPERIMENT_060_VLLM_VENV_FEASIBILITY.md`](EXPERIMENT_060_VLLM_VENV_FEASIBILITY.md); venv install + import/smoke probe only; **not** vLLM integration |
| vLLM version compatibility sweep | Phase 15B-unblock / Exp 061 | [`EXPERIMENT_061_VLLM_VERSION_SWEEP.md`](EXPERIMENT_061_VLLM_VERSION_SWEEP.md); versioned isolated venvs; **not** vLLM integration |
| vLLM container/CUDA-13 feasibility | Phase 15C-env / Exp 062 | [`EXPERIMENT_062_VLLM_CONTAINER_FEASIBILITY.md`](EXPERIMENT_062_VLLM_CONTAINER_FEASIBILITY.md); vLLM template probe only; **not** vLLM integration |
| vLLM API surface reconnaissance | Phase 15C / Exp 063 | [`EXPERIMENT_063_VLLM_API_SURFACE_RECON.md`](EXPERIMENT_063_VLLM_API_SURFACE_RECON.md); import/object visibility only; **not** vLLM integration |
| vLLM KV/cache visibility probe | Phase 15D / Exp 064 | [`EXPERIMENT_064_VLLM_KV_VISIBILITY_PROBE.md`](EXPERIMENT_064_VLLM_KV_VISIBILITY_PROBE.md); metadata-only object inspection; **not** vLLM integration |
| Idle-GPU vLLM object KV probe | Phase 15E / Exp 065 | [`EXPERIMENT_065_IDLE_VLLM_OBJECT_KV_PROBE.md`](EXPERIMENT_065_IDLE_VLLM_OBJECT_KV_PROBE.md); **deferred** on auto-serving RunPod template; **not** vLLM integration |
| Streaming quantized-KV attention feasibility | Phase 16A / Exp 066 | [`EXPERIMENT_066_STREAMING_QUANT_ATTENTION_FEASIBILITY.md`](EXPERIMENT_066_STREAMING_QUANT_ATTENTION_FEASIBILITY.md); tensor-level reference only; **not** inference integration |
| Generation-shadow observer smoke | Phase 16K / Exp 076 | [`EXPERIMENT_076_GENERATION_SHADOW_OBSERVER_SMOKE.md`](EXPERIMENT_076_GENERATION_SHADOW_OBSERVER_SMOKE.md); external post-hoc observer; **not** generation integration |
| Prompt+generated generation-shadow panel | Phase 16L / Exp 077 | [`EXPERIMENT_077_GENERATION_SHADOW_PROMPT_PLUS_GENERATED_PANEL.md`](EXPERIMENT_077_GENERATION_SHADOW_PROMPT_PLUS_GENERATED_PANEL.md); fixed-sequence post-hoc replay; **not** generation integration |
| Expanded generation-shadow panel | Phase 16M / Exp 078 | [`EXPERIMENT_078_GENERATION_SHADOW_EXPANDED_PANEL.md`](EXPERIMENT_078_GENERATION_SHADOW_EXPANDED_PANEL.md); broader prompt/compressor panel; **not** generation integration |
| Decode-prefix ladder shadow observer | Phase 16N / Exp 079 | [`EXPERIMENT_079_DECODE_PREFIX_LADDER_SHADOW_OBSERVER.md`](EXPERIMENT_079_DECODE_PREFIX_LADDER_SHADOW_OBSERVER.md); post-hoc prefix ladder; **not** live decode integration |
| ExactKV round-log shadow observer | Phase 16O / Exp 080 | [`EXPERIMENT_080_ROUND_LOG_SHADOW_OBSERVER.md`](EXPERIMENT_080_ROUND_LOG_SHADOW_OBSERVER.md); post-hoc round-log replay; **not** live decode integration |
| Live round observer smoke | Phase 16P / Exp 081 | [`EXPERIMENT_081_LIVE_ROUND_OBSERVER_SMOKE.md`](EXPERIMENT_081_LIVE_ROUND_OBSERVER_SMOKE.md); opt-in instrumentation; **default runtime unchanged** |
| Live observer + post-hoc shadow panel | Phase 16Q / Exp 082 | [`EXPERIMENT_082_LIVE_OBSERVER_SHADOW_PANEL.md`](EXPERIMENT_082_LIVE_OBSERVER_SHADOW_PANEL.md); live snapshots + post-hoc shadow; **not decode-time integration** |
| Guarded decode-time shadow observer dry-run | Phase 16R / Exp 083 | [`EXPERIMENT_083_GUARDED_DECODE_TIME_SHADOW_SMOKE.md`](EXPERIMENT_083_GUARDED_DECODE_TIME_SHADOW_SMOKE.md); callback-time shadow; **not streaming-attention integration** |
| Expanded guarded decode-time shadow panel | Phase 16S / Exp 084 | [`EXPERIMENT_084_GUARDED_DECODE_TIME_SHADOW_PANEL.md`](EXPERIMENT_084_GUARDED_DECODE_TIME_SHADOW_PANEL.md); broader panel; **diagnostic-only** |
| Phase 16 closeout & claim freeze | Phase 16T / Exp 085 | [`PHASE_16_CLOSEOUT.md`](PHASE_16_CLOSEOUT.md); **Phase 16 complete**; claim freeze before Phase 17 |
| Claim-safe demo packaging | Phase 17A / Exp 086 | [`PHASE_17_CLAIM_SAFE_DEMO.md`](PHASE_17_CLAIM_SAFE_DEMO.md); demo narrative + cards; **not new runtime** |
| Broader model validation panel | Phase 17B / Exp 087 | [`PHASE_17B_BROADER_MODEL_VALIDATION.md`](PHASE_17B_BROADER_MODEL_VALIDATION.md); **panel-scoped**; not model-family claim |
| Longer-context validation panel | Phase 17C / Exp 088 | [`PHASE_17C_LONG_CONTEXT_VALIDATION.md`](PHASE_17C_LONG_CONTEXT_VALIDATION.md); **context-length-scoped**; not long-context support claim |
| Integration design review | Phase 17D / Exp 089 | [`PHASE_17D_INTEGRATION_DESIGN_REVIEW.md`](PHASE_17D_INTEGRATION_DESIGN_REVIEW.md); L0–L5 levels + gate policy; **not implementation** |
| Integration safety spec | Phase 18A / Exp 090 | [`PHASE_18A_INTEGRATION_SAFETY_SPEC.md`](PHASE_18A_INTEGRATION_SAFETY_SPEC.md); invariants + proposal validator; **not implementation** |
| L3 guarded draft-shadow no-commit scaffold | Phase 18B / Exp 091 | [`PHASE_18B_GUARDED_DRAFT_SHADOW_NO_COMMIT.md`](PHASE_18B_GUARDED_DRAFT_SHADOW_NO_COMMIT.md); proposals diagnostic only; **not L4** |
| L3 guarded draft-shadow panel validation | Phase 18C / Exp 092 | [`PHASE_18C_GUARDED_DRAFT_SHADOW_PANEL_VALIDATION.md`](PHASE_18C_GUARDED_DRAFT_SHADOW_PANEL_VALIDATION.md); proposal coverage; **panel-scoped** |
| L3 shadow top-1 extraction hardening | Phase 18D / Exp 093 | [`PHASE_18D_SHADOW_TOP1_EXTRACTION_HARDENING.md`](PHASE_18D_SHADOW_TOP1_EXTRACTION_HARDENING.md); provenance-aware extraction; **panel-scoped** |
| L3 shadow proposal provenance audit | Phase 18E / Exp 094 | [`PHASE_18E_SHADOW_PROPOSAL_PROVENANCE_AUDIT.md`](PHASE_18E_SHADOW_PROPOSAL_PROVENANCE_AUDIT.md); decision gate; **panel-scoped** |
| L3 round-log draft proposal source | Phase 19A / Exp 095 | [`PHASE_19A_ROUND_LOG_DRAFT_PROPOSAL_SOURCE.md`](PHASE_19A_ROUND_LOG_DRAFT_PROPOSAL_SOURCE.md); alternative proposal source; **panel-scoped** |
| L3 proposal source comparison panel | Phase 19B / Exp 096 | [`PHASE_19B_ROUND_LOG_PROPOSAL_SOURCE_COMPARISON.md`](PHASE_19B_ROUND_LOG_PROPOSAL_SOURCE_COMPARISON.md); round-log vs shadow top-1; **panel-scoped** |
| L3 promoted round-log source validation | Phase 19C / Exp 097 | [`PHASE_19C_L3_PROMOTED_SOURCE_VALIDATION.md`](PHASE_19C_L3_PROMOTED_SOURCE_VALIDATION.md); viability gates; **panel-scoped** |
| Pre-L4 safety gate review | Phase 20A / Exp 098 | [`PHASE_20A_PRE_L4_SAFETY_GATE_REVIEW.md`](PHASE_20A_PRE_L4_SAFETY_GATE_REVIEW.md); L4 design spec gate only; **not implementation** |
| L4 verifier-mediated design spec | Phase 20B / Exp 099 | [`PHASE_20B_L4_VERIFIER_MEDIATED_DESIGN_SPEC.md`](PHASE_20B_L4_VERIFIER_MEDIATED_DESIGN_SPEC.md); contracts only; **not implementation** |
| L4 contract tests (no runtime) | Phase 20C / Exp 100 | [`PHASE_20C_L4_CONTRACT_TESTS_NO_RUNTIME.md`](PHASE_20C_L4_CONTRACT_TESTS_NO_RUNTIME.md); synthetic contract tests; **not runtime** |
| L4 integration plan review | Phase 20D / Exp 101 | [`PHASE_20D_L4_INTEGRATION_PLAN_REVIEW.md`](PHASE_20D_L4_INTEGRATION_PLAN_REVIEW.md); staged plan; **not runtime** |
| L4 no-op opt-in scaffold | Phase 21A / Exp 102 | [`PHASE_21A_L4_NOOP_OPT_IN_SCAFFOLD.md`](PHASE_21A_L4_NOOP_OPT_IN_SCAFFOLD.md); stage 1 no-op; **not commit** |
| L4 no-op scaffold panel validation | Phase 21B / Exp 103 | [`PHASE_21B_L4_NOOP_SCAFFOLD_PANEL_VALIDATION.md`](PHASE_21B_L4_NOOP_SCAFFOLD_PANEL_VALIDATION.md); panel parity; **not commit** |
| L4 trace-only dry-run design | Phase 21C / Exp 104 | [`PHASE_21C_L4_TRACE_ONLY_DRY_RUN_DESIGN.md`](PHASE_21C_L4_TRACE_ONLY_DRY_RUN_DESIGN.md); Stage 2 design; **not runtime** |
| Shard restricted external-drafter probe | Exp 039–041 RunPod | 32-prompt stress + ablation + combined stress; max divergence 56.25% (Exp 041); **not** integration claim |
| Leaderboard tier separation | Phase 8f | “Full-panel, restricted, smoke-only, and future candidates are separated.” |
| Token-level acceptance rate | Per compressor × panel | Quote mean acceptance with panel name; not universal ranking |
| Sequential verification is default | Code + docs | Default path; span is optional / non-default |
| SnapKV smoke exactness | Exp 032b only | “8-cell smoke: `exactkv_failures == 0`” — **not** full-suite |
| SpectralQuant tensor smoke | Exp 042 only | “Synthetic K/V tensor smoke pass” — **not** generation integration |
| SpectralQuant restricted adapter panel | Exp 045 only | “12-prompt panel: mean accept 0.481, exactkv_failures=0” — **small panel**; materializing factory-only adapter |
| SpectralQuant adapter smoke | Exp 044 only | “4-prompt smoke: exactkv_failures=0” — **not** full-panel |
| Llama-3.1-8B small-suite exactness | Exp 033 only | “12-prompt small suite, 48 cells” — **not** full V10 |
| Timing diagnostic honesty | Exp 030 | “Diagnostic only: ExactKV slower on tested panel” — not a benefit claim |
| Memory diagnostic honesty | Exp 031 | “Diagnostic only: no active VRAM savings at tested scale” |
| Restricted backends evaluated with caveats | Exp 008–010, 014 | Factory-only; list caveat; no production claim |
| `_sim` compressors are simulated | Exp 003+ | INT8 containers; not packed-bit production storage |
| External paper results ≠ ExactKV | Exp 032 addendum | Shard/SpectralQuant/SnapKV paper numbers are not ExactKV results |

---

## 2. Forbidden claims

Do **not** use these in README, visuals, demos, leaderboard, release notes, or social copy unless explicitly negating them.

| Forbidden claim | Why |
|---|---|
| **Speedup** | Exp 030: ExactKV adds overhead on tested panel |
| **Throughput improvement** | Not measured as a benefit; diagnostic timing only |
| **Latency improvement** | Not measured as a benefit |
| **Tokens/sec improvement** | Not measured as a benefit |
| **Runtime improvement** | Verifier loop adds work |
| **Active GPU memory savings** | Exp 031: no savings at tested scale |
| **VRAM savings / VRAM reduction** | Peak dominated by weights; no compressed-active path |
| **Production serving** | Exp 017: sidecar probe only; no integration |
| **vLLM / LMCache / PagedAttention integration** | D11/D12/D16: no-go or deferred |
| **Model accuracy improvement** | ExactKV preserves greedy output; does not improve model quality |
| **Shard restricted external-drafter metrics** | Exp 039–041 leaderboard tier; accepted-prefix mean + divergence rate; Exp 041 combined 56.25%; **not** full-panel compressor acceptance |
| **Shard ExactKV integration** | External drafter probe only — not default registry |
| **Full VeriCache system reproduction** | Phase 11A audit — algorithm yes, serving/throughput **no** |
| **VeriCache throughput or memory benefits** | Not measured as ExactKV benefits; Exp 030/031 honesty |
| **SpectralQuant tensor smoke (leaderboard)** | Exp 042 | Superseded by Exp 045 RESTRICTED BACKEND row — do not cite smoke tier |
| SpectralQuant external probe | Exp 042 | Tensor smoke on synthetic K/V; import OK; **not** generation integration |
| **SpectralQuant restricted adapter (Exp 045)** | Exp 045 | 12-prompt panel; materializing factory-only; mean accept 0.481; **not** full-panel ranking |
| **SnapKV full-suite performance** | Smoke-only (8 cells) |
| **SnapKV ranked vs INT8 full panel** | Apples-to-oranges; tiers forbid this |
| **Real packed INT4/INT2 storage** | `_sim` uses INT8 containers |
| **Universal benchmark coverage** | V10 suites are evaluation panels |
| **“Best compressor” leaderboard** | Crash-test lab ranks when compressors lie on cited panels |
| **Public launch ready / v1.0 ready** | Phase 9A audit: not ready |
| **TurboQuant/KIVI/KVQuant as production backends** | Restricted factory-only adapters |
| **KIVI CUDA/Triton production path** | Offline adapter only in ExactKV |
| **KVQuant deployment CUDA** | simquant adapter in published runs |

---

## 3. Required disclaimers (use when relevant)

Include at least one of these near any public demo, leaderboard, or results table:

1. “Not a timing or memory benchmark.”
2. “No speedup, throughput, latency, tokens/sec, active GPU memory savings, production serving, or model accuracy improvement claim.”
3. “ExactKV preserves full-greedy output while using lossy KV only as a draft.”
4. “Full-KV verifier remains authoritative.”
5. “External Shard, SpectralQuant, SnapKV paper, or kvpress results are not ExactKV results.”
6. “`_sim` compressors are simulated INT8 containers, not real packed-bit backends.”

---

## 4. Asset-specific review checklist

| Asset | Pass criteria |
|---|---|
| `README.md` | No forbidden claims; launch deferred stated |
| `docs/leaderboard.md` / `.html` | Tiers visible; no cross-tier ranking headline |
| `scripts/exactkv_terminal_crash_test.py` | **Primary** replay; disclaimers in doc |
| `scripts/exactkv_terminal_longbench_drift.py` | **Secondary** replay; LongBench-style disclaimer required |
| `docs/PUBLIC_VISUAL_PACKAGE.md` | Timing/memory cards labeled diagnostic |
| `public_*.png` | No speedup/VRAM headline |
| `docs/EXPERIMENT_*.md` | Per-experiment scope and forbidden footer |
| Release notes (future) | Claims audit sign-off required |

---

## 5. Audit procedure (Phase 9B)

1. Grep public docs for forbidden terms: `speedup`, `throughput`, `latency`, `tokens/sec`, `VRAM`, `production serving`, `v1.0`, `launch ready`.
2. Verify each numeric claim links to an experiment ID.
3. Verify leaderboard tiers are not collapsed into a single ranked list.
4. Verify SnapKV/Shard/SpectralQuant wording matches integration status.
5. Sign off in [`PRELAUNCH_HARDENING_PLAN.md`](PRELAUNCH_HARDENING_PLAN.md) claims-audit row.

---

## 6. Status

| Item | Status |
|---|---|
| Claims audit document | ✅ Created (Phase 9A) |
| Full repo grep pass | ✅ Phase 9B + 10C (`audit_public_claims.py` PASSED) |
| README sign-off | ✅ Primary/secondary demo hierarchy documented |
| Visual package sign-off | ⏳ |
| Parallel work integration | ✅ Phase 10C — [`PARALLEL_WORK_INTEGRATION_REPORT.md`](PARALLEL_WORK_INTEGRATION_REPORT.md) |
| Launch approval | ❌ **Not granted** |
