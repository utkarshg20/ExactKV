# ExactKV Deferred Work Register

**Status:** Living register of major deferred items. **Nothing listed here is
implemented** unless a linked experiment or release note says otherwise.

> Guardrails: `exactkv_failures == 0` on every published experiment; no throughput,
> latency, speedup, runtime, or production-serving claims; `_sim` ≠ packed-bit storage;
> external paper results are **not** ExactKV results.

**Version path:** V9 ✅ → **V10 ✅ (`v0.10.0`)** → **V11 ✅ (`v0.11.0`)** → **V12 (Phases 0–7 ✅)** → **V13 (Phase 2 ✅; Exp 029 grid ✅)** → v1.0.0 (public launch).

**V9 scope:** [`V9_SCOPE_STATEMENT.md`](V9_SCOPE_STATEMENT.md) — **complete** (`v0.9.0`).
**V10 scope:** [`V10_SCOPE_STATEMENT.md`](V10_SCOPE_STATEMENT.md) — **complete** (`v0.10.0`).
**V10 readiness:** [`V10_READINESS_ASSESSMENT.md`](V10_READINESS_ASSESSMENT.md).
**V11 scope:** [`V11_SCOPE_STATEMENT.md`](V11_SCOPE_STATEMENT.md) — **complete** (`v0.11.0`).
**V11 readiness:** [`V11_LAUNCH_READINESS.md`](V11_LAUNCH_READINESS.md).
**V12 scope:** [`V12_SCOPE_STATEMENT.md`](V12_SCOPE_STATEMENT.md) — Phases 0–7 complete; Phase 8 release package planned.
**V13 scope:** [`V13_SCOPE_STATEMENT.md`](V13_SCOPE_STATEMENT.md) — Phase 2 complete; Exp 029 grid passed.
**Experiment 014:** [`EXPERIMENT_014_REAL_BACKEND_SPOTCHECKS.md`](EXPERIMENT_014_REAL_BACKEND_SPOTCHECKS.md).
**V10 suites:** [`V10_PROMPT_SUITES.md`](V10_PROMPT_SUITES.md).
**Experiment 012:** [`EXPERIMENT_012_EVAL_SUITE_EXPANSION.md`](EXPERIMENT_012_EVAL_SUITE_EXPANSION.md).
**Experiment 013:** [`EXPERIMENT_013_SENSITIVITY_FORENSICS.md`](EXPERIMENT_013_SENSITIVITY_FORENSICS.md).

---

## V9 — Real backend integration gauntlet (complete)

| ID | Item | Status | V9 phase | Success criteria |
|---|---|---|---|---|
| D1 | **TurboQuant full integration** | **Evaluated (Phase C)** | ✅ | Exp 008: 272 cells, `exactkv_failures == 0`; accept **0.435** — [`EXPERIMENT_008_TURBOQUANT_PYTHON.md`](EXPERIMENT_008_TURBOQUANT_PYTHON.md) |
| D2 | **TurboQuant+ full integration** | **Evaluated (Python path only)** | ✅ | Production llama.cpp/MLX deferred — [`RELEASE_NOTES_V0.9.0.md`](RELEASE_NOTES_V0.9.0.md) |
| D3 | **KIVI adapter** | **Evaluated (Phase D3)** | ✅ | Exp 009: accept **0.012** — [`EXPERIMENT_009_KIVI_OFFLINE.md`](EXPERIMENT_009_KIVI_OFFLINE.md) |
| D4 | **KVQuant-style adapter** | **Evaluated (Phase D6)** | ✅ | Exp 010: accept **0.792** — [`EXPERIMENT_010_KVQUANT_SIM.md`](EXPERIMENT_010_KVQUANT_SIM.md) |
| D5 | **KVTC / Palu feasibility** | Deferred (optional) | — | Not in V9 scope |
| D15 | **Larger-model RunPod validation** | **Complete (Phase E)** | ✅ | Exp 011: 1.5B, 238 cells — [`EXPERIMENT_011_LARGER_MODEL_VALIDATION.md`](EXPERIMENT_011_LARGER_MODEL_VALIDATION.md) |

---

## V10 — Evaluation suite hardening and divergence forensics (complete)

| ID | Item | Status | V10 phase | Success criteria |
|---|---|---|---|---|
| D26 | **`core_v2` + category benchmark suites** | **Complete** | Phase 1 ✅ | 128 prompts; [`V10_PROMPT_SUITES.md`](V10_PROMPT_SUITES.md); validator + tests |
| D27 | **Draft length sensitivity (2/4/8)** | **Complete** | Phase 3 ✅ | Exp 013: 2160 cells, `exactkv_failures == 0` |
| D28 | **Generation length sensitivity (16/32/64)** | **Complete** | Phase 3 ✅ | Exp 013 full 3×3 grid |
| D29 | **Category-stratified divergence forensics** | **Complete** | Phase 3 ✅ | Token-type + structured-output heuristics; no attention weights |
| — | **Per-category leaderboards + prompt win/loss** | **Complete** | Phase 2 ✅ | Exp 012: 896 cells, `exactkv_failures == 0` |
| — | **Real-backend category spot-checks** | **Complete** | Phase 4 ✅ | Exp 014: 280 cells — [`EXPERIMENT_014_REAL_BACKEND_SPOTCHECKS.md`](EXPERIMENT_014_REAL_BACKEND_SPOTCHECKS.md) |
| — | **V10 readiness assessment** | **Complete** | Phase 5 ✅ | [`V10_READINESS_ASSESSMENT.md`](V10_READINESS_ASSESSMENT.md) |
| D6 | **Sparse V dequantization** | **Moved to V11 / research** | — | Acceptance-only evaluation |
| D7 | **True attention logging** | **Moved to V11** | — | Small subset; no fabricated weights |
| D8 | **Per-layer/head/token divergence forensics** | **Moved to V11** | — | Requires attention weights or documented blocker |
| D9 | **Pre-RoPE key quantization experiments** | Deferred | — | Compare vs post-RoPE baselines |
| D10 | **Boundary / layer-policy extensions** | Deferred | — | N>4 only with explicit approval |
| — | **1.5B on expanded V10 suites** | **Moved to V11** | — | Exp 011 is legacy `core` only |

**V10 exit:** tag **`v0.10.0`** — not v1.0.0. See [`RELEASE_NOTES_V0.10.0.md`](RELEASE_NOTES_V0.10.0.md).

---

## V11 — Final launch hardening (complete — `v0.11.0`)

| ID | Item | Status | V11 phase | Success criteria |
|---|---|---|---|---|
| — | **V11 scope statement** | **Complete** | Phase 0 | [`V11_SCOPE_STATEMENT.md`](V11_SCOPE_STATEMENT.md) |
| — | **1.5B on V10 expanded suites** | **Complete** | Phase 1 / Exp 015 | [`EXPERIMENT_015_QWEN15B_V10_SUITES.md`](EXPERIMENT_015_QWEN15B_V10_SUITES.md); `exactkv_failures == 0` |
| — | **Optional 3B built-in stretch** | **Complete** | Phase 2 / Exp 016 | [`EXPERIMENT_016_QWEN3B_V10_SUITES.md`](EXPERIMENT_016_QWEN3B_V10_SUITES.md); `exactkv_failures == 0` |
| — | **Optional 1.5B real-backend panel** | Deferred | — | Not run in Phase 2; may revisit post-Exp 017 |
| D13 | **vLLM / LMCache sidecar probe** | **Complete** | Phase 3 / Exp 017 | [`EXPERIMENT_017_SERVING_SIDECAR_PROBE.md`](EXPERIMENT_017_SERVING_SIDECAR_PROBE.md); sidecar pass; direct integration **no-go reaffirmed** |
| D14 | **Active GPU memory methodology** | **Complete** | Phase 4 / Exp 018 | [`GPU_MEMORY_METHODOLOGY.md`](GPU_MEMORY_METHODOLOGY.md); pilot **success** — not added to standard schema |
| D7 | **True attention logging** | **Deferred (partial)** | Phase 5 / Exp 019 | sdpa backend lacks `output_attentions`; no fabricated weights — [`EXPERIMENT_019_DIVERGENCE_AUTOPSY.md`](EXPERIMENT_019_DIVERGENCE_AUTOPSY.md) |
| D8 | **Per-layer/head divergence forensics** | **Partial complete** | Phase 5 / Exp 019 | Per-layer KV error in Exp 019; per-head deferred without attention weights |
| — | **Repair-policy pilot (Exp 019 hypotheses)** | **Complete** | Phase 5b / Exp 020 | [`EXPERIMENT_020_REPAIR_POLICY_PILOT.md`](EXPERIMENT_020_REPAIR_POLICY_PILOT.md); policies **not** in core ExactKV |
| D17 | **Raw report bundle** | **Policy complete** | Phase 6 | [`RAW_ARTIFACT_POLICY.md`](RAW_ARTIFACT_POLICY.md); physical zip optional until v1.0.0 |
| D18 | **Launch narrative** | **Draft complete** | Phase 6 | [`LAUNCH_NARRATIVE_DRAFT.md`](LAUNCH_NARRATIVE_DRAFT.md); not for public posting until v1.0.0 review |
| D6 | **Sparse V dequantization** | Deferred (out of V11 scope) | — | Not in V11 unless explicitly approved |
| D11 | **Direct vLLM integration** | No-go (Phase A) | — | Out of V11 scope |
| D12 | **LMCache integration** | No-go (Phase A) | — | Out of V11 scope |
| D16 | **PagedAttention kernel integration** | Deferred | — | Out of V11 scope |
| D1–D4 | **Production TurboQuant / KIVI CUDA / KVQuant CUDA** | Evaluated (V9) | — | **Not** in V11 unless factory-only re-panel in Exp 016 |

V11 covers multi-model validation, serving/profiling probes, and launch package prep — **not** a performance benchmark or production integration release.

---

## V12 — Deferred Work Completion Gauntlet (Phases 0–7 complete)

| ID | Item | Status | V12 phase | Success criteria |
|---|---|---|---|---|
| — | **V12 scope statement** | **Complete** | Phase 0 ✅ | [`V12_SCOPE_STATEMENT.md`](V12_SCOPE_STATEMENT.md) |
| D2 | **TurboQuant llama.cpp / GGUF / production-fidelity** | **External probe complete (Mode B)** | Phase 1–2 ✅ | [`EXPERIMENT_022_TURBOQUANT_LLAMACPP_PROBE.md`](EXPERIMENT_022_TURBOQUANT_LLAMACPP_PROBE.md); BackendAdapter **no-go**; Mode B **go with restrictions** |
| D4 | **KVQuant 1.5B/3B real-backend validation** | **1.5B complete** | Phase 3 ✅ / Exp 023 | [`EXPERIMENT_023_KVQUANT_LARGER_MODEL.md`](EXPERIMENT_023_KVQUANT_LARGER_MODEL.md); 1.5B accept **0.609**; 3B stretch not run |
| D3 | **KIVI CUDA/Triton packed path** | **Feasibility complete (`B_restricted_go`)** | Phase 4 ✅ / Exp 024 | [`EXPERIMENT_024_KIVI_CUDA_TRITON_FEASIBILITY.md`](EXPERIMENT_024_KIVI_CUDA_TRITON_FEASIBILITY.md); Triton pack OK; `dequant_cuda` missing; no Qwen model; BackendAdapter **restricted_future_only** |
| — | **Full-suite repair-policy validation** | **Complete** | Phase 5 ✅ / Exp 025 | [`EXPERIMENT_025_FULL_SUITE_REPAIR_POLICY.md`](EXPERIMENT_025_FULL_SUITE_REPAIR_POLICY.md); 768 cells 0.5B; `exactkv_failures == 0`; 1.5B optional not run |
| D7 | **True attention logging** | **Restricted go (eager prefill-only)** | Phase 6 ✅ / Exp 026 | [`EXPERIMENT_026_ATTENTION_LOGGING_FEASIBILITY.md`](EXPERIMENT_026_ATTENTION_LOGGING_FEASIBILITY.md); sdpa blocked; eager prefill OK on Qwen2.5-0.5B |
| D8 | **Per-head divergence forensics** | **Partial — prefill-only path** | Phase 6 ✅ / Exp 026 | Per-layer KV in Exp 019; per-head via eager prefill snapshots only; decode-step/default runtime blocked |
| — | **Performance/memory truth boundary** | **Complete** | Phase 7 ✅ / Exp 027 | [`EXPERIMENT_027_PERFORMANCE_MEMORY_TRUTH_BOUNDARY.md`](EXPERIMENT_027_PERFORMANCE_MEMORY_TRUTH_BOUNDARY.md); speed/VRAM savings **forbidden**; **V13 recommended** |
| D17 | **Physical raw report bundle** | Planned | Phase 8 | Optional until v1.0.0; policy in [`RAW_ARTIFACT_POLICY.md`](RAW_ARTIFACT_POLICY.md) |
| D18 | **Launch narrative (final)** | Planned | Phase 8 | Review/approve [`LAUNCH_NARRATIVE_DRAFT.md`](LAUNCH_NARRATIVE_DRAFT.md) |
| D11 | **Direct vLLM integration** | No-go (Phase A) | — | Out of V12/V13 unless scope changes |
| D12 | **LMCache integration** | No-go (Phase A) | — | Out of V12/V13 unless scope changes |
| D16 | **PagedAttention kernel integration** | Deferred | — | Out of V12/V13 scope unless explicitly approved |
| D6 | **Sparse V dequantization** | Deferred | — | Out of V12/V13 unless explicitly approved |
| D9/D10 | **Pre-RoPE / boundary N>4** | Deferred | — | Out of V12/V13 scope |

V12 closed deferred backend, policy, forensics, and claim-boundary tracks — **not** practicality proof.

---

## V13 — Practicality Proof (Phase 2 complete; Exp 029 grid passed)

| ID / track | Item | Status | V13 phase | Success criteria |
|---|---|---|---|---|
| — | **V13 scope statement** | **Complete** | Phase 0 ✅ | [`V13_SCOPE_STATEMENT.md`](V13_SCOPE_STATEMENT.md) |
| D21 | **Parallel / span verification** | **Exactness grid complete** | Phases 2 ✅ / Exp 028–029 | Smoke + [`EXPERIMENT_029_SPAN_VERIFICATION_GRID.md`](EXPERIMENT_029_SPAN_VERIFICATION_GRID.md); 600 cells, 0 failures; default sequential |
| — | **Performance proof (diagnostic timing)** | **Complete (diagnostic only)** | Phase 3 ✅ / Exp 030 | [`EXPERIMENT_030_DIAGNOSTIC_TIMING.md`](EXPERIMENT_030_DIAGNOSTIC_TIMING.md); ExactKV ~2.66× slower than full greedy on A5000 fp16; span ≡ sequential wall-clock; no general speed claim |
| — | **Active GPU memory isolation** | **Complete (diagnostic)** | Phase 4 ✅ / Exp 031 | [`EXPERIMENT_031_GPU_MEMORY_ISOLATION.md`](EXPERIMENT_031_GPU_MEMORY_ISOLATION.md); exactness gate pass; peak indistinguishable from full greedy; **no savings claim** |
| — | **Hot adapter (SnapKV / Shard / SpectralQuant)** | **Feasibility complete + addendum** | Phase 5 ✅ / Exp 032 + addendum | [`EXPERIMENT_032_ADDENDUM_SHARD_SPECTRALQUANT.md`](EXPERIMENT_032_ADDENDUM_SHARD_SPECTRALQUANT.md); SnapKV **B** primary 5b; Shard **B** Llama drafter; SpectralQuant **B** deferred 5c |
| — | **SnapKV experimental adapter** | **Complete (smoke)** | Phase 5b ✅ / Exp 032b | [`EXPERIMENT_032B_SNAPKV_EXPERIMENTAL_SMOKE.md`](EXPERIMENT_032B_SNAPKV_EXPERIMENTAL_SMOKE.md); factory-only; 8 cells, 0 failures; not in default registry |
| — | **Shard Llama external-drafter probe** | **Bounded probe complete** | Phase 10B–10C ✅ | Exp 038–041: probe + stress + ablation + combined stress; RESTRICTED BACKEND; **stop_shard_bounded_probe_complete** |
| — | **SpectralQuant experimental adapter** | **Restricted panel complete** | Phase 10D–10G ✅ | Exp 042–045: probe → real KV → adapter smoke → 12-prompt panel; RESTRICTED BACKEND (Exp 045); **not** default registry |
| — | **External methods consolidation** | **Complete** | Phase 10H ✅ | [`PHASE_10_EXTERNAL_METHODS_SUMMARY.md`](PHASE_10_EXTERNAL_METHODS_SUMMARY.md) |
| — | **Llama-3.1-8B small suite** | **Complete** | Phase 6 ✅ / Exp 033 | [`EXPERIMENT_033_LLAMA31_8B_SMALL_SUITE.md`](EXPERIMENT_033_LLAMA31_8B_SMALL_SUITE.md); 48 cells, 0 failures; RunPod A5000 bfloat16 |
| — | **Killer correction demo (Markdown trace)** | **Complete** | Phase 7 ✅ / Exp 034 | [`EXPERIMENT_034_KILLER_CORRECTION_DEMO.md`](EXPERIMENT_034_KILLER_CORRECTION_DEMO.md); `tj_002` × `int4_sim`; source data for live demo |
| — | **Live correction terminal demo** | **Complete** | Phase 7b ✅ | [`DEMO_EXACTKV_LIVE_CORRECTION.md`](DEMO_EXACTKV_LIVE_CORRECTION.md); `scripts/demo_exactkv_live_correction.py`; Exp 034 trace replay |
| — | **Visual plot package + leaderboard** | **Complete** | Phase 8 ✅ / Exp 035 | [`EXPERIMENT_035_VISUAL_PLOTS_AND_LEADERBOARD.md`](EXPERIMENT_035_VISUAL_PLOTS_AND_LEADERBOARD.md); internal `exp035_*.png` |
| — | **Public visual polish package** | **Complete** | Phase 8b ✅ / Exp 036 | [`PUBLIC_VISUAL_PACKAGE.md`](PUBLIC_VISUAL_PACKAGE.md); `public_*.png` launch cards |
| — | **Tiered crash-test leaderboard** | **Complete** | Phase 8d ✅ | [`leaderboard.md`](leaderboard.md); FULL / RESTRICTED / SMOKE / FUTURE tiers |
| — | **Cinematic crash-test video demo** | **Complete (optional)** | Phase 8c ✅ / Exp 036b | [`EXACTKV_CRASH_TEST_VIDEO.md`](EXACTKV_CRASH_TEST_VIDEO.md); 120s MP4 + HTML player; secondary to terminal demo |
| — | **Terminal-native crash-test demo** | **Complete** | Phase 8e ✅ / Exp 034b | [`EXACTKV_TERMINAL_CRASH_TEST.md`](EXACTKV_TERMINAL_CRASH_TEST.md); `pharm_001` semantic trace (`drop` → `pickup`); primary public demo |
| — | **Terminal + HTML crash-test leaderboard** | **Complete** | Phase 8f ✅ | `scripts/exactkv_leaderboard.py`; [`leaderboard.md`](leaderboard.md) · [`leaderboard.html`](leaderboard.html) |
| — | **Headline number audit** | **Complete (Phase 9A)** | Phase 9A ✅ | [`CLAIMS_AUDIT.md`](CLAIMS_AUDIT.md); full repo grep pass ⏳ Phase 9B |
| — | **Launch readiness gap audit** | **Complete** | Phase 9A ✅ | [`LAUNCH_READINESS_GAP_AUDIT.md`](LAUNCH_READINESS_GAP_AUDIT.md); decision: **not ready** |
| — | **Prelaunch hardening plan** | **Infrastructure complete** | Phase 9B ✅ | [`PRELAUNCH_HARDENING_REPORT.md`](PRELAUNCH_HARDENING_REPORT.md) |
| — | **Smoke test + audits** | **Complete** | Phase 9B ✅ | `scripts/smoke_test.sh`, `audit_public_claims.py`, `check_docs_links.py`, `check_report_hygiene.py` |
| — | **Repro checklist** | **Complete** | Phase 9A ✅ | [`REPRO_CHECKLIST.md`](REPRO_CHECKLIST.md) |
| — | **LongBench-style score-preserving drift demo** | **Complete (secondary)** | Phase 10A ✅ | Exp 037: `lb_md_001` × `int4_sim`; **not** primary demo — [`EXPERIMENT_037_LONGBENCH_STYLE_DRIFT_DEMO.md`](EXPERIMENT_037_LONGBENCH_STYLE_DRIFT_DEMO.md) |
| — | **Parallel work integration** | **Complete** | Phase 10C ✅ | [`PARALLEL_WORK_INTEGRATION_REPORT.md`](PARALLEL_WORK_INTEGRATION_REPORT.md) |
| — | **Compressed-active-KV memory path** | **Blocker documented (Exp 031)** | Phase 4 ✅ | Model weights dominate CUDA peak; V5 KV accounting does not translate to active VRAM savings at 0.5B scale |
| D11/D12 | **Production serving / vLLM / LMCache** | No-go | — | Remain deferred; see [`VERICACHE_SYSTEMS_ROADMAP.md`](VERICACHE_SYSTEMS_ROADMAP.md) Stages 5–6 |
| D22 | **Multi-request batching** | Deferred | — | Serving-scale; see VeriCache roadmap Stage 5+ |
| — | **VeriCache full system parity** | **Audit complete (11A)** | Phase 11A ✅ | [`VERICACHE_PARITY_AUDIT.md`](VERICACHE_PARITY_AUDIT.md) — algorithm **mostly done**; serving/throughput **missing** |
| — | **VeriCache systems roadmap** | **Planned** | Phase 11A ✅ doc | [`VERICACHE_SYSTEMS_ROADMAP.md`](VERICACHE_SYSTEMS_ROADMAP.md) Stages 1–10 |
| — | **Dual-cache contract (Stage 1)** | **Complete** | Phase 11B ✅ | [`DUAL_CACHE_ABSTRACTION.md`](DUAL_CACHE_ABSTRACTION.md); not wired to generator |
| — | **Full-KV storage manager (Stage 2 spike)** | **Complete** | Phase 11C ✅ | [`FULL_KV_STORAGE_MANAGER.md`](FULL_KV_STORAGE_MANAGER.md); tiny payload smoke only |
| — | **Materialized compressed-draft backend (Stage 3 spike)** | **Complete** | Phase 11D ✅ | [`MATERIALIZED_COMPRESSED_DRAFT_BACKEND.md`](MATERIALIZED_COMPRESSED_DRAFT_BACKEND.md) |
| — | **Extended verification scheduler (Stage 4 spike)** | **Complete** | Phase 11E ✅ | [`EXTENDED_VERIFICATION_SCHEDULER.md`](EXTENDED_VERIFICATION_SCHEDULER.md) |
| — | **vLLM prototype path (Stage 5 spike)** | **Complete** | Phase 11F ✅ | [`VLLM_PROTOTYPE_PATH.md`](VLLM_PROTOTYPE_PATH.md); contract-only; **not** integrated |
| — | **LMCache prototype path (Stage 6 spike)** | **Complete** | Phase 11G ✅ | [`LMCACHE_PROTOTYPE_PATH.md`](LMCACHE_PROTOTYPE_PATH.md); contract-only; **not** integrated |
| — | **Remote prefix cache semantics (Stage 7 spike)** | **Complete** | Phase 11H ✅ | [`REMOTE_PREFIX_CACHE_SEMANTICS.md`](REMOTE_PREFIX_CACHE_SEMANTICS.md); loopback mock only; **not** remote runtime |
| — | **Throughput benchmark harness (Stage 8 spike)** | **Complete** | Phase 11I ✅ | [`THROUGHPUT_BENCHMARK_HARNESS.md`](THROUGHPUT_BENCHMARK_HARNESS.md); methodology only; **not** speedup claim |
| — | **Paper-like reproduction panel (Stage 9 spike)** | **Complete** | Phase 11J ✅ | [`PAPER_LIKE_REPRODUCTION_PANEL.md`](PAPER_LIKE_REPRODUCTION_PANEL.md); contract only; **not** paper run |
| — | **VeriCache parity RC claim gate (Stage 10 spike)** | **Complete** | Phase 11K ✅ | [`VERICACHE_PARITY_CLAIM_GATE.md`](VERICACHE_PARITY_CLAIM_GATE.md); classification only; **not** RC certified |
| — | **Full-KV restore smoke (Stage 2 real HF KV)** | **Complete** | Phase 12A ✅ | [`EXPERIMENT_046_FULL_KV_RESTORE_SMOKE.md`](EXPERIMENT_046_FULL_KV_RESTORE_SMOKE.md); 4-prompt smoke; **not** runtime wired |
| — | **Full-KV restore panel hardening (Stage 2 panel)** | **Complete** | Phase 12B ✅ | [`EXPERIMENT_047_FULL_KV_RESTORE_PANEL.md`](EXPERIMENT_047_FULL_KV_RESTORE_PANEL.md); 12-prompt panel; optional CUDA; **not** serving |
| — | **Offline verifier restore smoke (Stage 2 verifier)** | **Complete** | Phase 12C ✅ | [`EXPERIMENT_048_OFFLINE_VERIFIER_RESTORE_SMOKE.md`](EXPERIMENT_048_OFFLINE_VERIFIER_RESTORE_SMOKE.md); reloaded full-KV verifier; **not** runtime wired |
| — | **Offline verifier lossy draft (Stage 2 dual-cache smoke)** | **Complete** | Phase 12D ✅ | [`EXPERIMENT_049_OFFLINE_VERIFIER_LOSSY_DRAFT.md`](EXPERIMENT_049_OFFLINE_VERIFIER_LOSSY_DRAFT.md); int8/int4_sim/k8_v4_sim; **not** serving |
| — | **Offline restored-verifier drift stress (Stage 2 drift panel)** | **Complete** | Phase 12E ✅ | [`EXPERIMENT_050_OFFLINE_RESTORED_VERIFIER_DRIFT_STRESS.md`](EXPERIMENT_050_OFFLINE_RESTORED_VERIFIER_DRIFT_STRESS.md); drift-prone panel; **not** serving |
| — | **Offline verifier CUDA drift panel (Stage 2 CUDA exactness)** | **Complete** | Phase 12F ✅ | [`EXPERIMENT_051_OFFLINE_VERIFIER_CUDA_DRIFT_PANEL.md`](EXPERIMENT_051_OFFLINE_VERIFIER_CUDA_DRIFT_PANEL.md); CUDA fp16/bf16; **not** serving |
| — | **Restored-verifier runner consolidation (Stage 2 API)** | **Complete** | Phase 12G ✅ | [`RESTORED_VERIFIER_RUNNER.md`](RESTORED_VERIFIER_RUNNER.md); isolated runner; **not** serving |
| — | **Runner-backed drift panel (Stage 2 canonical path)** | **Complete** | Phase 12H ✅ | [`EXPERIMENT_053_RESTORED_VERIFIER_RUNNER_PANEL.md`](EXPERIMENT_053_RESTORED_VERIFIER_RUNNER_PANEL.md); via `run_restored_verifier()`; **not** serving |
| — | **Experimental restored-verifier runtime (Stage 3 opt-in)** | **Complete** | Phase 13A ✅ | [`EXPERIMENTAL_RESTORED_VERIFIER_RUNTIME.md`](EXPERIMENTAL_RESTORED_VERIFIER_RUNTIME.md); explicit opt-in; **not** default runtime |
| — | **Explicit CLI experimental restored-verifier (Stage 3 CLI)** | **Complete** | Phase 13B ✅ | [`EXPERIMENT_055_EXPERIMENTAL_RESTORED_VERIFIER_CLI.md`](EXPERIMENT_055_EXPERIMENTAL_RESTORED_VERIFIER_CLI.md); flag-only opt-in; **not** default CLI |
| — | **CUDA experimental restored-verifier runtime gate** | **Complete** | Phase 14A ✅ | [`EXPERIMENT_056_CUDA_RESTORED_VERIFIER_RUNTIME_GATE.md`](EXPERIMENT_056_CUDA_RESTORED_VERIFIER_RUNTIME_GATE.md); CUDA exactness via runtime API; **not** default runtime |

V13 builds and measures missing practicality pieces — **not** public launch by default.

**Public launch:** ❌ **Not ready** (Phase 9A audit). **Launch decision deferred** until [`PRELAUNCH_HARDENING_PLAN.md`](PRELAUNCH_HARDENING_PLAN.md) must-fix items complete.

**Explicitly future work (post-launch or research):** speed/runtime path, compressed-active KV, active memory savings, serving integration (vLLM/LMCache), SnapKV full-suite, broader Llama/Mistral panels, SpectralQuant/Shard **full-panel** expansion (restricted probes complete for Phase 10).

---

## v1.0.0 — Public launch tag (after V13 exit — **NOT READY**)

| ID | Item | Status | Success criteria |
|---|---|---|---|
| D19 | **Project status v1.0.0** | Deferred | [`LAUNCH_READINESS_GAP_AUDIT.md`](LAUNCH_READINESS_GAP_AUDIT.md) must-fix blockers cleared |
| D20 | **Git tag `v1.0.0`** | Deferred | Phase 9B hardening + explicit launch approval |

_D17 and D18 may finalize in V12 Phase 8; **public posting** waits for V13 Phase 9 claim decision._

---

## Cross-cutting (ongoing)

| ID | Item | Status | Notes |
|---|---|---|---|
| D21 | **Sampling / parallel verify / bonus tokens** | **Span GPU fp16 parity fixed** | Exp 030b ✅ | Batched span restored via math-only SDPA + cache_position; parity guard kept; **rerun Exp 030 timing** before span speed claims |
| D22 | **Multi-request batching** | Deferred | Serving-scale feature |
| D23 | **CPU offload / CUDA kernels** | Deferred | No custom kernels in ExactKV today |
| D24 | **Broader kvpress** | Deferred | KnormPress only (V6) |
| D25 | **RESEARCH_BACKLOG sync** | Ongoing | Keep aligned with this register |

---

## Phase C reminder (V8)

vLLM and LMCache **direct integration** were judged **no-go** in
[`SERVING_CONTEXT_FEASIBILITY.md`](SERVING_CONTEXT_FEASIBILITY.md). Items D11–D12
remain **deferred, not forgotten**.

---

## Related

- [`V13_SCOPE_STATEMENT.md`](V13_SCOPE_STATEMENT.md) — V13 formal scope (Phase 1 complete)
- [`SPAN_VERIFICATION_DESIGN.md`](SPAN_VERIFICATION_DESIGN.md) — span verification design (Phase 1)
- [`V12_SCOPE_STATEMENT.md`](V12_SCOPE_STATEMENT.md) — V12 formal scope (Phases 0–7 complete)
- [`V11_SCOPE_STATEMENT.md`](V11_SCOPE_STATEMENT.md) — V11 formal scope (complete)
- [`V11_LAUNCH_READINESS.md`](V11_LAUNCH_READINESS.md) — v0.11.0 gate decision
- [`V10_SCOPE_STATEMENT.md`](V10_SCOPE_STATEMENT.md) — V10 formal scope (complete)
- [`V10_READINESS_ASSESSMENT.md`](V10_READINESS_ASSESSMENT.md) — Phase 5 readiness
- [`RELEASE_NOTES_V0.10.0.md`](RELEASE_NOTES_V0.10.0.md) — v0.10.0 changelog
- [`V10_SCOPE_DRAFT.md`](V10_SCOPE_DRAFT.md) — superseded draft
- [`RELEASE_NOTES_V0.9.0.md`](RELEASE_NOTES_V0.9.0.md) — what shipped in v0.9.0
- [`ROADMAP.md`](ROADMAP.md) — version planning
- [`RESEARCH_BACKLOG.md`](RESEARCH_BACKLOG.md) — experiment ideas
