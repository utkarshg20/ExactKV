# Launch Readiness Gap Audit (V13 Phase 9A)

**Date:** 2026-06-09  
**Status:** **NOT READY FOR PUBLIC LAUNCH**  
**Decision:** V13 has strong correctness evidence and demos, but important prelaunch work remains. Public launch is **deferred** until must-fix blockers in [`PRELAUNCH_HARDENING_PLAN.md`](PRELAUNCH_HARDENING_PLAN.md) are resolved.

> This document is an **internal readiness audit**, not a launch announcement.
> ExactKV is **not** public-launch-ready. **v1.0.0 is not approved.**

Companion docs: [`CLAIMS_AUDIT.md`](CLAIMS_AUDIT.md) · [`REPRO_CHECKLIST.md`](REPRO_CHECKLIST.md) · [`PRELAUNCH_HARDENING_PLAN.md`](PRELAUNCH_HARDENING_PLAN.md)

---

## Executive summary

ExactKV is a **credible correctness-first KV-cache compression crash-test lab** with:

- A working draft → verify → commit loop
- Published `exactkv_failures == 0` on documented panels
- A recordable terminal semantic crash-test demo
- A tiered leaderboard that avoids apples-to-oranges ranking
- Honest timing/memory diagnostics that **disprove** speed/VRAM headlines today

It is **not** a production KV serving system, a speedup library, or a universal benchmark suite. README density, external install friction, missing LongBench-style drift evidence, and incomplete adapter integration all block a confident public launch narrative.

**Launch readiness:** ❌ **Not ready** — proceed to Phase 9B prelaunch hardening only.

---

## 1. What is genuinely ready

| Area | Evidence | Notes |
|---|---|---|
| **ExactKV correctness concept** | Core loop in `exactkv/runtime/` | Draft on lossy KV; verify on full FP KV; commit authoritative tokens |
| **Full-KV verifier** | Default generation path | Verifier remains authoritative; rejected drafts never committed |
| **Sequential verification** | Exp 012–034, default in generator | Production code path for all published sweeps |
| **Span verification exactness grid** | [`EXPERIMENT_029_SPAN_VERIFICATION_GRID.md`](EXPERIMENT_029_SPAN_VERIFICATION_GRID.md) | 600 cells, `exactkv_failures == 0`; span ≡ sequential on exactness |
| **Terminal-native semantic crash-test demo** | [`EXACTKV_TERMINAL_CRASH_TEST.md`](EXACTKV_TERMINAL_CRASH_TEST.md) | `pharm_001` `drop` → `pickup`; replay mode, no inference required |
| **Tiered leaderboard** | [`leaderboard.md`](leaderboard.md) · [`leaderboard.html`](leaderboard.html) | FULL PANEL / RESTRICTED / SMOKE / FUTURE separation |
| **Public visual assets** | [`PUBLIC_VISUAL_PACKAGE.md`](PUBLIC_VISUAL_PACKAGE.md) | Hero, killer demo, exactness wall, timing/memory truth cards |
| **Llama-3.1-8B small suite** | [`EXPERIMENT_033_LLAMA31_8B_SMALL_SUITE.md`](EXPERIMENT_033_LLAMA31_8B_SMALL_SUITE.md) | 48 cells, 0 failures; 12-prompt panel only |
| **SnapKV smoke-only adapter** | [`EXPERIMENT_032B_SNAPKV_EXPERIMENTAL_SMOKE.md`](EXPERIMENT_032B_SNAPKV_EXPERIMENTAL_SMOKE.md) | 8 cells, 0 failures; factory-only; not default registry |
| **Honest timing diagnostics** | [`EXPERIMENT_030_DIAGNOSTIC_TIMING.md`](EXPERIMENT_030_DIAGNOSTIC_TIMING.md) | ExactKV slower than full greedy on tested panel; diagnostic only |
| **Honest memory diagnostics** | [`EXPERIMENT_031_GPU_MEMORY_ISOLATION.md`](EXPERIMENT_031_GPU_MEMORY_ISOLATION.md) | No active VRAM savings at tested scale; diagnostic only |
| **Restricted backend gauntlet** | Exp 008–010, 014, 022, 023 | Documented with caveats; TurboQuant/KIVI/KVQuant factory-only |
| **Claim boundary review** | [`EXPERIMENT_027_PERFORMANCE_MEMORY_TRUTH_BOUNDARY.md`](EXPERIMENT_027_PERFORMANCE_MEMORY_TRUTH_BOUNDARY.md) | Speed/VRAM/serving claims explicitly forbidden |
| **V10–V13 experiment corpus** | [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md) | Reproducible experiment reports with hard gate |

---

## 2. What is not ready

| Gap | Status | Why it blocks launch narrative |
|---|---|---|
| **No speedup** | Documented (Exp 030) | ExactKV adds verifier overhead; cannot claim faster inference |
| **No active GPU memory savings** | Documented (Exp 031) | Peak dominated by weights; no compressed-active-KV path |
| **No production serving path** | No-go (Exp 017) | Sidecar probe only; not integrated serving |
| **No vLLM / LMCache integration** | No-go (D11/D12) | Direct integration explicitly deferred |
| **No compressed-attention runtime** | Not built | Full materialize + recompress remains default |
| **No CUDA / Triton acceleration path** | Not built | CPU/GPU PyTorch path only for core loop |
| **SnapKV is smoke-only** | 8 cells | Cannot rank or headline vs full-suite INT8 |
| **Shard not integrated** | Feasibility only (Exp 032 addendum) | External drafter; no ExactKV panel numbers |
| **SpectralQuant not integrated** | Feasibility only | No adapter; no ExactKV results |
| **TurboQuant / KIVI / KVQuant restricted** | Factory-only adapters | Low acceptance on some; not production integrations |
| **LongBench-style score-preserving drift demo** | Not implemented | No public-facing benchmark-score drift story |
| **External install / repro hardening** | Partial | Model download, optional GPU, large test suite friction |
| **README / docs density** | Too dense for outsiders | Research repo feel; not launch onboarding |
| **Terminal demo screen recording** | Script ready; recording optional | No polished published `.cast` / launch clip in repo |
| **Universal benchmark coverage** | Not claimed | V10 suites are evaluation panels, not industry benchmarks |
| **`_sim` compressors** | INT8 containers | Not real packed INT4/INT2 storage |
| **v1.0.0 / public launch tag** | Deferred | Gates in V13 §18 not met |

---

## 3. Launch blockers

### Must fix before public launch

| # | Blocker | Rationale |
|---|---|---|
| M1 | **Clean install instructions** | Outsider must `pip install` and run without tribal knowledge |
| M2 | **One-command smoke test** | Fast proof the package works after clone |
| M3 | **One-command terminal demo** | `exactkv_terminal_crash_test.py --no-delay --plain` documented as entry point |
| M4 | **One-command leaderboard** | `exactkv_leaderboard.py` works from committed docs or documented CSV expectation |
| M5 | **Claims audit pass** | All public-facing text reviewed against [`CLAIMS_AUDIT.md`](CLAIMS_AUDIT.md) |
| M6 | **README narrative cleanup** | Shorter outsider path: what it is, what it is not, how to run demo + leaderboard |
| M7 | **No broken links / assets** | Verify PNG, HTML, MD cross-links from README |
| M8 | **No raw report files accidentally committed** | `reports/*.json` / `reports/*.csv` stay gitignored; policy enforced |

### Should fix before public launch

| # | Item | Rationale |
|---|---|---|
| S1 | **LongBench-style score-preserving drift demo** | Stronger “when compression starts lying” story on familiar benchmark |
| S2 | **Stronger terminal demo recording** | Published asciinema or screen-record guide artifact |
| S3 | **HTML leaderboard polish** | Mobile layout, accessibility, link from README hero |
| S4 | **Clearer limitations page** | Single “what ExactKV is not” page for outsiders |
| S5 | **Release notes draft** | `RELEASE_NOTES` for research preview tag (not v1.0.0) |

### Can defer after public launch

| # | Item | Notes |
|---|---|---|
| D1 | Speed / runtime optimization path | Future work; Exp 030 documents current overhead |
| D2 | True compressed attention / active KV path | Research engineering |
| D3 | Active memory savings proof | Requires compressed-active-KV architecture |
| D4 | Serving integration (vLLM, LMCache, PagedAttention) | Explicitly no-go today |
| D5 | Shard / SpectralQuant adapters | Future crash-test candidates |
| D6 | Broader Llama / Mistral panels | Beyond Exp 033 small suite |
| D7 | SnapKV full-suite integration | Smoke-only today |
| D8 | CUDA/Triton kernels | Performance path |
| D9 | Multi-request batching | Serving-scale |

---

## 4. Launch readiness decision

| Question | Answer |
|---|---|
| Is ExactKV scientifically interesting? | **Yes** — correctness-first KV compression evaluation is differentiated |
| Is the core exactness story proven on cited panels? | **Yes** — `exactkv_failures == 0` on published experiments |
| Are demos and visuals strong enough to show the idea? | **Yes** — terminal crash-test + tiered leaderboard |
| Is the project ready for a public **v1.0.0 launch**? | **No** |
| Is the project ready for a **soft research preview** after hardening? | **Maybe** — only after must-fix blockers |
| Recommended next step | **Phase 9B: prelaunch hardening** per [`PRELAUNCH_HARDENING_PLAN.md`](PRELAUNCH_HARDENING_PLAN.md) |

---

## 5. What we will not do in Phase 9A

- No launch posts or social copy
- No git tag or release
- No “public-launch-ready” language in README
- No v1.0.0 readiness claim
- No new benchmarks unless explicitly scoped in a later phase
- No changes to generation or verification logic

---

## 6. References

| Doc | Role |
|---|---|
| [`V13_SCOPE_STATEMENT.md`](V13_SCOPE_STATEMENT.md) | Phase status and exit criteria |
| [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md) | Deferred items register |
| [`EXPERIMENT_027_PERFORMANCE_MEMORY_TRUTH_BOUNDARY.md`](EXPERIMENT_027_PERFORMANCE_MEMORY_TRUTH_BOUNDARY.md) | Performance/memory claim boundary |
| [`CLAIMS_AUDIT.md`](CLAIMS_AUDIT.md) | Allowed vs forbidden claims |
| [`REPRO_CHECKLIST.md`](REPRO_CHECKLIST.md) | Reproduction commands |
| [`PRELAUNCH_HARDENING_PLAN.md`](PRELAUNCH_HARDENING_PLAN.md) | Hardening task table |
