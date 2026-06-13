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
| Terminal demo shows verified semantic drift correction | Exp 034b / `pharm_001` | “Replay of verified trace: lossy `drop` rejected, verifier `pickup` committed.” |
| Leaderboard tier separation | Phase 8f | “Full-panel, restricted, smoke-only, and future candidates are separated.” |
| Token-level acceptance rate | Per compressor × panel | Quote mean acceptance with panel name; not universal ranking |
| Sequential verification is default | Code + docs | Default path; span is optional / non-default |
| SnapKV smoke exactness | Exp 032b only | “8-cell smoke: `exactkv_failures == 0`” — **not** full-suite |
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
| **Shard ExactKV results** | Not integrated; external drafter only |
| **SpectralQuant ExactKV results** | Not integrated |
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
| `scripts/exactkv_terminal_crash_test.py` | Replay only; disclaimers in doc |
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
| Full repo grep pass | ⏳ Phase 9B |
| README sign-off | ⏳ Phase 9B |
| Visual package sign-off | ⏳ Phase 9B |
| Launch approval | ❌ **Not granted** |
