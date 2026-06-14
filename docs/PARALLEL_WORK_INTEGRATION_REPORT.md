# Parallel Work Integration Report (V13 Phase 10H)

**Date:** 2026-06-09  
**Status:** External-method consolidation complete — **research-demo-ready, not production/serving-ready**

> Internal consolidation after Shard (038–041) and SpectralQuant (042–045) external-method work.
> **v1.0.0 is not approved.** `v0.13.0-rc` remains a **future possibility only**.

Companion: [`PHASE_10_EXTERNAL_METHODS_SUMMARY.md`](PHASE_10_EXTERNAL_METHODS_SUMMARY.md) · [`LAUNCH_READINESS_GAP_AUDIT.md`](LAUNCH_READINESS_GAP_AUDIT.md) · [`CLAIMS_AUDIT.md`](CLAIMS_AUDIT.md)

---

## 1. Workstreams integrated

| Phase | Experiment / deliverable | Status |
|---|---|---|
| **10A** | Exp 037 — LongBench-style drift demo | ✅ Secondary demo |
| **10B–10B4** | Exp 038–041 — Shard external-drafter probe series | ✅ **Complete** — bounded probe stopped |
| **10C** | Shard RESTRICTED BACKEND leaderboard | ✅ |
| **10D–10G** | Exp 042–045 — SpectralQuant probe → restricted panel | ✅ **Complete** |
| **10H** | External methods consolidation | ✅ This report + [`PHASE_10_EXTERNAL_METHODS_SUMMARY.md`](PHASE_10_EXTERNAL_METHODS_SUMMARY.md) |
| **10I** | Benchmark gap analysis | ✅ [`BENCHMARK_GAP_ANALYSIS.md`](BENCHMARK_GAP_ANALYSIS.md) |
| **8f** | Terminal + HTML crash-test leaderboard | ✅ |

---

## 2. What changed (Phase 10H)

| Area | Change |
|---|---|
| **Shard** | Exp 038–041 documented; RESTRICTED BACKEND row; `stop_shard_bounded_probe_complete` |
| **SpectralQuant** | Exp 042–045 documented; RESTRICTED BACKEND adapter row (Exp 045); materializing factory-only |
| **Leaderboard** | Two external restricted rows: Shard + SpectralQuant; SnapKV remains SMOKE ONLY |
| **Docs** | Phase 10 summary, experiment index 043–045, claims audit, deferred register, roadmap, V13 scope |
| **Launch** | Label: **research-demo-ready**; public launch **not approved** |

No generation logic, verification logic, new adapters, or new evaluations in Phase 10H.

---

## 3. Primary demo decision

**Pharmacy terminal semantic crash test** — unchanged primary public demo (Exp 034b `pharm_001`).

---

## 4. Secondary demo decision

**LongBench-style score-preserving drift** — secondary / research appendix (Exp 037).

---

## 5. Shard status (final)

| Field | Value |
|---|---|
| Experiments | Exp 038–041 |
| Integration | **Not** default registry · **external drafter only** · Llama-only |
| exactkv_failures | **0** across all Shard runs |
| Strongest stress | Exp 041: **18/32 draft divergences (56.25%)** with stream_bits=4 + 128tok |
| Leaderboard | RESTRICTED BACKEND · Exp 039–041 |
| Recommendation | **`stop_shard_bounded_probe_complete`** |

---

## 5b. SpectralQuant status (final)

| Field | Value |
|---|---|
| Experiments | Exp 042–045 |
| Integration | **Not** default registry · **factory-only materializing adapter** |
| Exp 045 panel | 12 prompts · **exactkv_failures=0** · mean accept **0.481** |
| Draft divergence | 11/12 prompts (verifier corrected; final output exact) |
| Calibration | 6 prompts (minimal; not paper-scale) |
| Reconstruction | Layer-0 key max err ~39 (disclosed) |
| Leaderboard | RESTRICTED BACKEND · Exp 045 |
| Recommendation | **Restricted panel complete** — no full-panel promotion |

---

## 6. Leaderboard status

| Tier | Shard | SpectralQuant | SnapKV |
|---|---|---|---|
| RESTRICTED BACKEND | ✅ external-drafter | ✅ experimental adapter | — |
| SMOKE ONLY | — | — (promoted from 10E) | ✅ 8-cell smoke |

Regenerate: `python3 scripts/exactkv_leaderboard.py --md --html`

---

## 7. Claims allowed

- ExactKV repairs draft drift on tested panels; `exactkv_failures == 0` on cited runs.
- Shard and SpectralQuant have **caveated restricted external-method results**.
- SpectralQuant adapter is **materializing** and **factory-only**.
- Shard is **external-drafter only** (not integrated compressor).
- **Research-demo-ready** — not production/serving-ready.
- Outcome benchmarks and ExactKV answer **different questions** ([`BENCHMARK_GAP_ANALYSIS.md`](BENCHMARK_GAP_ANALYSIS.md)).

---

## 8. Benchmark gap (Phase 10I)

| Message | Doc |
|---|---|
| Outcome scores can stay green while KV path drifted | [`BENCHMARK_GAP_ANALYSIS.md`](BENCHMARK_GAP_ANALYSIS.md) |
| ExactKV is **complementary** to LongBench/RULER-style evaluation | Same |
| No claim that outcome benchmarks are flawed or replaced | Same |

---

## 9. Claims forbidden

- Speedup, throughput, latency, active GPU memory savings, production serving.
- vLLM / LMCache integration; model accuracy improvement.
- Full-panel ranking comparability for restricted rows.
- Default registry for Shard or SpectralQuant.
- Outcome benchmarks replace ExactKV or vice versa.
- Public launch ready / v1.0 ready.

---

## 10. Remaining blockers

| # | Blocker | Status |
|---|---|---|
| 1 | Clean-clone validation | ⏳ |
| 2 | Terminal demo screen recording | ⏳ |
| 3 | Shard RunPod probe | ✅ 038–041 complete |
| 4 | SpectralQuant restricted panel | ✅ Exp 045 |
| 5 | Explicit launch decision (9C) | ❌ not granted |

---

## 11. Recommended next action

**Phase 9C — launch validation** (not launch):

1. Clean-clone repro per [`REPRO_CHECKLIST.md`](REPRO_CHECKLIST.md).
2. Record pharmacy terminal demo.
3. Re-run audits from clean clone.
4. Optional `v0.13.0-rc` research preview — **not v1.0.0**.

**Recommended next research (not implemented):** public demo polish + clean release prep.

---

## 11. Launch readiness label

**Research-demo-ready, not production/serving-ready.**
