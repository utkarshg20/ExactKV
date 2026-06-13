# Parallel Work Integration Report (V13 Phase 10C)

**Date:** 2026-06-09  
**Status:** Integration checkpoint complete — **public launch still blocked**

> Internal consolidation after parallel agent workstreams. Not a launch announcement.
> **v1.0.0 is not approved.** `v0.13.0-rc` remains a **future possibility only**.

Companion: [`LAUNCH_READINESS_GAP_AUDIT.md`](LAUNCH_READINESS_GAP_AUDIT.md) · [`PRELAUNCH_HARDENING_REPORT.md`](PRELAUNCH_HARDENING_REPORT.md) · [`CLAIMS_AUDIT.md`](CLAIMS_AUDIT.md)

---

## 1. Workstreams integrated

| Phase | Experiment / deliverable | Status |
|---|---|---|
| **10A** | Exp 037 — LongBench-style score-preserving drift demo | ✅ Integrated as **secondary** demo |
| **10B** | Exp 038 — Shard external-drafter probe feasibility | ✅ Integrated — RunPod `pass`; `restricted_go` (4-prompt panel) |
| **8f polish** | Terminal + HTML crash-test leaderboard dashboard | ✅ Accepted |
| **10C** | This integration checkpoint | ✅ Complete |

---

## 2. What changed

| Area | Change |
|---|---|
| **Demos** | Primary: pharmacy terminal crash test (`pharm_001`). Secondary: LongBench-style drift (`lb_md_001`). |
| **Docs** | README, V13 scope, experiment index, deferred register, roadmap, launch audit, hardening report, claims audit reconciled. |
| **Shard** | `scripts/probe_shard_external_drafter.py` + Exp 038 doc; default run `probe_status=blocked` (no `SHARD_REPO_PATH`). |
| **Leaderboard** | `exactkv_leaderboard.py` supports `--plain`, `--summary`, `--watch --once`; `leaderboard.md` + `leaderboard.html` present. |
| **Launch** | Still **not approved**; must-fix infra from 9B stands; should-fix items partially cleared. |

No generation logic, verification logic, compressors, or new benchmarks were added in this phase.

---

## 3. Primary demo decision

**Pharmacy terminal semantic crash test** (`scripts/exactkv_terminal_crash_test.py`) remains the **primary public demo**.

- Trace: Exp 034b `pharm_001` × `k8_v4_sim` — `drop` rejected → `pickup` committed.
- Clearest structured-output “compressed KV lied” story for outsiders.
- Recordable terminal dashboard; replay mode (no inference).

---

## 4. Secondary demo decision

**LongBench-style score-preserving drift** (`scripts/exactkv_terminal_longbench_drift.py`) is a **secondary / research appendix** demo.

- Trace: Exp 037 `lb_md_001` × `int4_sim` — outcome heuristic green on full/lossy/ExactKV; token path drifted (`billing` → `answer`).
- **LongBench-style only** — not official LongBench evaluation.
- Outcome benchmarks and ExactKV answer **different questions**.
- Useful for researchers; not promoted above pharmacy demo.

---

## 5. Shard status

| Field | Value |
|---|---|
| Experiment | Exp 038–040 — Shard external-drafter probe + stress + ablation |
| Integration | **Not** a default ExactKV compressor |
| Feasibility (038) | **`pass`** — 4-prompt × 16-token |
| Stress panel (039) | **`pass`** — 6/32 divergences; `exactkv_failures=0` |
| Ablation (040) | **`pass`** — length 128 ↑ drift to 31%; `stream_bits=4` modest ↑; all `exactkv_failures=0` |
| Recommendation | **`expand_shard_lossy_ablation`** — continue bounded Shard; consider SpectralQuant in parallel |
| Next step | Optional `stream_bits=4` + `max_new_tokens=128`; still no registry entry |

Shard README throughput/memory numbers are **external results**, not ExactKV results.

---

## 6. Leaderboard status

| Item | Status |
|---|---|
| `scripts/exactkv_leaderboard.py` | ✅ `--plain`, `--summary`, `--watch --once --plain` |
| `docs/leaderboard.md` | ✅ Tiered (FULL / RESTRICTED / SMOKE / FUTURE) |
| `docs/leaderboard.html` | ✅ Present |
| Tier caveats | ✅ No cross-tier ranking; restricted/smoke/future labeled |
| Claims | ✅ No speedup / VRAM / serving headlines |

---

## 7. Claims allowed

- Pharmacy terminal demo as primary public replay trace.
- LongBench-**style** secondary demo with transparent heuristic (not official LongBench score).
- “Outcome benchmarks and ExactKV answer different questions.”
- Tiered leaderboard with documented panel scope.
- `exactkv_failures == 0` on cited experiment panels.
- Shard **restricted_go** feasibility probe (RunPod `pass`; 4-prompt panel).
- Diagnostic timing/memory honesty (Exp 030/031) as **negations**, not benefits.
- Public launch **deferred**; v1.0 **not approved**.

---

## 8. Claims forbidden

- Speedup, throughput, latency, tokens/sec, runtime improvement.
- Active GPU memory savings / VRAM savings.
- Production serving, vLLM/LMCache integration.
- Model accuracy improvement.
- **Shard ExactKV panel numbers** beyond the 4-prompt Exp 038 probe (no speedup/serving extrapolation).
- **SpectralQuant ExactKV results** (not integrated).
- **SnapKV full-suite** performance (smoke-only).
- **Official LongBench** score or ranking.
- **Public launch ready** / **v1.0 ready**.
- Promoting LongBench-style demo to **primary** without explicit re-decision.

---

## 9. Remaining blockers

| # | Blocker | Status |
|---|---|---|
| 1 | Clean-clone validation by independent reviewer | ⏳ |
| 2 | Terminal demo screen recording (`.cast` / clip) | ⏳ |
| 3 | Shard `--try-run` on RunPod (Llama + `SHARD_REPO_PATH`) | ✅ `pass` (restricted_go) |
| 4 | Dedicated limitations page | ⏳ |
| 5 | Research preview release notes (`v0.13.0-rc` candidate) | ⏳ future only |
| 6 | Speed / VRAM / serving paths | ❌ future work |
| 7 | Explicit launch decision (Phase 9C) | ❌ not granted |

---

## 10. Recommended next action

**Phase 9C — launch validation** (not launch):

1. Clean-clone repro per [`REPRO_CHECKLIST.md`](REPRO_CHECKLIST.md).
2. Record pharmacy terminal demo (`--speed cinematic`).
3. On RunPod: Exp 037 GPU confirm + Shard `--try-run` (feasibility only).
4. Re-run `bash scripts/smoke_test.sh` + audits from clean clone.
5. Only then consider `v0.13.0-rc` research preview tag — **not v1.0.0**.

---

## 11. LongBench-style demo placement validation

| Criterion | Result |
|---|---|
| Documented as LongBench-**style**, not official LongBench | ✅ |
| Outcome vs behavior framing present | ✅ |
| Secondary to pharmacy demo in README + scope | ✅ |
| No official LongBench score claimed | ✅ |
| No speedup / VRAM / serving claims | ✅ |
| `exactkv_failures == 0` on selected trace | ✅ (Exp 037 search) |

**Decision:** Placement validated as **secondary / research appendix**. Pharmacy remains primary.

---

## 12. Consolidated validation (Phase 10C run)

```bash
python3 scripts/exactkv_terminal_crash_test.py --no-delay --plain          # OK
python3 scripts/exactkv_terminal_longbench_drift.py --no-delay --plain     # OK
python3 scripts/exactkv_leaderboard.py --plain                           # OK
python3 scripts/exactkv_leaderboard.py --summary                         # OK
python3 scripts/exactkv_leaderboard.py --watch --once --plain            # OK
python3 scripts/probe_shard_external_drafter.py                          # blocked (expected)
python3 scripts/audit_public_claims.py                                   # PASSED
python3 scripts/check_docs_links.py                                      # PASSED
python3 scripts/check_report_hygiene.py --require-public                 # PASSED
pytest tests/test_exactkv_terminal_crash_test.py \
     tests/test_exactkv_terminal_longbench_drift.py \
     tests/test_search_longbench_style_drift_demo.py \
     tests/test_exactkv_leaderboard.py \
     tests/test_shard_external_probe.py -q                               # 60 passed, 1 skipped
git diff --check                                                         # clean
```

---

## 13. Version tags

| Tag | Status |
|---|---|
| `v1.0.0` | **Not approved** |
| `v0.13.0-rc` | Future research-preview possibility only — after 9C validation |
| `v0.11.0` | Latest tagged release (per scope statement) |
