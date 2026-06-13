# Prelaunch Hardening Report (V13 Phase 9B)

**Date:** 2026-06-09  
**Status:** Infrastructure complete — **launch still blocked**

> This is an internal hardening report, not a launch announcement.
> Public launch and v1.0 remain **not approved**.

Companion: [`LAUNCH_READINESS_GAP_AUDIT.md`](LAUNCH_READINESS_GAP_AUDIT.md) · [`PRELAUNCH_HARDENING_PLAN.md`](PRELAUNCH_HARDENING_PLAN.md)

---

## 1. What was hardened (Phase 9B)

| Area | Deliverable | Status |
|---|---|---|
| Clean install | [`INSTALL.md`](INSTALL.md) | ✅ |
| Quickstart (5 commands) | [`QUICKSTART.md`](QUICKSTART.md) | ✅ |
| One-command smoke test | `scripts/smoke_test.sh` + `make smoke` | ✅ |
| Terminal demo entry | `make demo` / documented in README | ✅ |
| Leaderboard entry | `make leaderboard` / documented in README | ✅ |
| Claims audit | `scripts/audit_public_claims.py` + tests | ✅ |
| Link/asset audit | `scripts/check_docs_links.py` + tests | ✅ |
| Report hygiene | `scripts/check_report_hygiene.py` + tests | ✅ |
| README quick-start | Top-of-README commands + status | ✅ |
| Makefile shortcuts | `Makefile` (`smoke`, `demo`, `leaderboard`, `audit`) | ✅ |

---

## 2. Commands added

```bash
bash scripts/smoke_test.sh              # full prelaunch smoke
make smoke                              # same
make demo                               # terminal crash-test (fast)
make leaderboard                        # tiered leaderboard
make audit                              # claims + links + hygiene
python3 scripts/audit_public_claims.py
python3 scripts/check_docs_links.py
python3 scripts/check_report_hygiene.py --require-public
```

---

## 3. Audit scope

| Audit | Scans | Pass criteria |
|---|---|---|
| **Claims** | Public-facing README + launch docs (9 files) | No forbidden *positive* claims; negations allowlisted |
| **Links** | All `docs/*.md` + `docs/*.html` + README | Local paths resolve |
| **Hygiene** | git tracked files + `.gitignore` | No `reports/*.json`/`*.csv` tracked; public leaderboard files exist |

Internal research docs (`V*_SCOPE`, `EXPERIMENT_*`, `METRICS.md`) are excluded from automated claims scan — reviewed manually via [`CLAIMS_AUDIT.md`](CLAIMS_AUDIT.md).

---

## 4. Must-fix blockers — Phase 9B status

| # | Blocker | 9B status |
|---|---|---|
| M1 | Clean install instructions | ✅ `INSTALL.md` |
| M2 | One-command smoke test | ✅ `smoke_test.sh` |
| M3 | One-command terminal demo | ✅ README + Makefile |
| M4 | One-command leaderboard | ✅ README + Makefile |
| M5 | Claims audit pass | ✅ automated on public files |
| M6 | README narrative cleanup | ✅ top quick-start (not full launch README) |
| M7 | No broken links/assets | ✅ link audit passes |
| M8 | Raw report hygiene | ✅ hygiene audit passes |

**Caveat:** Must-fix items are **implemented** but still require validation on a **clean clone** by a human reviewer before launch approval.

---

## 5. Remaining blockers (launch still blocked)

| Category | Item | Status |
|---|---|---|
| **Validation** | Clean-clone smoke pass by independent reviewer | ⏳ |
| **Should-fix** | LongBench-style score-preserving drift demo | ✅ Exp 037 complete — **secondary** demo |
| **Should-fix** | Polished terminal demo screen recording (`.cast` / clip) | ⏳ script ready |
| **Should-fix** | HTML leaderboard polish (mobile/a11y) | ✅ dashboard accepted (Phase 8f / 10C) |
| **Should-fix** | Shard external-drafter `--try-run` | ⏳ blocked locally; RunPod pending |
| **Should-fix** | Dedicated limitations page | ⏳ |
| **Should-fix** | Release notes for research preview tag | ⏳ |
| **Deferred** | Speed/runtime path, VRAM savings, serving | Future work |
| **Deferred** | Shard/SpectralQuant/SnapKV full-suite | Shard probe blocked (Exp 038); SnapKV smoke-only |

---

## 6. Launch decision

| Question | Answer |
|---|---|
| Is Phase 9B infrastructure complete? | **Yes** |
| Is public launch approved? | **No** |
| Is v1.0 approved? | **No** |
| Can we tag a research preview? | **Not yet** — after clean-clone validation + demo recording review |

**Conclusion:** Launch remains **blocked** until smoke/install/audit commands pass from a clean clone **and** the terminal demo is screen-recorded and reviewed for public legibility.

---

## 7. Recommended next phase

**Phase 9C — launch validation & should-fix**

1. Clean-clone repro on fresh machine (document in [`REPRO_CHECKLIST.md`](REPRO_CHECKLIST.md))
2. Record **primary** pharmacy terminal demo (`asciinema` or screen capture)
3. RunPod: Shard `--try-run` + optional Exp 037 GPU confirm
4. Re-run [`LAUNCH_READINESS_GAP_AUDIT.md`](LAUNCH_READINESS_GAP_AUDIT.md) with **ready / not ready** decision
5. Only then consider `v0.13.0-rc` research preview tag — **not v1.0.0**

See also [`PARALLEL_WORK_INTEGRATION_REPORT.md`](PARALLEL_WORK_INTEGRATION_REPORT.md) (Phase 10C).

---

## 8. Reproduction

See [`REPRO_CHECKLIST.md`](REPRO_CHECKLIST.md). Minimal:

```bash
bash scripts/smoke_test.sh
make audit
```
