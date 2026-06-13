# Prelaunch Hardening Plan (V13 Phase 9B)

**Status:** Phase 9B infrastructure **complete** — see [`PRELAUNCH_HARDENING_REPORT.md`](PRELAUNCH_HARDENING_REPORT.md). Launch **still blocked** pending clean-clone validation.

> This is an internal execution plan, not a launch announcement.
> Public launch remains **deferred**.

---

## Priority order

1. Clean install + smoke test  
2. README narrative cleanup  
3. Terminal demo recording  
4. Claims audit pass  
5. Leaderboard polish  
6. LongBench-style drift demo  
7. Release candidate tag (research preview — **not v1.0.0**)

---

## Hardening table

| # | Item | Why it matters | Effort | Risk | Owner / status | Required before launch? |
|---|---|---|---|---|---|---|
| 1 | **Clean install instructions** | Outsiders must reproduce without maintainer help | M | Low | **Done** — [`INSTALL.md`](INSTALL.md) | **Yes** |
| 2 | **One-command smoke test** | Fast CI/local gate after clone | M | Low | **Done** — `scripts/smoke_test.sh` | **Yes** |
| 3 | **One-command terminal demo** | Primary “aha” moment | S | Low | **Done** — README + `make demo` | **Yes** |
| 4 | **One-command leaderboard** | Shows crash-test lab credibility | S | Low | **Done** — README + `make leaderboard` | **Yes** |
| 5 | **Claims audit pass** | Prevents overclaim on launch | M | **High** if skipped | **Done** — `audit_public_claims.py` (public files) | **Yes** |
| 6 | **README narrative cleanup** | Current README is research-dense | L | Med | **Partial** — top quick-start only | **Yes** |
| 7 | **No broken links/assets** | Broken hero links destroy trust | S | Med | **Done** — `check_docs_links.py` | **Yes** |
| 8 | **Raw report commit guard** | Accidental `reports/*.json` leaks | S | Low | **Done** — `check_report_hygiene.py` | **Yes** |
| 9 | **Terminal demo screen recording** | Shard/TurboQuant-quality launch clip | M | Low | Script ✅; `.cast` ⏳ | No (should-fix) |
| 10 | **HTML leaderboard polish** | Mobile, a11y, README link | M | Low | **Done** — terminal dashboard + `--summary`/`--watch` | No (should-fix) |
| 11 | **Limitations page** | Single “what we are not” for outsiders | M | Low | Planned | No (should-fix) |
| 12 | **LongBench-style drift demo** | Familiar benchmark score-preserving story | L | Med | **Done** — Exp 037; **secondary** demo | No (should-fix) |
| 13 | **Release notes (research preview)** | Document what shipped in preview tag | M | Low | Planned | No (should-fix) |
| 14 | **Release candidate tag** | `v0.12.0` or `v0.13.0-rc` — **not v1.0.0** | S | Med | Deferred | No |
| 15 | **Speed / runtime path** | Exp 030 shows gap | XL | High | Future work | No |
| 16 | **Compressed-active KV / VRAM** | Exp 031 shows gap | XL | High | Future work | No |
| 17 | **Serving integration** | Exp 017 no-go | XL | High | Deferred | No |
| 18 | **Shard external-drafter probe** | Exp 038 feasibility | L | Med | **Blocked locally** — RunPod `--try-run` pending | No |
| 19 | **SnapKV full-suite** | Smoke-only today | L | Med | Deferred | No |
| 20 | **Broader Llama/Mistral panels** | Beyond Exp 033 small suite | L | Med | Deferred | No |

**Effort key:** S = hours, M = 1–2 days, L = multi-day, XL = multi-week research

---

## Phase 9B milestones

| Milestone | Deliverable | Gate |
|---|---|---|
| **9B.1** | `scripts/smoke_test.sh` or Makefile target | §1 of [`REPRO_CHECKLIST.md`](REPRO_CHECKLIST.md) passes on clean clone |
| **9B.2** | README “Quick start” section (5 commands) | M1–M4 satisfied |
| **9B.3** | Claims grep + fix pass | [`CLAIMS_AUDIT.md`](CLAIMS_AUDIT.md) sign-off |
| **9B.4** | Link/asset audit | M7 pass |
| **9B.5** | Optional: terminal `.cast` recording | S2 |
| **9B.6** | Launch decision review | Re-run [`LAUNCH_READINESS_GAP_AUDIT.md`](LAUNCH_READINESS_GAP_AUDIT.md) |

---

## Launch decision criteria (not met today)

Public launch (or v1.0.0 tag) requires **all** of:

- [ ] All **Yes** rows in hardening table complete
- [ ] Claims audit signed off
- [ ] Clean-clone repro passes ([`REPRO_CHECKLIST.md`](REPRO_CHECKLIST.md) §2)
- [ ] Launch readiness audit updated with **ready** decision
- [ ] Explicit reviewer approval in V13 Phase 9 completion note

**Current decision:** ❌ **Not ready** — remain in prelaunch hardening.

---

## What Phase 9B will not do

- No launch posts or marketing copy
- No v1.0.0 tag
- No speed/VRAM/serving claims
- No new GPU benchmarks unless a later phase explicitly scopes them
- No generation/verification logic changes (unless correctness bugfix)

---

## Recommended sequencing

```text
Phase 9A (this audit) ✅
    ↓
Phase 9B.1–9B.4 (must-fix hardening)
    ↓
Phase 9B.5–9B.6 (should-fix + decision review)
    ↓
Optional: soft research preview tag (v0.12.x / v0.13.0-rc) — NOT v1.0.0
    ↓
Future: runtime/memory/serving research (deferred register)
    ↓
v1.0.0 only after explicit exit criteria in V13 §18
```
