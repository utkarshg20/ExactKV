# Phase 20D: L4 Integration Plan Review

**Status:** run `scripts/research/run_exp101_l4_integration_plan_review.py`.

> This is an L4 integration plan review, not L4 runtime implementation.  
> ExactKV default generation remains unchanged.  
> ExactKVGenerator remains unchanged.  
> Runtime generation path remains unchanged.  
> CLI flag is planned but not implemented.  
> Stage 1 may only be a no-op opt-in scaffold.  
> Stage 4 runtime commit remains blocked.  
> Full verification must remain the required source of truth before any future compressed draft acceptance.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_101_L4_INTEGRATION_PLAN_REVIEW.md`](EXPERIMENT_101_L4_INTEGRATION_PLAN_REVIEW.md)

---

## 1. Purpose

Create a detailed L4 integration plan review documenting future change targets, interfaces, flag/trace/fallback/rollback plans, staged implementation, and risks — without runtime changes.

---

## 2. Relation to Phase 20C

Phase 20C validated L4 contracts on synthetic cases. Phase 20D plans how a future runtime scaffold would integrate, stage by stage.

---

## 3. Future change targets

Eight targets documented (ExactKVGenerator, verifier path, traces, L3 policy, reports, CLI, tests, claims audit) — all `not_modified`.

---

## 4. Future interfaces

Seven interfaces: `L4DraftProposalProvider`, `L4FullVerifier`, `L4AcceptanceDecision`, `L4RollbackController`, `L4FallbackController`, `L4TraceRecorder`, `L4SafetyGateEvaluator`. None affect token commit now.

---

## 5. Future opt-in flag plan

`--experimental-l4-verifier-mediated-draft`, disabled by default, with experimental warnings. Not implemented.

---

## 6. Future trace plan

Opt-in L4 trace fields on `ExactKVResult` / round traces; default trace unchanged.

---

## 7. Future fallback plan

Triggers: proposal exception, missing verifier evidence, safety gate failure. Must restore baseline without proposal-state dependency. Not implemented.

---

## 8. Future rollback plan

Triggers: verifier mismatch, partial rejection, hidden divergence. Must preserve state and surface mismatch. Not implemented.

---

## 9. Staged implementation plan

| Stage | Status |
|-------|--------|
| stage_0_current_no_runtime | Current |
| stage_1_noop_opt_in_scaffold | Future; no-op only |
| stage_2_trace_only_l4_dry_run | Future |
| stage_3_verifier_mediated_dry_run | Future |
| stage_4_runtime_commit_candidate | **Blocked** |

---

## 10. Risk register

11 risks including default runtime change, verifier bypass, direct commit, hidden divergence, fallback/rollback failures, trace gaps, CLI accidents, test coverage, and overclaims.

---

## 11. Integration plan decision

`ready_for_stage_1_noop_opt_in_scaffold_design` — authorizes future no-op scaffold design only.

---

## 12. What this authorizes

- Phase 21A no-op opt-in scaffold: `phase21a_l4_noop_opt_in_scaffold`

---

## 13. What this does not authorize

- L4 runtime commit
- CUDA/vLLM/LMCache integration
- Performance or memory benchmarks

---

## 14. Remaining blockers

Runtime fallback/rollback, stages 1–3 implementation, L4 parity panel, exactkv_failures gate run, GPU memory measurement, performance benchmark, serving integration.

---

## 15. Recommended next phase

**Phase 21A:** L4 no-op opt-in scaffold (`phase21a_l4_noop_opt_in_scaffold`).

---

## 16. Claim boundaries

Integration plan documentation only; no speed, throughput, latency, serving, memory, or VeriCache reproduction claims.

---

## Run

```bash
python3 scripts/research/run_exp101_l4_integration_plan_review.py
```

Report: `reports/experiment_101_l4_integration_plan_review.json` (gitignored).

```bash
pytest tests/test_exp101_l4_integration_plan_review.py -q
```

**Run summary:**

| Metric | Value |
|--------|-------|
| Status | `plan_review_complete` |
| Decision | `ready_for_stage_1_noop_opt_in_scaffold_design` |
| L4 runtime commit authorized | false |
| Allowed next phase | `phase21a_l4_noop_opt_in_scaffold` |
| Change targets | 8 (all not_modified) |
| Implementation stages | 5 (stage 4 blocked) |
