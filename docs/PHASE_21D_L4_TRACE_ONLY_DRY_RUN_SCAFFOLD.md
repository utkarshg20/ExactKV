# Phase 21D: L4 Trace-Only Dry-Run Scaffold

**Status:** run `scripts/research/run_exp105_l4_trace_only_dry_run_scaffold.py`.

> This is an L4 trace-only dry-run scaffold, not L4 runtime implementation.  
> ExactKV default generation remains unchanged.  
> ExactKVGenerator remains unchanged.  
> Production CLI remains unchanged.  
> Trace-only decisions are diagnostic only.  
> Trace-only decisions cannot affect token commits.  
> Trace-only decisions are not exposed to generator decisions.  
> Missing verifier evidence blocks a decision rather than fabricating one.  
> Runtime fallback/rollback behavior is not implemented.  
> L4 runtime commit remains blocked.  
> Passing this scaffold authorizes only trace-only panel validation, not runtime commit integration.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_105_L4_TRACE_ONLY_DRY_RUN_SCAFFOLD.md`](EXPERIMENT_105_L4_TRACE_ONLY_DRY_RUN_SCAFFOLD.md)

---

## 1. Purpose

Implement Stage 2 trace-only dry-run scaffold: compute diagnostic accept/reject decisions from explicit trace evidence without affecting generation.

---

## 2. Relation to Phase 21C

Phase 21C defined the trace-only dry-run design. Phase 21D implements the pure diagnostic evaluator and synthetic suite.

---

## 3. Stage 2 scaffold boundary

- Pure evaluator in `exactkv/safety/` only
- Not imported by runtime generation
- No ExactKVGenerator changes
- No commit or generator exposure

---

## 4. Synthetic scaffold suite

8 synthetic records covering all required decision statuses without model downloads.

---

## 5. Trace-only evaluator behavior

Longest verified prefix computation; blocks on missing evidence; fails on hidden divergence and direct commit attempts.

---

## 6. Evidence extraction rules

Explicit `proposal_token_ids` and `verifier_evidence_token_ids` only. Forbidden: guessed tokens, retokenized text, committed/baseline as proposal source.

---

## 7. Missing evidence behavior

Empty proposal → `blocked_missing_proposal`. Empty verifier → `blocked_missing_verifier_evidence`. Never fabricates tokens.

---

## 8. Safety gates

All decisions: `dry_run_decision_used_for_token_commit=false`, `exposed_to_generator=false`, `verifier_source_of_truth=true`.

---

## 9. Validation behavior

`validate_l4_trace_only_scaffold_report` fails on commit exposure, missing-evidence-as-match, or safety invariant violations.

---

## 10. What this authorizes

- Phase 21E trace-only panel validation: `phase21e_l4_trace_only_dry_run_panel_validation`

---

## 11. What this does not authorize

- L4 runtime commit integration
- Verifier-mediated compressed draft at runtime

---

## 12. Remaining blockers

Stage 2 panel validation, stage 3 verifier dry-run, runtime fallback/rollback, L4 commit parity panel, exactkv_failures gate run, serving integration.

---

## 13. Recommended next phase

**Phase 21E:** L4 trace-only dry-run panel validation — complete. See [`PHASE_21E_L4_TRACE_ONLY_DRY_RUN_PANEL_VALIDATION.md`](PHASE_21E_L4_TRACE_ONLY_DRY_RUN_PANEL_VALIDATION.md).

**Next:** Phase 21F per panel `decision_recommendation`.

---

## 14. Claim boundaries

Diagnostic dry-run claims only; no speed, throughput, latency, serving, memory, or VeriCache reproduction claims.

---

## Run

```bash
python3 scripts/research/run_exp105_l4_trace_only_dry_run_scaffold.py
```

Report: `reports/experiment_105_l4_trace_only_dry_run_scaffold.json` (gitignored).

```bash
pytest tests/test_exp105_l4_trace_only_dry_run_scaffold.py -q
```

**Run summary:**

| Metric | Value |
|--------|-------|
| Status | `scaffold_complete` |
| Synthetic decisions | 8 |
| All statuses covered | true |
| Runtime commit authorized | false |
| Allowed next phase | `phase21e_l4_trace_only_dry_run_panel_validation` |
