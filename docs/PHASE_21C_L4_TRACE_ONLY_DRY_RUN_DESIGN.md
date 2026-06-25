# Phase 21C: L4 Trace-Only Dry-Run Design

**Status:** run `scripts/research/run_exp104_l4_trace_only_dry_run_design.py`.

> This is L4 trace-only dry-run design, not L4 runtime implementation.  
> ExactKV default generation remains unchanged.  
> ExactKVGenerator remains unchanged.  
> Production CLI remains unchanged.  
> Trace-only decisions are diagnostic only.  
> Trace-only decisions cannot affect token commits.  
> Trace-only decisions are not exposed to generator decisions.  
> Missing verifier evidence blocks a decision rather than fabricating one.  
> Runtime fallback/rollback behavior is not implemented.  
> L4 runtime commit remains blocked.  
> Passing this design authorizes only a trace-only dry-run scaffold, not runtime commit integration.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_104_L4_TRACE_ONLY_DRY_RUN_DESIGN.md`](EXPERIMENT_104_L4_TRACE_ONLY_DRY_RUN_DESIGN.md)

---

## 1. Purpose

Define Stage 2 L4 trace-only dry-run: how to compute verifier-mediated accept/reject decisions from existing round traces without affecting commits or generation.

---

## 2. Relation to Phase 21B

Phase 21B validated Stage 1 no-op scaffold panel parity. Phase 21C designs Stage 2 trace-only dry-run behavior.

---

## 3. Stage 2 trace-only boundary

- Read existing round traces post-generation
- Compute diagnostic decisions only
- No ExactKVGenerator changes
- No commit effect
- No generator exposure

---

## 4. Intended dry-run behavior

10-step flow: unchanged generation → read traces → explicit proposal/verifier evidence → compute prefix decision → write diagnostics → block on missing evidence.

---

## 5. Evidence source plan

**Allowed:** `exactkv_round_log_draft_tokens`, explicit verifier/full-KV trace fields.

**Forbidden:** retokenized text, guessed tokens, committed/baseline tokens as proposals.

**Missing evidence:** blocks decision; never fabricates tokens.

---

## 6. Future decision schema

Fields: round_index, proposal/verifier/accepted/rejected tokens, decision_status, block_reason, safety flags, trace_complete, interpretation_note.

**Statuses:** all_match, partial_match, first_token_mismatch, blocked_missing_proposal, blocked_missing_verifier_evidence, failed_hidden_divergence, failed_direct_commit_attempt, invalid_trace.

---

## 7. Safety gates

10 gates including trace_only, no_commit_effect, no_generator_exposure, missing_evidence_blocks, claim_boundary.

---

## 8. Risk register

9 risks including commit influence, silent match on missing evidence, dry-run overclaim, prefix match overclaim.

---

## 9. Design decision

`trace_only_dry_run_design_complete` — Stage 2 scaffold design authorized; runtime commit blocked.

---

## 10. What this authorizes

- Phase 21D trace-only dry-run scaffold: `phase21d_l4_trace_only_dry_run_scaffold`

---

## 11. What this does not authorize

- L4 runtime commit integration
- Verifier-mediated compressed draft at runtime
- Production CLI changes

---

## 12. Remaining blockers

Stage 2 scaffold implementation, stages 3–4, L4 commit parity panel, exactkv_failures gate run, runtime fallback/rollback, GPU memory measurement, performance benchmark, serving integration.

---

## 13. Recommended next phase

**Phase 21D:** L4 trace-only dry-run scaffold — complete. See [`PHASE_21D_L4_TRACE_ONLY_DRY_RUN_SCAFFOLD.md`](PHASE_21D_L4_TRACE_ONLY_DRY_RUN_SCAFFOLD.md).

**Next:** Phase 21E trace-only panel validation (`phase21e_l4_trace_only_dry_run_panel_validation`).

---

## 14. Claim boundaries

Trace-only diagnostic design claims only; no speed, throughput, latency, serving, memory, or VeriCache reproduction claims.

---

## Run

```bash
python3 scripts/research/run_exp104_l4_trace_only_dry_run_design.py
```

Report: `reports/experiment_104_l4_trace_only_dry_run_design.json` (gitignored).

```bash
pytest tests/test_exp104_l4_trace_only_dry_run_design.py -q
```

**Run summary:**

| Metric | Value |
|--------|-------|
| Status | `design_complete` |
| Outcome | `trace_only_dry_run_design_complete` |
| Runtime commit authorized | false |
| Allowed next phase | `phase21d_l4_trace_only_dry_run_scaffold` |
| Safety gates | 10 |
