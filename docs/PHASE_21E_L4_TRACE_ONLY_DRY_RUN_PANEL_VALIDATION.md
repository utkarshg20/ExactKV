# Phase 21E: L4 Trace-Only Dry-Run Panel Validation

**Status:** run `scripts/research/run_exp106_l4_trace_only_dry_run_panel_validation.py`.

> This is L4 trace-only dry-run panel validation, not L4 runtime implementation.  
> ExactKV default generation remains unchanged.  
> ExactKVGenerator remains unchanged.  
> Production CLI remains unchanged.  
> Trace-only decisions are diagnostic only.  
> Trace-only decisions cannot affect token commits.  
> Trace-only decisions are not exposed to generator decisions.  
> Missing verifier evidence blocks a decision rather than fabricating one.  
> Low verifier evidence coverage is not a runtime failure; it means trace schema work is needed.  
> Runtime fallback/rollback behavior is not implemented.  
> L4 runtime commit remains blocked.  
> Passing this panel does not authorize runtime commit integration.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_106_L4_TRACE_ONLY_DRY_RUN_PANEL_VALIDATION.md`](EXPERIMENT_106_L4_TRACE_ONLY_DRY_RUN_PANEL_VALIDATION.md)

---

## 1. Purpose

Validate the Phase 21D trace-only dry-run scaffold on panel generation traces: measure decision compute rate, verifier evidence coverage, and no-commit safety gate integrity.

---

## 2. Relation to Phase 21D

Phase 21D implemented the pure diagnostic evaluator and synthetic suite. Phase 21E applies it to real or injected panel round traces after unchanged generation.

---

## 3. Stage 2 panel validation boundary

- Generation runs unchanged via existing baseline path
- Round traces read post-generation
- Dry-run decisions computed from explicit evidence only
- No ExactKVGenerator changes
- No commit or generator exposure

---

## 4. Panel dimensions

Default: `Qwen/Qwen2.5-0.5B` (optional `--include-instruct`), CPU, float32, 4 prompts, compressors `noop,int8,int4_sim,k8_v4_sim`, max_new_tokens `4,8`.

---

## 5. Trace evidence extraction

**Proposal:** explicit `draft_tokens` / `proposal_token_ids` with source `exactkv_round_log_draft_tokens`.

**Verifier:** explicit top-level verifier fields or `acceptance.verifier_tokens` (not `accepted_tokens`).

---

## 6. Proposal evidence coverage

`proposal_evidence_coverage_rate = rounds_with_proposal / trace_inputs_built`.

---

## 7. Verifier evidence coverage

`verifier_evidence_coverage_rate = rounds_with_verifier / trace_inputs_built`.

---

## 8. Decision status counts

Aggregated across panel cells and broken down by model, compressor, prompt, max_new_tokens, and decision status.

---

## 9. Missing verifier evidence behavior

`decision_status = blocked_missing_verifier_evidence`, `block_reason = no explicit verifier evidence in trace`. No fabricated accept/reject decision.

---

## 10. Safety gates

All decisions: `dry_run_decision_used_for_token_commit=false`, `exposed_to_generator=false`, `verifier_source_of_truth=true`.

---

## 11. Validation result

`validate_exp106_panel_report` passes when safety gates hold and token/text parity holds for completed cells. Low verifier coverage does not fail validation.

---

## 12. Decision recommendation

| Condition | Recommendation |
|-----------|----------------|
| Verifier coverage zero | `phase21f_l4_verifier_evidence_trace_schema_design` |
| Verifier coverage partial | `phase21f_l4_trace_only_panel_repeat_with_evidence` |
| Verifier coverage sufficient + safety OK | `phase21f_stage3_verifier_mediated_dry_run_design` |

Forbidden: `l4_runtime_commit_implementation`.

---

## 13. What this authorizes

- Phase 21F verifier evidence trace schema design, panel repeat, or stage 3 dry-run design (per recommendation)

---

## 14. What this does not authorize

- L4 runtime commit integration
- Verifier-mediated compressed draft at runtime
- Production CLI changes

---

## 15. Remaining blockers

Stage 3 verifier-mediated dry-run, runtime fallback/rollback, L4 commit parity panel, exactkv_failures gate run, serving integration, performance benchmarks.

---

## 16. Recommended next phase

**Phase 21F:** L4 verifier evidence trace schema design — complete. See [`PHASE_21F_L4_VERIFIER_EVIDENCE_TRACE_SCHEMA_DESIGN.md`](PHASE_21F_L4_VERIFIER_EVIDENCE_TRACE_SCHEMA_DESIGN.md).

**Next:** Phase 21G schema scaffold (`phase21g_l4_verifier_evidence_trace_schema_scaffold`).

---

## 17. Claim boundaries

Panel-scoped diagnostic claims only; no speed, throughput, latency, serving, memory, or VeriCache reproduction claims.

---

## Run

```bash
python3 scripts/research/run_exp106_l4_trace_only_dry_run_panel_validation.py
```

```bash
pytest tests/test_exp106_l4_trace_only_dry_run_panel_validation.py -q
```

Report: `reports/experiment_106_l4_trace_only_dry_run_panel_validation.json` (gitignored).
