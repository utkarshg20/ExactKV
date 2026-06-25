# Phase 21F: L4 Verifier Evidence Trace Schema Design

**Status:** run `scripts/research/run_exp107_l4_verifier_evidence_trace_schema_design.py`.

> This is verifier evidence trace schema design, not runtime instrumentation.  
> ExactKV default generation remains unchanged.  
> ExactKVGenerator remains unchanged.  
> Production CLI remains unchanged.  
> Verifier evidence must be explicit.  
> Proposal evidence and verifier evidence must be separate fields.  
> Round-log draft tokens are proposal evidence, not verifier evidence.  
> Missing verifier evidence blocks dry-run decisions rather than fabricating one.  
> Committed tokens cannot be used as verifier evidence unless a future trace explicitly marks them as full-KV verifier evidence.  
> Runtime instrumentation is not authorized in this phase.  
> L4 runtime commit remains blocked.  
> Passing this design authorizes only a schema scaffold phase, not runtime instrumentation or commit integration.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_107_L4_VERIFIER_EVIDENCE_TRACE_SCHEMA_DESIGN.md`](EXPERIMENT_107_L4_VERIFIER_EVIDENCE_TRACE_SCHEMA_DESIGN.md)

---

## 1. Purpose

Design explicit verifier evidence fields for future ExactKV round traces so Stage 2 trace-only dry-run can compute diagnostic accept/reject decisions without fabricating evidence or using committed tokens as verifier truth.

---

## 2. Relation to Phase 21E

Phase 21E panel validation showed 100% proposal coverage but 0% verifier evidence coverage. All dry-run decisions blocked with `blocked_missing_verifier_evidence`. This phase defines the schema needed to close that gap.

---

## 3. Why verifier evidence coverage was zero

Current round traces expose `draft_tokens` (proposal) but not explicit full-KV verifier token fields required by the Phase 21D evaluator. Acceptance counts and committed tokens exist but are comparison-only, not verifier evidence.

---

## 4. Required verifier evidence fields

14 verifier fields plus 6 metadata fields (`round_index`, `proposal_source`, `proposal_token_ids`, `trace_schema_version`, `created_by`, `diagnostic_only`).

Key fields: `verifier_evidence_available`, `verifier_evidence_source`, `verifier_evidence_token_ids`, `verifier_evidence_is_full_kv`, `verifier_evidence_is_authoritative`, prefix/suffix/mismatch fields, `verifier_block_reason`, `verifier_trace_complete`.

---

## 5. Proposal evidence vs verifier evidence separation

`proposal_token_ids` / `exactkv_round_log_draft_tokens` are proposal-only. Verifier fields must use distinct sources (`full_kv_verifier_output_tokens`, etc.). Same field cannot serve both roles.

---

## 6. Allowed verifier evidence sources

- Full-KV verifier output tokens
- Verifier comparison output for proposal tokens
- Verifier matching-prefix evidence
- Verifier mismatch evidence
- Verifier exception/block reason

Each source defines required fields, provenance rule, and validation rule.

---

## 7. Forbidden verifier evidence sources

Committed tokens (unmarked), accepted/rejected counts alone, baseline tokens, retokenized text, guessed IDs, compressed draft tokens, shadow top-1 proposals, round-log draft tokens as verifier.

---

## 8. Validation rules

10 rules including explicit evidence, full-KV-or-blocked, proposal/verifier distinct, missing evidence blocks, no committed-output inference, no accepted-count inference, immutable token IDs, source recorded, schema version, diagnostic-only flag.

---

## 9. Trace examples

7 examples: all_match, partial_match, first_mismatch, missing evidence (blocked), verifier exception, invalid committed-as-verifier, invalid counts-only.

---

## 10. Design decision

`verifier_evidence_trace_schema_design_complete` — schema scaffold authorized; runtime instrumentation and commit blocked.

---

## 11. What this authorizes

- Phase 21G schema scaffold: `phase21g_l4_verifier_evidence_trace_schema_scaffold`

---

## 12. What this does not authorize

- Runtime verifier instrumentation
- L4 runtime commit integration
- Stage 3 verifier-mediated dry-run

---

## 13. Remaining blockers

Schema scaffold implementation, runtime instrumentation, stage 3 dry-run, runtime fallback/rollback, L4 commit parity panel, serving integration.

---

## 14. Recommended next phase

**Phase 21G:** L4 verifier evidence trace schema scaffold — complete. See [`PHASE_21G_L4_VERIFIER_EVIDENCE_TRACE_SCHEMA_SCAFFOLD.md`](PHASE_21G_L4_VERIFIER_EVIDENCE_TRACE_SCHEMA_SCAFFOLD.md).

**Next:** Phase 21H schema-example dry-run validation (`phase21h_l4_trace_only_dry_run_with_schema_examples`).

---

## 15. Claim boundaries

Schema design claims only; no speed, throughput, latency, serving, memory, or VeriCache reproduction claims.

---

## Run

```bash
python3 scripts/research/run_exp107_l4_verifier_evidence_trace_schema_design.py
```

```bash
pytest tests/test_exp107_l4_verifier_evidence_trace_schema_design.py -q
```

Report: `reports/experiment_107_l4_verifier_evidence_trace_schema_design.json` (gitignored).
