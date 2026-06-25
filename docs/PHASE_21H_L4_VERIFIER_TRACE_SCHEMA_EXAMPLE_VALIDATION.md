# Phase 21H: L4 Verifier Trace Schema Example Validation

**Status:** run `scripts/research/run_exp109_l4_verifier_trace_schema_example_validation.py`.

> This is schema-example trace-only validation, not runtime instrumentation.  
> ExactKV default generation remains unchanged.  
> ExactKVGenerator remains unchanged.  
> Production CLI remains unchanged.  
> Verifier evidence must be explicit.  
> Proposal evidence and verifier evidence must be separate fields.  
> Round-log draft tokens are proposal evidence, not verifier evidence.  
> Missing verifier evidence blocks dry-run decisions rather than fabricating one.  
> Committed tokens cannot be used as verifier evidence unless a schema record explicitly marks them as full-KV verifier evidence.  
> Runtime instrumentation is not authorized in this phase.  
> L4 runtime commit remains blocked.  
> Passing this validation authorizes only schema stress/adversarial panel work, not runtime instrumentation or commit integration.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_109_L4_VERIFIER_TRACE_SCHEMA_EXAMPLE_VALIDATION.md`](EXPERIMENT_109_L4_VERIFIER_TRACE_SCHEMA_EXAMPLE_VALIDATION.md)

---

## 1. Purpose

Execute and validate all Phase 21F–21G schema examples through trace-only validation, conversion, and dry-run classification with enforcement-rule coverage reporting.

---

## 2. Relation to Phase 21G

Phase 21G implemented record validation and conversion. Phase 21H runs the full example suite, diagnostic probes, and coverage metrics to confirm schema correctness end-to-end.

---

## 3. Schema example validation boundary

- Synthetic examples only
- No ExactKVGenerator changes
- No runtime instrumentation
- Trace-only diagnostics

---

## 4. Trace record schema

Uses `l4_verifier_evidence_v1` records from Phase 21G `build_synthetic_schema_examples()`.

---

## 5. Validation behavior

Per-example: schema validation → conversion (if valid) → dry-run classification. Invalid traces rejected; blocked traces yield `blocked_missing_verifier_evidence`.

---

## 6. Conversion behavior

Valid records convert via Phase 21G `convert_verifier_trace_to_l4_trace_only_input`; invalid records do not convert without explicit diagnostic flag.

---

## 7. Synthetic examples

All 8 examples: 3 complete match variants, 2 blocked, 3 invalid.

---

## 8. Forbidden source rejection

Committed-as-verifier, counts-only, and proposal/verifier alias traces fail validation with 100% detection accuracy.

---

## 9. Proposal/verifier separation

Enforcement rule `proposal_verifier_separation` exercised across valid and invalid examples.

---

## 10. Dry-run compatibility

Classifications: `all_match`, `partial_match`, `first_token_mismatch`, `blocked_missing_verifier_evidence`, `invalid_trace` (rejected before dry-run).

---

## 11. Decision result

`schema_example_validation_complete` when all examples pass and all enforcement rules are exercised.

---

## 12. What this authorizes

- Phase 21I stress/adversarial panel: `phase21i_l4_trace_schema_stress_adversarial_trace_injection_panel`

---

## 13. What this does not authorize

- Runtime verifier instrumentation
- L4 runtime commit integration
- Stage 3 verifier-mediated dry-run

---

## 14. Remaining blockers

Stress/adversarial panel, runtime instrumentation, stage 3 dry-run, runtime fallback/rollback, L4 commit parity panel.

---

## 15. Recommended next phase

**Phase 21I:** L4 trace schema adversarial injection panel — complete. See [`PHASE_21I_L4_TRACE_SCHEMA_ADVERSARIAL_INJECTION_PANEL.md`](PHASE_21I_L4_TRACE_SCHEMA_ADVERSARIAL_INJECTION_PANEL.md).

**Next:** Phase 21J runtime instrumentation design (`phase21j_l4_verifier_evidence_runtime_instrumentation_design`).

---

## 16. Claim boundaries

Schema example validation claims only; no speed, throughput, latency, serving, memory, or VeriCache reproduction claims.

---

## Run

```bash
python3 scripts/research/run_exp109_l4_verifier_trace_schema_example_validation.py
```

```bash
pytest tests/test_exp109_l4_verifier_trace_schema_example_validation.py -q
```

Report: `reports/experiment_109_l4_verifier_trace_schema_example_validation.json` (gitignored).
