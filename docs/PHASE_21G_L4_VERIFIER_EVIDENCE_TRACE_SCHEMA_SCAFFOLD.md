# Phase 21G: L4 Verifier Evidence Trace Schema Scaffold

**Status:** run `scripts/research/run_exp108_l4_verifier_evidence_trace_schema_scaffold.py`.

> This is verifier evidence trace schema scaffold, not runtime instrumentation.  
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
> Passing this scaffold authorizes only schema-example dry-run validation, not runtime instrumentation or commit integration.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_108_L4_VERIFIER_EVIDENCE_TRACE_SCHEMA_SCAFFOLD.md`](EXPERIMENT_108_L4_VERIFIER_EVIDENCE_TRACE_SCHEMA_SCAFFOLD.md)

---

## 1. Purpose

Implement a no-runtime scaffold that validates explicit verifier evidence trace records and converts them to trace-only dry-run inputs without changing generation.

---

## 2. Relation to Phase 21F

Phase 21F designed required fields, allowed/forbidden sources, and validation rules. Phase 21G implements record validation, conversion, and synthetic example processing.

---

## 3. Schema scaffold boundary

- Pure `exactkv/safety/` module
- Not imported by runtime generation
- No ExactKVGenerator changes
- No runtime trace instrumentation

---

## 4. Trace record schema

Immutable `L4VerifierEvidenceTraceRecord` with all Phase 21F verifier and metadata fields plus proposal fields and `diagnostic_only=true`.

---

## 5. Validation behavior

`validate_verifier_evidence_trace_record` enforces field presence, source allow/forbid lists, proposal/verifier separation, diagnostic-only flag, and no counts-only inference.

---

## 6. Conversion behavior

`convert_verifier_trace_to_l4_trace_only_input` maps valid records to `L4TraceOnlyDryRunInput` without fabricating prefixes; invalid records do not convert unless explicitly requested for diagnostics.

---

## 7. Synthetic examples

8 examples: 3 complete match variants, 2 blocked (missing/exception), 3 invalid (committed-as-verifier, counts-only, proposal/verifier alias).

---

## 8. Forbidden source rejection

Invalid examples with forbidden sources fail validation; summary counts rejections.

---

## 9. Proposal/verifier separation

Same source string or aliased token list objects fail validation.

---

## 10. Dry-run compatibility

Valid records convert and evaluate via Phase 21D `evaluate_l4_trace_only_input`; blocked records yield `blocked_missing_verifier_evidence`.

---

## 11. Decision result

`schema_scaffold_complete` when all example expectations match.

---

## 12. What this authorizes

- Phase 21H schema-example dry-run validation: `phase21h_l4_trace_only_dry_run_with_schema_examples`

---

## 13. What this does not authorize

- Runtime verifier instrumentation
- L4 runtime commit integration
- Stage 3 verifier-mediated dry-run

---

## 14. Remaining blockers

Runtime instrumentation, schema-example panel validation, stage 3 dry-run, runtime fallback/rollback, L4 commit parity panel.

---

## 15. Recommended next phase

**Phase 21H:** L4 verifier trace schema example validation — complete. See [`PHASE_21H_L4_VERIFIER_TRACE_SCHEMA_EXAMPLE_VALIDATION.md`](PHASE_21H_L4_VERIFIER_TRACE_SCHEMA_EXAMPLE_VALIDATION.md).

**Next:** Phase 21I stress/adversarial panel (`phase21i_l4_trace_schema_stress_adversarial_trace_injection_panel`).

---

## 16. Claim boundaries

Schema scaffold claims only; no speed, throughput, latency, serving, memory, or VeriCache reproduction claims.

---

## Run

```bash
python3 scripts/research/run_exp108_l4_verifier_evidence_trace_schema_scaffold.py
```

```bash
pytest tests/test_exp108_l4_verifier_evidence_trace_schema_scaffold.py -q
```

Report: `reports/experiment_108_l4_verifier_evidence_trace_schema_scaffold.json` (gitignored).
