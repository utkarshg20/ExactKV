# Phase 20B: L4 Verifier-Mediated Compressed Draft Design Specification

**Status:** run `scripts/research/run_exp099_l4_verifier_mediated_design_spec.py`.

> This is an L4 design specification, not L4 implementation.  
> ExactKV default generation remains unchanged.  
> ExactKVGenerator remains unchanged.  
> L4 implementation is not authorized.  
> Full verification must remain the source of truth.  
> Draft proposals must never commit directly.  
> Only verifier-matched prefixes may be accepted in a future L4 implementation.  
> Rollback behavior must be defined and tested before implementation.  
> The future L4 opt-in flag must be disabled by default.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_099_L4_VERIFIER_MEDIATED_DESIGN_SPEC.md`](EXPERIMENT_099_L4_VERIFIER_MEDIATED_DESIGN_SPEC.md)

---

## 1. Purpose

Create a detailed L4 verifier-mediated compressed draft design specification. Answer: what would L4 look like, what safety contracts must it obey, what integration points would be touched, and what tests must pass before implementation?

---

## 2. Relation to Phase 20A

Phase 20A authorized L4 **design specification only** (`ready_for_l4_design_spec_only`). Phase 20B delivers that spec; L4 implementation remains blocked.

---

## 3. L4 intended flow

1. Default runtime unchanged unless `--experimental-l4-verifier-mediated-draft` (design-only flag).
2. Compressed draft proposes tokens from explicit proposal source.
3. Full-KV verifier evaluates proposals.
4. Only verified matching prefix may be accepted.
5. Mismatch triggers rollback to verifier output.
6. All commit decisions traced.
7. Safety failure falls back to baseline behavior.
8. `exactkv_failures > 0` fails gates.
9. No performance/memory/serving claims.

---

## 4. Draft proposal contract

Explicit source, provenance required, no direct commit, forbidden committed/baseline/verifier/retokenization sources. Promoted L3 source: `exactkv_round_log_draft_tokens`.

---

## 5. Full verifier contract

Full verifier is source of truth; cannot be bypassed; controls acceptance; mismatches surfaced.

---

## 6. Acceptance contract

Only longest verified matching prefix; accepted/rejected tokens traceable; no silent divergence.

---

## 7. Rollback contract

Rollback on mismatch, proposal exception, missing verifier evidence, and safety gate failure; restores baseline-safe behavior.

---

## 8. Fallback contract

Default runtime unchanged; opt-out equals existing behavior; fallback independent of compressed proposal state.

---

## 9. Opt-in contract

L4 disabled by default; proposed flag `--experimental-l4-verifier-mediated-draft` (not implemented).

---

## 10. Integration points

Documented touchpoints: `ExactKVGenerator`, verifier path, round traces, L3 policy, safety gates, reports, CLI flag — with risks and required tests before modification.

---

## 11. L4 test matrix

Unit, synthetic integration, and model tests defined. Performance/memory/serving/CUDA benchmarks forbidden for this phase.

---

## 12. Readiness gates

Eleven gates: default runtime, opt-in, verifier source of truth, no bypass, no direct commit, rollback, fallback, trace completeness, parity, exactkv failures, claim boundary.

---

## 13. What this authorizes

- Phase 20C contract tests (no runtime): `phase20c_l4_contract_tests_no_runtime`

---

## 14. What this does not authorize

- L4 runtime implementation
- CUDA/vLLM/LMCache integration
- Performance or memory benchmarks
- Token-commit changes in this phase

---

## 15. Remaining implementation blockers

ExactKVGenerator integration plan, fallback/rollback implementation, opt-in flag wiring, L4 parity panel, exactkv_failures gate run, GPU memory measurement, performance benchmark, serving integration.

---

## 16. Recommended next phase

**Phase 20C:** L4 contract tests without runtime — complete. See [`PHASE_20C_L4_CONTRACT_TESTS_NO_RUNTIME.md`](PHASE_20C_L4_CONTRACT_TESTS_NO_RUNTIME.md).

**Next:** Phase 20D integration plan review (`phase20d_l4_integration_plan_review`).

---

## 17. Claim boundaries

Design documentation and contract claims only; no speed, throughput, latency, serving, memory, or VeriCache reproduction claims.

---

## Run

```bash
python3 scripts/research/run_exp099_l4_verifier_mediated_design_spec.py
```

Report: `reports/experiment_099_l4_verifier_mediated_design_spec.json` (gitignored).

```bash
pytest tests/test_exp099_l4_verifier_mediated_design_spec.py -q
```

**Run summary:**

| Metric | Value |
|--------|-------|
| Status | `spec_complete` |
| Design outcome | `l4_design_spec_complete` |
| L4 implementation authorized | false |
| Allowed next phase | `phase20c_l4_contract_tests_no_runtime` |
| Readiness gates defined | 11 |
| ExactKVGenerator modified | false |
| Default runtime changed | false |
