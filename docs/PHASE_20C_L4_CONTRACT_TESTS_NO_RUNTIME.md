# Phase 20C: L4 Contract Tests with No Runtime Integration

**Status:** run `scripts/research/run_exp100_l4_contract_tests_no_runtime.py`.

> This is no-runtime L4 contract testing, not L4 runtime implementation.  
> ExactKV default generation remains unchanged.  
> ExactKVGenerator remains unchanged.  
> L4 implementation is not authorized.  
> Full verifier evidence is the source of truth in the synthetic contract evaluator.  
> Draft proposals must never commit directly.  
> Only verifier-matched prefixes are accepted in the synthetic contract tests.  
> Runtime fallback/rollback behavior is not implemented in this phase.  
> No model experiments are run.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_100_L4_CONTRACT_TESTS_NO_RUNTIME.md`](EXPERIMENT_100_L4_CONTRACT_TESTS_NO_RUNTIME.md)

---

## 1. Purpose

Validate L4 design contracts on synthetic token sequences using a pure contract evaluator. Confirm all-match, partial-match, mismatch, exception, missing-evidence, and hidden-divergence cases without touching generation runtime.

---

## 2. Relation to Phase 20B

Phase 20B defined L4 contracts and test matrix. Phase 20C implements synthetic contract tests only — no runtime wiring.

---

## 3. No-runtime boundary

- No `ExactKVGenerator` changes
- No default runtime changes
- No CLI opt-in
- No model experiments
- Contract evaluator is not imported by runtime generation

---

## 4. Synthetic contract cases

Seven cases: `all_match_accept_all`, `partial_match_accept_prefix`, `first_token_mismatch_accept_none`, `proposal_exception_fallback`, `missing_verifier_evidence_fallback`, `hidden_divergence_attempt_fails`, `direct_commit_attempt_fails`.

---

## 5. Contract evaluator behavior

`evaluate_l4_synthetic_contract_case` applies verifier-mediated prefix acceptance, fallback on exception/missing evidence, and rejects direct commit and hidden divergence attempts. All decisions traced.

---

## 6. Verifier source-of-truth rule

Full verifier token evidence controls acceptance. Proposal tokens never commit without verifier agreement.

---

## 7. Acceptance-prefix rule

Accepted prefix = longest prefix where proposal tokens match verifier tokens. First-token mismatch yields empty prefix.

---

## 8. Rollback/fallback cases

Proposal exception and missing verifier evidence trigger fallback (synthetic only). Runtime rollback not implemented.

---

## 9. Hidden divergence rejection

`hidden_divergence_attempt` cases must fail contract evaluation with trace evidence.

---

## 10. Direct commit rejection

`direct_commit_attempt` cases must fail contract evaluation; proposals never commit directly.

---

## 11. Trace completeness

Every case records proposal, verifier, accepted/rejected tokens, decision steps, and safety flags.

---

## 12. Suite summary

7/7 cases pass expectations; 2 fallback cases; 2 expected-fail safety violations detected.

---

## 13. What this authorizes

- Phase 20D integration plan review: `phase20d_l4_integration_plan_review`

---

## 14. What this does not authorize

- L4 runtime implementation
- CUDA/vLLM/LMCache integration
- Performance or memory benchmarks
- Model parity panels

---

## 15. Remaining blockers

ExactKVGenerator integration plan, runtime fallback/rollback, L4 parity panel, exactkv_failures gate run, GPU memory measurement, performance benchmark, serving integration.

---

## 16. Recommended next phase

**Phase 20D:** L4 integration plan review — complete. See [`PHASE_20D_L4_INTEGRATION_PLAN_REVIEW.md`](PHASE_20D_L4_INTEGRATION_PLAN_REVIEW.md).

**Next:** Phase 21A no-op opt-in scaffold (`phase21a_l4_noop_opt_in_scaffold`).

---

## 17. Claim boundaries

Synthetic contract-test claims only; no speed, throughput, latency, serving, memory, or VeriCache reproduction claims.

---

## Run

```bash
python3 scripts/research/run_exp100_l4_contract_tests_no_runtime.py
```

Report: `reports/experiment_100_l4_contract_tests_no_runtime.json` (gitignored).

```bash
pytest tests/test_exp100_l4_contract_tests_no_runtime.py -q
```

**Run summary:**

| Metric | Value |
|--------|-------|
| Status | `contract_tests_complete` |
| Cases | 7/7 pass |
| L4 runtime added | false |
| Allowed next phase | `phase20d_l4_integration_plan_review` |
