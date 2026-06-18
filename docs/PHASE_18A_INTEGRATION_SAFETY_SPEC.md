# Phase 18A: Integration Safety Spec

**Status:** run `scripts/research/run_exp090_integration_safety_spec.py`.

> This is an integration safety specification, not integration implementation.  
> ExactKV default generation remains unchanged.  
> Streaming attention is not integrated into token commit.  
> Full verification must remain the source of truth before any compressed draft token can be accepted.  
> Shadow output cannot directly commit tokens.  
> Top-k agreement is supplementary only and is not an exactness guarantee.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_090_INTEGRATION_SAFETY_SPEC.md`](EXPERIMENT_090_INTEGRATION_SAFETY_SPEC.md)

---

## 1. Purpose

Define a machine-readable and document-readable safety contract that must be satisfied before any L3/L4 integration work is allowed. Answer: what exact invariants, gates, schemas, and failure conditions must govern future draft-shadow or verifier-mediated token-commit research?

No runtime changes, no L3/L4/L5 implementation, no new model experiments.

---

## 2. Why this follows Phase 17D

Phase 17D defined integration levels L0–L5, gate policy, and risk register, and recommended this safety spec before any draft-shadow or verifier-mediated work. Phase 18A turns that policy into enforceable invariants, gate definitions, and a proposal validator.

---

## 3. Safety levels L2–L5

| Level | ID | Status |
|-------|-----|--------|
| L2 | `L2_DIAGNOSTIC_OBSERVER` | implemented |
| L3 | `L3_GUARDED_DRAFT_SHADOW_NO_COMMIT` | future |
| L4 | `L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT` | future |
| L5 | `L5_BACKEND_INTEGRATION` | deferred |

Each level defines allowed behavior, forbidden behavior, required gates, required tests, and claim boundary in `exactkv/safety/integration_safety_spec.py`.

---

## 4. Mandatory invariants

15 non-negotiable invariants including:

- Default runtime unchanged unless explicit opt-in
- Fallback restores baseline generation
- Full verifier remains source of truth for token commit
- Shadow cannot bypass verification; compressed draft cannot commit directly
- `exactkv_failures` zero in gate tests; baseline-vs-integrated token parity on fixed greedy
- All divergences surfaced; safety gate failures fail the report
- Top-k supplementary only; no perf/memory/serving/VeriCache claims without validation

---

## 5. Gate definitions

11 gates: `default_runtime_gate`, `fallback_gate`, `verifier_source_of_truth_gate`, `no_verifier_bypass_gate`, `no_direct_shadow_commit_gate`, `baseline_token_parity_gate`, `exactkv_failure_gate`, `divergence_visibility_gate`, `claim_boundary_gate`, `report_schema_gate`, `audit_gate`.

Each gate has name, purpose, required evidence, pass/fail conditions, and applicable levels.

---

## 6. Proposal validator

`validate_integration_proposal()` evaluates proposed integration plans against gates. Input fields include `proposed_level`, opt-in flags, verifier/shadow/commit booleans, divergence visibility, and claim flags. Output: pass/fail, `failed_gates`, `warnings`, `required_next_evidence`.

---

## 7. Passing synthetic proposals

- `l3_diagnostic_draft_shadow_no_commit` — opt-in, no shadow commit, fallback, reports failures
- `l4_verifier_mediated_compressed_draft_with_full_verifier` — L4 with `verifier_source_of_truth=true`

---

## 8. Failing synthetic proposals

- `shadow_direct_commit`
- `verifier_bypass`
- `default_runtime_changed`
- `hidden_token_divergence`
- `performance_claim_without_measurement`
- `memory_claim_without_active_measurement`
- `serving_claim_without_backend`
- `vericache_reproduction_overclaim`

---

## 9. Allowed claims

Phase 16 claim freeze (`ALLOWED_CLAIMS`) — diagnostic capabilities and tested-panel parity only.

---

## 10. Forbidden claims

Phase 16 claim freeze (`FORBIDDEN_CLAIMS`) — speed, memory, serving, VeriCache reproduction, shadow exactness, production-ready.

---

## 11. Deferred work

L3/L4 implementation blocked until gates pass; L5 and backend integration deferred; vLLM/LMCache direct integration no-go per deferred work register.

---

## 12. Recommended next phase

**Phase 18B (complete):** L3 guarded draft-shadow no-commit scaffold — see [`PHASE_18B_GUARDED_DRAFT_SHADOW_NO_COMMIT.md`](PHASE_18B_GUARDED_DRAFT_SHADOW_NO_COMMIT.md).

**Phase 18C (proposed):** guarded draft-shadow panel validation — explicit approval required.

---

## Run

```bash
python3 scripts/research/run_exp090_integration_safety_spec.py
```

Report: `reports/experiment_090_integration_safety_spec.json` (gitignored).
