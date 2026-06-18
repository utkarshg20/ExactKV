# Experiment 089: Integration Design Review (Phase 17D)

**Experiment ID:** `exp089_integration_design_review`  
**Report:** `reports/experiment_089_integration_design_review.json`  
Companion: [`PHASE_17D_INTEGRATION_DESIGN_REVIEW.md`](PHASE_17D_INTEGRATION_DESIGN_REVIEW.md) · `exactkv/demo/integration_design_review.py`

> Integration design review only — not implementation.  
> ExactKV default generation remains unchanged.

---

## Run

```bash
python3 scripts/research/run_exp089_integration_design_review.py
```

No GPU, model downloads, or network required.

---

## Outputs

| Field | Description |
|-------|-------------|
| `integration_levels` | L0–L5 with status, evidence, risks, gates, claims |
| `current_implemented_level` | `L2_live_diagnostic_observer` |
| `gate_policy_before_token_commit_changes` | 10 gates before L4 research |
| `risk_register` | 10 risks with severity and mitigation |
| `recommended_next_phase` | `phase18a_integration_safety_spec` |
| `allowed_claims` / `forbidden_claims` | Phase 16 claim freeze |

---

## Integration levels summary

| ID | Status |
|----|--------|
| L0_demo_only | implemented |
| L1_external_shadow_observer | implemented |
| L2_live_diagnostic_observer | implemented |
| L3_guarded_draft_shadow_no_commit | not_implemented |
| L4_verifier_mediated_compressed_draft | not_implemented |
| L5_real_backend_integration | deferred |

---

## Claim scope

No speed, throughput, latency, serving, active GPU memory, or production-memory claims. ExactKV does not reproduce VeriCache throughput or serving results. Top-k agreement is supplementary only.

---

## Tests

```bash
pytest tests/test_exp089_integration_design_review.py -q
```
