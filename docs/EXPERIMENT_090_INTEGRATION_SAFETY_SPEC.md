# Experiment 090: Integration Safety Spec (Phase 18A)

**Experiment ID:** `exp090_integration_safety_spec`  
**Report:** `reports/experiment_090_integration_safety_spec.json`  
Companion: [`PHASE_18A_INTEGRATION_SAFETY_SPEC.md`](PHASE_18A_INTEGRATION_SAFETY_SPEC.md) · `exactkv/safety/integration_safety_spec.py`

> Integration safety specification only — not implementation.

---

## Run

```bash
python3 scripts/research/run_exp090_integration_safety_spec.py
```

No GPU, model downloads, or network required.

---

## Outputs

| Field | Description |
|-------|-------------|
| `safety_levels` | L2–L5 with behavior, gates, tests, claim boundaries |
| `mandatory_invariants` | 15 non-negotiable invariants |
| `gates` | 11 gate definitions with pass/fail conditions |
| `passing_synthetic_proposals` | Validator results for L3/L4 passing plans |
| `failing_synthetic_proposals` | Validator results for 8 rejected plans |
| `recommended_next_phase` | `phase18b_guarded_draft_shadow_no_commit_spec_or_scaffold` |

---

## Proposal validator

```python
from exactkv.safety import IntegrationProposal, validate_integration_proposal
```

---

## Tests

```bash
pytest tests/test_exp090_integration_safety_spec.py -q
```
