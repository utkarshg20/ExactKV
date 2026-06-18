# Experiment 094: Shadow Proposal Provenance Audit (Phase 18E)

**Experiment ID:** `exp094_shadow_proposal_provenance_audit`

Companion: [`PHASE_18E_SHADOW_PROPOSAL_PROVENANCE_AUDIT.md`](PHASE_18E_SHADOW_PROPOSAL_PROVENANCE_AUDIT.md)

---

## Run

```bash
python3 scripts/research/run_exp094_shadow_proposal_provenance_audit.py \
  --guarded-draft-shadow-no-commit
```

---

## Core API

- `classify_proposal_audit_categories(...)`
- `build_proposal_audit_record(...)`
- `aggregate_provenance_audit(...)`
- `compute_decision_recommendation(...)`

---

## Report

`reports/experiment_094_shadow_proposal_provenance_audit.json`

Key fields: `audit_records`, `category_summary`, `decision_recommendation`, `decision_reason`, `match_rate_successful_extractions`.

---

## Tests

```bash
pytest tests/test_exp094_shadow_proposal_provenance_audit.py -q
```

Default tests do not require model downloads, CUDA, vLLM, or network.
