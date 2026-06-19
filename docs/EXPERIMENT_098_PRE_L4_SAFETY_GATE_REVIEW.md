# Experiment 098: Pre-L4 Safety Gate Review (Phase 20A)

**Experiment ID:** `exp098_pre_l4_safety_gate_review`

Companion: [`PHASE_20A_PRE_L4_SAFETY_GATE_REVIEW.md`](PHASE_20A_PRE_L4_SAFETY_GATE_REVIEW.md)

---

## Run

```bash
python3 scripts/research/run_exp098_pre_l4_safety_gate_review.py
```

No model downloads, GPU, or network required.

---

## Core API

- `inventory_evidence(...)`
- `evaluate_pre_l4_gates(...)`
- `compute_review_outcome(...)`
- `run_exp098_pre_l4_safety_gate_review(...)`
- `validate_exp098_report(...)`

---

## Report

`reports/experiment_098_pre_l4_safety_gate_review.json`

---

## Tests

```bash
pytest tests/test_exp098_pre_l4_safety_gate_review.py -q
```
