# Experiment 101: L4 Integration Plan Review (Phase 20D)

**Experiment ID:** `exp101_l4_integration_plan_review`

Companion: [`PHASE_20D_L4_INTEGRATION_PLAN_REVIEW.md`](PHASE_20D_L4_INTEGRATION_PLAN_REVIEW.md)

---

## Run

```bash
python3 scripts/research/run_exp101_l4_integration_plan_review.py
```

No model downloads, GPU, or network required.

---

## Core API

- `build_l4_integration_plan_review()`
- `evaluate_l4_integration_plan_decision(review)`
- `run_exp101_l4_integration_plan_review()`
- `validate_exp101_report(...)`

---

## Report

`reports/experiment_101_l4_integration_plan_review.json`

---

## Tests

```bash
pytest tests/test_exp101_l4_integration_plan_review.py -q
```
