# Experiment 109: L4 Verifier Trace Schema Example Validation (Phase 21H)

**Experiment ID:** `exp109_l4_verifier_trace_schema_example_validation`

Companion: [`PHASE_21H_L4_VERIFIER_TRACE_SCHEMA_EXAMPLE_VALIDATION.md`](PHASE_21H_L4_VERIFIER_TRACE_SCHEMA_EXAMPLE_VALIDATION.md)

---

## Run

```bash
python3 scripts/research/run_exp109_l4_verifier_trace_schema_example_validation.py
```

No model downloads, GPU, or network required.

---

## Core API

- `execute_schema_example(example)`
- `build_enforcement_rule_coverage(...)`
- `run_diagnostic_probes()`
- `run_exp109_l4_verifier_trace_schema_example_validation()`
- `validate_exp109_example_validation_report(...)`

---

## Report

`reports/experiment_109_l4_verifier_trace_schema_example_validation.json`

---

## Tests

```bash
pytest tests/test_exp109_l4_verifier_trace_schema_example_validation.py -q
```
