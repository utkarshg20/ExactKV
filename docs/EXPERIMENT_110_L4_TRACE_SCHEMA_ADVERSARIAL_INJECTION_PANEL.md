# Experiment 110: L4 Trace Schema Adversarial Injection Panel (Phase 21I)

**Experiment ID:** `exp110_l4_trace_schema_adversarial_injection_panel`

Companion: [`PHASE_21I_L4_TRACE_SCHEMA_ADVERSARIAL_INJECTION_PANEL.md`](PHASE_21I_L4_TRACE_SCHEMA_ADVERSARIAL_INJECTION_PANEL.md)

---

## Run

```bash
python3 scripts/research/run_exp110_l4_trace_schema_adversarial_injection_panel.py
```

No model downloads, GPU, or network required.

---

## Core API

- `build_adversarial_injection_cases()`
- `execute_adversarial_case(case)`
- `run_exp110_l4_trace_schema_adversarial_injection_panel()`
- `validate_exp110_adversarial_panel_report(...)`

---

## Report

`reports/experiment_110_l4_trace_schema_adversarial_injection_panel.json`

---

## Tests

```bash
pytest tests/test_exp110_l4_trace_schema_adversarial_injection_panel.py -q
```
