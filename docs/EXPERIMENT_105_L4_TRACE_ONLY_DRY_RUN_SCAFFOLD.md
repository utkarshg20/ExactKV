# Experiment 105: L4 Trace-Only Dry-Run Scaffold (Phase 21D)

**Experiment ID:** `exp105_l4_trace_only_dry_run_scaffold`

Companion: [`PHASE_21D_L4_TRACE_ONLY_DRY_RUN_SCAFFOLD.md`](PHASE_21D_L4_TRACE_ONLY_DRY_RUN_SCAFFOLD.md)

---

## Run

```bash
python3 scripts/research/run_exp105_l4_trace_only_dry_run_scaffold.py
```

Optional real trace extraction:

```bash
python3 scripts/research/run_exp105_l4_trace_only_dry_run_scaffold.py --try-real-traces
```

No model downloads or GPU required for default synthetic suite.

---

## Core API

- `evaluate_l4_trace_only_input(input)`
- `build_l4_trace_only_inputs_from_records(records)`
- `run_synthetic_trace_only_suite()`
- `run_exp105_l4_trace_only_dry_run_scaffold(...)`
- `validate_l4_trace_only_scaffold_report(...)`

---

## Report

`reports/experiment_105_l4_trace_only_dry_run_scaffold.json`

---

## Tests

```bash
pytest tests/test_exp105_l4_trace_only_dry_run_scaffold.py -q
```
