# Experiment 104: L4 Trace-Only Dry-Run Design (Phase 21C)

**Experiment ID:** `exp104_l4_trace_only_dry_run_design`

Companion: [`PHASE_21C_L4_TRACE_ONLY_DRY_RUN_DESIGN.md`](PHASE_21C_L4_TRACE_ONLY_DRY_RUN_DESIGN.md)

---

## Run

```bash
python3 scripts/research/run_exp104_l4_trace_only_dry_run_design.py
```

No model downloads, GPU, or network required.

---

## Core API

- `build_l4_trace_only_dry_run_design()`
- `evaluate_l4_trace_only_design_decision(design)`
- `run_exp104_l4_trace_only_dry_run_design()`
- `validate_exp104_report(...)`

---

## Report

`reports/experiment_104_l4_trace_only_dry_run_design.json`

---

## Tests

```bash
pytest tests/test_exp104_l4_trace_only_dry_run_design.py -q
```
