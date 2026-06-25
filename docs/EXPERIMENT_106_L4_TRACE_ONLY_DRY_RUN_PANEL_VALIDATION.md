# Experiment 106: L4 Trace-Only Dry-Run Panel Validation (Phase 21E)

**Experiment ID:** `exp106_l4_trace_only_dry_run_panel_validation`

Companion: [`PHASE_21E_L4_TRACE_ONLY_DRY_RUN_PANEL_VALIDATION.md`](PHASE_21E_L4_TRACE_ONLY_DRY_RUN_PANEL_VALIDATION.md)

---

## Run

```bash
python3 scripts/research/run_exp106_l4_trace_only_dry_run_panel_validation.py
```

Optional instruct model:

```bash
python3 scripts/research/run_exp106_l4_trace_only_dry_run_panel_validation.py --include-instruct
```

Default panel: `Qwen/Qwen2.5-0.5B`, CPU, 4 prompts, 4 compressors, max_new_tokens 4 and 8.

---

## Core API

- `run_exp106_l4_trace_only_dry_run_panel_validation(...)`
- `validate_exp106_panel_report(...)`
- `build_round_trace_records_for_cell(...)`
- `extract_verifier_evidence_from_round_trace(...)`
- `compute_phase21f_recommendation(...)`

---

## Report

`reports/experiment_106_l4_trace_only_dry_run_panel_validation.json`

---

## Tests

```bash
pytest tests/test_exp106_l4_trace_only_dry_run_panel_validation.py -q
```

No model downloads required for default unit tests.
