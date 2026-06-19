# Experiment 103: L4 No-Op Scaffold Panel Validation (Phase 21B)

**Experiment ID:** `exp103_l4_noop_scaffold_panel_validation`

Companion: [`PHASE_21B_L4_NOOP_SCAFFOLD_PANEL_VALIDATION.md`](PHASE_21B_L4_NOOP_SCAFFOLD_PANEL_VALIDATION.md)

---

## Run

```bash
python3 scripts/research/run_exp103_l4_noop_scaffold_panel_validation.py \
  --experimental-l4-verifier-mediated-draft
```

Default panel: 2 models × 4 prompts × 4 compressors × 2 max_new_tokens = 64 cells.

---

## Core API

- `run_exp103_l4_noop_scaffold_panel_validation(...)`
- `validate_exp103_panel_report(...)`
- `aggregate_noop_panel_breakdowns(...)`

---

## Report

`reports/experiment_103_l4_noop_scaffold_panel_validation.json`

---

## Tests

```bash
pytest tests/test_exp103_l4_noop_scaffold_panel_validation.py -q
```
