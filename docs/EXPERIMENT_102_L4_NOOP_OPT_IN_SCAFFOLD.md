# Experiment 102: L4 No-Op Opt-In Scaffold (Phase 21A)

**Experiment ID:** `exp102_l4_noop_opt_in_scaffold`

Companion: [`PHASE_21A_L4_NOOP_OPT_IN_SCAFFOLD.md`](PHASE_21A_L4_NOOP_OPT_IN_SCAFFOLD.md)

---

## Run

```bash
python3 scripts/research/run_exp102_l4_noop_opt_in_scaffold.py \
  --experimental-l4-verifier-mediated-draft
```

---

## Core API

- `default_l4_noop_opt_in_config()`
- `run_noop_scaffold_generation_external(...)`
- `run_exp102_l4_noop_opt_in_scaffold(...)`
- `validate_l4_noop_scaffold_report(...)`

---

## Report

`reports/experiment_102_l4_noop_opt_in_scaffold.json`

---

## Tests

```bash
pytest tests/test_exp102_l4_noop_opt_in_scaffold.py -q
```
