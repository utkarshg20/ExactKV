# Experiment 097: L3 Promoted Source Validation (Phase 19C)

**Experiment ID:** `exp097_l3_promoted_source_validation`

Companion: [`PHASE_19C_L3_PROMOTED_SOURCE_VALIDATION.md`](PHASE_19C_L3_PROMOTED_SOURCE_VALIDATION.md)

---

## Run

```bash
python3 scripts/research/run_exp097_l3_promoted_source_validation.py \
  --guarded-draft-shadow-no-commit
```

Default proposal source: `exactkv_round_log_draft_tokens`

Default models: `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-0.5B-Instruct`

---

## Core API

- `build_promoted_source_policy()`
- `evaluate_cell_source_viability_gates(...)`
- `aggregate_source_viability_gate_summary(...)`
- `aggregate_promoted_source_breakdowns(...)`
- `compute_promoted_source_decision(...)`
- `run_exp097_l3_promoted_source_validation(...)`
- `validate_exp097_report(...)`

---

## Report

`reports/experiment_097_l3_promoted_source_validation.json`

---

## Tests

```bash
pytest tests/test_exp097_l3_promoted_source_validation.py -q
```

Default tests do not require model downloads, CUDA, vLLM, or network.
