# Experiment 108: L4 Verifier Evidence Trace Schema Scaffold (Phase 21G)

**Experiment ID:** `exp108_l4_verifier_evidence_trace_schema_scaffold`

Companion: [`PHASE_21G_L4_VERIFIER_EVIDENCE_TRACE_SCHEMA_SCAFFOLD.md`](PHASE_21G_L4_VERIFIER_EVIDENCE_TRACE_SCHEMA_SCAFFOLD.md)

---

## Run

```bash
python3 scripts/research/run_exp108_l4_verifier_evidence_trace_schema_scaffold.py
```

No model downloads, GPU, or network required.

---

## Core API

- `validate_verifier_evidence_trace_record(record)`
- `convert_verifier_trace_to_l4_trace_only_input(record)`
- `build_synthetic_schema_examples()`
- `run_exp108_l4_verifier_evidence_trace_schema_scaffold()`
- `validate_exp108_scaffold_report(...)`

---

## Report

`reports/experiment_108_l4_verifier_evidence_trace_schema_scaffold.json`

---

## Tests

```bash
pytest tests/test_exp108_l4_verifier_evidence_trace_schema_scaffold.py -q
```
