# Experiment 107: L4 Verifier Evidence Trace Schema Design (Phase 21F)

**Experiment ID:** `exp107_l4_verifier_evidence_trace_schema_design`

Companion: [`PHASE_21F_L4_VERIFIER_EVIDENCE_TRACE_SCHEMA_DESIGN.md`](PHASE_21F_L4_VERIFIER_EVIDENCE_TRACE_SCHEMA_DESIGN.md)

---

## Run

```bash
python3 scripts/research/run_exp107_l4_verifier_evidence_trace_schema_design.py
```

No model downloads, GPU, or network required.

---

## Core API

- `build_l4_verifier_evidence_trace_schema_design()`
- `validate_verifier_evidence_trace_record(...)`
- `validate_trace_example(...)`
- `run_exp107_l4_verifier_evidence_trace_schema_design()`
- `validate_exp107_report(...)`

---

## Report

`reports/experiment_107_l4_verifier_evidence_trace_schema_design.json`

---

## Tests

```bash
pytest tests/test_exp107_l4_verifier_evidence_trace_schema_design.py -q
```
