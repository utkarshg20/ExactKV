# Experiment 113: L4 Stage 3 Verifier-Mediated Dry-Run Scaffold

**Phase:** 21L  
**Script:** `scripts/research/run_exp113_l4_stage3_verifier_mediated_dry_run_scaffold.py`  
**Module:** `exactkv/safety/l4_stage3_verifier_mediated_dry_run_scaffold.py`  
**Report:** `reports/experiment_113_l4_stage3_verifier_mediated_dry_run_scaffold.json` (gitignored)

> **Trace-only scaffold.** No model execution. No GPU. No runtime commit.  
> ExactKVGenerator unchanged. Verifier not executed — trace mapping only.

---

## Purpose

Execute the Stage 3 verifier-mediated dry-run decision graph on synthetic trace records, producing deterministic `L4Stage3DryRunResult` objects without model inference.

---

## Core layers

1. **Proposal ingestion** — read-only L3 round-log draft tokens  
2. **Verifier trace mapping** — schema v1 field mapping (no verifier execution)  
3. **Decision graph evaluator** — prefix walk simulation  
4. **Rollback simulation** — conceptual only  
5. **Panel report** — aggregated scaffold results  

---

## Terminal states

| State | Trigger |
|---|---|
| `ACCEPT_PREFIX` | Full prefix match |
| `REJECT` | Token mismatch |
| `BLOCK_MISSING_EVIDENCE` | Missing verifier evidence |
| `INVALID_TRACE` | Schema/alias/corruption |

---

## Scaffold cases (5)

- full match  
- partial mismatch  
- missing verifier  
- corrupted trace  
- adversarial aliasing  

---

## Authorization

**Allowed next phase:** `phase21m_l4_stage3_verifier_mediated_dry_run_panel_validation`

---

## Run

```bash
python3 scripts/research/run_exp113_l4_stage3_verifier_mediated_dry_run_scaffold.py
```

```bash
pytest tests/test_exp113_l4_stage3_verifier_mediated_dry_run_scaffold.py -q
```
