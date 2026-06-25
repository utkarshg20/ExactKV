# Experiment 112: L4 Stage 3 Verifier-Mediated Dry-Run Design

**Phase:** 21K  
**Script:** `scripts/research/run_exp112_l4_stage3_verifier_mediated_dry_run_design.py`  
**Module:** `exactkv/safety/l4_stage3_verifier_mediated_dry_run_design.py`  
**Report:** `reports/experiment_112_l4_stage3_verifier_mediated_dry_run_design.json` (gitignored)

> **Design only.** No runtime execution. No model execution.  
> ExactKVGenerator unchanged. Default runtime unchanged.  
> Verifier is not executed — trace field reconciliation only.  
> No performance/memory/serving claims.

---

## Purpose

Design Stage 3 of the L4 system: a verifier-mediated dry-run execution model that consumes L3 proposals and L4 verifier evidence schema v1, producing a deterministic decision graph and prefix-level accept/reject simulation — without modifying runtime or executing models.

---

## Core design object

`L4Stage3VerifierMediatedDryRunDesign` includes:

- proposal ingestion model (L3 round-log)
- verifier evidence mapping model (schema v1)
- evidence reconciliation layer
- decision graph model
- prefix acceptance simulation (conceptual)
- mismatch handling policy
- rollback simulation (no execution)
- trace-only output schema (`L4Stage3DryRunResult`)

---

## Decision graph terminal states

- `ACCEPT_PREFIX`
- `REJECT`
- `BLOCK_MISSING_EVIDENCE`
- `INVALID_TRACE`

---

## Safety invariants (all true)

- `default_runtime_unchanged`
- `no_token_commit`
- `no_generator_exposure`
- `verifier_is_not_executed`
- `trace_only`
- `deterministic_only`
- `no_external_effects`

---

## Failure modes (8)

All map to `BLOCK_DRY_RUN_DECISION`.

---

## Synthetic test matrix (6 cases)

Expected outcomes only — no runtime execution.

---

## Authorization

| Authorized | Blocked |
|---|---|
| Stage 3 dry-run **scaffold** design | Runtime execution |
| | L4 commit |
| | Default runtime modification |

**Allowed next phase:** `phase21l_l4_stage3_verifier_mediated_dry_run_scaffold`

---

## Run

```bash
python3 scripts/research/run_exp112_l4_stage3_verifier_mediated_dry_run_design.py
```

```bash
pytest tests/test_exp112_l4_stage3_verifier_mediated_dry_run_design.py -q
```
