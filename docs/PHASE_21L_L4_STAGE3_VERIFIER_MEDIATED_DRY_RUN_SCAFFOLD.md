# Phase 21L: L4 Stage 3 Verifier-Mediated Dry-Run Scaffold

**Status:** run `scripts/research/run_exp113_l4_stage3_verifier_mediated_dry_run_scaffold.py`.

> This is Stage 3 dry-run **scaffold execution** on synthetic traces only.  
> ExactKV default generation remains unchanged.  
> ExactKVGenerator remains unchanged.  
> Verifier is **not executed** — trace field mapping only.  
> No model inference. No GPU. No token commit.  
> L4 runtime commit remains blocked.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.

Companion: [`EXPERIMENT_113_L4_STAGE3_VERIFIER_MEDIATED_DRY_RUN_SCAFFOLD.md`](EXPERIMENT_113_L4_STAGE3_VERIFIER_MEDIATED_DRY_RUN_SCAFFOLD.md)

---

## 1. Purpose

Implement the trace-only execution engine for Stage 3: consume L3 proposals and L4 verifier evidence schema v1 traces, execute the decision graph in pure logic, and emit `L4Stage3DryRunResult` records.

---

## 2. Relation to Phase 21K

Phase 21K defined the execution model design. Phase 21L implements the scaffold evaluator — still trace-only, still no model execution.

---

## 3. Execution layers

| Layer | Function |
|---|---|
| Proposal ingestion | Read `proposal_token_ids` from round-log source |
| Verifier mapping | Map schema v1 verifier fields |
| Decision graph | Prefix walk → terminal state |
| Rollback simulation | Conceptual baseline restore on non-accept |
| Report builder | Aggregated panel JSON |

---

## 4. Outputs

- `L4Stage3DryRunResult` per case  
- `L4Stage3DecisionGraphTrace` per decision  
- `L4Stage3RollbackSimulation` per case  
- Panel report with classification summary  

---

## 5. Scaffold panel cases

| Case | Expected |
|---|---|
| `scaffold_full_match` | `ACCEPT_PREFIX` |
| `scaffold_partial_mismatch` | `REJECT` |
| `scaffold_missing_verifier` | `BLOCK_MISSING_EVIDENCE` |
| `scaffold_corrupted_trace` | `INVALID_TRACE` |
| `scaffold_adversarial_aliasing` | `INVALID_TRACE` |

---

## 6. What this authorizes

- Phase 21M panel validation: `phase21m_l4_stage3_verifier_mediated_dry_run_panel_validation`

---

## 7. What this does not authorize

- Runtime model execution  
- L4 runtime commit  
- ExactKVGenerator modification  
- Verifier-in-loop execution  

---

## Run

```bash
python3 scripts/research/run_exp113_l4_stage3_verifier_mediated_dry_run_scaffold.py
```

```bash
pytest tests/test_exp113_l4_stage3_verifier_mediated_dry_run_scaffold.py -q
```

Report: `reports/experiment_113_l4_stage3_verifier_mediated_dry_run_scaffold.json` (gitignored).
