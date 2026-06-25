# Experiment 111: L4 Verifier Runtime Instrumentation Design

**Phase:** 21J  
**Script:** `scripts/research/run_exp111_l4_verifier_runtime_instrumentation_design.py`  
**Module:** `exactkv/safety/l4_verifier_runtime_instrumentation_design.py`  
**Report:** `reports/experiment_111_l4_verifier_runtime_instrumentation_design.json` (gitignored)

> **Design only.** No runtime hooks implemented.  
> ExactKVGenerator unchanged. Default runtime unchanged.  
> No L4 commit path. No verifier-in-loop execution.  
> No model experiments. No performance/memory/serving claims.

---

## Purpose

Define the full runtime instrumentation architecture for L4 verifier evidence tracking — how L4 **would** integrate if ever implemented — without writing execution code.

---

## Design scope

1. **Runtime hooks (conceptual)** — attach points in the generation loop  
2. **Instrumentation points** — pre-generation, per-token (not implemented), post-generation, verifier comparison  
3. **Data flow** — proposal → trace → verifier → decision → rollback (concept only)  
4. **Safety boundaries** — what cannot change in default runtime  
5. **Integration points** — ExactKVGenerator, round-log, verifier schema, trace-only dry-run  

---

## Outputs

- Text-based architecture diagram  
- Hook definitions (all `implemented: false`)  
- Data flow description  
- Safety boundary matrix  
- Failure mode analysis (8 modes)  
- Incorrect-enablement scenarios (6 cases)  

---

## Authorization

| Authorized | Blocked |
|---|---|
| Stage 3 verifier-mediated dry-run **design** | Runtime hook **implementation** |
| Architecture documentation | L4 runtime commit |
| | Default runtime modification |
| | Verifier-in-loop execution |

**Allowed next phase:** `phase21k_l4_stage3_verifier_mediated_dry_run_design`

---

## Run

```bash
python3 scripts/research/run_exp111_l4_verifier_runtime_instrumentation_design.py
```

```bash
pytest tests/test_exp111_l4_verifier_runtime_instrumentation_design.py -q
```
