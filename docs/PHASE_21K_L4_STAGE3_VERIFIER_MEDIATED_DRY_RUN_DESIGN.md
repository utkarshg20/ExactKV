# Phase 21K: L4 Stage 3 Verifier-Mediated Dry-Run Design

**Status:** run `scripts/research/run_exp112_l4_stage3_verifier_mediated_dry_run_design.py`.

> This is Stage 3 dry-run **architecture design**, not implementation or execution.  
> ExactKV default generation remains unchanged.  
> ExactKVGenerator remains unchanged.  
> Verifier is **not executed** — only trace field reconciliation is designed.  
> Prefix acceptance and rollback are simulation models only.  
> No token commit. No generator exposure.  
> Runtime execution remains blocked.  
> L4 runtime commit remains blocked.  
> Passing this design authorizes only Stage 3 scaffold, not runtime execution or commit.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.

Companion: [`EXPERIMENT_112_L4_STAGE3_VERIFIER_MEDIATED_DRY_RUN_DESIGN.md`](EXPERIMENT_112_L4_STAGE3_VERIFIER_MEDIATED_DRY_RUN_DESIGN.md)

---

## 1. Purpose

Define how Stage 3 verifier-mediated dry-run **would** consume L3 proposal sources and L4 verifier evidence trace schema v1 to produce a deterministic decision graph and prefix-level accept/reject simulation — without runtime implementation.

---

## 2. Relation to Phase 21J

Phase 21J defined where runtime instrumentation would attach. Phase 21K defines what the Stage 3 dry-run execution model would compute from ingested traces — still design-only.

---

## 3. Inputs

- L3 proposal sources (`exactkv_round_log_draft_tokens`)
- L4 verifier evidence trace schema v1 fields

---

## 4. Outputs

- Deterministic decision graph (nodes = alignment points, edges = match/mismatch)
- `L4Stage3DryRunResult` trace-only record
- Prefix match length simulation
- Terminal state classification

---

## 5. Decision graph

| Terminal state | Meaning |
|---|---|
| `ACCEPT_PREFIX` | Full or partial prefix match accepted (simulated) |
| `REJECT` | Mismatch detected |
| `BLOCK_MISSING_EVIDENCE` | Verifier evidence missing |
| `INVALID_TRACE` | Schema/alias/corruption |

---

## 6. Evidence reconciliation

1. Validate schema version  
2. Load proposal from explicit source  
3. Load verifier evidence fields  
4. Detect aliasing  
5. Walk tokens building graph  
6. Emit terminal state (no generation influence)  

---

## 7. Safety invariants

All seven invariants required `true`: default runtime unchanged, no token commit, no generator exposure, verifier not executed, trace-only, deterministic-only, no external effects.

---

## 8. Failure taxonomy

Eight failure modes; all `required_response = BLOCK_DRY_RUN_DECISION`.

---

## 9. Synthetic test matrix

| Test | Expected terminal |
|---|---|
| full match | `ACCEPT_PREFIX` |
| partial mismatch | `REJECT` |
| missing verifier | `BLOCK_MISSING_EVIDENCE` |
| corrupted proposal | `INVALID_TRACE` |
| adversarial aliasing | `INVALID_TRACE` |
| conflicting sources | `BLOCK_DRY_RUN_DECISION` |

---

## 10. What this authorizes

**Phase 21L:** Stage 3 verifier-mediated dry-run scaffold — complete. See [`PHASE_21L_L4_STAGE3_VERIFIER_MEDIATED_DRY_RUN_SCAFFOLD.md`](PHASE_21L_L4_STAGE3_VERIFIER_MEDIATED_DRY_RUN_SCAFFOLD.md).

**Next:** Phase 21M panel validation (`phase21m_l4_stage3_verifier_mediated_dry_run_panel_validation`).

---

## 11. What this does not authorize

- Runtime dry-run execution  
- Model/inference execution  
- L4 runtime commit  
- ExactKVGenerator modification  

---

## Run

```bash
python3 scripts/research/run_exp112_l4_stage3_verifier_mediated_dry_run_design.py
```

```bash
pytest tests/test_exp112_l4_stage3_verifier_mediated_dry_run_design.py -q
```

Report: `reports/experiment_112_l4_stage3_verifier_mediated_dry_run_design.json` (gitignored).
