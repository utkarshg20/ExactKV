# Phase 21J: L4 Verifier Runtime Instrumentation Design

**Status:** run `scripts/research/run_exp111_l4_verifier_runtime_instrumentation_design.py`.

> This is runtime instrumentation **architecture design**, not implementation.  
> ExactKV default generation remains unchanged.  
> ExactKVGenerator remains unchanged.  
> Production CLI remains unchanged.  
> Verifier evidence must be explicit.  
> Proposal evidence and verifier evidence must be separate fields.  
> Missing verifier evidence blocks decisions rather than fabricating one.  
> Runtime instrumentation is **not** implemented in this phase.  
> L4 runtime commit remains blocked.  
> Passing this design authorizes only Stage 3 dry-run design, not hook implementation or commit.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_111_L4_VERIFIER_RUNTIME_INSTRUMENTATION_DESIGN.md`](EXPERIMENT_111_L4_VERIFIER_RUNTIME_INSTRUMENTATION_DESIGN.md)

---

## 1. Purpose

Move from schema correctness (21F–21H) and adversarial robustness (21I) to a full **runtime integration design** that defines how L4 verifier evidence instrumentation would attach to the generation loop — without implementing any of it.

---

## 2. Relation to Phase 21I

Phase 21I stress-tested trace schema enforcement. Phase 21J defines where future instrumentation would live, what it would observe, and what safety gates must hold before any code is written.

---

## 3. Design boundary

- Conceptual hooks and instrumentation points only  
- No ExactKVGenerator changes  
- No runtime hook implementation  
- No L4 commit path  
- Per-token hook defined but explicitly **not implemented**  

---

## 4. Runtime hooks (conceptual)

| Hook ID | Attach location |
|---|---|
| `hook_pre_generation_session` | `ExactKVGenerator.generate()` entry |
| `hook_round_proposal_intercept` | Round-log proposal emission |
| `hook_per_token_generation` | Inner generation loop (observe only) |
| `hook_verifier_comparison` | After full-KV verifier output |
| `hook_trace_record_emit` | Round trace write path |
| `hook_post_generation_finalize` | `generate()` exit |
| `hook_rollback_decision` | After mismatch/block (concept only) |

All hooks: `implemented: false`.

---

## 5. Instrumentation points

| Point | Phase |
|---|---|
| `pre_generation` | Before first token |
| `per_token` | Each step (design only — not implemented) |
| `verifier_comparison` | After verifier output |
| `post_generation` | After generation completes |

---

## 6. Data flow (conceptual)

```
proposal_capture → trace_record_write → verifier_evidence_capture
    → comparison_decision → rollback_concept
```

None of these steps execute at runtime in Phase 21J.

---

## 7. Safety boundary matrix

| Boundary | Protected component |
|---|---|
| `default_runtime_unchanged` | ExactKVGenerator default path |
| `verifier_non_authoritative_until_l4` | Verifier evidence fields |
| `proposal_verifier_separation` | Proposal vs verifier trace fields |
| `no_direct_proposal_commit` | Token commit path |
| `trace_only_no_commit_wiring` | Dry-run evaluator |
| `opt_in_gate_required` | `--experimental-l4-verifier-mediated-draft` |

---

## 8. Integration points (design only)

- **ExactKVGenerator** — hook attach points at `generate()` boundary  
- **Round-log system** — proposal intercept from `exactkv_round_log_draft_tokens`  
- **Verifier evidence schema** — `l4_verifier_evidence_v1` trace records  
- **Trace-only dry-run** — offline evaluator feed; non-authoritative  

---

## 9. Failure modes

Eight failure modes defined with detection signals and required responses — all `blocks_commit: true`.

---

## 10. What happens if instrumentation is enabled incorrectly

Six incorrect-enablement scenarios document expected harm and required mitigations (e.g. hooks active without opt-in, proposal committed without verifier, dry-run wired to commit).

---

## 11. What this authorizes

**Phase 21K:** Stage 3 verifier-mediated dry-run design — complete. See [`PHASE_21K_L4_STAGE3_VERIFIER_MEDIATED_DRY_RUN_DESIGN.md`](PHASE_21K_L4_STAGE3_VERIFIER_MEDIATED_DRY_RUN_DESIGN.md).

**Next:** Phase 21L Stage 3 dry-run scaffold (`phase21l_l4_stage3_verifier_mediated_dry_run_scaffold`).

---

## 12. What this does not authorize

- Runtime hook implementation  
- L4 runtime commit integration  
- Default runtime modification  
- Verifier-in-loop execution  

---

## Run

```bash
python3 scripts/research/run_exp111_l4_verifier_runtime_instrumentation_design.py
```

```bash
pytest tests/test_exp111_l4_verifier_runtime_instrumentation_design.py -q
```

Report: `reports/experiment_111_l4_verifier_runtime_instrumentation_design.json` (gitignored).
