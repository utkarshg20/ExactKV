# Phase 18B: L3 Guarded Draft-Shadow No-Commit Scaffold

**Status:** run `scripts/research/run_exp091_guarded_draft_shadow_no_commit_scaffold.py --guarded-draft-shadow-no-commit`.

> This is an L3 guarded draft-shadow no-commit scaffold, not L4 verifier-mediated compressed draft.  
> ExactKV default generation remains unchanged.  
> Draft-shadow proposals are diagnostic only.  
> Draft-shadow proposals cannot affect token commits.  
> Draft-shadow proposals are not exposed to generator decisions.  
> Full verification remains the required source of truth before any future compressed draft acceptance.  
> Top-k or proposal match rate is supplementary only and is not an exactness guarantee.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_091_GUARDED_DRAFT_SHADOW_NO_COMMIT_SCAFFOLD.md`](EXPERIMENT_091_GUARDED_DRAFT_SHADOW_NO_COMMIT_SCAFFOLD.md)

---

## 1. Purpose

Implement an L3 guarded draft-shadow no-commit scaffold: a proposal interface and diagnostic observer path where draft tokens can be proposed and compared, but never committed. Answer whether ExactKV can represent and validate draft-shadow proposals while proving they cannot affect generated tokens.

---

## 2. Relation to Phase 18A

Phase 18A defined invariants, gates, and a proposal validator. Phase 18B implements the L3 scaffold that must pass that validator before any deeper integration.

---

## 3. What L3 means

Draft/shadow compressed-attention diagnostics may run during generation, but proposals cannot commit tokens or influence the generator.

---

## 4. What no-commit means

Every proposal records `proposal_used_for_token_commit=false`, `proposal_exposed_to_generator=false`, and related safety gates. Observer return values are ignored; exceptions cannot change generation.

---

## 5. Proposal dataclasses

Immutable dataclasses in `exactkv/safety/guarded_draft_shadow.py`:

- `GuardedDraftShadowProposal`
- `GuardedDraftShadowDecision`
- `GuardedDraftShadowCell`
- `GuardedDraftShadowSafetyResult`

---

## 6. Proposal source behavior

| Source | Behavior |
|--------|----------|
| `synthetic_shadow_provider` | Test provider; mirrors committed tokens diagnostically |
| `decode_time_shadow_top1` | Extracts `other_top1_token_id` from post-hoc shadow when available |
| `blocked_no_provider` | Records blocked status; does not fake proposals |

---

## 7. Safety spec validation

Report self-validates against Phase 18A `validate_integration_proposal()` as L3 no-commit. Failure marks report `failed`.

---

## 8. Baseline vs draft-shadow parity

Per cell: baseline generation → guarded draft-shadow path (existing observer, no generator changes) → token/text parity comparison.

---

## 9. Proposal match diagnostics

`proposal_match_summary` records match counts and `proposal_match_rate` for diagnostics only — not exactness.

---

## 10. What this proves

ExactKV can represent draft-shadow proposals in the diagnostic path while maintaining baseline-vs-draft-shadow generation parity and no-commit safety gates in tested cells.

---

## 11. What this does not prove

L4 verifier-mediated acceptance, general exact-generation preservation, speed/memory/serving claims, or VeriCache reproduction.

---

## 12. Allowed claims

Phase 16 claim freeze plus panel-scoped L3 scaffold diagnostic claims only.

---

## 13. Forbidden claims

Speed, throughput, latency, memory savings, serving, VeriCache reproduction, draft-shadow used for token commit, L4 implemented.

---

## 14. Recommended next phase

**Phase 18C (complete):** guarded draft-shadow panel validation — see [`PHASE_18C_GUARDED_DRAFT_SHADOW_PANEL_VALIDATION.md`](PHASE_18C_GUARDED_DRAFT_SHADOW_PANEL_VALIDATION.md).

**Phase 18D (proposed):** shadow top-1 extraction hardening.

---

## Run

```bash
python3 scripts/research/run_exp091_guarded_draft_shadow_no_commit_scaffold.py \
  --guarded-draft-shadow-no-commit
```

Report: `reports/experiment_091_guarded_draft_shadow_no_commit_scaffold.json` (gitignored).

**Run summary (CPU, float32, 2 prompts × 2 compressors, decode_time_shadow_top1, max_new_tokens=4):**

| Metric | Value |
|--------|-------|
| Status | `scaffold_complete` |
| Safety spec validation | pass |
| Baseline generation successful | 4/4 |
| Draft-shadow generation successful | 4/4 |
| Baseline-vs-draft-shadow token match | 4/4 |
| Baseline-vs-draft-shadow text match | 4/4 |
| Total proposals | 16 |
| Successful / blocked proposals | 4 / 12 |
| Safety gates | 4/4 OK |
| Proposals used for token commit | false |
| Proposals exposed to generator | false |

---
