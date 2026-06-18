# Phase 18C: L3 Guarded Draft-Shadow Panel Validation

**Status:** run `scripts/research/run_exp092_guarded_draft_shadow_panel_validation.py --guarded-draft-shadow-no-commit`.

> This is L3 guarded draft-shadow no-commit panel validation, not L4 verifier-mediated compressed draft.  
> ExactKV default generation remains unchanged.  
> Draft-shadow proposals are diagnostic only.  
> Draft-shadow proposals cannot affect token commits.  
> Draft-shadow proposals are not exposed to generator decisions.  
> Blocked proposals are reported, not fabricated.  
> Proposal match rate is supplementary only and is not an exactness guarantee.  
> Full verification remains the required source of truth before any future compressed draft acceptance.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_092_GUARDED_DRAFT_SHADOW_PANEL_VALIDATION.md`](EXPERIMENT_092_GUARDED_DRAFT_SHADOW_PANEL_VALIDATION.md)

---

## 1. Purpose

Expand the L3 guarded draft-shadow no-commit scaffold into a broader panel and diagnose proposal coverage. Answer: across a broader panel, how often can the scaffold produce draft-shadow proposals, why do proposals block, and do all no-commit safety gates remain intact?

---

## 2. Relation to Phase 18B

Phase 18B introduced the L3 scaffold (2 prompts × 2 compressors, 4 cells). Phase 18C expands to 32 cells and adds proposal coverage and block-reason diagnostics.

---

## 3. Panel dimensions

| Dimension | Default |
|-----------|---------|
| Model | `Qwen/Qwen2.5-0.5B` |
| Prompts | 4 deterministic |
| Compressors | noop, int8, int4_sim, k8_v4_sim |
| max_new_tokens | 4, 8 |
| Cells | 32 |

---

## 4. Proposal source behavior

| Source | Use |
|--------|-----|
| `decode_time_shadow_top1` | Default for real-model panel |
| `blocked_no_provider` | Explicit blocked provider |
| `synthetic_shadow_provider` | Tests only — not used for real panel |

Committed tokens are never used as proposals.

---

## 5. Proposal coverage diagnostics

Per cell: `total_proposals`, `successful_proposals`, `blocked_proposals`, `proposal_block_reasons`, `first_successful_proposal_round`, `first_blocked_proposal_round`, `proposal_match_summary`.

Top-level: `proposal_coverage_rate`, `proposal_block_reason_summary`.

---

## 6. Proposal block reasons

When `other_top1_token_id` or shadow output is unavailable, proposals are marked blocked with exact reasons recorded — no fabricated token IDs.

---

## 7. Safety spec validation

Self-validates against Phase 18A `validate_integration_proposal()` as L3 no-commit. Failure marks report `failed`.

---

## 8. Generation parity

Baseline vs draft-shadow token and text parity required per cell. Mismatch fails cell and report.

---

## 9. Proposal match diagnostics

`proposal_match_rate` and match counts are supplementary only — not exactness guarantees.

---

## 10. What this proves

Across the expanded panel, generation parity and no-commit safety gates hold while proposal coverage and block reasons are measured transparently.

---

## 11. What this does not prove

L4 verifier-mediated acceptance, general exact-generation preservation, speed/memory/serving claims, or VeriCache reproduction.

---

## 12. Allowed claims

Panel-scoped L3 diagnostic claims: proposal coverage measured, block reasons recorded, parity in tested cells.

---

## 13. Forbidden claims

Draft-shadow used for commit, L4 implemented, speed/throughput/latency/memory/serving/VeriCache claims.

---

## 14. Recommended next phase

**`phase18d_shadow_top1_extraction_hardening`** — improve safe top-1 extraction to reduce blocked proposals without L4 integration.

---

## Run

```bash
python3 scripts/research/run_exp092_guarded_draft_shadow_panel_validation.py \
  --guarded-draft-shadow-no-commit
```

Report: `reports/experiment_092_guarded_draft_shadow_panel_validation.json` (gitignored).

**Run summary (CPU, float32, 4 prompts × 4 compressors × 2 max_new_tokens, decode_time_shadow_top1):**

| Metric | Value |
|--------|-------|
| Status | `panel_complete` |
| Safety spec validation | pass |
| Total cells | 32 |
| Baseline / draft-shadow successful | 32/32 |
| Token / text parity | 32/32 |
| Total proposals | 152 |
| Successful / blocked proposals | 53 / 99 |
| Proposal coverage rate | ~0.35 |
| Block reason | `no safe top1 extraction from shadow output` (99) |
| Safety gates | 32/32 OK |
| Proposals used for token commit | false |
| Proposals exposed to generator | false |

**Recommended next phase:** Phase 18D shadow top-1 extraction hardening — see [`PHASE_18D_SHADOW_TOP1_EXTRACTION_HARDENING.md`](PHASE_18D_SHADOW_TOP1_EXTRACTION_HARDENING.md).
