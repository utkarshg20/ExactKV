# Phase 18E: L3 Shadow Proposal Provenance Audit

**Status:** run `scripts/research/run_exp094_shadow_proposal_provenance_audit.py --guarded-draft-shadow-no-commit`.

> This is an L3 shadow proposal provenance audit, not L4 verifier-mediated compressed draft.  
> ExactKV default generation remains unchanged.  
> Proposal tokens are diagnostic only.  
> Proposal tokens cannot affect token commits.  
> Committed tokens may be used for comparison only, never as proposal sources.  
> Blocked proposals are reported, not fabricated.  
> Proposal match rate is supplementary only and is not an exactness guarantee.  
> If decode_time_shadow_top1 has low coverage and zero match rate, it should be replaced rather than promoted.  
> Full verification remains the required source of truth before any future compressed draft acceptance.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_094_SHADOW_PROPOSAL_PROVENANCE_AUDIT.md`](EXPERIMENT_094_SHADOW_PROPOSAL_PROVENANCE_AUDIT.md)

---

## 1. Purpose

Audit L3 shadow proposal provenance and decide whether the current `decode_time_shadow_top1` proposal source is viable for future work. Answer: are failed proposal matches caused by true shadow drift, missing diagnostic fields, alignment issues, or a bad proposal source definition?

---

## 2. Relation to Phase 18D

Phase 18D hardened safe top-1 extraction and compared coverage to Exp092 (delta 0.0; 53/152 successful; zero match rate). Phase 18E classifies each audited round, aggregates diagnostics by panel dimension, and applies a decision gate.

---

## 3. Why provenance audit is needed

Low coverage alone does not explain whether `decode_time_shadow_top1` should continue. The audit separates missing diagnostic fields, shadow drift (mismatch), alignment issues, and unsafe-source rejection to inform a replace/continue/stop recommendation.

---

## 4. Taxonomy

| Category | Meaning |
|----------|---------|
| `safe_shadow_top1_available` | Safe shadow top-1 extracted |
| `missing_shadow_top1_field` | Blocked due to absent diagnostic top-1 field |
| `shadow_top1_mismatches_committed` | Extracted shadow top-1 ≠ committed (comparison only) |
| `shadow_top1_matches_committed` | Extracted shadow top-1 = committed (supplementary) |
| `round_alignment_unknown` | Cannot verify post-hoc cell alignment |
| `round_alignment_mismatch` | Post-hoc cell count/index mismatch |
| `non_comparable_round` | No valid comparison context |
| `blocked_no_safe_extraction` | Blocked without safe extraction |
| `unsafe_source_rejected` | Forbidden source rejected |

---

## 5. Proposal-source vs committed-token comparison separation

Proposal tokens are extracted only from allowed shadow diagnostic fields. `committed_token_id_for_comparison` is recorded for supplementary match diagnostics only and is never used as a proposal source.

---

## 6. Panel dimensions

Same as Exp092/Exp093: Qwen/Qwen2.5-0.5B, CPU float32, 4 prompts, 4 compressors, max_new_tokens 4/8, 32 cells, `decode_time_shadow_top1`.

---

## 7. Match diagnostics

- `match_rate_successful_extractions` — matches among safe extractions only
- `match_rate_total_rounds` — matches among all audited rounds
- `matched_committed_count` / `mismatched_committed_count`

Match rate is supplementary only; not an exactness guarantee.

---

## 8. Block diagnostics

- `missing_top1_field_count` — rounds blocked for absent shadow top-1 field
- `blocked_count` — rounds with `blocked_no_safe_extraction`
- `unsafe_rejected_count` — forbidden source rejections

---

## 9. Decision gate

Allowed recommendations: `continue_with_decode_time_shadow_top1`, `replace_proposal_source`, `stop_l3_top1_path`, `needs_more_evidence`.

---

## 10. Decision recommendation

Given Phase 18D evidence (low coverage, zero match rate, no unsafe dependency), the expected recommendation is **`replace_proposal_source`**: `decode_time_shadow_top1` should be replaced rather than promoted.

---

## 11. What this proves

- Per-round provenance categories can be assigned with committed-token separation.
- Match and block diagnostics aggregate across panel dimensions.
- A decision gate can recommend replacing a non-viable L3 proposal source without L4 integration.

---

## 12. What this does not prove

- L4 verifier-mediated compressed draft viability.
- Production serving, throughput, or memory savings.
- That proposal match rate implies exactness.

---

## 13. Allowed claims

- L3 provenance audit with taxonomy and decision gate.
- Panel-scoped viability assessment of `decode_time_shadow_top1`.
- Replace recommendation when coverage is low and match rate is zero.

---

## 14. Forbidden claims

- Speedup, throughput, latency, tokens-per-second.
- Active GPU memory or production memory savings.
- Production serving or VeriCache reproduction.
- Token-commit integration or generator exposure.

---

## 15. Recommended next phase

**Phase 19A (complete):** round-log draft proposal source — see [`PHASE_19A_ROUND_LOG_DRAFT_PROPOSAL_SOURCE.md`](PHASE_19A_ROUND_LOG_DRAFT_PROPOSAL_SOURCE.md).

---

## Run

```bash
python3 scripts/research/run_exp094_shadow_proposal_provenance_audit.py \
  --guarded-draft-shadow-no-commit
```

Report: `reports/experiment_094_shadow_proposal_provenance_audit.json` (gitignored).

```bash
pytest tests/test_exp094_shadow_proposal_provenance_audit.py -q
```

**Run summary (CPU, float32, 4 prompts × 4 compressors × 2 max_new_tokens, decode_time_shadow_top1):**

| Metric | Value |
|--------|-------|
| Status | `audit_complete` |
| Safety spec validation | pass |
| Total audited rounds | 152 |
| Safe extractions | 53 |
| Missing top-1 field | 99 |
| Unsafe rejected | 0 |
| Matched / mismatched committed | 0 / 53 |
| Match rate (successful extractions) | 0.0 |
| Decision | `replace_proposal_source` |
| Baseline / draft-shadow parity | 32/32 |
| Safety gates | 32/32 OK |
| Proposals used for token commit | false |
| Proposals exposed to generator | false |
