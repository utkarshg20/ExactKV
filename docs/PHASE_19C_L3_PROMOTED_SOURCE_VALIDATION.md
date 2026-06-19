# Phase 19C: L3 Promoted-Source Validation

**Status:** run `scripts/research/run_exp097_l3_promoted_source_validation.py --guarded-draft-shadow-no-commit`.

> This is L3 promoted-source validation, not L4 verifier-mediated compressed draft.  
> ExactKV default generation remains unchanged.  
> Round-log draft proposals are diagnostic only.  
> Round-log draft proposals cannot affect token commits.  
> Round-log draft proposals are not exposed to generator decisions.  
> Committed tokens may be used for comparison only, never as proposal sources.  
> Proposal coverage and prefix match rate are supplementary only and are not exactness guarantees.  
> Promoting a proposal source for L3 diagnostics does not authorize L4 commit integration.  
> Full verification remains the required source of truth before any future compressed draft acceptance.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_097_L3_PROMOTED_SOURCE_VALIDATION.md`](EXPERIMENT_097_L3_PROMOTED_SOURCE_VALIDATION.md)

---

## 1. Purpose

Validate `exactkv_round_log_draft_tokens` as the promoted L3 no-commit proposal source across an expanded panel. Answer: when used as the promoted diagnostic source, does it maintain coverage, parity, safety gates, and claim boundaries?

---

## 2. Relation to Phase 19B

Phase 19B compared round-log draft tokens with `decode_time_shadow_top1` and recommended `promote_round_log_draft_tokens_as_l3_source`. Phase 19C formalizes that promotion policy and validates the promoted source on a larger 96-cell panel (6 prompts vs 4).

---

## 3. Promoted source policy

| Field | Value |
|-------|-------|
| `promoted_source` | `exactkv_round_log_draft_tokens` |
| L3 diagnostics only | yes |
| L4 token commit | not authorized |
| Generator exposure | none |
| Accept/reject/commit | not used |

---

## 4. Demoted sources

`decode_time_shadow_top1` is demoted due to:

- Low coverage on Phase 19B comparison panel
- Zero prefix match rate on tested panel
- Source disagreement with round-log draft proposals

---

## 5. Validation panel

Default: 2 models × 6 prompts × 4 compressors × `max_new_tokens` {4, 8} = **96 cells**; CPU `float32`.

---

## 6. Source viability gates

| Gate | Criterion |
|------|-----------|
| `proposal_coverage_gate` | coverage ≥ 0.95 |
| `proposal_block_gate` | no unexplained blocked rounds |
| `proposal_provenance_gate` | round-log draft only; no committed/baseline/verifier sources |
| `proposal_isolation_gate` | proposals do not affect commits or generator |
| `generation_parity_gate` | baseline vs promoted-source token/text match |
| `exactkv_failure_gate` | `exactkv_failures == 0` on both paths |
| `claim_boundary_gate` | no performance/memory/serving/VeriCache claims |

Prefix match rate is diagnostic only and does not gate promotion as an exactness claim.

---

## 7. Coverage diagnostics

`proposal_coverage_rate`, `rounds_with_draft_tokens`, `rounds_missing_draft_tokens`, `total_proposed_tokens`.

---

## 8. Prefix-match diagnostics

`matching_committed_prefix`, `not_matching_committed_prefix`, `prefix_match_rate` — supplementary only.

---

## 9. Accepted/rejected token summaries

`accepted_token_count_summary`, `rejected_or_corrected_token_count_summary` from round-log acceptance metadata.

---

## 10. Breakdown summaries

By model, compressor, prompt, `max_new_tokens`, and round index.

---

## 11. Safety spec validation

Self-validates against Phase 18A `validate_integration_proposal()` as `L3_GUARDED_DRAFT_SHADOW_NO_COMMIT`.

---

## 12. Decision recommendation

| Value | When |
|-------|------|
| `l3_source_promoted` | Required viability gates pass on full panel |
| `l3_source_needs_more_validation` | Blocked cells/models without safety failure |
| `l3_source_not_promoted` | Safety or provenance failure |

---

## 13. What this proves

- Promoted round-log draft source maintains L3 no-commit safety on expanded panel.
- Source viability gates can be evaluated systematically.
- Demotion policy for shadow top-1 is documented with reasons.

---

## 14. What this does not prove

- L4 verifier-mediated compressed draft acceptance.
- That prefix match implies exactness or production viability.
- Serving, throughput, latency, or memory improvements.

---

## 15. Allowed claims

- L3 promoted round-log source validation with viability gates.
- Panel-scoped coverage and prefix-match diagnostics.
- Formal demotion of `decode_time_shadow_top1` for L3 diagnostics.

---

## 16. Forbidden claims

- Speedup, throughput, latency, tokens-per-second, `runtime_seconds`.
- Active GPU memory or production memory savings.
- Production serving or VeriCache reproduction.
- Token-commit integration.

---

## 17. Recommended next phase

**Phase 20A:** pre-L4 safety gate review — design review before any verifier-mediated compressed draft work.

---

## Run

```bash
python3 scripts/research/run_exp097_l3_promoted_source_validation.py \
  --guarded-draft-shadow-no-commit
```

Report: `reports/experiment_097_l3_promoted_source_validation.json` (gitignored).

```bash
pytest tests/test_exp097_l3_promoted_source_validation.py -q
```

**Run summary (CPU, float32, 2 models × 6 prompts × 4 compressors × 2 max_new_tokens):**

| Metric | Value |
|--------|-------|
| Status | `validation_complete` |
| Safety spec validation | pass |
| Models loaded / blocked | 2 / 0 |
| Generation cells | 96 / 96 successful |
| Proposal coverage | 1.0 |
| Prefix match rate | ~0.68 |
| Rounds with / missing draft tokens | 167 / 0 |
| All required viability gates | pass |
| Decision | `l3_source_promoted` |
| Baseline / promoted-source parity | 96/96 |
| Safety gates | 96/96 OK |
| Proposals used for token commit | false |
| Proposals exposed to generator | false |
