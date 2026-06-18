# Phase 19A: L3 Round-Log Draft Proposal Source

**Status:** run `scripts/research/run_exp095_round_log_draft_proposal_source.py --guarded-draft-shadow-no-commit`.

> This is an L3 round-log draft proposal source scaffold, not L4 verifier-mediated compressed draft.  
> ExactKV default generation remains unchanged.  
> Round-log draft proposals are diagnostic only.  
> Round-log draft proposals cannot affect token commits.  
> Round-log draft proposals are not exposed to generator decisions.  
> Committed tokens may be used for comparison only, never as proposal sources.  
> Proposal coverage and match rate are supplementary only and are not exactness guarantees.  
> Full verification remains the required source of truth before any future compressed draft acceptance.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_095_ROUND_LOG_DRAFT_PROPOSAL_SOURCE.md`](EXPERIMENT_095_ROUND_LOG_DRAFT_PROPOSAL_SOURCE.md)

---

## 1. Purpose

Introduce and validate `exactkv_round_log_draft_tokens` as a safer alternative L3 proposal source. Answer: can existing ExactKV round-log draft tokens serve as a better diagnostic L3 proposal source while preserving no-commit safety?

---

## 2. Relation to Phase 18E

Phase 18E audited `decode_time_shadow_top1` and recommended `replace_proposal_source` due to low coverage (~35%) and zero match rate. Phase 19A implements the replacement scaffold using draft token IDs already recorded in ExactKV round logs / live snapshots.

---

## 3. Why decode_time_shadow_top1 was rejected

Low proposal coverage, missing shadow top-1 diagnostic fields on many rounds, zero supplementary match rate among successful extractions, and shadow top-1 mismatches when extraction did succeed.

---

## 4. Why round-log draft tokens are a safer alternative proposal source

Draft token IDs are explicitly recorded during ExactKV verification rounds in `VerificationTrace.draft_tokens` and `LiveRoundSnapshot.draft_token_ids`. They are not inferred from committed tokens, generated output, or retokenized text.

---

## 5. Allowed extraction sources

| Source | Field path |
|--------|------------|
| ExactKV result traces | `exactkv_traces[{round_index}].draft_tokens` |
| Live round snapshots | `live_snapshots[{round_index}].draft_token_ids` |

---

## 6. Forbidden extraction sources

Committed token IDs, accepted tokens as proposals, verifier tokens, full-KV output tokens, baseline generated tokens, retokenized generated text, guessed token IDs.

---

## 7. Proposal-source vs comparison-field separation

`committed_token_ids_for_comparison`, `accepted_token_count_for_comparison`, and `rejected_or_corrected_token_count_for_comparison` are diagnostic comparison fields only. They are derived from round commit deltas or acceptance metadata and never used as proposal sources.

---

## 8. Coverage and match diagnostics

- `proposal_coverage_rate` — rounds with draft tokens / total rounds with logs
- `proposal_prefix_match_rate` — among rounds with prefix comparison available
- `match_rate_total_rounds` — prefix matches / total rounds (supplementary only)

---

## 9. Comparison against Exp094

When `reports/experiment_094_shadow_proposal_provenance_audit.json` exists, Exp095 reports `previous_source_comparison` with coverage and match-rate deltas vs `decode_time_shadow_top1`. Missing Exp094 report does not fail the run.

---

## 10. Safety spec validation

Self-validates against Phase 18A `validate_integration_proposal()` as `L3_GUARDED_DRAFT_SHADOW_NO_COMMIT`.

---

## 11. What this proves

- Round-log draft tokens can be extracted with explicit provenance.
- L3 no-commit safety gates remain intact on the same 32-cell panel.
- An alternative proposal source scaffold exists without L4 integration.

---

## 12. What this does not prove

- L4 verifier-mediated compressed draft acceptance.
- That prefix match rate implies exactness or production viability.
- Serving, throughput, or memory improvements.

---

## 13. Allowed claims

- L3 round-log draft proposal source scaffold with provenance.
- Panel-scoped coverage and prefix-match diagnostics.
- Comparison vs Exp094 `decode_time_shadow_top1` when that report exists.

---

## 14. Forbidden claims

- Speedup, throughput, latency, tokens-per-second.
- Active GPU memory or production memory savings.
- Production serving or VeriCache reproduction.
- Token-commit integration.

---

## 15. Recommended next phase

**Phase 19B:** round-log proposal panel validation — expanded comparison of round-log draft proposals vs shadow top-1 across panel dimensions without L4 commit integration.

---

## Run

```bash
python3 scripts/research/run_exp095_round_log_draft_proposal_source.py \
  --guarded-draft-shadow-no-commit
```

Report: `reports/experiment_095_round_log_draft_proposal_source.json` (gitignored).

```bash
pytest tests/test_exp095_round_log_draft_proposal_source.py -q
```

**Run summary (CPU, float32, 4 prompts × 4 compressors × 2 max_new_tokens, exactkv_round_log_draft_tokens):**

| Metric | Value |
|--------|-------|
| Status | `scaffold_complete` |
| Safety spec validation | pass |
| Total rounds with logs | 53 |
| Rounds with / missing draft tokens | 53 / 0 |
| Proposal coverage rate | 1.0 |
| Prefix match rate | ~0.72 |
| vs Exp094 coverage delta | ~+0.65 |
| vs Exp094 match-rate delta | ~+0.72 |
| Baseline / draft-shadow parity | 32/32 |
| Safety gates | 32/32 OK |
| Proposals used for token commit | false |
| Proposals exposed to generator | false |
