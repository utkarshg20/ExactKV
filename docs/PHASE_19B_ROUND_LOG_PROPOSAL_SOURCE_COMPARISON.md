# Phase 19B: L3 Round-Log Proposal Source Comparison

**Status:** run `scripts/research/run_exp096_round_log_proposal_source_comparison_panel.py --guarded-draft-shadow-no-commit`.

> This is L3 proposal source comparison, not L4 verifier-mediated compressed draft.  
> ExactKV default generation remains unchanged.  
> Proposal tokens are diagnostic only.  
> Proposal tokens cannot affect token commits.  
> Proposal tokens are not exposed to generator decisions.  
> Committed tokens may be used for comparison only, never as proposal sources.  
> Proposal coverage and match rate are supplementary only and are not exactness guarantees.  
> Promoting a proposal source for L3 does not authorize L4 commit integration.  
> Full verification remains the required source of truth before any future compressed draft acceptance.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_096_ROUND_LOG_PROPOSAL_SOURCE_COMPARISON_PANEL.md`](EXPERIMENT_096_ROUND_LOG_PROPOSAL_SOURCE_COMPARISON_PANEL.md)

---

## 1. Purpose

Validate and compare L3 proposal sources side-by-side across a broader no-commit panel. Answer: is `exactkv_round_log_draft_tokens` consistently the stronger L3 proposal source compared with `decode_time_shadow_top1`, and what diagnostics explain differences by model, prompt, compressor, round, and `max_new_tokens`?

---

## 2. Relation to Phase 19A

Phase 19A introduced `exactkv_round_log_draft_tokens` on a single-model 32-cell panel with 100% coverage and ~72% prefix match vs Exp094 shadow top-1 (~35% coverage, 0% match). Phase 19B runs one generation per cell and evaluates both sources on matched round boundaries without changing generation or commit behavior.

---

## 3. Proposal sources compared

| Source | Role |
|--------|------|
| `exactkv_round_log_draft_tokens` | Draft token IDs from ExactKV round logs / live snapshots |
| `decode_time_shadow_top1` | Post-hoc shadow top-1 extraction when safely available |
| `synthetic_shadow_provider` | Test-only optional source |

Committed tokens, baseline tokens, verifier tokens, and retokenized generated text are never proposal sources.

---

## 4. Panel dimensions

Default real-model panel:

- Models: `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-0.5B-Instruct`
- Device: `cpu`; dtype: `float32`
- Prompts: 4 deterministic prompts
- Compressors: `noop`, `int8`, `int4_sim`, `k8_v4_sim`
- `max_new_tokens`: `4`, `8`
- Proposal sources: `exactkv_round_log_draft_tokens`, `decode_time_shadow_top1`

Blocked models and blocked proposal extractions are recorded; results are not fabricated.

---

## 5. Side-by-side round comparison

For each generation cell, both sources are evaluated on the same ExactKV round indexes where possible. Per-round records include availability status, proposed token IDs, agreement flag, and comparison-only committed token IDs.

---

## 6. Coverage diagnostics

Per-source aggregates: `total_possible_rounds`, `successful_proposals`, `blocked_proposals`, `proposal_coverage_rate`, `block_reason_summary`, `source_field_summary`.

---

## 7. Match diagnostics

Per-source prefix / committed-token match counts and `prefix_match_rate`. Side-by-side summary includes rounds where both, only round-log, only shadow, or neither source is available, plus source agreement counts and `match_rate_delta`.

---

## 8. Safety spec validation

Self-validates against Phase 18A `validate_integration_proposal()` as `L3_GUARDED_DRAFT_SHADOW_NO_COMMIT` with `opt_in_only`, `modifies_default_runtime: false`, and no performance/memory/serving claims.

---

## 9. Decision recommendation

| Value | When |
|-------|------|
| `promote_round_log_draft_tokens_as_l3_source` | Round-log materially higher coverage and match with safety gates OK |
| `keep_comparing_sources` | Mixed performance across dimensions |
| `replace_both_sources` | Both sources low coverage or unsafe extraction |
| `insufficient_evidence` | Too many blocked cells or safety failures |

---

## 10. What this proves

- Two L3 proposal sources can be compared on the same generation cells without commit integration.
- Round-log draft coverage and prefix-match diagnostics can be contrasted directly with shadow top-1.
- L3 no-commit safety gates and baseline/draft-shadow parity can hold across a multi-model panel.

---

## 11. What this does not prove

- L4 verifier-mediated compressed draft acceptance.
- That higher proposal match rate implies exactness or production viability.
- Serving, throughput, latency, or memory improvements.

---

## 12. Allowed claims

- L3 side-by-side proposal source comparison with provenance.
- Panel-scoped coverage, prefix-match, and source-agreement diagnostics.
- Decision recommendation for L3 source promotion (diagnostic only).

---

## 13. Forbidden claims

- Speedup, throughput, latency, tokens-per-second, `runtime_seconds`.
- Active GPU memory or production memory savings.
- Production serving or VeriCache reproduction.
- Token-commit integration or generator exposure of proposals.

---

## 14. Recommended next phase

**Phase 19C:** L3 promoted-source validation for `exactkv_round_log_draft_tokens`. See [`PHASE_19C_L3_PROMOTED_SOURCE_VALIDATION.md`](PHASE_19C_L3_PROMOTED_SOURCE_VALIDATION.md).

---

## Run

```bash
python3 scripts/research/run_exp096_round_log_proposal_source_comparison_panel.py \
  --guarded-draft-shadow-no-commit
```

Report: `reports/experiment_096_round_log_proposal_source_comparison_panel.json` (gitignored).

```bash
pytest tests/test_exp096_round_log_proposal_source_comparison_panel.py -q
```

**Run summary (CPU, float32, 2 models × 4 prompts × 4 compressors × 2 max_new_tokens):**

| Metric | Value |
|--------|-------|
| Status | `panel_complete` |
| Safety spec validation | pass |
| Models loaded / blocked | 2 / 0 |
| Generation cells | 64 / 64 successful |
| Round-log coverage / prefix match | 1.0 / ~0.67 |
| Shadow top-1 coverage / prefix match | ~0.33 / 0.0 |
| Rounds both sources available | 112 |
| Rounds sources agree | 0 |
| Decision | `promote_round_log_draft_tokens_as_l3_source` |
| Baseline / draft-shadow parity | 64/64 |
| Safety gates | 64/64 OK |
| Proposals used for token commit | false |
| Proposals exposed to generator | false |
