# Phase 18D: Shadow Top-1 Extraction Hardening

**Status:** run `scripts/research/run_exp093_shadow_top1_extraction_hardening.py --guarded-draft-shadow-no-commit`.

> This is shadow top-1 extraction hardening for L3 no-commit diagnostics, not L4 verifier-mediated compressed draft.  
> ExactKV default generation remains unchanged.  
> Extracted shadow top-1 proposals are diagnostic only.  
> Extracted proposals cannot affect token commits.  
> Extracted proposals are not exposed to generator decisions.  
> Proposal token IDs are never fabricated from committed tokens.  
> Blocked proposals are reported, not hidden.  
> Proposal coverage is not exactness.  
> Proposal match rate is supplementary only and is not an exactness guarantee.  
> Full verification remains the required source of truth before any future compressed draft acceptance.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_093_SHADOW_TOP1_EXTRACTION_HARDENING.md`](EXPERIMENT_093_SHADOW_TOP1_EXTRACTION_HARDENING.md)

---

## 1. Purpose

Improve safe diagnostic top-1 extraction from shadow output so the L3 guarded draft-shadow scaffold can produce more draft-shadow proposals without fabricating token IDs and without changing generation behavior.

Answer: can we safely extract shadow-derived top-1 proposal candidates from existing diagnostic shadow outputs with clear provenance and no token-commit effect?

---

## 2. Relation to Phase 18C

Phase 18C ran the 32-cell L3 panel and found ~35% proposal coverage with 99/152 proposals blocked by `no safe top1 extraction from shadow output`. Phase 18D hardens extraction by exposing explicit diagnostic shadow top-1 fields and a provenance-aware `extract_shadow_top1_candidate()` function, then re-runs the same panel dimensions for coverage comparison.

---

## 3. Why proposal coverage was low

Post-hoc shadow cells did not always surface explicit `shadow_top1_token_id` / `shadow_topk_token_ids` fields even when offline replay computed streaming top-1 internally. Extraction relied on a single nested path (`streaming_vs_materialized_metrics.other_top1_token_id`) and blocked when `shadow_status != shadow_complete` before checking alternate safe diagnostic fields.

---

## 4. Safe extraction sources

| Source field | Notes |
|--------------|-------|
| `shadow_top1_token_id` | Explicit diagnostic-only field |
| `diagnostic_proposal_token_id` | Shadow diagnostics proposal field |
| `topk_agreement_metrics.shadow_top1_token_id` | Enriched agreement metrics |
| `streaming_top1_token_id` | Offline replay streaming path |
| `streaming_vs_materialized_logit_metrics.other_top1_token_id` | Streaming shadow logit top-1 |
| `streaming_vs_materialized_metrics.other_top1_token_id` | Post-hoc cell metrics alias |
| `shadow_topk_token_ids[0]` | Rank-0 from diagnostic top-k list |
| `streaming_top5_token_ids[0]` | Rank-0 from streaming top-5 |

---

## 5. Forbidden extraction sources

| Source | Reason |
|--------|--------|
| `committed_token_id` / `generated_token_id` | Committed generation path |
| `baseline_token_id` | Baseline generation path |
| `verifier_committed_token_id` | Verifier commit path |
| `full_top1_token_id` / `materialized_top1_token_id` | Reference/materialized paths |
| `reference_top1_token_id` | Full-model reference |
| Unsafe retokenization markers | Disabled by default |

---

## 6. Provenance requirements

Every successful extraction records:

- exact source field path (e.g. `shadow_top1_token_id`)
- `is_shadow_derived=true`
- `uses_committed_token=false`
- `uses_baseline_token=false`

If provenance requirements are not met, extraction fails as `unsafe_rejected`.

---

## 7. Coverage comparison

When `reports/experiment_092_guarded_draft_shadow_panel_validation.json` exists, Exp093 reports `previous_coverage`, `current_coverage`, and `coverage_delta`. If the Exp092 report is missing, previous coverage is marked unknown and the run does not fail.

---

## 8. Safety spec validation

Self-validates against Phase 18A `validate_integration_proposal()` as `L3_GUARDED_DRAFT_SHADOW_NO_COMMIT`. Failure marks report `failed`.

---

## 9. What this proves

- Hardened extraction can surface more shadow-derived top-1 proposals when explicit diagnostic fields are present.
- Provenance is recorded per extraction.
- L3 no-commit safety gates remain intact across the same 32-cell panel.
- Baseline vs draft-shadow generation parity is preserved.

---

## 10. What this does not prove

- L4 verifier-mediated compressed draft acceptance.
- Production serving, throughput, or memory savings.
- General model-output or exact-generation preservation.
- That proposal match rate implies exactness.

---

## 11. Allowed claims

- L3 diagnostic shadow top-1 extraction hardening with provenance.
- Panel-scoped coverage comparison vs Exp092 when that report exists.
- Blocked and unsafe-rejected extractions are reported explicitly.

---

## 12. Forbidden claims

- Speedup, throughput, latency, or tokens-per-second improvements.
- Active GPU memory or production memory savings.
- Production serving or VeriCache reproduction.
- Token-commit integration or generator exposure of proposals.

---

## 13. Recommended next phase

**Phase 18E:** L3 shadow proposal provenance audit — expand provenance logging and match diagnostics across additional panel dimensions without L4 commit integration.

---

## Run

```bash
python3 scripts/research/run_exp093_shadow_top1_extraction_hardening.py \
  --guarded-draft-shadow-no-commit
```

Report: `reports/experiment_093_shadow_top1_extraction_hardening.json` (gitignored).

```bash
pytest tests/test_exp093_shadow_top1_extraction_hardening.py -q
```

**Run summary (CPU, float32, 4 prompts × 4 compressors × 2 max_new_tokens, decode_time_shadow_top1):**

| Metric | Value |
|--------|-------|
| Status | `hardening_complete` |
| Safety spec validation | pass |
| Total cells | 32 |
| Baseline / draft-shadow successful | 32/32 |
| Token / text parity | 32/32 |
| Total extractions | 152 |
| Successful / blocked extractions | 53 / 99 |
| Unsafe rejected | 0 |
| Extraction coverage rate | ~0.35 |
| Extraction source | `shadow_top1_token_id` (53) |
| Block reason | `no explicit shadow top-1 diagnostic field available` (99) |
| Coverage delta vs Exp092 | 0.0 |
| Safety gates | 32/32 OK |
| Proposals used for token commit | false |
| Proposals exposed to generator | false |
