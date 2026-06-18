# Experiment 095: Round-Log Draft Proposal Source (Phase 19A)

**Experiment ID:** `exp095_round_log_draft_proposal_source`

Companion: [`PHASE_19A_ROUND_LOG_DRAFT_PROPOSAL_SOURCE.md`](PHASE_19A_ROUND_LOG_DRAFT_PROPOSAL_SOURCE.md)

---

## Run

```bash
python3 scripts/research/run_exp095_round_log_draft_proposal_source.py \
  --guarded-draft-shadow-no-commit
```

Default proposal source: `exactkv_round_log_draft_tokens`

---

## Core API

- `build_round_log_draft_proposals(...)`
- `round_log_proposal_to_report_dict(...)`
- `aggregate_round_log_proposal_coverage(...)`
- `load_exp094_previous_source_comparison(...)`

---

## Report

`reports/experiment_095_round_log_draft_proposal_source.json`

---

## Tests

```bash
pytest tests/test_exp095_round_log_draft_proposal_source.py -q
```

Default tests do not require model downloads, CUDA, vLLM, or network.
