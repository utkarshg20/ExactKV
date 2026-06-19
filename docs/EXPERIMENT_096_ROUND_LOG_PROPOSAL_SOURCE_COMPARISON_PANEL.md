# Experiment 096: Round-Log Proposal Source Comparison Panel (Phase 19B)

**Experiment ID:** `exp096_round_log_proposal_source_comparison_panel`

Companion: [`PHASE_19B_ROUND_LOG_PROPOSAL_SOURCE_COMPARISON.md`](PHASE_19B_ROUND_LOG_PROPOSAL_SOURCE_COMPARISON.md)

---

## Run

```bash
python3 scripts/research/run_exp096_round_log_proposal_source_comparison_panel.py \
  --guarded-draft-shadow-no-commit
```

Default proposal sources: `exactkv_round_log_draft_tokens`, `decode_time_shadow_top1`

Default models: `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-0.5B-Instruct`

---

## Core API

- `run_exp096_round_log_proposal_source_comparison_panel(...)`
- `summarize_proposal_source_rounds(...)`
- `build_side_by_side_round_records(...)`
- `aggregate_side_by_side_summary(...)`
- `compute_comparison_decision(...)`
- `validate_exp096_report(...)`

---

## Report

`reports/experiment_096_round_log_proposal_source_comparison_panel.json`

---

## Tests

```bash
pytest tests/test_exp096_round_log_proposal_source_comparison_panel.py -q
```

Default tests do not require model downloads, CUDA, vLLM, or network.
