# Experiment 091: L3 Guarded Draft-Shadow No-Commit Scaffold (Phase 18B)

**Experiment ID:** `exp091_guarded_draft_shadow_no_commit_scaffold`  
**Report:** `reports/experiment_091_guarded_draft_shadow_no_commit_scaffold.json`  
Companion: [`PHASE_18B_GUARDED_DRAFT_SHADOW_NO_COMMIT.md`](PHASE_18B_GUARDED_DRAFT_SHADOW_NO_COMMIT.md) · `exactkv/safety/guarded_draft_shadow.py`

> L3 scaffold only — proposals cannot affect token commits.

---

## Run

```bash
python3 scripts/research/run_exp091_guarded_draft_shadow_no_commit_scaffold.py \
  --guarded-draft-shadow-no-commit
```

---

## Defaults

| Parameter | Default |
|-----------|---------|
| Model | `Qwen/Qwen2.5-0.5B` |
| Prompts | 2 |
| Compressors | noop, int8 |
| max_new_tokens | 4 |
| proposal-source | `decode_time_shadow_top1` |

---

## CLI flags

| Flag | Purpose |
|------|---------|
| `--guarded-draft-shadow-no-commit` | Required |
| `--proposal-source` | synthetic / decode_time_shadow_top1 / blocked_no_provider |
| `--model-id` / `--device` / `--dtype` | Model settings |
| `--max-prompts` / `--max-new-tokens` / `--compressors` | Panel size |
| `--allow-provider-blocked` | Record blocked provider without hard fail |

---

## Tests

```bash
pytest tests/test_exp091_guarded_draft_shadow_no_commit_scaffold.py -q
```

CPU-only; no model downloads required for default tests.
