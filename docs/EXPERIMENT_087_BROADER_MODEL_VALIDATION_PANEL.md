# Experiment 087: Broader Model Validation Panel (Phase 17B)

**Status:** `--guarded-decode-time-shadow` required.

> This is broader model validation for diagnostic guarded shadow, not production model-family support.  
> Results are model-scoped and panel-scoped.  
> ExactKV default generation remains unchanged.  
> Shadow output cannot affect token commits.

Companion: [`PHASE_17B_BROADER_MODEL_VALIDATION.md`](PHASE_17B_BROADER_MODEL_VALIDATION.md) · `exactkv/demo/broader_model_validation.py`

---

## Run

```bash
python3 scripts/research/run_exp087_broader_model_validation_panel.py \
  --guarded-decode-time-shadow
```

Optional larger models:

```bash
python3 scripts/research/run_exp087_broader_model_validation_panel.py \
  --guarded-decode-time-shadow --include-optional-models
```

Report: `reports/experiment_087_broader_model_validation_panel.json` (gitignored).
