# Experiment 092: L3 Guarded Draft-Shadow Panel Validation (Phase 18C)

**Experiment ID:** `exp092_guarded_draft_shadow_panel_validation`  
**Report:** `reports/experiment_092_guarded_draft_shadow_panel_validation.json`  
Companion: [`PHASE_18C_GUARDED_DRAFT_SHADOW_PANEL_VALIDATION.md`](PHASE_18C_GUARDED_DRAFT_SHADOW_PANEL_VALIDATION.md)

---

## Run

```bash
python3 scripts/research/run_exp092_guarded_draft_shadow_panel_validation.py \
  --guarded-draft-shadow-no-commit
```

---

## Default panel

4 prompts × 4 compressors × 2 max_new_tokens = **32 cells**

---

## Tests

```bash
pytest tests/test_exp092_guarded_draft_shadow_panel_validation.py -q
```
