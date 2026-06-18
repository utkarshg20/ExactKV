# Experiment 088: Longer-Context Validation Panel (Phase 17C)

**Experiment ID:** `exp088_long_context_validation_panel`  
**Report:** `reports/experiment_088_long_context_validation_panel.json`  
Companion: [`PHASE_17C_LONG_CONTEXT_VALIDATION.md`](PHASE_17C_LONG_CONTEXT_VALIDATION.md) · `exactkv/demo/long_context_validation.py`

> Longer-context validation for diagnostic guarded shadow only — not production long-context support.  
> Results are model-scoped, panel-scoped, and context-length-scoped.

---

## Run

```bash
python3 scripts/research/run_exp088_long_context_validation_panel.py \
  --guarded-decode-time-shadow
```

Optional instruct model:

```bash
python3 scripts/research/run_exp088_long_context_validation_panel.py \
  --guarded-decode-time-shadow \
  --include-instruct
```

---

## Panel defaults

| Parameter | Default |
|-----------|---------|
| Model | `Qwen/Qwen2.5-0.5B` |
| Target lengths | 128, 256, 512 |
| Families | factual, structured, code |
| Compressors | noop, int8 |
| max_new_tokens | 4 |
| Cells | 18 |

---

## CLI flags

| Flag | Purpose |
|------|---------|
| `--guarded-decode-time-shadow` | Required; enables guarded validation path |
| `--model-id` | Override default model |
| `--include-instruct` | Add `Qwen/Qwen2.5-0.5B-Instruct` |
| `--target-context-tokens` | Comma-separated lengths (default `128,256,512`) |
| `--prompt-families` | factual, structured, code |
| `--compressors` | noop, int8 |
| `--max-new-tokens` | Decode length (default 4) |
| `--max-cells` | Optional cap |
| `--device` / `--dtype` | Default cpu / float32 |
| `--local-files-only` | HF local cache only |
| `--allow-model-blocked` | Record blocked models without hard fail |

---

## Report schema (top-level)

`experiment_id`, `status`, `models_requested`, `models_loaded`, `models_blocked`, `target_context_tokens`, `prompt_families`, `compressors_requested`, `compressors_run`, `max_new_tokens`, `total_cells`, `successful_cells`, `failed_cells`, `blocked_cells`, `baseline_vs_guarded_token_match_cells`, `baseline_vs_guarded_text_match_cells`, `exactkv_failure_summary`, `decode_time_shadow_callback_summary`, `decode_time_vs_posthoc_shadow_match_summary`, `safety_gate_summary`, `context_length_summary`, `claim_scope_note`, `blockers`, `limitations`, `no_performance_claims_note`, `model_results`.

Per cell: `prompt_family`, `target_context_tokens`, `actual_prompt_token_count`, parity fields, shadow callbacks, `safety_gates`, `blockers`.

---

## Claim scope

Passing this panel does **not** prove general long-context support, production readiness, speed, memory, serving, or VeriCache reproduction. No performance metrics are recorded.

---

## Tests

```bash
pytest tests/test_exp088_long_context_validation_panel.py -q
```

CPU-only; no model downloads or network required.
