# Experiment 093: Shadow Top-1 Extraction Hardening (Phase 18D)

**Experiment ID:** `exp093_shadow_top1_extraction_hardening`

Companion: [`PHASE_18D_SHADOW_TOP1_EXTRACTION_HARDENING.md`](PHASE_18D_SHADOW_TOP1_EXTRACTION_HARDENING.md)

---

## Run

```bash
python3 scripts/research/run_exp093_shadow_top1_extraction_hardening.py \
  --guarded-draft-shadow-no-commit
```

Optional flags match Exp092: `--model-id`, `--device`, `--dtype`, `--max-prompts`, `--max-new-tokens-values`, `--compressors`, `--proposal-source`, `--local-files-only`, `--allow-provider-blocked`.

---

## Panel dimensions (default)

| Dimension | Value |
|-----------|-------|
| Model | `Qwen/Qwen2.5-0.5B` |
| Device / dtype | `cpu` / `float32` |
| Prompts | 4 |
| Compressors | noop, int8, int4_sim, k8_v4_sim |
| max_new_tokens | 4, 8 |
| Cells | 32 |
| Proposal source | `decode_time_shadow_top1` |

---

## Core API

- `extract_shadow_top1_candidate(shadow_output) -> ShadowTop1ExtractionResult`
- `diagnostic_shadow_top1_fields(shadow_cell)` in `generation_shadow_observer.py`

---

## Report

`reports/experiment_093_shadow_top1_extraction_hardening.json`

Key fields: `previous_coverage`, `current_coverage`, `coverage_delta`, `extraction_source_summary`, `extraction_block_reason_summary`, `successful_extractions`, `blocked_extractions`, `unsafe_extractions_rejected`.

---

## Tests

```bash
pytest tests/test_exp093_shadow_top1_extraction_hardening.py -q
```

Default tests do not require model downloads, CUDA, vLLM, or network.
