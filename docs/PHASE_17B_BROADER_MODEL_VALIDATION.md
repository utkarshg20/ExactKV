# Phase 17B: Broader Model Validation

**Status:** run `scripts/research/run_exp087_broader_model_validation_panel.py --guarded-decode-time-shadow`.

> This is broader model validation for diagnostic guarded shadow, not production model-family support.  
> Results are model-scoped and panel-scoped.  
> ExactKV default generation remains unchanged.  
> Shadow output cannot affect token commits.  
> Passing this panel does not prove speed, throughput, latency, serving, memory savings, or VeriCache reproduction.  
> Top-k agreement is supplementary only and is not an exactness guarantee.  
> No CUDA/Triton kernel is implemented.  
> No vLLM integration is implemented.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_087_BROADER_MODEL_VALIDATION_PANEL.md`](EXPERIMENT_087_BROADER_MODEL_VALIDATION_PANEL.md)

---

## 1. Purpose

Validate whether the guarded decode-time shadow pipeline holds on a slightly broader Qwen model set (base + instruct 0.5B) without widening public claims.

---

## 2. Relation to Phase 17A

Phase 17A packaged the Phase 16 story for demos. Phase 17B runs a small multi-model panel using the same guarded observer path from Phase 16R–16S.

---

## 3. Model panel

| Tier | Models |
|------|--------|
| Default | `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-0.5B-Instruct` |
| Optional (`--include-optional-models`) | `Qwen/Qwen2.5-1.5B`, `Qwen/Qwen2.5-1.5B-Instruct` |

Default cells: 2 models × 2 prompts × 2 compressors = **8 cells**.

---

## 4. Validation path

Per cell: baseline generation → guarded `GuardedDecodeTimeShadowObserver` generation → token/text parity → decode-time vs post-hoc shadow comparison → safety gates.

---

## 5. Safety gates

Completion, parity, `decode_time_shadow_used_for_token_commit=false`, `generation_modified_by_decode_time_shadow=false`, `default_runtime_changed=false`, observer return values ignored, shadow exceptions cannot affect generation, shadow not exposed to generator.

---

## 6. Results

```bash
python3 scripts/research/run_exp087_broader_model_validation_panel.py \
  --guarded-decode-time-shadow
```

**Run summary (CPU, float32, 2 default models × 2 prompts × 2 compressors, max_new_tokens=4):**

| Metric | Value |
|--------|-------|
| Status | `diagnostic_complete` |
| Models loaded | 2/2 |
| Models blocked | 0 |
| Total cells | 8 |
| Successful cells | 8 |
| Baseline-vs-guarded token match | 8/8 |
| Baseline-vs-guarded text match | 8/8 |
| Safety gates | 8/8 OK |

Report: `reports/experiment_087_broader_model_validation_panel.json` (gitignored).

---

## 7. What this proves

The existing guarded diagnostic shadow pipeline can run on the default two-model panel while preserving token parity and safety gates in tested cells.

---

## 8. What this does not prove

General model-family support, production readiness, speed/memory/serving claims, or VeriCache reproduction.

---

## 9. Claim scope

Results are **model-scoped and panel-scoped only**.

---

## 10. Blocked models

Models that fail to load are recorded with `blocked_reason`; success is not faked.

---

## 11. Recommended next step

**Phase 17C (complete):** longer-context guarded-shadow validation — see [`PHASE_17C_LONG_CONTEXT_VALIDATION.md`](PHASE_17C_LONG_CONTEXT_VALIDATION.md).

**Phase 17D (proposed):** integration design review — explicit approval required; claim boundaries unchanged.
