# Phase 17C: Longer-Context Validation

**Status:** run `scripts/research/run_exp088_long_context_validation_panel.py --guarded-decode-time-shadow`.

> This is longer-context validation for diagnostic guarded shadow, not production long-context support.  
> Results are model-scoped, panel-scoped, and context-length-scoped.  
> ExactKV default generation remains unchanged.  
> Shadow output cannot affect token commits.  
> Passing this panel does not prove speed, throughput, latency, serving, memory savings, or VeriCache reproduction.  
> Top-k agreement is supplementary only and is not an exactness guarantee.  
> No CUDA/Triton kernel is implemented.  
> No vLLM integration is implemented.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_088_LONG_CONTEXT_VALIDATION_PANEL.md`](EXPERIMENT_088_LONG_CONTEXT_VALIDATION_PANEL.md)

---

## 1. Purpose

Run a small longer-context validation panel for the guarded diagnostic shadow pipeline. Answer whether the existing guarded-shadow validation path remains stable on longer deterministic prompts while preserving token parity and safety gates.

This is not a benchmark suite, production validation, or performance/memory test.

---

## 2. Relation to Phase 17B

Phase 17B validated the guarded shadow path on a two-model short-prompt panel (8 cells). Phase 17C extends the same validation path to longer deterministic prompts at approximate target lengths 128, 256, and 512 tokens across three prompt families.

---

## 3. Long-context panel design

| Setting | Default |
|---------|---------|
| Model | `Qwen/Qwen2.5-0.5B` |
| Optional (`--include-instruct`) | `Qwen/Qwen2.5-0.5B-Instruct` |
| Target context lengths | 128, 256, 512 tokens (approximate) |
| Prompt families | factual, structured, code |
| Compressors | noop, int8 |
| max_new_tokens | 4 |

Default cells: 1 model × 3 lengths × 3 families × 2 compressors = **18 cells**.

---

## 4. Prompt families

Deterministic synthetic fillers (repeated until approximate target length, then truncated):

| Family | Content |
|--------|---------|
| `factual` | Repeated factual sentences |
| `structured` | JSON-like key/value filler |
| `code` | Python-like function segments |

Each cell records `target_context_tokens`, `actual_prompt_token_count`, and `prompt_family`. Exact token counts are not guaranteed.

---

## 5. Validation path

Per cell:

1. Tokenize deterministic long prompt
2. Run baseline generation
3. Run guarded-shadow generation (`GuardedDecodeTimeShadowObserver`)
4. Compare baseline vs guarded token IDs and text
5. Record ExactKV failures and token_exact_match when available
6. Record decode-time shadow callbacks
7. Compare decode-time shadow vs post-hoc shadow when available
8. Enforce safety gates

Mismatches mark the cell and report as failed; mismatches are not hidden.

---

## 6. Safety gates

Per cell: `baseline_generation_completed`, `guarded_shadow_generation_completed`, `baseline_vs_guarded_token_match`, `baseline_vs_guarded_text_match`, `decode_time_shadow_used_for_token_commit=false`, `generation_modified_by_decode_time_shadow=false`, `default_runtime_changed=false`, `observer_return_value_ignored=true`, `shadow_exception_affects_generation=false`, `shadow_result_exposed_to_generator=false`.

---

## 7. Results

```bash
python3 scripts/research/run_exp088_long_context_validation_panel.py \
  --guarded-decode-time-shadow
```

Report: `reports/experiment_088_long_context_validation_panel.json` (gitignored).

**Run summary (CPU, float32, 1 model × 3 lengths × 3 families × 2 compressors, max_new_tokens=4):**

| Metric | Value |
|--------|-------|
| Status | `diagnostic_complete` |
| Models requested | 1 (`Qwen/Qwen2.5-0.5B`) |
| Models loaded | 1/1 |
| Models blocked | 0 |
| Target lengths | 128, 256, 512 |
| Prompt families | factual, structured, code |
| Compressors | noop, int8 |
| Total cells | 18 |
| Successful cells | 18 |
| Failed cells | 0 |
| Baseline-vs-guarded token match | 18/18 |
| Baseline-vs-guarded text match | 18/18 |
| ExactKV failures | 0 baseline, 0 guarded |
| Decode-time shadow callbacks | 18/18 successful |
| Decode-time vs post-hoc shadow match | 18/18 |
| Safety gates | 18/18 OK |
| Context length summary | 6/6 per length (128, 256, 512) |

---

## 8. What this proves

The guarded diagnostic shadow validation path can run on longer deterministic prompts in this panel while preserving baseline-vs-guarded token/text parity and safety gates in tested cells.

---

## 9. What this does not prove

General long-context support, production readiness, speed/throughput/latency improvement, memory savings, serving capability, or VeriCache reproduction.

---

## 10. Claim scope

Results are **model-scoped, panel-scoped, and context-length-scoped only**.

---

## 11. Blocked models/cells

Models that fail to load are recorded with `blocked_reason`; success is not faked. Cells with parity or safety-gate failure are marked failed.

---

## 12. Recommended next step

**Phase 17D (complete):** integration design review — see [`PHASE_17D_INTEGRATION_DESIGN_REVIEW.md`](PHASE_17D_INTEGRATION_DESIGN_REVIEW.md).

**Phase 18A (proposed):** integration safety spec — explicit approval required; claim boundaries unchanged.
