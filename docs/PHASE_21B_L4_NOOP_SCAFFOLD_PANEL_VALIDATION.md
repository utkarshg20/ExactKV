# Phase 21B: L4 No-Op Scaffold Panel Validation

**Status:** run `scripts/research/run_exp103_l4_noop_scaffold_panel_validation.py --experimental-l4-verifier-mediated-draft`.

> This is L4 no-op scaffold panel validation, not L4 runtime implementation.  
> ExactKV default generation remains unchanged.  
> ExactKVGenerator remains unchanged unless explicitly reported otherwise.  
> Production CLI remains unchanged.  
> The experimental flag remains research-script-only in this phase.  
> No verifier-mediated acceptance is performed.  
> No proposal can affect token commits.  
> No proposal is exposed to generator decisions.  
> Runtime fallback/rollback behavior is not implemented.  
> L4 runtime commit remains blocked.  
> Passing this panel authorizes only a trace-only dry-run design phase, not runtime commit integration.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_103_L4_NOOP_SCAFFOLD_PANEL_VALIDATION.md`](EXPERIMENT_103_L4_NOOP_SCAFFOLD_PANEL_VALIDATION.md)

---

## 1. Purpose

Validate the L4 no-op opt-in scaffold on a real-model panel: baseline vs no-op scaffold token/text parity with all safety gates holding.

---

## 2. Relation to Phase 21A

Phase 21A implemented the no-op scaffold. Phase 21B validates it across models, compressors, prompts, and max_new_tokens.

---

## 3. Real-model panel

Default: `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-0.5B-Instruct`, CPU, float32, 4 prompts, compressors `noop,int8,int4_sim,k8_v4_sim`, max_new_tokens 4 and 8 → 64 cells.

---

## 4. Research-script-only flag

`--experimental-l4-verifier-mediated-draft` on the exp103 research script only.

---

## 5. No-op safety gates

Every cell records scaffold enabled, no runtime commit, no verifier-mediated acceptance, no proposal commit/exposure, no rollback/fallback runtime, default runtime unchanged.

---

## 6. Baseline vs no-op parity

Token IDs and text compared per cell; mismatch fails the cell and report.

---

## 7. ExactKV failure summary

Per-cell `exactkv_failures_baseline` and `exactkv_failures_noop_scaffold` aggregated in report.

---

## 8. Breakdown summaries

By model, compressor, prompt, and max_new_tokens.

---

## 9. Validation result

`validate_exp103_panel_report` enforces schema, safety invariants, and parity on completed cells.

---

## 10. What this authorizes

- Phase 21C trace-only dry-run design: `phase21c_l4_trace_only_dry_run_design`

---

## 11. What this does not authorize

- L4 runtime commit integration
- Verifier-mediated compressed draft
- Production CLI flag

---

## 12. Remaining blockers

Runtime fallback/rollback, stages 2–4, L4 commit parity panel, exactkv_failures gate run, GPU memory measurement, performance benchmark, serving integration.

---

## 13. Recommended next phase

**Phase 21C:** L4 trace-only dry-run design (`phase21c_l4_trace_only_dry_run_design`).

---

## 14. Claim boundaries

Panel-scoped parity claims only; no speed, throughput, latency, serving, memory, or VeriCache reproduction claims.

---

## Run

```bash
python3 scripts/research/run_exp103_l4_noop_scaffold_panel_validation.py \
  --experimental-l4-verifier-mediated-draft
```

Report: `reports/experiment_103_l4_noop_scaffold_panel_validation.json` (gitignored).

```bash
pytest tests/test_exp103_l4_noop_scaffold_panel_validation.py -q
```

**Run summary:**

| Metric | Value |
|--------|-------|
| Default cells | 64 |
| Stage | `stage_1_noop_opt_in_scaffold` |
| Allowed next phase | `phase21c_l4_trace_only_dry_run_design` |
