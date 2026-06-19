# Phase 21A: L4 No-Op Opt-In Scaffold

**Status:** run `scripts/research/run_exp102_l4_noop_opt_in_scaffold.py --experimental-l4-verifier-mediated-draft`.

> This is a no-op L4 opt-in scaffold, not L4 runtime implementation.  
> ExactKV default generation remains unchanged.  
> ExactKVGenerator remains unchanged unless explicitly reported otherwise.  
> Production CLI remains unchanged.  
> The experimental flag is research-script-only in this phase.  
> No verifier-mediated acceptance is performed.  
> No proposal can affect token commits.  
> No proposal is exposed to generator decisions.  
> Runtime fallback/rollback behavior is not implemented.  
> L4 runtime commit remains blocked.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_102_L4_NOOP_OPT_IN_SCAFFOLD.md`](EXPERIMENT_102_L4_NOOP_OPT_IN_SCAFFOLD.md)

---

## 1. Purpose

Implement Stage 1 no-op L4 opt-in scaffold that records experimental opt-in and trace metadata externally while proving baseline and scaffold paths produce identical tokens.

---

## 2. Relation to Phase 20D

Phase 20D authorized `ready_for_stage_1_noop_opt_in_scaffold_design`. Phase 21A implements that scaffold without runtime commit integration.

---

## 3. No-op scaffold boundary

- External wrapper around unchanged baseline generation
- No L4 interfaces invoked during generation
- No ExactKVGenerator modification
- No production CLI changes

---

## 4. Research-script-only flag

`--experimental-l4-verifier-mediated-draft` on `run_exp102_l4_noop_opt_in_scaffold.py` only.

---

## 5. Why production CLI is not modified

Stage 1 requires explicit experimental opt-in with warnings; production CLI must remain unchanged until later gated phases.

---

## 6. Safety invariants

Every cell/report records: no runtime commit, no verifier-mediated acceptance, no proposal commit/exposure, no rollback/fallback runtime, default runtime unchanged, research-script flag only.

---

## 7. Real-model smoke setup

Default: `Qwen/Qwen2.5-0.5B`, CPU, float32, 2 prompts, compressors `noop,int8`, max_new_tokens 4.

---

## 8. Validation behavior

`validate_l4_noop_scaffold_report` fails on any safety invariant violation or token/text parity failure in completed cells.

---

## 9. What this authorizes

- Phase 21B panel validation: `phase21b_l4_noop_scaffold_panel_validation`

---

## 10. What this does not authorize

- L4 runtime commit
- Verifier-mediated compressed draft
- Production CLI flag

---

## 11. Remaining blockers

Runtime fallback/rollback, stages 2–4, L4 parity panel, exactkv_failures gate run, GPU memory measurement, performance benchmark, serving integration, production CLI flag.

---

## 12. Recommended next phase

**Phase 21B:** L4 no-op scaffold panel validation — complete. See [`PHASE_21B_L4_NOOP_SCAFFOLD_PANEL_VALIDATION.md`](PHASE_21B_L4_NOOP_SCAFFOLD_PANEL_VALIDATION.md).

**Next:** Phase 21C trace-only dry-run design (`phase21c_l4_trace_only_dry_run_design`).

---

## 13. Claim boundaries

Scaffold and panel-scoped parity claims only; no speed, throughput, latency, serving, memory, or VeriCache reproduction claims.

---

## Run

```bash
python3 scripts/research/run_exp102_l4_noop_opt_in_scaffold.py \
  --experimental-l4-verifier-mediated-draft
```

Report: `reports/experiment_102_l4_noop_opt_in_scaffold.json` (gitignored).

```bash
pytest tests/test_exp102_l4_noop_opt_in_scaffold.py -q
```

**Run summary:**

| Metric | Value |
|--------|-------|
| Stage | `stage_1_noop_opt_in_scaffold` |
| Mode | `noop_trace_only` |
| L4 runtime commit | false |
| Allowed next phase | `phase21b_l4_noop_scaffold_panel_validation` |
