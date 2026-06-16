# Experiment 080: ExactKV Round-Log Shadow Observer (Phase 16O)

**Status:** post-hoc round-log shadow observer — run `scripts/research/run_exp080_round_log_shadow_observer.py --generation-shadow-observer`.

> This is a **post-hoc ExactKV round-log shadow observer**, not live decode integration.  
> Existing ExactKVResult round logs are used when available.  
> ExactKV generation runs first and remains unchanged.  
> Shadow analysis runs after generation and cannot affect token commits.  
> Round-boundary replay is fixed-sequence post-hoc analysis, not token generation.  
> Shadow logits/top-k are diagnostic only and are not exactness guarantees.  
> Streaming attention is not wired into ExactKV generation.  
> No live per-round decode hooks are implemented.  
> No CUDA/Triton kernel is implemented.  
> No vLLM integration is implemented.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_079_DECODE_PREFIX_LADDER_SHADOW_OBSERVER.md`](EXPERIMENT_079_DECODE_PREFIX_LADDER_SHADOW_OBSERVER.md) · `exactkv/attention/generation_shadow_observer.py`

---

## 1. Purpose

Phase 16O uses existing `ExactKVResult.traces` to run offline shadow diagnostics at actual ExactKV decode/verification round boundaries. The goal is to see how tolerance statuses and top-k diagnostics evolve at real round boundaries and whether changes align descriptively with accepted/corrected token boundaries.

---

## 2. Relation to Phase 16N

Phase 16N built a post-hoc prefix ladder (k=0..N generated tokens). Phase 16O uses true ExactKV round metadata when available, aligning shadow replay with actual draft/verify/commit rounds rather than uniform prefix steps.

---

## 3. Why existing round logs come before live hooks

`ExactKVResult` already exposes per-round `VerificationTrace` entries without generator modification. Using them first avoids live hook risk while still anchoring diagnostics to real round boundaries.

---

## 4. Round-log extraction

Per round, when available:

- round index
- prefix length before/after round
- draft length
- accepted and rejected/corrected token counts
- generated token IDs up to round boundary

Missing fields are recorded as null/unknown without fabrication. If no round log exists, the cell is `blocked_missing_round_log` unless `--fallback-prefix-ladder` is explicitly set.

---

## 5. Round-boundary shadow replay

For each extracted round:

- Build fixed sequence = prompt + generated tokens through `prefix_length_after_round`
- Run Phase 16F-style offline shadow replay
- Apply Phase 16I tolerance policy
- Record top-k agreement (supplementary)

---

## 6. Boundary/event analysis

Aggregates include:

- tolerance policy summary by round index
- top-k agreement summary by round index
- first round where tolerance status changes
- first round where top-1 mismatch appears
- accepted-prefix correlation summary (descriptive only; no causality claim)

---

## 7. Safety gates

Per generation cell:

- `generation_completed`
- `generated_output_unchanged = true`
- `shadow_ran_after_generation = true` (when shadow runs)
- `shadow_used_for_token_commit = false`
- `generation_modified_by_shadow = false`
- `default_runtime_changed = false`

---

## 8. Results

```bash
python3 scripts/research/run_exp080_round_log_shadow_observer.py --generation-shadow-observer
```

Report: `reports/experiment_080_round_log_shadow_observer.json` (gitignored).

---

## 9. What this proves

- Post-hoc round-log shadow replay is feasible using existing `ExactKVResult` traces.
- Tolerance/top-k evolution can be tracked at real ExactKV round boundaries externally.
- Accepted-prefix overlap with mismatch rounds can be recorded descriptively.

---

## 10. What this does not prove

- Exact generation preservation or model-output preservation.
- Causal links between acceptance events and shadow diagnostics.
- That streaming attention is ready for live per-round decode integration.
- Speed, throughput, latency, serving, or GPU memory claims.
- VeriCache throughput or serving reproduction.

---

## 11. Relation to ExactKV restored verification

Independent from restored-verifier experimental runtime.

---

## 12. Relation to VeriCache parity

Does not reproduce VeriCache throughput, serving, or memory panels.

---

## 13. Next step

**Phase 16P (proposed):** live per-round decode hooks — only with explicit approval; still no unapproved `ExactKVGenerator` modification.
