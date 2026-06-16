# Experiment 079: Decode-Prefix Ladder Shadow Observer (Phase 16N)

**Status:** post-hoc decode-prefix ladder observer — run `scripts/research/run_exp079_decode_prefix_ladder_shadow_observer.py --generation-shadow-observer`.

> This is a **post-hoc decode-prefix ladder observer**, not live decode integration.  
> ExactKV generation runs first and remains unchanged.  
> Shadow analysis runs after generation and cannot affect token commits.  
> Prefix ladder replay is fixed-sequence post-hoc analysis, not token generation.  
> Shadow logits/top-k are diagnostic only and are not exactness guarantees.  
> Streaming attention is not wired into ExactKV generation.  
> No live per-round decode hooks are implemented.  
> No CUDA/Triton kernel is implemented.  
> No vLLM integration is implemented.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_078_GENERATION_SHADOW_EXPANDED_PANEL.md`](EXPERIMENT_078_GENERATION_SHADOW_EXPANDED_PANEL.md) · `exactkv/attention/generation_shadow_observer.py`

---

## 1. Purpose

Phase 16N approximates decode-time shadow behavior externally by running offline shadow diagnostics on every generated-token prefix after generation completes:

- `k=0`: prompt only
- `k=1`: prompt + first generated token
- …
- `k=N`: prompt + all generated tokens

The goal is to see where tolerance statuses and top-k diagnostics remain stable or change as generated tokens are appended.

---

## 2. Relation to Phase 16M

Phase 16M validated the expanded external panel across prompts, lengths, and compressors with two fixed shadow modes. Phase 16N adds a per-prefix ladder on each completed generation cell without modifying `ExactKVGenerator` or default runtime.

---

## 3. Why this is post-hoc, not live per-round decode

- Generation completes first with default `ExactKVGenerator` behavior unchanged.
- Prefix ladder is reconstructed from captured prompt + generated token IDs.
- Shadow runs post-hoc on fixed sequences and cannot affect commit decisions.
- No in-loop draft/verify/commit hooks are added.

---

## 4. Prefix ladder construction

Requires prompt token IDs and generated token IDs. If generated token IDs are missing:

- Cell is marked `blocked_no_round_data`
- No default retokenization from generated text (unsafe)

`--ladder-stride` (default `1`) subsamples prefix steps; the final `k=N` step is always included.

---

## 5. Round source classification

| Value | Meaning |
|---|---|
| `posthoc_prefix_ladder` | Ladder built from prompt + generated token IDs (default) |
| `exactkv_round_log` | `ExactKVResult.traces` captured without generator modification |
| `blocked_no_round_data` | Missing token IDs; ladder cannot be built |

---

## 6. Generation safety gates

Per generation cell `safety_gates`:

- `generation_completed`
- `generated_output_unchanged = true`
- `shadow_ran_after_generation`
- `shadow_used_for_token_commit = false`
- `generation_modified_by_shadow = false`
- `default_runtime_changed = false`

---

## 7. Prefix-level diagnostics

Each prefix step records:

- `generated_prefix_length` (k)
- `shadow_sequence_length`
- `tolerance_policy_status`
- streaming-vs-materialized and full-vs-streaming metrics
- top-k agreement metrics (supplementary)
- interpretation note and blockers

---

## 8. Aggregated status/top-k summaries

- `tolerance_policy_summary_by_prefix_length`
- `topk_agreement_summary_by_prefix_length`
- `first_status_change_summary` — first k where tolerance status differs from k=0
- `first_top1_mismatch_summary` — first k where top-1 agreement is false
- Max drift by prefix length (full-vs-streaming and streaming-vs-materialized)

---

## 9. Results

```bash
python3 scripts/research/run_exp079_decode_prefix_ladder_shadow_observer.py --generation-shadow-observer
```

Report: `reports/experiment_079_decode_prefix_ladder_shadow_observer.json` (gitignored).

---

## 10. What this proves

- Post-hoc prefix ladder shadow replay is feasible without generation integration.
- Tolerance/top-k stability can be tracked across generated prefixes externally.
- True round traces from `ExactKVResult` can be classified when already exposed.

---

## 11. What this does not prove

- Exact generation preservation or model-output preservation.
- That shadow top-k agreement implies numeric equivalence.
- That streaming attention is ready for live per-round decode integration.
- Speed, throughput, latency, serving, or GPU memory claims.
- VeriCache throughput or serving reproduction.

---

## 12. Relation to ExactKV restored verification

Independent from restored-verifier experimental runtime.

---

## 13. Relation to VeriCache parity

Does not reproduce VeriCache throughput, serving, or memory panels.

---

## 14. Next step

**Phase 16O (proposed):** live per-round decode hooks — only with explicit approval; still no unapproved `ExactKVGenerator` modification.
