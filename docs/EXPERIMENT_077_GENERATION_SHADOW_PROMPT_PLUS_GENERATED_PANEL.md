# Experiment 077: Prompt+Generated Generation-Shadow Panel (Phase 16L)

**Status:** external prompt-plus-generated shadow panel — run `scripts/research/run_exp077_generation_shadow_prompt_plus_generated_panel.py --generation-shadow-observer`.

> This is an **external prompt-plus-generated shadow observer**, not generation integration.  
> ExactKV generation runs first and remains unchanged.  
> Shadow analysis runs after generation and cannot affect token commits.  
> Prompt+generated replay is fixed-sequence post-hoc analysis, not token generation.  
> Shadow logits/top-k are diagnostic only and are not exactness guarantees.  
> Streaming attention is not wired into ExactKV generation.  
> No CUDA/Triton kernel is implemented.  
> No vLLM integration is implemented.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_076_GENERATION_SHADOW_OBSERVER_SMOKE.md`](EXPERIMENT_076_GENERATION_SHADOW_OBSERVER_SMOKE.md) · `exactkv/attention/generation_shadow_observer.py`

---

## 1. Purpose

Phase 16L extends the Phase 16K external observer by running post-hoc shadow diagnostics on:

- prompt-only prefix (`prompt_prefix_only`)
- prompt + generated tokens (`prompt_plus_generated_tokens`)

The goal is to validate safe reconstruction of the prompt+generated token-ID sequence after generation completes.

---

## 2. Relation to Phase 16K

Phase 16K proved an external observer can run generation unchanged and run shadow after the fact. Phase 16L adds a panel that compares two fixed-sequence shadow modes without per-round decode hooks.

---

## 3. Why prompt+generated is safer than per-round decode hooks

- Generation completes first with the default `ExactKVGenerator` behavior unchanged.
- Shadow runs post-hoc on a fixed token-ID sequence and cannot affect commit decisions.
- Avoids any in-loop draft/verify/commit wiring, which is higher risk for accidental behavior changes.

---

## 4. Shadow modes

| Mode | Sequence used for offline replay |
|---|---|
| `prompt_prefix_only` | prompt token IDs only |
| `prompt_plus_generated_tokens` | prompt token IDs + generated token IDs (when available) |

---

## 5. Token reconstruction behavior

For `prompt_plus_generated_tokens`, the panel uses generated token IDs captured from `ExactKVResult.output_ids` when available.

If generated token IDs are unavailable:

- **Do not fake token IDs**
- Mark the shadow cell blocked as `blocked_missing_generated_token_ids`

No retokenization-from-text fallback is used by default (unsafe).

---

## 6. Generation safety gates

Per report and per prompt:

- `generation_modified_by_shadow = false`
- `shadow_used_for_token_commit = false`
- `default_runtime_changed = false`
- `shadow_ran_after_generation = true` for completed shadow cells

---

## 7. Results

```bash
python3 scripts/research/run_exp077_generation_shadow_prompt_plus_generated_panel.py --generation-shadow-observer
```

Report: `reports/experiment_077_generation_shadow_prompt_plus_generated_panel.json` (gitignored).

---

## 8. What this proves

- Prompt+generated fixed-sequence shadow replay is feasible as a post-hoc observer.
- The wrapper can safely reuse generated token IDs (when available) without touching generation internals.

---

## 9. What this does not prove

- Exact generation preservation or model-output preservation.
- That shadow top-k agreement implies numeric equivalence or exactness.
- That streaming attention is ready for in-loop draft/verify/commit integration.
- Speed, throughput, latency, serving, or GPU memory claims.

---

## 10. Relation to ExactKV restored verification

Independent from restored-verifier experimental runtime. No restored-verifier interaction is required for this phase.

---

## 11. Relation to VeriCache parity

Does not reproduce VeriCache throughput, serving, or memory panels.

---

## 12. Next step

**Phase 16M (complete):** expanded external panel across prompts, `max_new_tokens`, and compressors — see [`EXPERIMENT_078_GENERATION_SHADOW_EXPANDED_PANEL.md`](EXPERIMENT_078_GENERATION_SHADOW_EXPANDED_PANEL.md).

**Phase 16N (proposed):** per-round decode observer — only with explicit approval; still no `ExactKVGenerator` modification.

