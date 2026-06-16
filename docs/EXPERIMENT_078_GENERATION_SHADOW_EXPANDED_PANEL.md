# Experiment 078: Expanded Generation-Shadow Panel (Phase 16M)

**Status:** external expanded generation-shadow panel — run `scripts/research/run_exp078_generation_shadow_expanded_panel.py --generation-shadow-observer`.

> This is an **external expanded generation-shadow panel**, not generation integration.  
> ExactKV generation runs first and remains unchanged.  
> Shadow analysis runs after generation and cannot affect token commits.  
> Prompt+generated replay is fixed-sequence post-hoc analysis, not token generation.  
> Shadow logits/top-k are diagnostic only and are not exactness guarantees.  
> Streaming attention is not wired into ExactKV generation.  
> Compressor-specific results are reported only when exposed cleanly by current APIs.  
> No CUDA/Triton kernel is implemented.  
> No vLLM integration is implemented.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_077_GENERATION_SHADOW_PROMPT_PLUS_GENERATED_PANEL.md`](EXPERIMENT_077_GENERATION_SHADOW_PROMPT_PLUS_GENERATED_PANEL.md) · `exactkv/attention/generation_shadow_observer.py`

---

## 1. Purpose

Phase 16M expands the Phase 16L external observer across:

- **8 deterministic prompts** (factual, JSON-like, code-like, arithmetic-text, long-context)
- **`max_new_tokens`:** 4 and 8
- **Compressors:** `noop`, `int8`, `int4_sim`, `k8_v4_sim` when `get_compressor` exposes them
- **Shadow modes:** `prompt_prefix_only`, `prompt_plus_generated_tokens`

The goal is to answer whether the external observer remains stable on a broader prompt/compressor/continuation panel and whether generated token IDs remain available for safe prompt+generated shadow replay.

---

## 2. Relation to Phase 16L

Phase 16L validated prompt+generated fixed-sequence replay on 4 prompts with noop compressor only. Phase 16M scales prompt count, continuation lengths, and compressor coverage without modifying `ExactKVGenerator` or default runtime.

---

## 3. Panel dimensions

| Dimension | Default values |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` |
| Prompts | 8 deterministic (`default_exp078_prompts`) |
| `max_new_tokens` | 4, 8 |
| Compressors requested | `noop`, `int8`, `int4_sim`, `k8_v4_sim` |
| Shadow modes | `prompt_prefix_only`, `prompt_plus_generated_tokens` |

Total generation cells = prompts × `max_new_tokens` × compressors_run. Each generation cell has two shadow cells (one per mode).

---

## 4. Compressor handling

Compressors are resolved via `exactkv.compressors.get_compressor`. Names not in the registry are recorded in `compressors_blocked` with reason `blocked_compressor_api_missing`. No fake compressor results are recorded.

---

## 5. Generation safety gates

Per report and per generation cell:

- `generation_modified_by_shadow = false`
- `shadow_used_for_token_commit = false`
- `default_runtime_changed = false`

If any safety gate is false, the cell is treated as failed/blocked.

---

## 6. Shadow modes

| Mode | Sequence used for offline replay |
|---|---|
| `prompt_prefix_only` | prompt token IDs only |
| `prompt_plus_generated_tokens` | prompt token IDs + generated token IDs (when available) |

Shadow runs post-hoc via Phase 16F-style `run_exp071_logit_cell` replay with Phase 16I tolerance policy.

---

## 7. Token ID availability

Each generation cell records `generation_output_token_ids_available` and `generation_output_token_count`. Prompt+generated shadow requires generated token IDs; missing IDs block that shadow cell as `blocked_missing_generated_token_ids` without retokenization fallback.

---

## 8. ExactKV failure summary

When generation completes, each cell may record `exactkv_failures` and `token_exact_match` from comparison against full greedy baseline (`generate_full_greedy`). The report aggregates `exactkv_failure_summary` across cells; unknown status is tracked when generation fails or baseline compare is unavailable.

---

## 9. Tolerance policy summary

Shadow cells apply Phase 16I tolerance policy (`strict` and depth-aware thresholds). `tolerance_policy_summary` counts policy outcomes across successful shadow cells.

---

## 10. Top-k agreement summary

`topk_agreement_summary` reports supplementary top-1 agreement counts from streaming-vs-materialized metrics. Top-k is **not** an exactness guarantee.

---

## 11. Results

```bash
python3 scripts/research/run_exp078_generation_shadow_expanded_panel.py --generation-shadow-observer
```

Report: `reports/experiment_078_generation_shadow_expanded_panel.json` (gitignored).

---

## 12. What this proves

- The external observer scales to a broader prompt × length × compressor panel without generation integration.
- Generated token IDs can be captured for prompt+generated post-hoc replay when generation succeeds.
- Compressor API gaps are reported honestly without faking results.

---

## 13. What this does not prove

- Exact generation preservation or model-output preservation.
- That shadow top-k agreement implies numeric equivalence.
- That streaming attention is ready for in-loop draft/verify/commit integration.
- Speed, throughput, latency, serving, or GPU memory claims.
- VeriCache throughput or serving reproduction.

---

## 14. Relation to ExactKV restored verification

Independent from restored-verifier experimental runtime. No restored-verifier interaction is required.

---

## 15. Relation to VeriCache parity

Does not reproduce VeriCache throughput, serving, or memory panels.

---

## 16. Next step

**Phase 16N (complete):** post-hoc decode-prefix ladder observer — see [`EXPERIMENT_079_DECODE_PREFIX_LADDER_SHADOW_OBSERVER.md`](EXPERIMENT_079_DECODE_PREFIX_LADDER_SHADOW_OBSERVER.md).

**Phase 16O (proposed):** live per-round decode hooks — opt-in only; still no `ExactKVGenerator` modification.
