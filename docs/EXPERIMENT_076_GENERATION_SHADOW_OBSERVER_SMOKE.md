# Experiment 076: Generation-Shadow Observer Smoke (Phase 16K)

**Status:** external L1 observer smoke — run `scripts/research/run_exp076_generation_shadow_observer_smoke.py --generation-shadow-observer` for report.

> This is an **external generation-shadow observer**, not generation integration.  
> ExactKV generation runs first and remains **unchanged**.  
> Shadow analysis runs **after** generation and **cannot** affect token commits.  
> Shadow logits/top-k are **diagnostic only** and are **not** exactness guarantees.  
> Streaming attention is **not wired into ExactKV generation**.  
> No CUDA kernel is implemented.  
> No Triton kernel is implemented.  
> No vLLM integration is implemented.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_075_GENERATION_SHADOW_WIRING_REVIEW.md`](EXPERIMENT_075_GENERATION_SHADOW_WIRING_REVIEW.md) · `exactkv/attention/generation_shadow_observer.py`

---

## 1. Purpose

Phase 16K implements an external L1 generation-shadow observer wrapper that runs ExactKV generation unchanged, then post-hoc offline shadow replay/logit diagnostics.

---

## 2. Relation to Phase 16J

Phase 16J recommended `L1_generation_observer` as the safest next step. Phase 16K implements that level externally without modifying `ExactKVGenerator`.

---

## 3. Why this is an external observer

The wrapper calls `ExactKVGenerator.generate` through the public API, captures output, then runs Phase 16F-style offline replay separately. No generator internals are modified.

---

## 4. Generation safety gates

- `shadow_used_for_token_commit`: always **false**
- `generation_modified_by_shadow`: always **false**
- `default_runtime_changed`: always **false**
- Shadow runs only when `--generation-shadow-observer` is set

---

## 5. Shadow modes

| Mode | Description |
|---|---|
| `prompt_prefix_only` | **Default** — shadow replay on prompt token IDs only |
| `prompt_plus_generated_tokens` | Shadow on prompt + generated token IDs when available |
| `blocked_missing_tokens` | Automatic when token IDs cannot be reconstructed |

---

## 6. What the wrapper observes

- Streaming vs materialized hidden/logit drift (Phase 16F)
- Full vs streaming drift
- Top-k agreement (supplementary)
- Tolerance policy status (Phase 16I)

---

## 7. What the wrapper cannot affect

- Generated token IDs or text
- Draft/verify/commit loop
- Acceptance or rejection decisions
- Default CLI/runtime behavior

---

## 8. Results

```bash
python3 scripts/research/run_exp076_generation_shadow_observer_smoke.py --generation-shadow-observer
```

Report: `reports/experiment_076_generation_shadow_observer_smoke.json` (gitignored).

---

## 9. What this proves

- ExactKV can run generation and record post-hoc shadow diagnostics in one external flow
- Safety invariants (no token commit from shadow) are enforced in the report schema
- Phase 16F–16I utilities compose with generation output capture

---

## 10. What this does not prove

- Exact generation preservation or model-output preservation
- Production correctness of streaming attention in generation
- Per-round decode-step shadow during multi-round ExactKV loops
- Speed, throughput, latency, or GPU memory savings

---

## 11. Relation to ExactKV restored verification

Independent from restored-verifier experimental API. Shadow observer does not interact with `VerificationEngine` commit paths.

---

## 12. Relation to VeriCache parity

Does not reproduce VeriCache throughput, serving, or memory panels.

---

## 13. Next step

**Phase 16L (proposed):** per-round decode observer or optional 1.5B panel extension — still opt-in; still no `ExactKVGenerator` modification.

---

## Claims boundary

| Allowed | Forbidden |
|---|---|
| External post-hoc shadow observer | Generation integration |
| Diagnostic shadow metrics | Shadow as exactness proof |
| Unchanged generation output | Token commit from shadow |
