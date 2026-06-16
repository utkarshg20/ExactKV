# Experiment 075: Generation-Shadow Wiring Review (Phase 16J)

**Status:** generation-shadow wiring review — run `scripts/research/run_exp075_generation_shadow_wiring_review.py` for report.

> This is a **generation-shadow wiring review**, not generation integration.  
> Streaming attention is **not wired into ExactKV generation**.  
> Future shadow mode must be **opt-in** and must **not affect generated tokens**.  
> Shadow logits/top-k are **diagnostic only** and are **not** exactness guarantees.  
> No CUDA kernel is implemented.  
> No Triton kernel is implemented.  
> No vLLM integration is implemented.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_074_ATTENTION_TOLERANCE_POLICY_PANEL.md`](EXPERIMENT_074_ATTENTION_TOLERANCE_POLICY_PANEL.md) · `exactkv/attention/generation_shadow_review.py`

---

## 1. Purpose

Phase 16J reviews ExactKV hook points and defines staged shadow-mode levels (L0–L4) plus safety gates for a future opt-in generation observer — without implementing generation integration.

---

## 2. Why this follows Phase 16I

Phase 16I formalized how to interpret offline attention drift. Before broader model claims or runtime wiring, ExactKV needs a concrete, claim-safe plan for observing streaming-vs-materialized drift **beside** generation without affecting tokens.

---

## 3. Shadow-mode levels L0–L4

| Level | ID | Status |
|---|---|---|
| L0 | `L0_offline_replay` | **Implemented** (Phases 16F–16H) |
| L1 | `L1_generation_observer` | **Not implemented** — **recommended next** |
| L2 | `L2_draft_shadow` | Not implemented |
| L3 | `L3_restored_verifier_shadow` | Not implemented |
| L4 | `L4_runtime_integration` | **Forbidden for now** |

---

## 4. Hook-point review

**Prompt entry:** `ExactKVGenerator.generate`, `prefill_to_full_state`, `ModelRuntime.encode`

**Generation output:** `ExactKVResult.output_ids`, `output_text`

**Logits today:** `ModelRuntime.forward` exposes logits; generator does not persist per-step shadow logits

**Proposed L1 wrapper (future):**
1. Run `ExactKVGenerator.generate` unchanged
2. Reconstruct prefix `input_ids`
3. Run offline 16F–16G replay/trace
4. Apply 16I tolerance policy
5. Emit separate shadow report

**Future opt-in flag (not added in 16J):** `--generation-shadow-observer`

Precedent: `exactkv/runtime/experimental_cli.py` (`--experimental-restored-verifier`)

---

## 5. Recommended next safe level

**`L1_generation_observer`** — external wrapper only; generated tokens unaffected; shadow diagnostics after generation.

---

## 6. Safety gates

- Opt-in only
- Default runtime unchanged
- Generated tokens unaffected
- No streaming result used for token commit
- No speed/memory claims
- Shadow output diagnostic only
- Exactness judged against full generation/verifier

---

## 7. Allowed claims

- Wiring review completed
- L1 identified as safest next step
- Default generation unchanged
- Shadow would be diagnostic only
- Tolerance policy applies to shadow metrics

---

## 8. Forbidden claims

- Exact generation / model-output preservation
- Shadow exactness guarantee
- Throughput, latency, speedup, GPU memory savings
- Default runtime integration (L4)
- VeriCache throughput reproduction

---

## 9. Blockers

- Phase 16J is review-only — no generator hooks added
- Streaming attention not in `ExactKVGenerator`
- L1 prefix-shadow does not cover per-round decode drift
- Qwen2.5-centric offline probes

---

## 10. What this proves

- ExactKV has inspected hook points and a staged shadow roadmap
- L1 external observer is the safest next engineering step
- Safety gates and forbidden claims are documented before any wiring

---

## 11. What this does not prove

- Production correctness of streaming attention in generation
- That shadow top-k implies exactness
- That draft or verifier paths are ready for streaming shadow (L2/L3)

---

## 12. Relation to ExactKV restored verification

Restored verifier experimental API (`restored_verifier_runner`, `--experimental-restored-verifier`) is a separate track. L3 would interact in the future; not implemented in 16J.

---

## 13. Relation to VeriCache parity

Does not reproduce VeriCache throughput, serving, or memory panels.

---

## 14. Next step

**Phase 16K (complete):** [`EXPERIMENT_076_GENERATION_SHADOW_OBSERVER_SMOKE.md`](EXPERIMENT_076_GENERATION_SHADOW_OBSERVER_SMOKE.md) — external L1 generation-shadow observer.

**Phase 16L (proposed):** per-round decode observer or optional 1.5B panel — still opt-in; still no `ExactKVGenerator` modification.

---

## Claims boundary

| Allowed | Forbidden |
|---|---|
| Shadow wiring review | Generation integration |
| L1 observer plan | Altering generated tokens |
| Diagnostic shadow metrics | Shadow as exactness proof |
| Opt-in flag specification | Default-on shadow mode |
