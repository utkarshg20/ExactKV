# Experiment 074: Attention Tolerance Policy Panel (Phase 16I)

**Status:** offline attention-diagnostics policy — run `scripts/research/run_exp074_attention_tolerance_policy_panel.py` for report.

> This is an **offline attention-diagnostics policy**, not model generation integration.  
> Streaming attention is **not wired into ExactKV generation**.  
> Depth-aware tolerance is **diagnostic only** and is **not** a production correctness guarantee.  
> Top-k agreement is supplementary and is **not** exactness.  
> No CUDA kernel is implemented.  
> No Triton kernel is implemented.  
> No vLLM integration is implemented.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_073_QWEN_FAMILY_DIVERGENCE_PANEL.md`](EXPERIMENT_073_QWEN_FAMILY_DIVERGENCE_PANEL.md) · `exactkv/attention/tolerance_policy.py`

---

## 1. Purpose

Phase 16I formalizes how ExactKV interprets offline streaming-vs-materialized attention results from Phases 16E–16H without overstating them.

---

## 2. Why tolerance policy is needed after Phases 16E–16H

Phases 16E–16H showed:

- strict tolerance fails at full depth while teacher-forced local alignment holds,
- free-running divergence accumulates,
- top-k agreement can remain perfect despite numeric drift.

A single pass/fail gate is insufficient; this phase documents layered interpretation rules.

---

## 3. Strict numeric tolerance

- Base strict tolerance: **`5e-4`** (Phase 16D gate)
- Applies to all metrics; required for `attention_context` and shallow-prefix hidden checks.

---

## 4. Depth-aware diagnostic tolerance

- Formula: **`5e-4 × sqrt(prefix_layers)`**
- Allowed for **hidden** and **logits** metrics only.
- **Diagnostic only** — not a production correctness guarantee.
- Strict failures remain visible when depth-aware passes.

---

## 5. Teacher-forced local alignment

When teacher-forced layer inputs are shared, streaming vs materialized attention/context errors should stay below **`1e-4`** attn / **`1e-3`** post-MLP for local alignment pass.

---

## 6. Free-running accumulation

When teacher-forced local alignment holds but free-running post-MLP errors grow with depth and root cause is `free_running_accumulation`, classify as cumulative hidden-state amplification — not a per-layer attention bug.

---

## 7. Top-k agreement as supplementary only

Top-1/top-5/top-10 agreement may be recorded when numeric drift exceeds strict tolerance. **Top-k never upgrades a cell to exactness** or exact generation preservation.

---

## 8. Optional 1.5B panel handling

`--include-optional-models` attempts `Qwen/Qwen2.5-1.5B` and `Qwen/Qwen2.5-1.5B-Instruct` with conservative settings (length 32, 1 prompt, chunks 16/64). Load failures are recorded as blocked entries without failing the phase.

---

## 9. Results

```bash
python3 scripts/research/run_exp074_attention_tolerance_policy_panel.py
```

Report: `reports/experiment_074_attention_tolerance_policy_panel.json` (gitignored).

Re-applies policy to existing Exp 070–073 reports when present; otherwise runs a synthetic mock panel.

**Local run (all four prior reports present):**

| Metric | Value |
|---|---|
| Reports loaded | exp070, exp071, exp072, exp073 |
| Cells evaluated | **112** |
| Strict numeric pass | **64** |
| Strict fail / depth-aware pass | **8** |
| Local alignment + free-running accumulation | **28** |
| Top-k agrees / numeric drift present | **6** |
| Local attention mismatch | **0** |
| Blocked | **0** |

---

## 10. What this proves

- ExactKV has a documented, tested offline interpretation policy for attention drift experiments.
- Strict vs depth-aware vs supplementary top-k roles are separated.
- Phase 16G/16H patterns can be classified consistently across reports.

---

## 11. What this does not prove

- Production correctness of streaming attention in ExactKV generation.
- Exact generation preservation or model-output preservation.
- Speed, throughput, latency, or GPU memory savings.
- VeriCache throughput or serving reproduction.

---

## 12. Relation to ExactKV restored verification

Policy module is research-only. Does not change `ExactKVGenerator` or default runtime.

---

## 13. Relation to VeriCache parity

Does not reproduce VeriCache throughput, serving, or memory panels.

---

## 14. Next step

**Phase 16J (complete):** [`EXPERIMENT_075_GENERATION_SHADOW_WIRING_REVIEW.md`](EXPERIMENT_075_GENERATION_SHADOW_WIRING_REVIEW.md) — generation-shadow wiring review.

**Phase 16K (complete):** [`EXPERIMENT_076_GENERATION_SHADOW_OBSERVER_SMOKE.md`](EXPERIMENT_076_GENERATION_SHADOW_OBSERVER_SMOKE.md) — external L1 generation-shadow observer.

**Phase 16L (proposed):** per-round decode observer — still no `ExactKVGenerator` modification.

---

## Claims boundary

| Allowed | Forbidden |
|---|---|
| Documented tolerance policy | Production correctness guarantee |
| Depth-aware diagnostic interpretation | Hiding strict failures |
| Supplementary top-k recording | Top-k as exactness proof |
| Theoretical memory accounting | Measured active GPU memory savings |
