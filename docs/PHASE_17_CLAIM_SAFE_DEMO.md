# Phase 17: Claim-Safe Demo

**Status:** Phase 17A complete — see [`PHASE_17_CLAIM_SAFE_DEMO.md`](PHASE_17_CLAIM_SAFE_DEMO.md) and [`EXPERIMENT_086_CLAIM_SAFE_DEMO_PACKAGING.md`](EXPERIMENT_086_CLAIM_SAFE_DEMO_PACKAGING.md).

> This is claim-safe demo packaging, not new runtime functionality.  
> ExactKV has guarded diagnostic shadow infrastructure, not streaming-attention token-commit integration.  
> Guarded decode-time shadow was tested as diagnostic-only observer work.  
> Shadow output cannot affect token commits in the tested path.  
> Top-k agreement is supplementary only and is not an exactness guarantee.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

---

## Demo purpose

Package Phase 16 evidence into a clear, reusable story for demos, README hero copy, and technical walkthroughs — without overstating performance, memory, serving, or VeriCache claims.

---

## Hook

**Everyone is racing to shrink KV caches. ExactKV tells you when they start lying.**

---

## Problem

**KV compression should not be trusted. It should be crash-tested.**

---

## Why benchmark outcomes are insufficient

Outcome benchmarks can tell you whether the answer scored well. ExactKV tells you whether KV compression changed the model's path before the answer looked fine.

---

## What ExactKV checks

- Offline streaming-attention drift on Qwen2.5 probes
- Diagnostic tolerance policy (local alignment vs free-running accumulation)
- External and round-log generation-shadow replay
- Opt-in live round snapshots with baseline parity
- Guarded decode-time shadow dry-run (diagnostic-only callback)

---

## Phase 16 evidence highlights

| Track | Highlight |
|-------|-----------|
| Attention drift | Qwen-family divergence panel; multi-layer numerics audit |
| Tolerance policy | Exp 074 depth-aware policy labels |
| Generation shadow | Prefix ladder + expanded panels (post-hoc) |
| Live observer | Exp 082 parity + post-hoc shadow from snapshots |
| Guarded decode-time | Exp 084: 32/32 parity, 53/53 callbacks |

---

## Demo cards

Six reusable cards: `attention_drift_card`, `tolerance_policy_card`, `external_generation_shadow_card`, `live_round_observer_card`, `guarded_decode_shadow_card`, `claim_freeze_card`.

Built by `exactkv/demo/phase17_claim_safe_demo.py`; report: `reports/experiment_086_claim_safe_demo_packaging.json`.

---

## Allowed claims

See Phase 16 claim freeze ([`PHASE_16_CLOSEOUT.md`](PHASE_16_CLOSEOUT.md)): offline diagnostics, Qwen probes, tolerance policy, shadow observers, tested-panel parity, zero exactkv_failures on cited panels.

---

## Forbidden claims

Speed, throughput, latency, active GPU/production memory savings, VeriCache serving/throughput reproduction, streaming attention in token commit, shadow/top-k as exactness guarantees, production-ready.

---

## Deferred work

CUDA/Triton kernels, vLLM, LMCache, measured active GPU memory savings, production serving, broader model/context validation, real compressed-attention token-commit path.

---

## Recommended next step

**Phase 17B (complete):** broader model validation — see [`PHASE_17B_BROADER_MODEL_VALIDATION.md`](PHASE_17B_BROADER_MODEL_VALIDATION.md).

**Phase 17C (complete):** longer-context validation — see [`PHASE_17C_LONG_CONTEXT_VALIDATION.md`](PHASE_17C_LONG_CONTEXT_VALIDATION.md).

**Phase 17D (complete):** integration design review — see [`PHASE_17D_INTEGRATION_DESIGN_REVIEW.md`](PHASE_17D_INTEGRATION_DESIGN_REVIEW.md).

**Phase 18A (proposed):** integration safety spec — explicit approval required.

Scripts: [`PHASE_17_DEMO_SCRIPT.md`](PHASE_17_DEMO_SCRIPT.md)
