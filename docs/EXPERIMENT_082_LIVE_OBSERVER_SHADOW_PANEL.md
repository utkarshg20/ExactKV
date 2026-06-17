# Experiment 082: Live Observer + Post-Hoc Shadow Panel (Phase 16Q)

**Status:** live observer + post-hoc shadow panel — run `scripts/research/run_exp082_live_observer_shadow_panel.py --live-round-observer --generation-shadow-observer`.

> This is a **live observer plus post-hoc shadow panel**, not decode-time shadow integration.  
> ExactKV default generation remains unchanged.  
> Observer output is ignored and cannot affect token commits.  
> Shadow analysis runs after generation and cannot affect token commits.  
> Streaming attention is not wired into ExactKV generation.  
> No CUDA/Triton kernel is implemented.  
> No vLLM integration is implemented.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_081_LIVE_ROUND_OBSERVER_SMOKE.md`](EXPERIMENT_081_LIVE_ROUND_OBSERVER_SMOKE.md) · `exactkv/attention/live_round_observer.py`

---

## 1. Purpose

Phase 16Q combines Phase 16P live round observer instrumentation with post-hoc round-boundary shadow diagnostics. Live snapshots collected during generation are used as the shadow round source after generation completes.

---

## 2. Relation to Phase 16P

Phase 16P proved baseline-vs-observer parity and snapshot-vs-round-log agreement without running shadow. Phase 16Q adds mandatory post-hoc shadow using `live_round_observer` as the round source.

---

## 3. Why shadow is still post-hoc

Shadow replay runs only after observer-enabled generation completes. No streaming attention or shadow computation runs inside the token commit loop.

---

## 4. Baseline vs observer parity design

Each cell runs baseline generation (`round_observer=None`) then observer-enabled generation. Token IDs and text must match before shadow analysis proceeds.

---

## 5. Live snapshot collection

`LiveRoundObserver` records immutable `LiveRoundSnapshot` entries at post-commit round boundaries during observer-enabled generation.

---

## 6. Post-hoc shadow from live snapshots

`run_posthoc_shadow_from_live_snapshots` replays fixed sequences at each live snapshot boundary with Phase 16I tolerance policy. Missing live snapshots block the cell (no silent fallback).

---

## 7. Safety gates

Per cell: baseline/observer completion, token/text parity, `observer_used_for_token_commit=false`, `shadow_used_for_token_commit=false`, `generation_modified_by_observer=false`, `generation_modified_by_shadow=false`, `default_runtime_changed=false`, `observer_return_value_ignored=true`.

---

## 8. Results

```bash
python3 scripts/research/run_exp082_live_observer_shadow_panel.py \
  --live-round-observer --generation-shadow-observer
```

**Run summary (Qwen/Qwen2.5-0.5B, CPU, float32, 4 prompts × 4 compressors, max_new_tokens=8):**

| Metric | Value |
|--------|-------|
| Status | `diagnostic_complete` |
| Baseline generation successful | 16/16 |
| Observer generation successful | 16/16 |
| Baseline-vs-observer token match | 16/16 |
| Baseline-vs-observer text match | 16/16 |
| Live snapshots | 34 total |
| Observer exceptions | 0 |
| Snapshot-vs-result-round-log match | 16/16 |
| Post-hoc shadow successful | 34/34 |
| Post-hoc shadow blocked | 0 |
| ExactKV failures (baseline/observer) | 0 / 0 |
| Tolerance policy | `local_alignment_pass_free_running_accumulation` at all rounds |
| Top-1 agreement | 34/34 true, 0 false |
| First status change | 0 cells |
| First top-1 mismatch | 0 cells |

Report: `reports/experiment_082_live_observer_shadow_panel.json` (gitignored).

---

## 9. What this proves

- Live observer and post-hoc shadow can run together while preserving generation parity.
- Live snapshots suffice as a round source for post-hoc shadow diagnostics.
- Safety gates hold when both subsystems are active.

---

## 10. What this does not prove

- General exact generation preservation beyond tested parity cells.
- Decode-time shadow integration readiness.
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

**Phase 16R (proposed):** broader live-observer shadow panels or guarded decode-time shadow research — only with explicit approval.
