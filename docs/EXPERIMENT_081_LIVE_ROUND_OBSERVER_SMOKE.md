# Experiment 081: Live Round Observer Smoke (Phase 16P)

**Status:** opt-in live observer smoke — run `scripts/research/run_exp081_live_round_observer_smoke.py --live-round-observer`.

> This is **opt-in live observer instrumentation**, not streaming-attention generation integration.  
> ExactKV default generation remains unchanged.  
> Observer output is ignored and cannot affect token commits.  
> Live snapshots are diagnostic only.  
> Streaming attention is not wired into ExactKV generation.  
> No live per-round decode hooks that affect commits are implemented.  
> No CUDA/Triton kernel is implemented.  
> No vLLM integration is implemented.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_080_ROUND_LOG_SHADOW_OBSERVER.md`](EXPERIMENT_080_ROUND_LOG_SHADOW_OBSERVER.md) · `exactkv/attention/live_round_observer.py`

---

## 1. Purpose

Phase 16P adds disabled-by-default `ExactKVGenerator` instrumentation that records immutable live round snapshots at post-commit boundaries, then proves baseline-vs-observer generation parity on the same prompts/compressors/settings.

---

## 2. Relation to Phase 16O

Phase 16O used existing `ExactKVResult.traces` post-hoc. Phase 16P exposes the same round boundaries live during generation via an opt-in observer without changing token commits.

---

## 3. Why this is opt-in instrumentation

Default `round_observer=None` preserves the prior generation path. Instrumentation runs only when a `LiveRoundObserver` is attached explicitly (CLI `--live-round-observer`).

---

## 4. ExactKVGenerator change summary

- Added optional `round_observer` constructor argument (default `None`).
- After each round commit, `_notify_round_observer` builds an immutable `LiveRoundSnapshot` and calls `observer.observe()`.
- Observer exceptions are captured in observer state; generation continues unchanged.

---

## 5. Immutable snapshot design

`LiveRoundSnapshot` uses tuples for token IDs and frozen dataclass fields. Callbacks receive read-only snapshots and cannot mutate generator buffers through the snapshot API.

---

## 6. Observer safety gates

- `observer_used_for_token_commit = false`
- `generation_modified_by_observer = false`
- `default_runtime_changed = false`
- `observer_return_value_ignored = true`

---

## 7. Baseline vs observer parity smoke

Exp 081 runs each cell twice:

1. Baseline generation (`round_observer=None`)
2. Observer-enabled generation (`LiveRoundObserver` attached)

Compares generated token IDs, text, and exactness fields. Mismatches fail the report.

---

## 8. Snapshot vs existing round-log comparison

Live snapshots are compared to `ExactKVResult.traces` from the observer-enabled run. Count and per-round fields must agree.

---

## 9. Optional post-hoc shadow analysis

With `--generation-shadow-observer`, round-log shadow diagnostics run after generation using live-snapshot-derived round metadata. Shadow never runs during token commit.

---

## 10. Results

```bash
python3 scripts/research/run_exp081_live_round_observer_smoke.py --live-round-observer
# optional post-hoc shadow:
python3 scripts/research/run_exp081_live_round_observer_smoke.py --live-round-observer --generation-shadow-observer
```

Report: `reports/experiment_081_live_round_observer_smoke.json` (gitignored).

---

## 11. What this proves

- Live round snapshots can be recorded without changing generated tokens (when parity holds).
- Live snapshots agree with existing `ExactKVResult` round logs.
- Observer failures/exceptions do not block generation.

---

## 12. What this does not prove

- General exact generation preservation beyond tested parity cells.
- Causal links between observer diagnostics and acceptance decisions.
- Streaming-attention or live decode integration readiness.
- Speed, throughput, latency, serving, or GPU memory claims.
- VeriCache throughput or serving reproduction.

---

## 13. Relation to ExactKV restored verification

Independent from restored-verifier experimental runtime.

---

## 14. Relation to VeriCache parity

Does not reproduce VeriCache throughput, serving, or memory panels.

---

## 15. Next step

**Phase 16Q (proposed):** broader live-observer panels or guarded decode-time shadow wiring — only with explicit approval.
