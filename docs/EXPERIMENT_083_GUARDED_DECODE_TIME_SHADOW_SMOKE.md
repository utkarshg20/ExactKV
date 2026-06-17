# Experiment 083: Guarded Decode-Time Shadow Observer Smoke (Phase 16R)

**Status:** guarded decode-time shadow observer dry-run — run `scripts/research/run_exp083_guarded_decode_time_shadow_smoke.py --guarded-decode-time-shadow`.

> This is a **guarded decode-time shadow observer dry-run**, not streaming-attention generation integration.  
> ExactKV default generation remains unchanged.  
> Shadow diagnostics run inside an opt-in observer callback only.  
> Shadow results are diagnostic only and cannot affect token commits.  
> Observer return values are ignored.  
> Shadow exceptions are captured and cannot alter generated tokens.  
> Streaming attention is not used for token commit.  
> No CUDA/Triton kernel is implemented.  
> No vLLM integration is implemented.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_082_LIVE_OBSERVER_SHADOW_PANEL.md`](EXPERIMENT_082_LIVE_OBSERVER_SHADOW_PANEL.md) · `exactkv/attention/decode_time_shadow_observer.py`

---

## 1. Purpose

Phase 16R implements a guarded decode-time shadow observer dry-run. Shadow diagnostics execute inside the live round observer callback immediately after post-commit snapshots are emitted, while generation parity is verified against a baseline run without the observer.

---

## 2. Relation to Phase 16Q

Phase 16Q ran post-hoc shadow after generation using live snapshots as the round source. Phase 16R moves shadow execution into the observer callback (still diagnostic-only) and compares callback-time results with post-hoc replay on the same snapshots.

---

## 3. Why this is guarded decode-time shadow dry-run

`GuardedDecodeTimeShadowObserver` conforms to the Phase 16P callback protocol. Shadow runs only after snapshots are validated as post-commit. Results are stored in observer-owned state and never returned to the generator for decisions.

---

## 4. Why this is not generation integration

Streaming attention is not wired into token commit. Shadow logits are not used for argmax, acceptance, or verifier decisions. This is observer instrumentation only.

---

## 5. Observer callback safety design

- Snapshots are recorded immutably.
- Shadow diagnostics run inside `observe()` after post-commit validation.
- Exceptions are captured in observer state.
- Return values are ignored by `ExactKVGenerator`.
- `shadow_result_used_for_token_commit=false` on every decode-time shadow cell.

---

## 6. Post-commit snapshot requirement

`snapshot_is_post_commit()` requires `full_seq_len_before` / `full_seq_len_after` metadata matching prefix lengths. If timing cannot be identified as post-commit, the cell is blocked and decode-time shadow safety is not claimed.

---

## 7. Baseline vs guarded-shadow parity smoke

Each cell runs baseline generation (`round_observer=None`) then guarded decode-time shadow observer generation with identical model, prompt, compressor, and `max_new_tokens`. Token IDs, text, and exactkv failure counts are compared.

---

## 8. Decode-time vs post-hoc shadow comparison

After generation, post-hoc shadow replays the same live snapshots. Decode-time and post-hoc cells are compared on `shadow_status`, `tolerance_policy_status`, and top-1 agreement.

---

## 9. Results

```bash
python3 scripts/research/run_exp083_guarded_decode_time_shadow_smoke.py \
  --guarded-decode-time-shadow
```

**Run summary (Qwen/Qwen2.5-0.5B, CPU, float32, 2 prompts × 2 compressors, max_new_tokens=8):**

| Metric | Value |
|--------|-------|
| Status | `diagnostic_complete` |
| Baseline generation successful | 4/4 |
| Guarded-shadow generation successful | 4/4 |
| Baseline-vs-guarded token match | 4/4 |
| Baseline-vs-guarded text match | 4/4 |
| Decode-time shadow callbacks | 8 total, 8 successful, 0 exceptions |
| Decode-time vs post-hoc shadow match | 4/4 cells |
| ExactKV failures (baseline/guarded) | 0 / 0 |

Report: `reports/experiment_083_guarded_decode_time_shadow_smoke.json` (gitignored).

---

## 10. What this proves

- Shadow diagnostics can execute during the live observer callback while preserving baseline-vs-guarded generation parity.
- Decode-time shadow results match post-hoc shadow on comparable fields for tested cells.
- Safety gates hold: shadow cannot affect token commits or generator state.

---

## 11. What this does not prove

- General exact generation preservation beyond tested parity cells.
- Streaming-attention generation integration readiness.
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

**Phase 16S (complete):** expanded guarded decode-time shadow panel — see [`EXPERIMENT_084_GUARDED_DECODE_TIME_SHADOW_PANEL.md`](EXPERIMENT_084_GUARDED_DECODE_TIME_SHADOW_PANEL.md).

**Phase 16T (proposed):** deeper integration research only with explicit approval.
