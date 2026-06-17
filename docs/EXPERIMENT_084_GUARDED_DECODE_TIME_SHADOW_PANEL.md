# Experiment 084: Expanded Guarded Decode-Time Shadow Panel (Phase 16S)

**Status:** expanded guarded decode-time shadow panel — run `scripts/research/run_exp084_guarded_decode_time_shadow_panel.py --guarded-decode-time-shadow`.

> This is an **expanded guarded decode-time shadow panel**, not streaming-attention generation integration.  
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

Companion: [`EXPERIMENT_083_GUARDED_DECODE_TIME_SHADOW_SMOKE.md`](EXPERIMENT_083_GUARDED_DECODE_TIME_SHADOW_SMOKE.md) · `exactkv/attention/decode_time_shadow_observer.py`

---

## 1. Purpose

Phase 16S expands the Phase 16R guarded decode-time shadow dry-run across more prompts, compressors, and `max_new_tokens` values while verifying baseline-vs-guarded parity and decode-time vs post-hoc shadow agreement.

---

## 2. Relation to Phase 16R

Phase 16R validated callback-time shadow on a small 2×2 smoke panel. Phase 16S runs the same guarded observer pattern on a 4×4×2 panel (32 cells default).

---

## 3. Why this is still diagnostic-only

Shadow executes inside the observer callback after post-commit snapshots. Results are never exposed to the generator and cannot affect token commits.

---

## 4. Panel dimensions

Default: 4 prompts × 4 compressors (`noop`, `int8`, `int4_sim`, `k8_v4_sim`) × 2 `max_new_tokens` values (4, 8) = **32 cells**.

---

## 5. Baseline vs guarded-shadow parity design

Each cell runs baseline generation then guarded `GuardedDecodeTimeShadowObserver` generation with identical settings. Token IDs, text, and exactkv failure counts are compared.

---

## 6. Decode-time callback diagnostics

Per callback: `round_index`, `shadow_sequence_length`, `shadow_status`, `tolerance_policy_status`, top-k agreement, exception info, `posthoc_shadow_match`, interpretation note. No timing fields recorded.

---

## 7. Decode-time vs post-hoc comparison

After generation, post-hoc shadow replays the same live snapshots. Decode-time and post-hoc cells are compared on `shadow_status`, `tolerance_policy_status`, and top-1 agreement.

---

## 8. Safety gates

Per cell: `decode_time_shadow_used_for_token_commit=false`, `generation_modified_by_decode_time_shadow=false`, `default_runtime_changed=false`, `observer_return_value_ignored=true`, `shadow_exception_affects_generation=false`, `shadow_result_exposed_to_generator=false`.

---

## 9. Results

```bash
python3 scripts/research/run_exp084_guarded_decode_time_shadow_panel.py \
  --guarded-decode-time-shadow
```

**Run summary (Qwen/Qwen2.5-0.5B, CPU, float32, 4 prompts × 4 compressors × 2 max_new_tokens, default panel):**

| Metric | Value |
|--------|-------|
| Status | `diagnostic_complete` |
| Total cells | 32 |
| Baseline generation successful | 32/32 |
| Guarded-shadow generation successful | 32/32 |
| Baseline-vs-guarded token match | 32/32 |
| Baseline-vs-guarded text match | 32/32 |
| Decode-time shadow callbacks | 53 total, 53 successful, 0 exceptions |
| Decode-time vs post-hoc shadow match | 32/32 cells |
| ExactKV failures (baseline/guarded) | 0 / 0 |
| Safety gates | 32/32 cells all gates OK |

Report: `reports/experiment_084_guarded_decode_time_shadow_panel.json` (gitignored).

---

## 10. What this proves

- Guarded decode-time shadow remains safe across a broader ExactKV panel.
- Baseline-vs-guarded parity holds across prompts, compressors, and generation lengths.
- Decode-time shadow results remain consistent with post-hoc replay.

---

## 11. What this does not prove

- General exact generation preservation beyond tested cells.
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

**Phase 16T (proposed):** deeper integration research only with explicit approval.
