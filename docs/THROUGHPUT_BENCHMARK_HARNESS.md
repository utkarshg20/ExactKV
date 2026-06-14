# Throughput Benchmark Harness (Phase 11I)

**Status:** Benchmarking methodology and contract layer — **not a throughput result.**

> This is a **benchmarking methodology and contract layer**, not a throughput result.  
> **Current ExactKV timing diagnostics do not support a speedup claim.**  
> No speedup, latency improvement, throughput improvement, memory savings, or production-serving claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput results.  
> Future positive throughput claims require exactness gates and reproducible baseline comparisons.

Companion: [`VERICACHE_SYSTEMS_ROADMAP.md`](VERICACHE_SYSTEMS_ROADMAP.md) · [`EXPERIMENT_030_DIAGNOSTIC_TIMING.md`](EXPERIMENT_030_DIAGNOSTIC_TIMING.md) · `exactkv/benchmarks/throughput_contract.py` · `exactkv/metrics/timing.py`

---

## 1. Why throughput benchmarking is required for VeriCache parity

VeriCache reports **end-to-end tokens/sec** and memory panels after verify overhead. ExactKV V13 prioritized **exactness first** (Exp 029 grid). Stage 8 defines **what must be measured** before any throughput, latency, or speedup language is allowed — without claiming VeriCache throughput reproduction today.

---

## 2. Why current ExactKV does not claim speedup

Experiment **030** (diagnostic timing on a 20-prompt V10 panel, fp16 GPU) found:

| Arm | Mean tok/s (Exp 030) |
|---|---:|
| `full_greedy` | **54.4** |
| `exactkv_sequential` | **20.4** |
| `exactkv_span` | **18.4** |

ExactKV sequential throughput was **lower** than full greedy on that panel. These numbers are **diagnostic only** — panel-bound, hardware-specific, and **not** a benefit claim.

---

## 3. What is implemented now

| Component | Purpose |
|---|---|
| `ThroughputBenchmarkPlan` | Methodology metadata: mode, metrics, gates, claims |
| `ThroughputDiagnosticResult` | JSON schema for diagnostic timing rows |
| Validators | Block speed claims unless `CLAIM_ALLOWED` |
| `build_default_diagnostic_plan()` | Default `DIAGNOSTIC_ONLY` offline single-request plan |
| `scripts/diagnose_throughput_methodology.py` | Checklist + optional JSON stub (no inference) |
| `exactkv/metrics/timing.py` | Exp 030 helpers (`timed_call`, `summarize_trials`) — unchanged |

**Not wired** into public leaderboard or `ExactKVGenerator`.

---

## 4. What must be measured

| Requirement | Gate |
|---|---|
| **Exactness** | `exactkv_failures == 0` on timed panel before interpreting timing |
| **Warmup** | Discard initial runs before measured trials |
| **Synchronization** | CUDA sync when timing GPU paths |
| **Baseline** | Compare candidate vs named baseline arm (e.g. `full_greedy`) |
| **Named metrics** | `TOKENS_PER_SECOND`, `TOTAL_SECONDS`, `VERIFY_SECONDS`, `DECODE_SECONDS`, … |
| **Samples** | `sample_count >= 3` for claim-ready panels |
| **Hardware** | GPU name, dtype, torch, transformers versions |
| **Honesty** | `hide_negative_results` must stay `False` |

---

## 5. Exactness gate before performance claim

Timing is **invalid** if exactness fails. Exp 029 enabled `phase3_timing_allowed`; Exp 030 enforced the gate. `ThroughputClaimStatus.CLAIM_ALLOWED` requires `exactness_gate_required=True` and `exactness_passed=True` on results.

---

## 6. Warmup and synchronization

| Field | Default (contract) |
|---|---|
| `warmup_required` | `True` |
| `synchronization_required` | `True` |

Matches Exp 030 methodology (2 warmup, 3 measured trials, CUDA synchronize).

---

## 7. Baseline comparison requirements

| Field | Requirement |
|---|---|
| `baseline_arm` | Named on plan (e.g. `full_greedy`) |
| `baseline_name` / `candidate_name` | On each `ThroughputDiagnosticResult` |
| Ratio interpretation | Panel-bound; no universal speedup claim |

`CLAIM_ALLOWED` fails validation without baseline arm and hardware metadata.

---

## 8. Placeholder modes (not runtime)

| `ThroughputBenchmarkMode` | Phase 11I status |
|---|---|
| `OFFLINE_SINGLE_REQUEST` | Only executable diagnostic context today |
| `BATCHED_PLACEHOLDER` | Metadata only — no batching runtime |
| `SERVING_PLACEHOLDER` | Metadata only — no serving runtime |
| `REMOTE_PREFIX_PLACEHOLDER` | Metadata only — no remote prefix runtime |

`runtime_placeholder_active` must remain `False`.

---

## 9. Why this is not vLLM / LMCache / serving

- No vLLM or LMCache import or integration
- No remote prefix network runtime (Phase 11H loopback only)
- No multi-request serving or continuous batching
- Contract layer only — Exp 030 remains the cited diagnostic artifact

---

## 10. How Stage 9 builds on this

| Stage | Connection |
|---|---|
| **Stage 9** — Paper-like panel | Fixed compressor × model × benchmark panel with methodology sign-off |
| **Claim upgrade path** | `DIAGNOSTIC_ONLY` → `NEGATIVE_OR_NEUTRAL` → (future) `CLAIM_ALLOWED` only after gates |
| **VeriCache parity RC** | Stage 10 checklist — still not automatic speedup claim |

---

## 11. JSON schema (plan)

```json
{
  "mode": "OFFLINE_SINGLE_REQUEST",
  "metrics_required": ["TOKENS_PER_SECOND", "TOTAL_SECONDS", "VERIFY_SECONDS"],
  "exactness_gate_required": true,
  "warmup_required": true,
  "synchronization_required": true,
  "sample_count": 3,
  "baseline_arm": "full_greedy",
  "claim_status": "DIAGNOSTIC_ONLY",
  "claim_note": "..."
}
```

---

## 12. Claims boundary

| Allowed | Forbidden |
|---|---|
| Methodology contract exists | Speedup / throughput improvement |
| Panel-bound diagnostic numbers with exactness cite | Latency improvement claim |
| Negative results reported (Exp 030) | Hiding slower-than-baseline timing |
| Exactness gate documented | Production serving readiness |
| | VeriCache throughput reproduction |
| | Memory savings from timing harness |
