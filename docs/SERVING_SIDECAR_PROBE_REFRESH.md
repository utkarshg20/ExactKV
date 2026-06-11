# Serving Sidecar / Probe Refresh (V11 Phase 3)

**Status:** Complete — feasibility refresh + metadata-only sidecar probe (Experiment 017).
**Builds on:** [`SERVING_CONTEXT_FEASIBILITY.md`](SERVING_CONTEXT_FEASIBILITY.md) (V8 Phase A),
[`EXPERIMENT_007_SERVING_CONTEXT.md`](EXPERIMENT_007_SERVING_CONTEXT.md) (V8 Phase D),
[`SERVING_CACHE_LIFECYCLE_HARNESS.md`](SERVING_CACHE_LIFECYCLE_HARNESS.md).

> This is **not** production serving.
> This is **not** vLLM integration.
> This is **not** LMCache integration.
> This does **not** implement PagedAttention.
> This does **not** measure throughput, latency, speedup, runtime, tokens/sec,
> or active GPU memory.
> The **authoritative full-KV verifier remains separate**.
> ExactKV does **not** claim production readiness.
> External-paper serving results are **not** ExactKV results.

---

## 1. Purpose

Refresh the V8 serving-context feasibility conclusions after V9 (real-backend gauntlet),
V10 (suite hardening, Exp 012–014), and V11 (multi-model validation, Exp 015–016).
Determine whether a **safe metadata-only sidecar/probe** is viable, or whether the
correct V11 outcome is a **documented no-go refresh** for direct vLLM/LMCache integration.

---

## 2. What V8 concluded about vLLM and LMCache

Phase A ([`SERVING_CONTEXT_FEASIBILITY.md`](SERVING_CONTEXT_FEASIBILITY.md) §15–16):

| Stack | Phase C verdict | Core blocker |
|---|---|---|
| **vLLM** | **NO-GO** | No safe export of authoritative full-precision HF KV for per-step verification |
| **LMCache** | **NO-GO** | Tiering/async offload worsens synchronous authoritative KV availability |

Experiment **007** validated the **restricted local harness** (Mode B): 238 cells,
`exactkv_failures == 0`, all harness invariants passed. Direct integration was
explicitly deferred (`phase_c_status: no-go_deferred`).

---

## 3. Whether anything from V9/V10/V11 changes that conclusion

| V11-era work | Relevant to serving? | Changes V8 no-go? |
|---|---|---|
| Exp 008–010 real backends (TurboQuant/KIVI/KVQuant) | Draft-path adapters only | **No** — verify still on HF authoritative KV |
| Exp 011 1.5B validation | Scale | **No** — same HF runtime |
| Exp 012–014 V10 suites + spot-checks | Evaluation breadth | **No** — no serving stack touched |
| Exp 015–016 1.5B/3B V10 validation | Scale | **No** — RunPod HF path only |
| Exp 005 kvpress physical `<` logical | Harness precedent | **Supports** local probe; not vLLM export |

**Conclusion:** V9–V11 strengthen ExactKV's evaluation platform but **do not** provide
evidence that vLLM or LMCache can safely own authoritative full KV for verification.
The V8 no-go for **direct integration** stands.

What **does** change: V11 confirms that multi-model and expanded-suite work proceeds
entirely on the HF-centric path — reinforcing that serving integration is a **separate,
unvalidated** concern, not a side effect of scale validation.

---

## 4. What a safe sidecar/probe would mean

A **safe sidecar/probe** (Outcome A — restricted success):

1. **ExactKV verifier remains authoritative** — `VerificationEngine` uses full-precision
   `FullKVState` only; no serving stack mutates it.
2. **Sidecar is observational** — metadata-only lifecycle tracking via
   `ServingCacheLifecycleHarness` + `ServingSidecarProbe`.
3. **Compressed draft stays separate** — `compressed_draft` owner never replaces
   `authoritative_full`.
4. **Logical/physical mapping is explicit** — block tables, retained positions when
   physical `<` logical (kvpress precedent).
5. **No production-serving claim** — local compatibility evaluation only.

Experiment **017** implements this pattern: 32 cells (8 V10 prompts × 4 compressors),
`exactkv_failures == 0`, all sidecar probe invariants pass.

---

## 5. What remains unsafe

| Pattern | Status | Why |
|---|---|---|
| vLLM-owned authoritative KV for verify | **Unsafe** | Paged blocks ≠ HF `FullKVState`; reconstruction non-trivial |
| LMCache tier as authoritative store | **Unsafe** | Async restore, shared prefixes, compression-at-rest |
| Shared worker mutation (draft + vLLM decode) | **Unsafe** | Race on block table; exceeds kvpress hook-isolation lesson |
| Dual-runtime "verify on HF, draft on vLLM" | **Misleading** | Does not validate vLLM-owned verify path |
| Production serving scheduler integration | **Out of scope** | Batching, multi-request, sampling not in ExactKV |
| Performance measurement via sidecar | **Forbidden** | No throughput/latency/speedup/runtime fields |

---

## 6. What is allowed in V11

| Allowed | Deliverable |
|---|---|
| Feasibility refresh memo | This document |
| Metadata-only sidecar probe | `exactkv/serving/sidecar_probe.py` |
| Harness + probe experiment | Experiment 017 |
| Documented no-go reaffirmation | Exp 017 §4–5 |
| Additive probe metadata in reports | `sidecar_probe` cell field (strictly additive) |

---

## 7. What remains deferred to future work

| Item | Register | Notes |
|---|---|---|
| Direct vLLM integration | D11 no-go | Requires explicit scope revision |
| Direct LMCache integration | D12 no-go | Requires vLLM path first |
| PagedAttention kernel integration | D16 deferred | Out of V11 |
| Active GPU memory methodology | D14 / Exp 018 | Phase 4 — distinct from sidecar |
| Production TurboQuant/KIVI/KVQuant CUDA | D1–D4 | Factory-only; not serving |
| Multi-request batching / sampling | — | Not ExactKV scope |
| Read-only vLLM block-layout observation | V8 deferred | Metadata-only; not approved without re-scope |

---

## 8. Sidecar/probe architecture (implemented)

```
ExactKVGenerator (unchanged)
    │
    ├── VerificationEngine → authoritative FullKVState (deep-copy verify)
    ├── Draft path → CompressedKVState (materialize_for_draft)
    │
    └── ServingSidecarProbe (observational)
            └── ServingCacheLifecycleHarness
                    ├── authoritative_full entry (register only)
                    └── compressed_draft entry (register only)
```

**Invariants checked every commit round:**

- `verification_uses_authoritative_full`
- `owners_separate`
- `sidecar_observational_only`
- `compressed_draft_separate`
- `logical_alignment_maintained`

---

## 9. Go/no-go summary (V11 refresh)

| Candidate | V8 | V11 refresh | Experiment |
|---|---|---|---|
| Direct vLLM | NO-GO | **NO-GO reaffirmed** | 017 |
| Direct LMCache | NO-GO | **NO-GO reaffirmed** | 017 |
| Local harness | GO | **GO** (unchanged) | 007 |
| Metadata-only sidecar probe | Not implemented | **GO — implemented** | 017 |
| PagedAttention import | N/A | **Still N/A** | — |

**Overall V11 Phase 3 outcome:** **Outcome A + B combined** — restricted sidecar/probe
**succeeds**; direct vLLM/LMCache integration **remains no-go** with documented blockers.

---

## 10. Related documents

| Document | Role |
|---|---|
| [`EXPERIMENT_017_SERVING_SIDECAR_PROBE.md`](EXPERIMENT_017_SERVING_SIDECAR_PROBE.md) | Experiment report |
| [`EXPERIMENT_007_SERVING_CONTEXT.md`](EXPERIMENT_007_SERVING_CONTEXT.md) | V8 harness baseline |
| [`V11_SCOPE_STATEMENT.md`](V11_SCOPE_STATEMENT.md) | Phase 3 plan |
| [`V10_READINESS_ASSESSMENT.md`](V10_READINESS_ASSESSMENT.md) | D13 deferred to V11 |

## Attribution

- vLLM / PagedAttention: Kwon et al., arXiv:2309.06180, 2023
- LMCache: Liu et al., arXiv:2510.09665, 2025
- VeriCache / ExactKV: Yao et al., arXiv:2605.17613, 2026

External throughput and latency figures belong to those systems, **not** ExactKV.
