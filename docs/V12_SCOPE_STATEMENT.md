# V12 Scope Statement — Deferred Work Completion Gauntlet

**Status:** **Phase 1b complete** — TurboQuant toolchain prep on RunPod; Phase 2 (Exp 022) next.
**Builds on:** `v0.11.0` — V11 complete (Experiments 015–020; launch package prepared).
**Not public launch.** v1.0.0 deferred until V12 substance and exit criteria are met or honestly closed.

> V12 is a **deferred-work completion gauntlet** — not a performance benchmark, not
> production serving integration, and not a public launch.
> V12 preserves the exactness gate: `exactkv_failures == 0` on every published experiment.
> V12 must **finish or conclusively close** major deferred technical tracks before public launch.
> No throughput, latency, tokens/sec, speedup, `runtime_seconds`,
> `active_gpu_kv_bytes`, or production-serving claims — unless Experiment 027 approves
> an explicit, caveated methodology and still does not imply production readiness.
> If measurement is not robust, **document deferral** instead of claiming it.

---

## 1. Status

| Phase | Focus | Status |
|---|---|---|
| **0** | Formal scope statement (this document) | **Complete** |
| **1** | TurboQuant production-fidelity feasibility (Exp 021) | **Complete** — [`TURBOQUANT_PRODUCTION_FIDELITY_FEASIBILITY.md`](TURBOQUANT_PRODUCTION_FIDELITY_FEASIBILITY.md) |
| **1b** | TurboQuant toolchain prep (build, GGUF, smoke) | **Complete** — [`TURBOQUANT_PRODUCTION_TOOLCHAIN_PREP.md`](TURBOQUANT_PRODUCTION_TOOLCHAIN_PREP.md) |
| **2** | TurboQuant llama.cpp / GGUF probe or documented no-go (Exp 022) | Planned |
| **3** | KVQuant larger-model real-backend validation (Exp 023) | Planned |
| **4** | KIVI CUDA/Triton packed-path feasibility or documented no-go (Exp 024) | Planned |
| **5** | Full-suite repair-policy validation (Exp 025) | Planned |
| **6** | True attention logging feasibility or documented no-go (Exp 026) | Planned |
| **7** | Performance/memory truth boundary review (Exp 027) | Planned |
| **8** | V12 release package and public-launch decision | Planned |

**Latest release:** `v0.11.0`. **V11 exit docs:**
[`V11_LAUNCH_READINESS.md`](V11_LAUNCH_READINESS.md),
[`PROJECT_STATUS_V0.11.0.md`](PROJECT_STATUS_V0.11.0.md),
[`RELEASE_NOTES_V0.11.0.md`](RELEASE_NOTES_V0.11.0.md).

**Do not start Phase 1 until this scope is reviewed and committed.**

---

## 2. Why V12 is needed after V11

V11 closed scale, serving-context, profiling, forensics, and launch-documentation
gaps for an **internal** `v0.11.0` milestone. ExactKV is impressive as a
correctness-first KV-compression evaluation framework, but **not yet ready for public
launch** because major deferred tracks remain unfinished or only partially closed:

| Gap | Evidence |
|---|---|
| **No speed claim** | ExactKV never measured or claimed throughput, latency, or speedup |
| **No active GPU memory savings claim** | Exp 018 pilot documented methodology; `active_gpu_kv_bytes` not in standard schema |
| **No production serving claim** | Exp 017 sidecar pass; direct vLLM/LMCache **no-go** reaffirmed |
| **TurboQuant Python ≠ production TurboQuant** | Exp 008 evaluated restricted Python adapter only; llama.cpp / GGUF / MLX deferred (D2) |
| **KIVI offline ≠ KIVI CUDA/Triton** | Exp 009 evaluated offline simulate path only; packed CUDA/Triton path untested |
| **KVQuant at 0.5B only** | Exp 010 on 0.5B; larger-model real-backend validation unfinished (1.5B/3B) |
| **Repair policies pilot-scale only** | Exp 020 on 25-prompt panel; not validated on full 128-prompt V10 suites |
| **True attention logging deferred** | Exp 019 blocked by sdpa `output_attentions`; D7 partial only |
| **Launch package not final** | D17 physical bundle optional; D18 narrative is draft only; D19/D20 v1.0.0 docs unpublished |

V12 explicitly collects, prioritizes, and defines a path to **finish or conclusively
close** these tracks — producing the highest-quality launch-worthy ExactKV result
without overclaiming.

---

## 3. What V11 proved

- **Exactness gate** holds through **Qwen2.5-3B** on full V10 suites (Exp 015–016).
- **V10 findings transfer** to 1.5B; scale story strengthened at 3B with shrinking layer-aware margin.
- **Serving sidecar probe** passes; direct vLLM/LMCache integration remains **no-go** (Exp 017).
- **GPU memory methodology** pilot succeeds; V5 `total_kv_footprint_bytes` remains primary memory story (Exp 018).
- **Divergence autopsy** localizes failure modes via logit margins, token types, and per-layer KV error (Exp 019).
- **Autopsy-guided repair policies** improve draft acceptance on a pilot panel while preserving exactness (Exp 020).
- **Launch documentation package** prepared: readiness assessment, artifact policy, narrative draft.

**Hard gate:** `exactkv_failures == 0` on Experiments 015–020.

---

## 4. What V11 did not prove

- Speed, throughput, latency, runtime, or tokens/sec improvement.
- Active GPU memory savings in production.
- Production serving readiness or vLLM/LMCache integration.
- TurboQuant llama.cpp / GGUF / production-fidelity equivalence to Python adapter results.
- KIVI CUDA/Triton packed-path draft usefulness.
- KVQuant simquant behaviour at 1.5B or 3B on V10 suites.
- Repair-policy survival at full V10-suite and multi-model scale.
- True attention-weight forensics (per-head divergence).
- Final public launch narrative or curated artifact bundle.
- Model accuracy improvement from compression.
- Universal benchmark coverage.

---

## 5. What remains deferred

| ID / track | Status after V11 | V12 target |
|---|---|---|
| D2 | TurboQuant llama.cpp / MLX / production-fidelity | Phases 1–2 / Exp 021–022 |
| D3 (CUDA path) | KIVI offline only (Exp 009) | Phase 4 / Exp 024 |
| D4 (larger model) | KVQuant simquant 0.5B only (Exp 010) | Phase 3 / Exp 023 |
| D7 | True attention logging — sdpa blocker | Phase 6 / Exp 026 |
| D8 | Per-head forensics — partial (per-layer KV only) | Phase 6 / Exp 026 |
| Exp 020 policies | Pilot on 25 prompts only | Phase 5 / Exp 025 |
| D14 (truth boundary) | Methodology pilot only; no savings claim | Phase 7 / Exp 027 |
| D17 | Policy complete; physical bundle optional | Phase 8 |
| D18 | Draft complete; not approved for posting | Phase 8 |
| D11/D12 | vLLM / LMCache direct integration **no-go** | Remain deferred unless scope changes |
| D16 | PagedAttention kernel integration | Out of V12 scope |
| D6 | Sparse V dequantization | Out of V12 scope unless explicitly approved |
| D9/D10 | Pre-RoPE / boundary N>4 extensions | Out of V12 scope |

---

## 6. V12 goals

1. **Finish or conclusively close** production-fidelity backend checks (TurboQuant, KIVI, KVQuant).
2. **Validate repair policies** at full V10-suite scale on 0.5B and 1.5B (optional 3B).
3. **Resolve or document no-go** for true attention logging on a tiny subset.
4. **Review performance/memory truth boundaries** — decide what claims remain forbidden vs methodology-gated.
5. **Prepare V12 release package** and an updated public-launch decision.
6. Preserve `exactkv_failures == 0` on all published V12 experiments.

V12 must answer:

> **Can ExactKV honestly close the major deferred backend, policy, forensics, and
> claim-boundary gaps — or document exact blockers — before public v1.0.0 launch?**

---

## 7. V12 non-goals

- **No public launch during Phase 0** or before Phase 8 gate review.
- **No new default-registry compressors** unless a phase explicitly approves a restricted adapter re-run.
- **No generation or verification logic changes.**
- **No report schema changes** (JSON/CSV field additions require separate approval).
- **No direct vLLM, LMCache, PagedAttention, Sparse V production, KVTC, or Palu integration** unless scope explicitly changes.
- **No infinite backend gauntlet** — selected tracks only; valid outcome is documented **no-go**.
- **No implication** that `_sim` compressors are real packed-bit backends.
- **No implication** that upstream paper results are ExactKV results.
- **No positive speed, latency, throughput, runtime, tokens/sec, active GPU memory savings, or production-serving claims** unless Exp 027 approves explicit methodology — and even then, no production readiness claim.
- **No model accuracy improvement claims.**

---

## 8. Phase plan

### Phase 0 — Scope statement (this document)

Formal V12 scope, experiment plans 021–027, policies, and gate criteria.
**No code. No experiments.**

### Phase 1 — TurboQuant production-fidelity feasibility (Experiment 021)

Research and environment audit: llama.cpp / GGUF / Qwen compatibility, adapter
interop requirements, isolated venv constraints. **Feasibility only** — no performance claims.

### Phase 2 — TurboQuant llama.cpp / GGUF probe or documented no-go (Experiment 022)

If Phase 1 is feasible: restricted probe comparing production-fidelity path against
Exp 008 Python results on a **small** prompt panel. If infeasible: document exact
blocker (model format, API mismatch, verifier incompatibility, env isolation).

### Phase 3 — KVQuant larger-model real-backend validation (Experiment 023)

KVQuant simquant on `Qwen2.5-1.5B` and optional `3B`. Start with hard V10 category
subset (`long_context`, `retrieval_copy`, `tool_json`); expand to full V10 suite if
quantizer artifacts and RunPod budget permit.

### Phase 4 — KIVI CUDA/Triton packed-path feasibility or documented no-go (Experiment 024)

Environment/API/model compatibility audit and restricted probe. If infeasible:
document exact blocker (CUDA kernel, Triton version, model arch, ExactKV adapter gap).

### Phase 5 — Full-suite repair-policy validation (Experiment 025)

Validate `baseline_k8_v4`, `baseline_boundary4`, `int8`, `category_adaptive`,
`fallback_int8_for_hard_categories` on **full 128-prompt V10 suites** at 0.5B and
1.5B; optional 3B if budget permits. Policies remain **experiment-layer only**.

### Phase 6 — True attention logging feasibility or documented no-go (Experiment 026)

Tiny prompt subset only; attempt `output_attentions` via eager/legacy path or
documented HF workaround. **No fabricated attention weights.** Extends D7/D8.

### Phase 7 — Performance/memory truth boundary review (Experiment 027)

Decide whether speed or active GPU memory claims remain **forbidden** or may be
reported under an approved, caveated methodology. Does **not** automatically authorize
positive claims — default remains forbidden unless evidence is robust.

### Phase 8 — V12 release package and public-launch decision

`V12_READINESS_ASSESSMENT.md`, `PROJECT_STATUS` / `RELEASE_NOTES` v0.12.0 (or
v1.0.0 if all gates pass), updated narrative, optional artifact bundle, v1.0.0
launch decision.

---

## 9. Experiment 021 plan — TurboQuant production-fidelity feasibility

| Parameter | Planned value |
|---|---|
| Focus | llama.cpp / GGUF / Qwen compatibility audit |
| Models | `Qwen2.5-0.5B` primary; document 1.5B constraints |
| Output | Feasibility memo + go/no-go for Exp 022 |
| Performance claims | **None** |

**Success criteria:**

- Document TurboQuant production path requirements vs Exp 008 Python adapter.
- Identify GGUF conversion, llama.cpp build, and ExactKV `BackendAdapter` interop gaps.
- `exactkv_failures == 0` if any probe cells are run; otherwise feasibility-only doc.

**References:** [`TURBOQUANT_INTEGRATION_RESEARCH.md`](TURBOQUANT_INTEGRATION_RESEARCH.md),
[`TURBOQUANT_ADAPTER_PROTOTYPE.md`](TURBOQUANT_ADAPTER_PROTOTYPE.md),
[`EXPERIMENT_008_TURBOQUANT_PYTHON.md`](EXPERIMENT_008_TURBOQUANT_PYTHON.md).

---

## 10. Experiment 022 plan — TurboQuant production-fidelity probe

| Parameter | Planned value |
|---|---|
| Prerequisite | Exp 021 go |
| Panel | Small shared panel (≤25 prompts); same compressors as Exp 008 where comparable |
| Compare | Production-fidelity path vs Exp 008 Python accept/exactness |
| If blocked | Exact blocker document (no silent skip) |

**Success criteria:**

- Restricted adapter preserves exactness on probe cells, **or** documented no-go with reproducible blocker.
- No llama.cpp / GGUF / production TurboQuant implied as already implemented before probe runs.
- No throughput, latency, or speedup claims.

---

## 11. Experiment 023 plan — KVQuant larger-model validation

| Parameter | Planned value |
|---|---|
| Models | `Qwen2.5-1.5B` required; `3B` optional |
| Backend | KVQuant simquant (restricted factory adapter) |
| Suites | Hard V10 subset first; optional full 128-prompt V10 |
| Quantizers | External pickles per model size (not committed) |

**Success criteria:**

- `exactkv_failures == 0` on published cells.
- Per-category accept vs Exp 010 (0.5B) and Exp 014 spot-check anchors.
- Document if quantizer unavailable for a model size (omit row, do not invent).

**References:** [`KVQUANT_RUNPOD_VALIDATION.md`](KVQUANT_RUNPOD_VALIDATION.md),
[`KVQUANT_ADAPTER_PROTOTYPE.md`](KVQUANT_ADAPTER_PROTOTYPE.md),
[`EXPERIMENT_010_KVQUANT_SIM.md`](EXPERIMENT_010_KVQUANT_SIM.md).

---

## 12. Experiment 024 plan — KIVI CUDA/Triton packed-path feasibility

| Parameter | Planned value |
|---|---|
| Focus | CUDA/Triton packed-path vs Exp 009 offline simulate |
| Models | `Qwen2.5-0.5B` primary; document larger-model constraints |
| Outcome | Probe cells **or** documented no-go |

**Success criteria:**

- Exactness on probe cells with restricted adapter, **or** exact environment/API/model blocker documented.
- No claim that KIVI CUDA/Triton is integrated before probe succeeds.
- Compare accept against Exp 009 offline baseline where both run.

**References:** [`KIVI_KVQUANT_INTEGRATION_RESEARCH.md`](KIVI_KVQUANT_INTEGRATION_RESEARCH.md),
[`KIVI_ADAPTER_PROTOTYPE.md`](KIVI_ADAPTER_PROTOTYPE.md),
[`EXPERIMENT_009_KIVI_OFFLINE.md`](EXPERIMENT_009_KIVI_OFFLINE.md).

---

## 13. Experiment 025 plan — Full-suite repair-policy validation

| Parameter | Planned value |
|---|---|
| Models | `0.5B` + `1.5B` required; `3B` optional |
| Prompts | **Full 128-prompt V10 suites** (not pilot subset) |
| Policies | `baseline_k8_v4`, `baseline_boundary4`, `int8`, `category_adaptive_policy`, `fallback_int8_for_hard_categories` |
| Layer | Experiment-layer only; **not** core ExactKV defaults |

**Success criteria:**

- `exactkv_failures == 0`.
- Compare accept vs Exp 020 pilot; document if pilot gains do or do not survive full suite.
- No claim of final model accuracy improvement.

**References:** [`EXPERIMENT_020_REPAIR_POLICY_PILOT.md`](EXPERIMENT_020_REPAIR_POLICY_PILOT.md),
[`EXPERIMENT_019_DIVERGENCE_AUTOPSY.md`](EXPERIMENT_019_DIVERGENCE_AUTOPSY.md).

---

## 14. Experiment 026 plan — True attention logging feasibility

| Parameter | Planned value |
|---|---|
| Panel | Tiny subset (≤5 prompts) |
| Goal | Obtain real `output_attentions` or document blocker |
| Constraint | **No fabricated attention weights** |

**Success criteria:**

- Real attention tensors logged on subset, **or** documented no-go (sdpa, memory, HF API).
- If successful: extend D8 per-head forensics on subset only.
- `exactkv_failures == 0` on any ExactKV cells run alongside logging.

---

## 15. Experiment 027 plan — Performance/memory truth boundary

| Parameter | Planned value |
|---|---|
| Focus | Claim-boundary review, not a speed benchmark |
| Inputs | Exp 018 methodology, V5 accounting, any V12 profiling artifacts |
| Output | Updated claim policy document |

**Success criteria:**

- Explicit decision: speed/memory claims remain **forbidden** vs **methodology-gated**.
- If methodology-gated: document baselines, hardware, isolation, and caveats required before any future positive claim.
- Does **not** authorize production serving or model accuracy improvement claims.
- Default outcome expected: **claims remain forbidden** unless evidence is exceptionally robust.

**References:** [`GPU_MEMORY_METHODOLOGY.md`](GPU_MEMORY_METHODOLOGY.md),
[`EXPERIMENT_018_GPU_MEMORY_PILOT.md`](EXPERIMENT_018_GPU_MEMORY_PILOT.md).

---

## 16. Backend completion policy

| Backend | What ExactKV tested | What it is **not** | V12 action |
|---|---|---|---|
| TurboQuant | Exp 008 Python adapter | llama.cpp / GGUF / MLX production | Exp 021–022 |
| KIVI | Exp 009 offline simulate | CUDA/Triton packed path | Exp 024 |
| KVQuant | Exp 010 simquant 0.5B | Deployment CUDA; post-RoPE production | Exp 023 |
| kvpress | Exp 005 KnormPress | Broader kvpress suite | Out of V12 |
| vLLM / LMCache | Exp 007/017 harness/sidecar | Direct integration | **No-go** unless scope changes |

**Rules:**

- TurboQuant Python is **not** production TurboQuant.
- KIVI offline is **not** KIVI CUDA/Triton.
- KVQuant simquant is **not** KVQuant deployment CUDA.
- Restricted adapters remain **factory-only**; not default registry.
- External paper results are **not** ExactKV results.

---

## 17. Repair-policy policy

- Experiment 020 was a **pilot only** (25 prompts, 300 cells).
- Policies select existing compressors; they do **not** change verification.
- Policies are **not enabled by default** in core ExactKV.
- V12 Phase 5 (Exp 025) tests whether pilot gains **survive** full V10-suite and larger-model validation.
- Valid outcomes: gains hold, gains shrink, or category-specific regression — all publishable with exactness gate intact.

---

## 18. Performance/memory policy

- **No speed claim** unless explicitly measured with approved methodology, baselines, hardware disclosure, and caveats — and still no production readiness claim.
- **No active GPU memory savings claim** unless robustly isolated (allocator noise, fragmentation, peak vs allocated documented).
- **V5 `total_kv_footprint_bytes`** remains the stable memory story unless Exp 027 replaces it with stronger evidence.
- Exp 018 pilot artifacts stay **outside** the standard schema unless separately approved.
- Forbidden by default: `tokens_per_second`, `throughput`, `latency`, `speedup`, `runtime_seconds`, `active_gpu_kv_bytes` as ExactKV launch claims.

---

## 19. Launch criteria after V12

Public v1.0.0 launch requires **all** of:

| Criterion | Required |
|---|---|
| V12 Phases 1–8 complete or honestly deferred with documented blockers |
| `exactkv_failures == 0` on all published V12 experiments |
| Backend tracks closed or no-go documented (Exp 021–024) |
| Repair policies validated or pilot-only status reaffirmed (Exp 025) |
| Attention logging resolved or no-go documented (Exp 026) |
| Performance/memory claim boundary finalized (Exp 027) |
| `V12_READINESS_ASSESSMENT.md` with public-launch decision |
| `PROJECT_STATUS` / `RELEASE_NOTES` v1.0.0 (D19/D20) |
| Launch narrative reviewed and approved (D18) |
| Optional curated artifact bundle (D17) |

**`v0.12.0` tag** may ship after V12 substance even if public v1.0.0 remains deferred.

---

## 20. Stop/revise criteria

**Stop V12 phase and document blocker** if:

- Environment isolation cannot be maintained (venv conflicts, CUDA/Triton version lock).
- Adapter cannot preserve `exactkv_failures == 0` on probe cells.
- RunPod budget exhausted before minimum panel completes.
- GGUF / llama.cpp / KIVI CUDA interop requires core generation or verification changes.

**Revise scope** (requires explicit approval) if:

- A phase expands beyond feasibility/probe into production integration.
- Report schema changes are requested.
- Positive performance claims are proposed without Exp 027 approval.

**Valid partial exit:** Documented no-go with reproducible blocker is a **successful** V12 outcome for that track.

---

## 21. Risks and unknowns

| Risk | Mitigation |
|---|---|
| llama.cpp / GGUF incompatible with ExactKV verifier loop | Exp 021 feasibility before Exp 022 cells |
| KVQuant quantizer missing for 1.5B/3B | Stratified subset; omit with blocker note |
| KIVI CUDA/Triton env irreproducible locally | RunPod isolated venv; document versions |
| Full-suite Exp 025 cost at 1.5B × 128 × 5 policies | Stratified validation first if needed |
| Attention logging OOM or sdpa-only | Tiny subset; eager fallback; honest no-go |
| Exp 027 pressure to claim speed/memory | Default forbidden; methodology bar intentionally high |
| Scope creep into vLLM/LMCache | D11/D12 remain no-go unless scope changes |
| Launch narrative overclaims | Explicit negation template; Phase 8 review gate |

---

## 22. No-performance-claim policy

V12 documents, experiment reports, and updated README/ROADMAP sections must **not**:

- Add or imply `tokens_per_second`, `throughput`, `latency`, `speedup`,
  `runtime_seconds`, or `active_gpu_kv_bytes` as ExactKV results **unless**
  Experiment 027 explicitly approves a caveated methodology — default remains forbidden.
- Claim production serving readiness or vLLM/LMCache **integration**.
- Present `_sim` compressors as real packed-bit backends.
- Cite TurboQuant, KIVI, or KVQuant **paper** results as ExactKV experiment results.
- Imply TurboQuant llama.cpp, KIVI CUDA/Triton, or KVQuant deployment CUDA are **already implemented**.
- Imply V10/V11/V12 suites are **universal** public benchmarks.
- Claim model accuracy improvement from compression.

Forbidden terms may appear **only** in explicit negation or future-methodology guardrails.
Acceptance, rejection, correction, divergence, and honest workspace-memory accounting
remain the default permitted metrics.

---

## Related

- [`V11_LAUNCH_READINESS.md`](V11_LAUNCH_READINESS.md) — prior launch gate (v0.11.0 ready; v1.0.0 not)
- [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md) — D1–D4, D7–D8, D11–D12, D17–D20
- [`ROADMAP.md`](ROADMAP.md) — version path
- [`RAW_ARTIFACT_POLICY.md`](RAW_ARTIFACT_POLICY.md) — artifact bundle policy
- [`LAUNCH_NARRATIVE_DRAFT.md`](LAUNCH_NARRATIVE_DRAFT.md) — deferred public narrative
