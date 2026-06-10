# V11 Scope Statement: Final Launch Hardening

**Status:** **Phase 1 complete** — Experiment 015 succeeded (`exactkv_failures == 0`). Phase 2 next.
**Builds on:** `v0.10.0` — V10 complete (Experiments 012–014; evaluation-suite hardening).
**Not public launch.** v1.0.0 deferred until V11 substance and launch package (D17–D20) exit criteria are met.

> V11 is **final pre-v1.0.0 hardening** — not a performance benchmark and not
> production serving integration.
> V11 preserves the exactness gate: `exactkv_failures == 0` on every published experiment.
> V11 must **not** broaden into infinite backend integration or universal benchmark expansion.
> No throughput, latency, tokens/sec, speedup, `runtime_seconds`,
> `active_gpu_kv_bytes`, or production-serving claims — unless active GPU memory is
> later reported under an **explicit, caveated methodology** approved in Phase 4/018.
> If measured GPU memory is not robust, **document deferral** instead of claiming it.

---

## 1. Status

| Phase | Focus | Status |
|---|---|---|
| **0** | Formal scope statement (this document) | **Complete** |
| **1** | 1.5B expanded-suite validation (Experiment 015) | **Complete** |
| **2** | Optional 3B stretch or 1.5B real-backend panel (Experiment 016) | Planned |
| **3** | Serving sidecar/probe feasibility refresh (Experiment 017) | Planned |
| **4** | Active GPU memory methodology (Experiment 018) | Planned |
| **5** | Optional attention logging / divergence deep dive | Planned |
| **6** | Raw report bundle + launch package readiness | Planned |

**Latest release:** `v0.10.0`. **V10 exit docs:**
[`V10_READINESS_ASSESSMENT.md`](V10_READINESS_ASSESSMENT.md),
[`PROJECT_STATUS_V0.10.0.md`](PROJECT_STATUS_V0.10.0.md),
[`RELEASE_NOTES_V0.10.0.md`](RELEASE_NOTES_V0.10.0.md).

---

## 2. V11 goal

Close the remaining **scale, serving-context, profiling, and launch-documentation**
gaps before a defensible **v1.0.0** tag — without changing the exactness gate,
generation logic, verification logic, or report JSON/CSV schemas.

V11 must answer:

> **Do ExactKV's V10 findings hold at 1.5B+ on expanded suites, is serving
> integration still honestly no-go or probe-feasible, can active GPU memory be
> measured with an approved methodology, and is the launch documentation package
> complete — still with `exactkv_failures == 0`?**

---

## 3. Why V11 is needed after V10

V10 hardened evaluation at **0.5B** on 128 versioned prompts with sensitivity,
forensics, and factory-only real-backend spot-checks. That is necessary but not
sufficient for public v1.0.0:

| Gap | Evidence |
|---|---|
| **Multi-model on V10 suites** | Experiment 011 validated 1.5B on legacy `core` (34 prompts) only; Experiments 012–014 are 0.5B |
| **Serving integration status stale** | V8 Phase A no-go for direct vLLM/LMCache; no post-V10 refresh |
| **Active GPU memory** | `total_kv_footprint_bytes` is accounting only; D14 unresolved |
| **Attention forensics depth** | Experiment 013 used heuristics only; D7/D8 deferred |
| **Launch package** | No curated raw report bundle (D17) or final public narrative (D18) |
| **3B scale unknown** | No published ExactKV cells at 3B |

V11 makes ExactKV **launch-ready as a research platform** — not faster or
production-deployed.

---

## 4. What V11 should add

1. **Experiment 015** — `Qwen/Qwen2.5-1.5B` on full or stratified **V10 suites** with per-category leaderboards.
2. **Experiment 016 (optional)** — `Qwen/Qwen2.5-3B` stretch and/or 1.5B restricted real-backend panel where RunPod budget and isolated envs permit.
3. **Experiment 017** — serving **sidecar/probe** or documented **no-go refresh** for vLLM/LMCache (metadata-only; no forced integration).
4. **Experiment 018** — **active GPU memory methodology** document + measured-memory pilot if feasible; honest deferral if not.
5. **Phase 5 (optional)** — true attention logging on a **tiny** prompt subset (D7); no fabricated weights.
6. **Phase 6** — raw report bundle policy, checksum manifest, launch package checklist, v1.0.0 readiness assessment.
7. **Documentation** — Experiment 015–018 reports; updated experiment index; no schema changes without separate approval.

---

## 5. What V11 explicitly should not add

- **No new default-registry compressors** unless a phase explicitly approves a restricted adapter re-run panel.
- **No generation or verification logic changes.**
- **No report schema changes** (JSON/CSV field additions require separate approval).
- **No direct vLLM, LMCache, llama.cpp, MLX, TurboQuant production runtime, KIVI CUDA/Triton,
  KVQuant deployment CUDA, Sparse V production, KVTC, or Palu integration.**
- **No PagedAttention kernel integration** (D16 remains deferred).
- **No throughput, latency, tokens/sec, speedup, `runtime_seconds`, or
  `active_gpu_kv_bytes` claims** unless Phase 4 defines an approved methodology and
  Experiment 018 publishes caveated results — otherwise defer.
- **No implication** that `_sim` compressors are real packed-bit backends.
- **No implication** that external paper results are ExactKV results.
- **No infinite backend gauntlet** — selected real backends only where environment permits.
- **No public launch post** until v1.0.0 gates are met (draft narrative may be prepared in Phase 6, clearly marked deferred).

---

## 6. Phase plan

### Phase 0 — Scope statement (this document)

Deliverable: `V11_SCOPE_STATEMENT.md`; README/ROADMAP/deferred-register updates.

### Phase 1 — 1.5B expanded-suite validation

- Run Experiment **015** on RunPod GPU (float16 CUDA).
- Primary model: `Qwen/Qwen2.5-1.5B`.
- Prompts: V10 suites (§12); anchor `draft_len=4`, `max_new_tokens=16`.
- Success: `exactkv_failures == 0`; per-category tables; comparison to Experiment 012 (0.5B).

### Phase 2 — Optional 3B stretch

- Run Experiment **016** if RunPod budget and stability permit.
- Options (document chosen path):
  - **A:** `Qwen/Qwen2.5-3B` built-in panel on stratified V10 subset.
  - **B:** 1.5B restricted real-backend panel (factory-only) on Exp 014–style hard-category subset.
- Omit entirely if infeasible — document blocker in deferred register.

### Phase 3 — Serving sidecar/probe refresh

- Run Experiment **017** or publish equivalent feasibility memo.
- Refresh [`SERVING_CONTEXT_FEASIBILITY.md`](SERVING_CONTEXT_FEASIBILITY.md) conclusions post-V10.
- Builds on Experiment **007** harness and V8 Phase A/B artifacts.
- Outcome may be **probe pass**, **metadata-only sidecar sketch**, or **reaffirmed no-go** — all are valid.

### Phase 4 — Active GPU memory methodology

- Run Experiment **018** methodology + pilot.
- Distinct from `total_kv_footprint_bytes` workspace accounting (V5).
- If measurement is not reproducible or well-defined, publish **deferral** — do not invent `active_gpu_kv_bytes` fields in reports.

### Phase 5 — Optional attention logging / divergence deep dive

- Tiny subset only (e.g. ≤5 prompts × ≤2 compressors).
- Supports D7/D8 if weights are obtainable without fabrication.
- If infeasible, extend Experiment 013 deferral note — no fake attention maps.

### Phase 6 — Raw report bundle + launch package readiness

- Publish raw artifact policy (§15).
- Curated zip/bundle for release (experiments 001–014+ manifests).
- Draft launch narrative (D18) — **deferred draft**, not public post.
- `V11_READINESS_ASSESSMENT.md` + v1.0.0 gate checklist.
- `PROJECT_STATUS` / `RELEASE_NOTES` v1.0.0 only if all gates pass.

---

## 7. Experiment 015 plan

**Name:** 1.5B V10 Suite Validation.

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B` |
| Prompt suites | V10 seven suites (§12) — full 128 if feasible |
| Compressors | Built-in panel (§13) |
| `draft_len` | 4 (anchor; match Exp 012) |
| `max_new_tokens` | 16 |
| Environment | RunPod GPU, float16 CUDA |
| Experiment class | `v11_qwen15b_v10_suites` |
| Deliverables | [`EXPERIMENT_015_QWEN15B_V10_SUITES.md`](EXPERIMENT_015_QWEN15B_V10_SUITES.md); gitignored JSON/CSV |
| Script | `scripts/run_experiment_015_qwen15b_v10_suites.py` (Phase 1) |

**Estimated cells (full panel):** 128 prompts × 7 compressors ≈ **896** (same shape as Exp 012).

**Success criteria:**

- `exactkv_failures == 0`
- Per-suite and per-category leaderboards
- Documented comparison to Experiment 012 (0.5B) and Experiment 011 (1.5B legacy `core`)
- No performance claims

---

## 8. Experiment 016 plan

**Name:** Optional 3B Stretch or 1.5B Real-Backend Panel.

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen2.5-3B` **or** `Qwen/Qwen2.5-1.5B` |
| Prompts | Stratified V10 subset (document ids; prefer harder categories from Exp 014) |
| Compressors | Built-ins + **selected** factory-only real backends if env permits |
| `draft_len` / `max_new_tokens` | 4 / 16 |
| Environment | RunPod; isolated venvs per backend |
| Experiment class | `v11_optional_scale_or_backend` |
| Deliverables | `EXPERIMENT_016_OPTIONAL_SCALE_BACKEND.md`; gitignored JSON/CSV |

**Success criteria:**

- `exactkv_failures == 0` on executed cells
- Explicit statement if 3B omitted or backends split across panels
- No production-backend claims

---

## 9. Experiment 017 plan

**Name:** Serving Sidecar / Probe Feasibility Refresh.

| Parameter | Value |
|---|---|
| Scope | vLLM and/or LMCache **sidecar** or metadata-only probe — **not** direct integration |
| Baseline | [`SERVING_CONTEXT_FEASIBILITY.md`](SERVING_CONTEXT_FEASIBILITY.md), [`EXPERIMENT_007_SERVING_CONTEXT.md`](EXPERIMENT_007_SERVING_CONTEXT.md), [`SERVING_CACHE_LIFECYCLE_HARNESS.md`](SERVING_CACHE_LIFECYCLE_HARNESS.md) |
| Experiment class | `v11_serving_probe` |
| Deliverables | `EXPERIMENT_017_SERVING_PROBE.md` or updated feasibility memo |

**Valid outcomes:**

1. **Probe succeeded** — documented hook points, ownership model, verify isolation preserved.
2. **No-go reaffirmed** — direct integration still unsafe; sidecar also deferred with evidence.
3. **Partial** — metadata/lifecycle compatibility only; integration remains future work.

**Success criteria:**

- Written decision with evidence
- No claim of production serving or ExactKV+vLLM shipping
- No throughput/latency claims

---

## 10. Experiment 018 plan

**Name:** Active GPU Memory Methodology and Pilot.

| Parameter | Value |
|---|---|
| Scope | Define methodology distinct from `total_kv_footprint_bytes`; optional measured pilot |
| Models | 0.5B and/or 1.5B on small stratified subset |
| Tools | Document chosen approach (e.g. `torch.cuda.max_memory_allocated` snapshots at defined lifecycle points — **not** peak serving memory claims) |
| Experiment class | `v11_active_gpu_memory` |
| Deliverables | `EXPERIMENT_018_ACTIVE_GPU_MEMORY.md`; methodology section even if pilot deferred |

**Success criteria:**

- Published methodology with explicit caveats
- If pilot runs: reproducible procedure; **no** `active_gpu_kv_bytes` in default report schema unless separately approved
- If pilot infeasible: deferral recorded; D14 status updated honestly

---

## 11. Model matrix

| Model | Role | Required? |
|---|---|---|
| `Qwen/Qwen2.5-0.5B` | Baseline reference (V10 Experiments 012–014) | Reference only |
| `Qwen/Qwen2.5-1.5B` | **Primary V11 validation target** on V10 suites | **Required** (Exp 015) |
| `Qwen/Qwen2.5-3B` | Optional scale stretch | Optional (Exp 016) |

**Policy:** 1.5B on V10 suites is the **minimum** multi-model bar for v1.0.0 readiness.
3B is stretch goals only — omit if RunPod unstable or cost-prohibitive.

---

## 12. Prompt suite matrix

| Suite set | Prompts | Use in V11 |
|---|---:|---|
| V10 full panel | 128 | Exp 015 default |
| V10 stratified subset | TBD (document ids) | Exp 016, 018, attention pilot if runtime constrained |
| Exp 014 hard-category subset | 40 | Optional anchor for 1.5B real-backend panel |
| Legacy `core` | 34 | Comparison anchor only (Exp 011) |

**Subset rule:** If runtime requires reduction, prefer **deterministic** stratified subsets
(e.g. first N ids per suite, or Exp 014-style hard categories) — not cherry-picked
per compressor.

---

## 13. Compressor panel

### Built-in (default)

| Compressor | Role |
|---|---|
| `noop` | Lossless baseline |
| `backend_passthrough` | V6 BackendAdapter PoC |
| `int8` | Real symmetric INT8 |
| `k8_v4_sim` | Simulated uniform K8/V4 |
| `k8_v4_boundary4_v8_sim` | Best simulated layer-aware policy |
| `k_full_v8` | Real INT8 V, full K |
| `k8_v_full` | Real INT8 K, full V |

### Restricted real backends (factory-only; Exp 016 / optional rows)

| Adapter | When |
|---|---|
| `kvquant_sim_qwen05b` / per-model quantizer | KVQuant isolated venv + quantizers pickle |
| `turboquant_python_k3_v3` | `PYTHONPATH=vendor/turboquant_plus` |
| `kivi_offline_k2_v2` | `PYTHONPATH` to KIVI repo |

**Not in default registry.** Cross-panel merge labelling if environments split (per Exp 014 precedent).

---

## 14. Serving / profiling requirements

| Requirement | Source |
|---|---|
| R1–R10 serving invariants preserved | [`SERVING_CONTEXT_FEASIBILITY.md`](SERVING_CONTEXT_FEASIBILITY.md) |
| Experiment 007 harness remains reference implementation | Mode B lifecycle |
| Verification uses **full-precision KV only** | Unchanged |
| Sidecar/probe must not mutate authoritative full KV | Exp 017 |
| Active GPU memory methodology **documented before** any measured numbers | Exp 018 |
| Measured memory ≠ `total_kv_footprint_bytes` | V5 honesty policy |
| No production deployment claims | Global |

---

## 15. Raw artifact policy

| Artifact | Git policy |
|---|---|
| Per-experiment JSON/CSV under `reports/` | **Gitignored** (unchanged) |
| Committed Markdown experiment reports | Yes |
| Curated release bundle (zip/tar) | Phase 6; hosted or attached to release — **not** full raw dump in git |
| Checksum manifest | SHA-256 per bundled file; `manifest.json` listing experiment id, model, cell count, `exactkv_failures` |
| Secrets / quantizer pickles / large logs | Never committed |

**Bundle scope target:** experiments **001–014** minimum plus V11 experiments with redacted paths where needed.

---

## 16. v1.0.0 launch gates

v1.0.0 public launch requires **V10 ✅ + V11 substance + launch package**:

| Gate | Owner | V11 contribution |
|---|---|---|
| Multi-model on expanded suites (1.5B minimum) | V11 | Experiment 015 |
| Optional 3B or 1.5B real-backend scale panel | V11 | Experiment 016 (or documented omit) |
| Serving sidecar probe or no-go refresh | V11 | Experiment 017 (D13) |
| Active GPU memory methodology or honest deferral | V11 | Experiment 018 (D14) |
| Attention logging or documented deferral | V11 | Phase 5 (D7/D8) |
| Curated raw report bundle policy + bundle | V11 Phase 6 | D17 |
| Public launch narrative (explicit negation of performance claims) | v1.0.0 | D18 — draft in Phase 6, publish at tag |
| `PROJECT_STATUS` + `RELEASE_NOTES` v1.0.0 | v1.0.0 | D19 |
| Git tag `v1.0.0` | v1.0.0 | D20 |

**`v0.10.0` is not v1.0.0.** V11 exit does not automatically trigger public launch —
only the **tag** after gate review.

---

## 17. Risks and unknowns

| Risk | Mitigation |
|---|---|
| 1.5B × 128 × 7 cell cost on RunPod | Stratified subset with documented ids; float16 CUDA |
| 3B OOM or instability | Optional Exp 016; omit with blocker note |
| KVQuant quantizer per model size | Separate pickles; omit row if unavailable |
| Sidecar probe scope creep → forced vLLM integration | Valid outcome is **no-go**; scope frozen in Exp 017 |
| Active GPU memory misleading (fragmentation, peak vs allocated) | Methodology doc + caveats; defer if not robust |
| Attention logging cost / HF path limitations | Tiny subset; defer D7 if blocked |
| Report bundle size | Curated manifests; checksums; no git bloat |
| Launch narrative overclaims | Explicit negation template; deferred until gates pass |

---

## 18. No-performance-claim policy

V11 documents, experiment reports, and updated README/ROADMAP sections must **not**:

- Add or imply `tokens_per_second`, `throughput`, `latency`, `speedup`,
  `runtime_seconds`, or `active_gpu_kv_bytes` as ExactKV results **unless**
  Experiment 018 publishes an approved, caveated methodology.
- Claim production serving readiness or vLLM/LMCache **integration** (probe ≠ integration).
- Present `_sim` compressors as real packed-bit backends.
- Cite TurboQuant, KIVI, or KVQuant **paper** results as ExactKV experiment results.
- Imply V10/V11 suites are **universal** public benchmarks.
- Imply active GPU memory has **already** been measured before Exp 018 completes.

Forbidden terms may appear **only** in explicit negation or future-methodology guardrails.
Acceptance, rejection, correction, divergence, and honest workspace-memory accounting
remain the default permitted metrics.

---

## Related

- [`V10_SCOPE_STATEMENT.md`](V10_SCOPE_STATEMENT.md) — prior phase (complete)
- [`V10_READINESS_ASSESSMENT.md`](V10_READINESS_ASSESSMENT.md) — v1.0.0 gate baseline
- [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md) — D6–D8, D13–D14, D17–D20
- [`SERVING_CONTEXT_FEASIBILITY.md`](SERVING_CONTEXT_FEASIBILITY.md) — V8 Phase A baseline
- [`EXPERIMENT_011_LARGER_MODEL_VALIDATION.md`](EXPERIMENT_011_LARGER_MODEL_VALIDATION.md) — 1.5B legacy `core`
- [`ROADMAP.md`](ROADMAP.md) — version path
