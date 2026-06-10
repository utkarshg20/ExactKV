# V9 Scope Statement: Real Backend Integration Gauntlet

**Status:** Phases 0–C complete for TurboQuant Python track. **TurboQuant not in
default registry.** Experiment 008 complete (`exactkv_failures == 0`). **Phase D
complete** (KIVI offline track D1–D3). **Phase D4 complete** (D4a static + **D4b
RunPod GPU validation** on L40S; Option A faithful adapter go). **Phase D5
complete** — restricted KVQuant simquant adapter prototype
([`KVQUANT_ADAPTER_PROTOTYPE.md`](KVQUANT_ADAPTER_PROTOTYPE.md)); factory-only,
not in default registry; RunPod smoke gate `exactkv_failures == 0`. **Phase D6
complete** — Experiment 010 KVQuant simquant evaluation
([`EXPERIMENT_010_KVQUANT_SIM.md`](EXPERIMENT_010_KVQUANT_SIM.md)); 272 cells,
`exactkv_failures == 0`. **Phase E complete** — larger-model RunPod validation
([`EXPERIMENT_011_LARGER_MODEL_VALIDATION.md`](EXPERIMENT_011_LARGER_MODEL_VALIDATION.md));
Qwen2.5-1.5B, 238 cells, `exactkv_failures == 0`.
**Builds on:** `v0.8.0` — V8 complete; serving harness (Experiment 007);
`BackendAdapter` + restricted kvpress (V6); simulated layer-aware policies (V7).
**Not public launch.** V9 is the **real backend credibility phase**.

> ExactKV does **not** implement TurboQuant, TurboQuant+, KIVI, KVQuant, KVTC,
> Palu, Sparse V, vLLM, LMCache, or PagedAttention in Phase 0 or by default in V9.
> No throughput, latency, tokens/sec, speedup, `runtime_seconds`,
> `active_gpu_kv_bytes`, or production-readiness claims as ExactKV results.
> External backends and papers cited below are **candidates and related work** —
> not current ExactKV capabilities until integrated and evaluated under the
> exactness gate.
> Simulated (`_sim`) compressors remain int8-container simulations; real packed-bit
> savings must not be implied for `_sim` rows.

---

## 1. V9 goal

Integrate and **evaluate real KV-compression backends** behind ExactKV's existing
`BackendAdapter` boundary — measuring **exactness, acceptance, divergence, and
honest workspace-memory accounting** only.

V9 must answer:

> **Can ExactKV wrap production-grade quantization backends (TurboQuant+,
> KIVI, KVQuant-style) without breaking `exactkv_output_ids == full_output_ids`,
> and what acceptance/memory behaviour do they exhibit versus simulated policies?**

Default V9 success is **credible real-backend evaluation with `exactkv_failures ==
0`**, or **documented incompatibility** with a clear blocker write-up. Time is
not a constraint; scope is not shrunk because integration is hard.

---

## 2. Why V9 exists after V8

After V1–V8, ExactKV has:

| Layer | What is established |
|---|---|
| V1–V5 | Draft-verify-commit; exactness gate; asymmetric `_sim`; V5 workspace memory |
| V6 | `BackendAdapter`; one real backend PoC (restricted kvpress KnormPress) |
| V7 | Simulated layer-aware V; boundary-depth ablation (006C) |
| V8 | Serving-context harness; Experiment 007 compatibility (`exactkv_failures == 0`) |

**Gap:** Only **one** restricted real backend (kvpress token-dropping) is integrated.
The project's simulated compressors (`k8_v4_sim`, `k8_v4_boundary4_v8_sim`) and
native `int8` do **not** represent TurboQuant+, KIVI, or KVQuant packed formats,
rotation-based V quantization, pre-RoPE key paths, or real residuals.

V9 closes the credibility gap: **attempt full integration** of major real backends,
not just simulate their ideas.

---

## 3. Why ExactKV is not public-launch final yet

v0.8.0 completed the serving-context chapter but [`PROJECT_STATUS_V0.8.0.md`](PROJECT_STATUS_V0.8.0.md)
states the project is **not launch-final**:

- No TurboQuant/TurboQuant+, KIVI, or KVQuant adapter evaluated on the core suite.
- No larger-model GPU validation (RunPod) beyond `Qwen/Qwen2.5-0.5B` CPU sweeps.
- Public narrative and raw report bundle intentionally deferred.
- Simulated policies dominate published leaderboards; external papers' claims are
  **not** ExactKV results.

V9 exists to make ExactKV **substantially more impressive** before public posting —
by integrating or **seriously attempting** major real backends and publishing
honest evaluation either way.

---

## 4. What “full backend integration” means

In V9, **full backend integration** means:

| Requirement | Meaning |
|---|---|
| **Real library code** | Compress/decompress (or encode/decode) delegated to the external backend's own implementation — not reimplemented in ExactKV |
| **`BackendAdapter` wrapper** | Thin adapter implementing `KVCompressor` via sealed public methods in `backend_adapter.py` |
| **Draft path only** | Backend affects `compress`, `materialize_for_draft`, `update_after_commit`, `stats` — **never** `VerificationEngine` |
| **Exactness preserved** | `exactkv_output_ids == full_output_ids`; `exactkv_failures == 0` on published cells |
| **Honest labelling** | `is_simulated`, `supports_real_bytes_claim`, `backend_name`, `backend_version`, `adapter_name` populated |
| **V5 memory fields** | `stored_kv_bytes`, `materialized_working_kv_bytes`, `metadata_bytes`, `temporary_workspace_bytes`, `total_kv_footprint_bytes` |
| **Environment isolation** | Optional extras (`[turboquant]`, `[kivi]`, etc.) — default `pip install -e ".[dev]"` unchanged |
| **Experiment report** | Markdown + gitignored JSON/CSV with manifest, capabilities, and disclaimers |

**Not required for “full integration”:** throughput parity, serving deployment,
custom CUDA kernels inside ExactKV, or reproduction of external-paper speedup claims.

**Valid partial outcome:** feasibility doc concluding **no-go** with documented
blocker (API, hook safety, cache format, device mismatch) — still valuable V9 output.

---

## 5. Candidate backends

| Backend | Type | ExactKV question | V9 priority |
|---|---|---|---|
| **TurboQuant / TurboQuant+** | Rotation + scalar quant; asymmetric K/V; optional Sparse V themes | Does real rotation-based V quant recover acceptance lost by naive `k8_v2_sim`? | **First** (Phases A–C) |
| **KIVI** | Per-channel K, per-token V, 2-bit + residual | Does KIVI granularity beat symmetric `int8` on acceptance at honest byte budget? | Phase D (Exp 009) |
| **KVQuant** | Pre-RoPE K quant, dense-and-sparse outliers | Does pre-RoPE key path preserve acceptance vs post-RoPE `int8`? | Phase D (Exp 009) |
| **KVTC** | Transform coding + entropy | Metadata-heavy storage; acceptance vs budget | Optional feasibility only |
| **Palu** | Low-rank KV + residual | Reconstruction workspace honesty | Optional feasibility only |

**Survey reference:** [`RELATED_WORK_KV_CACHE_COMPRESSION.md`](RELATED_WORK_KV_CACHE_COMPRESSION.md)
§§KIVI, KVQuant, TurboQuant, TurboQuant+.

ExactKV must **not** claim any backend's external accuracy or speedup numbers as its own.

---

## 6. Recommended first backend and why

**Recommended first: TurboQuant / TurboQuant+** (Phase A targets both; Phase B
prototype picks the feasible path).

| Reason | Detail |
|---|---|
| **Research alignment** | V7 layer-aware and 006A proxy analysis motivate **V-specific** and rotation-aware policies; TurboQuant+ is the closest real asymmetric V backend |
| **Naive sim gap** | `k8_v2_sim` (0.33 accept in Exp 003) ≠ TurboQuant `turbo2`; integrating real format tests whether aggressive V can work under verification |
| **Adapter precedent** | V6 `BackendAdapter` + kvpress lesson ([`EXPERIMENT_005_KVPRESS_KNORM.md`](EXPERIMENT_005_KVPRESS_KNORM.md)) — hook isolation, separate env, honest bytes |
| **HF compatibility** | Python `turboquant` bridge on HF `past_key_values` confirmed in Phase A ([`TURBOQUANT_INTEGRATION_RESEARCH.md`](TURBOQUANT_INTEGRATION_RESEARCH.md)); llama.cpp GGUF path is **not** the ExactKV integration route |
| **Honest evaluation** | ExactKV evaluates **acceptance + memory**, not TurboQuant+ paper throughput |

**Fallback:** If TurboQuant+ integration is blocked, Phase A documents why and
Phase D may prioritize KIVI or KVQuant — without abandoning TurboQuant attempt.

---

## 7. Required adapter interface

All V9 backends must use the **existing** V6 `BackendAdapter` sealed API
([`BACKEND_ADAPTER_INTERFACE.md`](BACKEND_ADAPTER_INTERFACE.md),
`exactkv/compressors/backend_adapter.py`):

| Method / field | Obligation |
|---|---|
| `compress()` | Clone authoritative tensors before `_backend_compress`; never mutate `FullKVState` |
| `materialize_for_draft()` | Deterministic HF-compatible `past_key_values`; no verify-path hooks |
| `update_after_commit()` | Re-compress from new full state; `logical_seq_len` aligned |
| `stats()` | All V5 workspace fields from values stored at compress time |
| `capabilities` | `backend_name`, `backend_version`, `adapter_name`, `adapter_version`; correct `is_simulated` / `supports_real_bytes_claim` |
| `verification_mode()` | If hooks used (kvpress lesson): context manager that **disables** hooks during verify |
| Registry policy | Real backends may be **opt-in** (like kvpress) until stable; document env extra |

**No changes** to `VerificationEngine`, `ExactKVGenerator` commit/verify logic, or
report schema except **additive** per-backend metadata fields approved per phase.

---

## 8. Required exactness contract

Hard gate (unchanged from V1):

```
exactkv_output_ids == full_output_ids   (greedy, per cell)
exactkv_failures == 0                   (every published experiment cell)
```

| Invariant | Enforcement |
|---|---|
| Verify uses **full-precision** authoritative KV only | `VerificationEngine` unchanged |
| Authoritative `FullKVState` not mutated by draft | Deep-copy in draft; adapter clones on compress |
| Cache alignment at round boundaries | `full_state.seq_len == compressed.logical_seq_len` |
| Hook safety | Zero attention forward hooks during verify (kvpress gate pattern) |
| Greedy only | No sampling, parallel verify, or bonus-token acceptance in V9 |

Any backend that cannot satisfy this contract is **rejected or isolated** — not
worked around by weakening verification.

---

## 9. Required memory-honesty contract

| Field | Rule |
|---|---|
| `stored_kv_bytes` | Real packed/quantised representation only |
| `materialized_working_kv_bytes` | Bytes needed for attention forward on draft path |
| `metadata_bytes` | Scales, codebooks, rotation matrices, sparse masks — counted honestly |
| `temporary_workspace_bytes` | Conservative scratch during compress/materialize |
| `total_kv_footprint_bytes` | Sum of above — **accounting sum, not measured peak GPU** |
| `supports_real_bytes_claim` | `True` only for genuinely smaller storage vs fp32 |
| `is_simulated` | `False` for real backends; `_sim` compressors unchanged |
| `memory_claim_note` | Per-row disclaimer in JSON/CSV reports |

**Forbidden:** `active_gpu_kv_bytes` unless a separate approved profiling sub-phase
runs under explicit methodology (not default V9).

---

## 10. Required environment isolation

Following Experiment 005 precedent:

| Rule | Rationale |
|---|---|
| **Optional extras** | e.g. `pip install -e ".[turboquant]"` in dedicated venv (`.venv-turboquant`) |
| **Default registry unchanged** | `import exactkv.compressors` must not import heavy backends |
| **Lazy import** | Backend library imported only inside adapter module construction |
| **Pinned versions** | `backend_version` in manifest; document transformers/torch pins |
| **Isolated compression model** | If backend mutates model state (hooks, buffers): `deepcopy` for compress path; verify model clean |
| **No global side effects in CI** | Default pytest suite runs without optional extras |

Experiment scripts (008, 009) document required venv and reproduce commands.

---

## 11. RunPod GPU requirements

V9 **does not reject** GPU-required backends. CPU remains the default for Phases
A–D smoke gates; **Phase E** adds RunPod validation.

| Phase / activity | GPU |
|---|---|
| Phase 0 (this document) | None |
| Phase A feasibility (install, API inspect) | **Recommended** — match backend's expected device (CUDA) |
| Phase B adapter smoke | CPU acceptable if backend supports CPU; else **GPU required** |
| Phase C Experiment 008 (0.5B) | CPU path if feasible; GPU for backend-faithful runs |
| Phase D KIVI/KVQuant | **Likely GPU** (KVQuant CUDA kernels in upstream) |
| **Phase E larger models** | **Required** — RunPod or equivalent |

### Phase E RunPod plan (illustrative)

| Parameter | Minimum | Stretch |
|---|---|---|
| Provider | RunPod (or documented equivalent) | — |
| GPU | 1× A100 40GB or L40S 48GB | A100 80GB for 7B |
| Models | `Qwen/Qwen2.5-1.5B` or `Qwen2.5-3B` | `Qwen2.5-7B` if memory allows |
| Suite | `core` subset (e.g. 16 prompts) for cost control | Full `core` if budget allows |
| Gate | `exactkv_failures == 0` on all cells | Same |
| Documentation | Hardware, CUDA version, backend version in manifest | — |

GPU sweeps are **manual/local/RunPod** — not required in default CI.

---

## 12. Tests and gates

| Gate | When |
|---|---|
| `exactkv_failures == 0` | Every published experiment cell |
| Exactness smoke | Phase B: 2 prompts × 2 draft lengths per new adapter |
| Hook-safety (if applicable) | Zero verify-path hooks; compress hooks restored |
| `BackendAdapter` unit tests | Clone safety, stats reconcile, capabilities honest |
| Memory reconcile | `total_kv_footprint_bytes` = sum of components |
| `_assert_no_forbidden_fields` | No performance keys in JSON/CSV |
| `validate_report` | On experiment JSON before report generation |
| Registry backward compat | Default `list_compressors()` unchanged without extras |

**Phase 0:** docs-only — `git diff --check` + prose audit only.

**Later phases:** targeted adapter tests + existing exactness gates; full pytest
only if shared runtime, verification, registry, or report schema changes broadly.

---

## 13. Experiment 008 plan — TurboQuant feasibility and adapter evaluation

**Name:** TurboQuant / TurboQuant+ Real-Backend Evaluation

**Prerequisite:** Phase A go (or documented conditional go with isolation requirements).

| Parameter | Illustrative value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` (baseline); optional 1.5B in Phase E |
| Suite | `core` (34 prompts) |
| Compressors | **TurboQuant adapter** + baselines: `noop`, `int8`, `k8_v4_sim`, `k8_v4_boundary4_v8_sim`, `k_full_v8`, `k8_v_full`, `backend_passthrough`; **optional** `kvpress_knorm_restricted` in isolated env |
| `draft_len` | 4 |
| `max_new_tokens` | 16 |
| Experiment class | `turboquant_real` or `turboquant_plus_real` |

**Deliverables:**

- `reports/experiment_008_turboquant.{json,csv}` — **gitignored**
- `docs/EXPERIMENT_008_TURBOQUANT.md`
- Per-compressor acceptance, divergence, V5 workspace table
- Explicit **real vs simulated** labelling; no TurboQuant paper claims

**Success:** `exactkv_failures == 0`. **Valid failure:** Phase A/B documents
blocker; Experiment 008 skipped with written no-go report.

---

## 14. Experiment 009 plan — KIVI / KVQuant comparison

**Name:** KIVI / KVQuant Real-Backend Comparison (if feasible)

**Prerequisite:** Phase D feasibility go for at least one of KIVI or KVQuant.

| Parameter | Illustrative value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B`; GPU likely required for KVQuant |
| Suite | `core` (34 prompts) |
| Compressors | **One or two** real adapters (KIVI, KVQuant-style) + same baseline panel as 008 |
| Comparison focus | Acceptance vs `int8`, `k_full_v8`, best TurboQuant row from 008 (if any) |

**Deliverables:**

- `reports/experiment_009_kivi_kvquant.{json,csv}` — **gitignored**
- `docs/EXPERIMENT_009_KIVI_KVQUANT.md`

**Valid outcomes:**

- Full experiment with `exactkv_failures == 0`
- Partial experiment (one backend only)
- Documented no-go for one backend with feasibility appendix

---

## 15. Larger-model validation plan (Phase E)

**Goal:** Confirm exactness gate holds on a **larger** Qwen2.5 variant with at
least one **real** integrated backend (prefer TurboQuant+ if Phase C succeeded).

| Step | Action |
|---|---|
| 1 | Select RunPod GPU tier per §11 |
| 2 | Load `Qwen2.5-1.5B` or `3B` (minimum); try `7B` if memory allows |
| 3 | Run reduced or full `core` suite with new real adapter + `noop`/`int8` controls |
| 4 | Report `exactkv_failures`, acceptance, workspace memory with hardware manifest |
| 5 | Document in Phase F release notes — **no throughput claims** |

This is **exactness and acceptance validation at scale**, not a serving benchmark.

---

## 16. Failure modes that count as valid findings

| Finding | Valid V9 outcome |
|---|---|
| Backend API incompatible with HF `DynamicCache` / Qwen2.5 | Phase A no-go doc |
| Hooks cannot be disabled during verify | Reject or isolate (kvpress pattern) |
| `materialize_for_draft` non-deterministic | Blocker until fixed or documented |
| Real bytes cannot be counted honestly | Adapter must not claim `supports_real_bytes_claim` |
| GPU required but unavailable | Defer experiment; document; do not fake CPU results as GPU-faithful |
| Acceptance very low but exactness holds | **Valid** — publish with honest labelling |
| Integration infeasible within scope | Written incompatibility report; V9 still delivers credibility via honesty |

**Invalid:** weakening verification, faking packed-bit savings for `_sim`, or citing
external paper throughput as ExactKV results.

---

## 17. What V9 explicitly does not claim

- TurboQuant, TurboQuant+, KIVI, or KVQuant **results** until integrated and swept
- Throughput, latency, speedup, tokens/sec, `runtime_seconds`, or production serving
- That ExactKV implements Sparse V, KVTC, Palu, vLLM, LMCache, or PagedAttention
- That simulated `_sim` compressors are real packed-bit backends
- That external paper accuracy or speedup numbers are reproduced
- Public launch or v1.0.0 readiness at `v0.9.0` tag

---

## 18. No-performance-claim policy

V9 inherits the global ExactKV policy ([`V8_SCOPE_STATEMENT.md`](V8_SCOPE_STATEMENT.md) §17).

**Forbidden as data fields, table columns, or metric keys:**

```
tokens_per_second
throughput
latency
speedup
runtime_seconds
active_gpu_kv_bytes
```

These terms may appear **only** in explicit negation prose or future methodology
requirements.

**V9 additionally states:**

- Real-backend integration evaluates **acceptance and memory honesty**, not speed.
- RunPod GPU usage is for **backend fidelity and larger-model exactness**, not
  throughput leaderboards.
- `total_kv_footprint_bytes` remains a conservative accounting sum unless §9
  profiling is separately approved.

---

## 19. Updated path to v1.0.0

| Version | Focus | Relationship to launch |
|---|---|---|
| **v0.8.0** ✅ | Serving harness (V8) | Research milestone; not launch |
| **v0.9.0** (V9 target) | Real backend gauntlet; Exp 008/009; RunPod Phase E | Credibility phase; **not** public launch |
| **v0.10.0** (V10) | Sparse V, attention logging, divergence forensics | Research depth |
| **v0.11.0** (V11) | Sidecar serving probes, GPU profiling methodology | Scale/serving context |
| **v1.0.0** (V12) | Raw bundle, launch narrative, `PROJECT_STATUS_v1.0.0` | **Public launch** — only when deferred register bars met |

**v0.9.0 tag criteria (Phase F):**

- [x] Phase A TurboQuant feasibility doc ([`TURBOQUANT_INTEGRATION_RESEARCH.md`](TURBOQUANT_INTEGRATION_RESEARCH.md))
- [x] Phase B adapter smoke (`exactkv_failures == 0`) — [`TURBOQUANT_ADAPTER_PROTOTYPE.md`](TURBOQUANT_ADAPTER_PROTOTYPE.md)
- [x] Experiment 008 complete — [`EXPERIMENT_008_TURBOQUANT_PYTHON.md`](EXPERIMENT_008_TURBOQUANT_PYTHON.md)
- [x] Phase D1 KIVI/KVQuant decision documented — [`KIVI_KVQUANT_INTEGRATION_RESEARCH.md`](KIVI_KVQUANT_INTEGRATION_RESEARCH.md)
- [x] Phase D2 KIVI offline adapter — [`KIVI_ADAPTER_PROTOTYPE.md`](KIVI_ADAPTER_PROTOTYPE.md)
- [x] Experiment 009 complete — [`EXPERIMENT_009_KIVI_OFFLINE.md`](EXPERIMENT_009_KIVI_OFFLINE.md)
- [x] Phase D4 KVQuant RunPod validation — [`KVQUANT_RUNPOD_VALIDATION.md`](KVQUANT_RUNPOD_VALIDATION.md)
- [x] Phase D5 KVQuant simquant adapter prototype — [`KVQUANT_ADAPTER_PROTOTYPE.md`](KVQUANT_ADAPTER_PROTOTYPE.md)
- [x] Phase D6 Experiment 010 — [`EXPERIMENT_010_KVQUANT_SIM.md`](EXPERIMENT_010_KVQUANT_SIM.md)
- [x] Phase E Experiment 011 — [`EXPERIMENT_011_LARGER_MODEL_VALIDATION.md`](EXPERIMENT_011_LARGER_MODEL_VALIDATION.md)
- [ ] Phase F release notes v0.9.0 **or** documented blocker
- [ ] `RELEASE_NOTES_V0.9.0.md`, updated experiment index
- [ ] No forbidden performance fields in code/docs/reports

**v1.0.0** requires V9–V11 substance plus launch package — see
[`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md).

---

## 20. Proposed V9 phases

| Phase | Scope | Deliverable | Status |
|---|---|---|---|
| **Phase 0** (this document) | Scope statement only; no code | `docs/V9_SCOPE_STATEMENT.md` | ✅ Complete |
| **Phase A** | TurboQuant / TurboQuant+ deep feasibility | Install, API/cache-format doc, device/model matrix, go/no-go | ✅ Complete — **restricted go** (Python path; see research doc §23) |
| **Phase B** | TurboQuant adapter prototype (if feasible) | Adapter code + smoke exactness tests | ✅ Complete — see [`TURBOQUANT_ADAPTER_PROTOTYPE.md`](TURBOQUANT_ADAPTER_PROTOTYPE.md) |
| **Phase C** | Experiment 008 + report | `EXPERIMENT_008_TURBOQUANT_PYTHON.md`; gitignored JSON/CSV | ✅ Complete — 272 cells, `exactkv_failures == 0` |
| **Phase D1** | KIVI / KVQuant feasibility (research only) | [`KIVI_KVQUANT_INTEGRATION_RESEARCH.md`](KIVI_KVQUANT_INTEGRATION_RESEARCH.md); scratch inspect script | ✅ Complete — **KIVI restricted go**; KVQuant deferred pending RunPod |
| **Phase D2** | KIVI restricted offline adapter prototype | [`KIVI_ADAPTER_PROTOTYPE.md`](KIVI_ADAPTER_PROTOTYPE.md); smoke `exactkv_failures == 0` | ✅ Complete |
| **Phase D3** | Experiment 009 — KIVI offline evaluation | [`EXPERIMENT_009_KIVI_OFFLINE.md`](EXPERIMENT_009_KIVI_OFFLINE.md); 272 cells, `exactkv_failures == 0` | ✅ Complete |
| **Phase D4** | KVQuant RunPod validation + adapter decision | [`KVQUANT_RUNPOD_VALIDATION.md`](KVQUANT_RUNPOD_VALIDATION.md); D4b L40S GPU | ✅ Complete (D4a static + D4b GPU) |
| **Phase D5** | KVQuant simquant adapter prototype (if approved) | [`KVQUANT_ADAPTER_PROTOTYPE.md`](KVQUANT_ADAPTER_PROTOTYPE.md); smoke `exactkv_failures == 0` | ✅ Complete — factory-only; not in default registry |
| **Phase D6** | Experiment 010 — KVQuant simquant evaluation | [`EXPERIMENT_010_KVQUANT_SIM.md`](EXPERIMENT_010_KVQUANT_SIM.md); 272 cells, `exactkv_failures == 0` | ✅ Complete |
| **Phase E** | RunPod larger-model validation | [`EXPERIMENT_011_LARGER_MODEL_VALIDATION.md`](EXPERIMENT_011_LARGER_MODEL_VALIDATION.md); 1.5B, 238 cells | ✅ Complete |
| **Phase F** | Release notes v0.9.0, index, bundle plan, v1.0.0 readiness decision | `RELEASE_NOTES_V0.9.0.md` | Pending F |

> Phases A–F require **separate explicit approval** before code or experiments.
> Phase 0 introduces no code and no behaviour change.

---

## 21. Related documents

| Document | Relevance |
|---|---|
| [`RELEASE_NOTES_V0.8.0.md`](RELEASE_NOTES_V0.8.0.md) | V9 baseline |
| [`BACKEND_ADAPTER_INTERFACE.md`](BACKEND_ADAPTER_INTERFACE.md) | Adapter contract |
| [`KVPRESS_KNORM_VALIDATION.md`](KVPRESS_KNORM_VALIDATION.md) | Hook/isolation gates |
| [`EXPERIMENT_005_KVPRESS_KNORM.md`](EXPERIMENT_005_KVPRESS_KNORM.md) | Real backend precedent |
| [`EXPERIMENT_006C_BOUNDARY_DEPTH_ABLATION.md`](EXPERIMENT_006C_BOUNDARY_DEPTH_ABLATION.md) | Sim baseline to beat |
| [`EXPERIMENT_007_SERVING_CONTEXT.md`](EXPERIMENT_007_SERVING_CONTEXT.md) | Serving harness baseline |
| [`RELATED_WORK_KV_CACHE_COMPRESSION.md`](RELATED_WORK_KV_CACHE_COMPRESSION.md) | Backend survey |
| [`RESEARCH_BACKLOG.md`](RESEARCH_BACKLOG.md) | B1–B4 backlog items |
| [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md) | V9–v1.0.0 tracker |
| [`TURBOQUANT_INTEGRATION_RESEARCH.md`](TURBOQUANT_INTEGRATION_RESEARCH.md) | Phase A feasibility + restricted go |
| [`TURBOQUANT_ADAPTER_PROTOTYPE.md`](TURBOQUANT_ADAPTER_PROTOTYPE.md) | Phase B restricted Python adapter |
| [`EXPERIMENT_008_TURBOQUANT_PYTHON.md`](EXPERIMENT_008_TURBOQUANT_PYTHON.md) | Phase C Experiment 008 results |
| [`KIVI_KVQUANT_INTEGRATION_RESEARCH.md`](KIVI_KVQUANT_INTEGRATION_RESEARCH.md) | Phase D1 KIVI/KVQuant feasibility + recommendation |
| [`KIVI_ADAPTER_PROTOTYPE.md`](KIVI_ADAPTER_PROTOTYPE.md) | Phase D2 restricted offline KIVI adapter |
| [`EXPERIMENT_009_KIVI_OFFLINE.md`](EXPERIMENT_009_KIVI_OFFLINE.md) | Phase D3 Experiment 009 results |
| [`KVQUANT_RUNPOD_VALIDATION.md`](KVQUANT_RUNPOD_VALIDATION.md) | Phase D4 KVQuant RunPod validation + D5 adapter decision |
| [`KVQUANT_ADAPTER_PROTOTYPE.md`](KVQUANT_ADAPTER_PROTOTYPE.md) | Phase D5 restricted KVQuant simquant adapter prototype |
| [`EXPERIMENT_010_KVQUANT_SIM.md`](EXPERIMENT_010_KVQUANT_SIM.md) | Phase D6 Experiment 010 KVQuant simquant evaluation |
| [`EXPERIMENT_011_LARGER_MODEL_VALIDATION.md`](EXPERIMENT_011_LARGER_MODEL_VALIDATION.md) | Phase E larger-model RunPod validation (Qwen2.5-1.5B) |
| [`PROJECT_STATUS_V0.8.0.md`](PROJECT_STATUS_V0.8.0.md) | Pre-V9 status |

---

## Attribution

**VeriCache** (draft-then-verify): Yao et al., arXiv:2605.17613, 2026.

**Backend references** (not implemented by ExactKV in Phase 0):

- TurboQuant: Zandieh et al., ICLR 2026, arXiv:2504.19874
- TurboQuant+: community workspace — evaluation context only
- KIVI: Liu et al., ICLR 2024, arXiv:2402.02750
- KVQuant: Hooper et al., NeurIPS 2024, arXiv:2401.18079

ExactKV does not reproduce or claim external-backend performance or accuracy results.
