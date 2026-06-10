# ExactKV v0.9.0 Release Notes

**Status:** V9 complete (Phases 0–F). **Tag:** `v0.9.0` — **research milestone, not public launch.**
**Base:** Builds on `v0.8.0` (serving-context harness, Experiment 007).

> **V9 is a real-backend credibility release, not a performance or production release.**
> V9 does **not** integrate TurboQuant production runtime (llama.cpp/MLX), KIVI CUDA/Triton,
> or KVQuant deployment CUDA. Real backends are **factory-only**, isolated-venv adapters —
> **not** in the default compressor registry.
> `total_kv_footprint_bytes` is a conservative accounting sum, not measured peak GPU memory.
> Active GPU memory is **not** reported.
> ExactKV does **not** claim speedup, throughput, latency, runtime, tokens/sec, or
> production readiness.
> External paper results are **not** ExactKV results.

---

## 1. V9 summary

V9 integrates and **evaluates** three restricted real-backend families behind
ExactKV's existing `BackendAdapter` boundary — TurboQuant Python, KIVI offline
simulate, and KVQuant simquant — while preserving the exactness gate on every
published cell. V9 also validates **Qwen/Qwen2.5-1.5B** on RunPod GPU (Experiment 011).

V9 delivers:

- **Phases A–C:** TurboQuant feasibility, adapter prototype, Experiment 008.
- **Phases D1–D3:** KIVI feasibility, offline adapter, Experiment 009.
- **Phases D4–D6:** KVQuant RunPod validation, simquant adapter, Experiment 010.
- **Phase E:** Larger-model RunPod validation — Experiment 011.
- **Phase F:** This file, project status, experiment index, V10 scope draft.

---

## 2. What V9 adds

| Deliverable | Location |
|---|---|
| V9 scope statement | [`docs/V9_SCOPE_STATEMENT.md`](V9_SCOPE_STATEMENT.md) |
| TurboQuant feasibility (Phase A) | [`docs/TURBOQUANT_INTEGRATION_RESEARCH.md`](TURBOQUANT_INTEGRATION_RESEARCH.md) |
| TurboQuant adapter prototype (Phase B) | [`docs/TURBOQUANT_ADAPTER_PROTOTYPE.md`](TURBOQUANT_ADAPTER_PROTOTYPE.md) |
| Experiment 008 (Phase C) | [`docs/EXPERIMENT_008_TURBOQUANT_PYTHON.md`](EXPERIMENT_008_TURBOQUANT_PYTHON.md) |
| KIVI/KVQuant feasibility (Phase D1) | [`docs/KIVI_KVQUANT_INTEGRATION_RESEARCH.md`](KIVI_KVQUANT_INTEGRATION_RESEARCH.md) |
| KIVI adapter prototype (Phase D2) | [`docs/KIVI_ADAPTER_PROTOTYPE.md`](KIVI_ADAPTER_PROTOTYPE.md) |
| Experiment 009 (Phase D3) | [`docs/EXPERIMENT_009_KIVI_OFFLINE.md`](EXPERIMENT_009_KIVI_OFFLINE.md) |
| KVQuant RunPod validation (Phase D4) | [`docs/KVQUANT_RUNPOD_VALIDATION.md`](KVQUANT_RUNPOD_VALIDATION.md) |
| KVQuant adapter prototype (Phase D5) | [`docs/KVQUANT_ADAPTER_PROTOTYPE.md`](KVQUANT_ADAPTER_PROTOTYPE.md) |
| Experiment 010 (Phase D6) | [`docs/EXPERIMENT_010_KVQUANT_SIM.md`](EXPERIMENT_010_KVQUANT_SIM.md) |
| Experiment 011 (Phase E) | [`docs/EXPERIMENT_011_LARGER_MODEL_VALIDATION.md`](EXPERIMENT_011_LARGER_MODEL_VALIDATION.md) |
| v0.9.0 documentation package (Phase F) | This file, [`PROJECT_STATUS_V0.9.0.md`](PROJECT_STATUS_V0.9.0.md), [`V10_SCOPE_DRAFT.md`](V10_SCOPE_DRAFT.md) |

**Unchanged:** generation logic, verification logic, default compressor registry (15
built-in), report schema (additive manifest fields in experiment JSON only).

---

## 3. TurboQuant track

| Phase | Deliverable | Result |
|---|---|---|
| **A** | Feasibility research | Restricted **go** — Python `KVCacheCompressor` path |
| **B** | `TurboQuantPythonAdapter` prototype | Smoke `exactkv_failures == 0`; factory-only |
| **C** | **Experiment 008** | **272 cells**, `exactkv_failures == 0` |

**Headline:** `turboquant_python_k3_v3` mean acceptance **0.435** vs `int8` **0.961** on
0.5B core suite (`draft_len=4`, `max_new_tokens=16`).

**Restrictions:** Restricted **Python adapter only** — not llama.cpp, MLX, GGUF, or
TurboQuant+ production runtime. `supports_real_bytes_claim=False`. Not in default registry.

---

## 4. KIVI track

| Phase | Deliverable | Result |
|---|---|---|
| **D1** | Feasibility research | Restricted **go** — offline `models.utils_quant` simulate |
| **D2** | `KIVIOfflineAdapter` prototype | Smoke `exactkv_failures == 0`; factory-only |
| **D3** | **Experiment 009** | **272 cells**, `exactkv_failures == 0` |

**Headline:** `kivi_offline_k2_v2` mean acceptance **0.012** vs `int8` **0.961** on
0.5B core suite.

**Restrictions:** Restricted **offline simulate path only** — not KIVI CUDA/Triton,
not `LlamaForCausalLM_KIVI`. `supports_real_bytes_claim=False`. Not in default registry.

---

## 5. KVQuant track

| Phase | Deliverable | Result |
|---|---|---|
| **D4a** | Static feasibility | Documented integration paths |
| **D4b** | RunPod GPU validation (L40S) | Quantizer pickle, `QuantLinearSim` forward, draft/verify isolation |
| **D5** | `KVQuantSimAdapter` prototype | Smoke `exactkv_failures == 0`; factory-only |
| **D6** | **Experiment 010** | **272 cells**, `exactkv_failures == 0` |

**Headline:** `kvquant_sim_qwen05b` mean acceptance **0.792** vs `int8` **0.966** on
0.5B core suite (RunPod CUDA fp16).

**Restrictions:** Restricted **simquant adapter only** — pre-RoPE `k_proj`/`v_proj`
`QuantLinearSim` on draft-model clone. **Not** KVQuant deployment CUDA. **Not** forked
transformers deployment. External quantizers pickle required (not committed).
`supports_real_bytes_claim=False`. Not in default registry.

---

## 6. Larger-model validation

**Experiment 011** — [`EXPERIMENT_011_LARGER_MODEL_VALIDATION.md`](EXPERIMENT_011_LARGER_MODEL_VALIDATION.md)

| Metric | Value |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B`, float16, CUDA (RunPod L40S) |
| Cells | **238** (34 prompts × 7 compressors) |
| **ExactKV failures** | **0** |
| `int8` accept | **0.980** |
| `k8_v4_boundary4_v8_sim` accept | **0.954** |
| `k8_v4_sim` accept | **0.945** |
| boundary4 − k8_v4_sim Δ | **+0.009** (vs **+0.051** on 0.5B Exp 010) |

Layer-aware boundary V **still beats** uniform `k8_v4_sim` on 1.5B, but by a **smaller margin**.
Optional Qwen2.5-3B stretch **not run**. KVQuant 1.5B quantizer artifact generated
post-sweep; not in primary 238-cell panel.

---

## 7. Cross-backend acceptance summary (0.5B, core suite)

_Reference panel: Experiments 008–010, `draft_len=4`, `max_new_tokens=16`._

| Compressor / backend | Mean accept (approx.) | Source |
|---|---:|---|
| `int8` | **0.966** | Exp 010 live baselines |
| `k8_v4_boundary4_v8_sim` | **0.950** | Exp 010 live baselines |
| `k_full_v8` | **0.990** | Exp 010 live baselines |
| `kvquant_sim_qwen05b` | **0.792** | Exp 010 |
| `turboquant_python_k3_v3` | **0.435** | Exp 008 |
| `kivi_offline_k2_v2` | **0.012** | Exp 009 |

Exactness gate holds for **all** rows. Acceptance spread is large — external backends
are **not** interchangeable under ExactKV's verify path without per-adapter evaluation.

---

## 8. Main implications

1. **Exactness is backend-agnostic** — three external quantizer families preserve
   `exactkv_output_ids == full_output_ids` when wrapped behind `BackendAdapter`.
2. **Acceptance is not** — KIVI offline (0.012) and TurboQuant Python (0.435) diverge
   sharply from INT8 (~0.96) and KVQuant simquant (0.792).
3. **Simulated layer-aware V remains strong** — `k8_v4_boundary4_v8_sim` (~0.95) beats
   all three restricted real backends on 0.5B acceptance (not memory claims).
4. **Scale transfers for exactness** — 1.5B RunPod validation passes with zero failures.
5. **Evaluation suite is still narrow** — 34-prompt core panel is controlled engineering
   evidence, not a comprehensive public benchmark (see V10).

---

## 9. What V9 proves

- ExactKV can wrap **TurboQuant Python**, **KIVI offline simulate**, and **KVQuant simquant**
  behind `BackendAdapter` while preserving `exactkv_failures == 0`.
- Honest workspace-memory accounting extends to non-registry real backends
  (`supports_real_bytes_claim=False` where appropriate).
- Layer-aware simulated V advantage (**boundary4 > k8_v4_sim**) holds on **1.5B** (smaller margin).
- Larger-model exactness gate passes on RunPod GPU without changing generation or verification logic.

---

## 10. What V9 does not prove

- TurboQuant+, KIVI, or KVQuant **production** paths (CUDA kernels, llama.cpp, MLX, deployment CUDA).
- Upstream paper throughput, bandwidth, perplexity, or memory claims.
- That restricted real backends beat simulated INT8 policies on acceptance in all settings.
- Production serving readiness, multi-request batching, or vLLM/LMCache compatibility.
- That the 34-prompt core suite supports broad public claims (V10 required).
- Active GPU memory peaks or speedup versus uncompressed inference.

---

## 11. Known limitations

- **Factory-only real backends** — TurboQuant, KIVI, KVQuant not in default registry or default deps.
- **Isolated venvs** — each real backend requires its own environment (Python path, KIVI repo, KVQuant venv).
- **Single primary model** for backend sweeps (`Qwen2.5-0.5B`); 1.5B validated separately (Exp 011).
- **`_sim` compressors** use int8 containers — not packed-bit storage.
- **No active GPU memory** reporting.
- **No public launch narrative** or curated raw report bundle (deferred to v1.0.0).
- **KVQuant** requires per-model quantizers pickle; 1.5B artifact exists but was not in Exp 011 primary sweep.

---

## 12. Why v1.0.0 is not next

v0.9.0 completes the **real-backend credibility chapter** but public launch requires
V10 evaluation-suite hardening and divergence forensics before credible broad claims:

| Gap | Target |
|---|---|
| Narrow 34-prompt core suite | V10 — `core_v2`, category suites, sensitivity sweeps |
| Proxy-only divergence analysis (006A) | V10 — category/token/layer forensics |
| Sparse V, true attention logging | V10 (research gauntlet) |
| Scale/serving probes | V11 |
| Launch narrative + raw bundle | v1.0.0 |

See [`V10_SCOPE_DRAFT.md`](V10_SCOPE_DRAFT.md) and [`PROJECT_STATUS_V0.9.0.md`](PROJECT_STATUS_V0.9.0.md).

---

## 13. V10 recommendation

Proceed to **V10: Evaluation Suite Hardening and Divergence Forensics** before any
public launch narrative. V10 should:

- Expand prompt taxonomy and benchmark suites (not throughput benchmarks).
- Run draft-length and generation-length sensitivity on validated model matrix.
- Upgrade divergence analysis beyond 006A proxies.
- Preserve `exactkv_failures == 0` as the hard gate.
- Re-evaluate restricted real backends under broader, category-stratified panels.

V10 is **not** a performance benchmark phase.

---

## 14. Reproduction and raw artifact policy

**Experiments 008–011 runners:**

```bash
# 008 — TurboQuant Python (isolated venv)
PYTHONPATH=vendor/turboquant_plus .venv-turboquant/bin/python scripts/run_experiment_008_turboquant_python.py

# 009 — KIVI offline (PYTHONPATH to KIVI repo)
PYTHONPATH=/path/to/KIVI .venv-turboquant/bin/python scripts/run_experiment_009_kivi_offline.py

# 010 — KVQuant simquant (KVQuant venv, CUDA)
EXACTKV_KVQUANT_QUANTIZERS=/path/to/quantizers.pickle python scripts/run_experiment_010_kvquant_sim.py

# 011 — Larger-model validation (RunPod CUDA)
python scripts/run_experiment_011_larger_model_validation.py
```

**Artifacts (gitignored — not committed):**

- `reports/experiment_008_turboquant_python.{json,csv}`
- `reports/experiment_009_kivi_offline.{json,csv}`
- `reports/experiment_010_kvquant_sim.{json,csv}`
- `reports/experiment_011_qwen15b_validation.{json,csv}`
- `quantizers*.pickle`, model weights, RunPod venvs

Published evidence is the **Markdown experiment reports** and this release package.

---

## 15. v0.9.0 tag readiness

| Criterion | Status |
|---|---|
| Phases A–E complete | ✅ |
| Experiments 008–011 (`exactkv_failures == 0`) | ✅ |
| Release notes (this file) | ✅ |
| Project status v0.9.0 | ✅ |
| V10 scope draft | ✅ |
| Experiment index 001–011 | ✅ |
| Deferred-work register updated | ✅ |
| No forbidden performance fields in docs | ✅ (audited) |
| Public launch narrative | ❌ **Deferred** (V10+ / v1.0.0) |

**Ready to tag `v0.9.0`.**

---

## Attribution

**VeriCache** (draft-then-verify): Yao et al., arXiv:2605.17613, 2026.

**Backend references** (evaluation context only; restricted adapters in V9):

- TurboQuant: Zandieh et al., ICLR 2026, arXiv:2504.19874
- KIVI: Liu et al., ICLR 2024, arXiv:2402.02750
- KVQuant: Hooper et al., NeurIPS 2024, arXiv:2401.18079

ExactKV does not reproduce or claim external-backend performance or accuracy results.
