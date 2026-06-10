# ExactKV Deferred Work Register

**Status:** Living register of major deferred items. **Nothing listed here is
implemented** unless a linked experiment or release note says otherwise.

> Guardrails: `exactkv_failures == 0` on every published experiment; no throughput,
> latency, speedup, runtime, or production-serving claims; `_sim` ≠ packed-bit storage;
> external paper results are **not** ExactKV results.

**Version path:** V9 ✅ → **V10 ✅ (`v0.10.0`)** → **V11 (active)** → v1.0.0 (public launch).

**V9 scope:** [`V9_SCOPE_STATEMENT.md`](V9_SCOPE_STATEMENT.md) — **complete** (`v0.9.0`).
**V10 scope:** [`V10_SCOPE_STATEMENT.md`](V10_SCOPE_STATEMENT.md) — **complete** (`v0.10.0`).
**V10 readiness:** [`V10_READINESS_ASSESSMENT.md`](V10_READINESS_ASSESSMENT.md).
**Experiment 014:** [`EXPERIMENT_014_REAL_BACKEND_SPOTCHECKS.md`](EXPERIMENT_014_REAL_BACKEND_SPOTCHECKS.md).
**V10 suites:** [`V10_PROMPT_SUITES.md`](V10_PROMPT_SUITES.md).
**Experiment 012:** [`EXPERIMENT_012_EVAL_SUITE_EXPANSION.md`](EXPERIMENT_012_EVAL_SUITE_EXPANSION.md).
**Experiment 013:** [`EXPERIMENT_013_SENSITIVITY_FORENSICS.md`](EXPERIMENT_013_SENSITIVITY_FORENSICS.md).

---

## V9 — Real backend integration gauntlet (complete)

| ID | Item | Status | V9 phase | Success criteria |
|---|---|---|---|---|
| D1 | **TurboQuant full integration** | **Evaluated (Phase C)** | ✅ | Exp 008: 272 cells, `exactkv_failures == 0`; accept **0.435** — [`EXPERIMENT_008_TURBOQUANT_PYTHON.md`](EXPERIMENT_008_TURBOQUANT_PYTHON.md) |
| D2 | **TurboQuant+ full integration** | **Evaluated (Python path only)** | ✅ | Production llama.cpp/MLX deferred — [`RELEASE_NOTES_V0.9.0.md`](RELEASE_NOTES_V0.9.0.md) |
| D3 | **KIVI adapter** | **Evaluated (Phase D3)** | ✅ | Exp 009: accept **0.012** — [`EXPERIMENT_009_KIVI_OFFLINE.md`](EXPERIMENT_009_KIVI_OFFLINE.md) |
| D4 | **KVQuant-style adapter** | **Evaluated (Phase D6)** | ✅ | Exp 010: accept **0.792** — [`EXPERIMENT_010_KVQUANT_SIM.md`](EXPERIMENT_010_KVQUANT_SIM.md) |
| D5 | **KVTC / Palu feasibility** | Deferred (optional) | — | Not in V9 scope |
| D15 | **Larger-model RunPod validation** | **Complete (Phase E)** | ✅ | Exp 011: 1.5B, 238 cells — [`EXPERIMENT_011_LARGER_MODEL_VALIDATION.md`](EXPERIMENT_011_LARGER_MODEL_VALIDATION.md) |

---

## V10 — Evaluation suite hardening and divergence forensics (complete)

| ID | Item | Status | V10 phase | Success criteria |
|---|---|---|---|---|
| D26 | **`core_v2` + category benchmark suites** | **Complete** | Phase 1 ✅ | 128 prompts; [`V10_PROMPT_SUITES.md`](V10_PROMPT_SUITES.md); validator + tests |
| D27 | **Draft length sensitivity (2/4/8)** | **Complete** | Phase 3 ✅ | Exp 013: 2160 cells, `exactkv_failures == 0` |
| D28 | **Generation length sensitivity (16/32/64)** | **Complete** | Phase 3 ✅ | Exp 013 full 3×3 grid |
| D29 | **Category-stratified divergence forensics** | **Complete** | Phase 3 ✅ | Token-type + structured-output heuristics; no attention weights |
| — | **Per-category leaderboards + prompt win/loss** | **Complete** | Phase 2 ✅ | Exp 012: 896 cells, `exactkv_failures == 0` |
| — | **Real-backend category spot-checks** | **Complete** | Phase 4 ✅ | Exp 014: 280 cells — [`EXPERIMENT_014_REAL_BACKEND_SPOTCHECKS.md`](EXPERIMENT_014_REAL_BACKEND_SPOTCHECKS.md) |
| — | **V10 readiness assessment** | **Complete** | Phase 5 ✅ | [`V10_READINESS_ASSESSMENT.md`](V10_READINESS_ASSESSMENT.md) |
| D6 | **Sparse V dequantization** | **Moved to V11 / research** | — | Acceptance-only evaluation |
| D7 | **True attention logging** | **Moved to V11** | — | Small subset; no fabricated weights |
| D8 | **Per-layer/head/token divergence forensics** | **Moved to V11** | — | Requires attention weights or documented blocker |
| D9 | **Pre-RoPE key quantization experiments** | Deferred | — | Compare vs post-RoPE baselines |
| D10 | **Boundary / layer-policy extensions** | Deferred | — | N>4 only with explicit approval |
| — | **1.5B on expanded V10 suites** | **Moved to V11** | — | Exp 011 is legacy `core` only |

**V10 exit:** tag **`v0.10.0`** — not v1.0.0. See [`RELEASE_NOTES_V0.10.0.md`](RELEASE_NOTES_V0.10.0.md).

---

## V11 — Scale, serving, and launch hardening (active)

| ID | Item | Status | Success criteria |
|---|---|---|---|
| — | **1.5B+ on V10 expanded suites** | **Next** | `exactkv_failures == 0`; per-category tables |
| D6 | **Sparse V dequantization** | Deferred | Acceptance-only evaluation |
| D7 | **True attention logging** | Deferred | Small subset; no fabricated weights |
| D8 | **Per-layer/head divergence forensics** | Deferred | Where weights exist |
| D11 | **Direct vLLM integration** | No-go (Phase A) | Re-approval only if safe full-KV export path demonstrated |
| D12 | **LMCache integration** | No-go (Phase A) | Re-approval only if ownership + verify isolation proven |
| D13 | **vLLM / LMCache sidecar probe** | **Blocker for v1.0.0** | Metadata-only or isolated sidecar evaluation |
| D14 | **Active GPU memory profiling** | **Blocker for v1.0.0** | Approved methodology; distinct from `total_kv_footprint_bytes` |
| D16 | **PagedAttention kernel integration** | Deferred | Local harness remains default |

V11 covers multi-model validation, serving/profiling probes, and optional forensics depth — **not** a performance benchmark release.

---

## v1.0.0 — Final public launch package (after V11)

| ID | Item | Status | Success criteria |
|---|---|---|---|
| D17 | **Raw report bundle** | **Blocker** | Curated archive for experiments 001–014+ with manifests |
| D18 | **Final public launch narrative** | **Blocker** | Reviewed post/docs; explicit negation of performance claims; draft only until gates met |
| D19 | **Project status v1.0.0** | Deferred | Supersedes [`PROJECT_STATUS_V0.10.0.md`](PROJECT_STATUS_V0.10.0.md) |
| D20 | **Git tag `v1.0.0`** | Deferred | V11 exit criteria met |

---

## Cross-cutting (ongoing)

| ID | Item | Status | Notes |
|---|---|---|---|
| D21 | **Sampling / parallel verify / bonus tokens** | Deferred | Out of scope until explicit future version |
| D22 | **Multi-request batching** | Deferred | Serving-scale feature |
| D23 | **CPU offload / CUDA kernels** | Deferred | No custom kernels in ExactKV today |
| D24 | **Broader kvpress** | Deferred | KnormPress only (V6) |
| D25 | **RESEARCH_BACKLOG sync** | Ongoing | Keep aligned with this register |

---

## Phase C reminder (V8)

vLLM and LMCache **direct integration** were judged **no-go** in
[`SERVING_CONTEXT_FEASIBILITY.md`](SERVING_CONTEXT_FEASIBILITY.md). Items D11–D12
remain **deferred, not forgotten**.

---

## Related

- [`V10_SCOPE_STATEMENT.md`](V10_SCOPE_STATEMENT.md) — V10 formal scope (complete)
- [`V10_READINESS_ASSESSMENT.md`](V10_READINESS_ASSESSMENT.md) — Phase 5 readiness
- [`RELEASE_NOTES_V0.10.0.md`](RELEASE_NOTES_V0.10.0.md) — v0.10.0 changelog
- [`V10_SCOPE_DRAFT.md`](V10_SCOPE_DRAFT.md) — superseded draft
- [`RELEASE_NOTES_V0.9.0.md`](RELEASE_NOTES_V0.9.0.md) — what shipped in v0.9.0
- [`ROADMAP.md`](ROADMAP.md) — version planning
- [`RESEARCH_BACKLOG.md`](RESEARCH_BACKLOG.md) — experiment ideas
