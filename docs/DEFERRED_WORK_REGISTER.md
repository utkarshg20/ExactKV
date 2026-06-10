# ExactKV Deferred Work Register

**Status:** Living register of major deferred items. **Nothing listed here is
implemented** unless a linked experiment or release note says otherwise.

> Guardrails: `exactkv_failures == 0` on every published experiment; no throughput,
> latency, speedup, runtime, or production-serving claims; `_sim` ≠ packed-bit storage;
> external paper results are **not** ExactKV results.

**Version path:** V9 ✅ → **V10 (active, Phase 1 complete)** → V11 → V12/v1.0.0 (public launch).

**V9 scope:** [`V9_SCOPE_STATEMENT.md`](V9_SCOPE_STATEMENT.md) — **complete** (`v0.9.0`).
**V10 scope:** [`V10_SCOPE_STATEMENT.md`](V10_SCOPE_STATEMENT.md) — **Phase 1 complete**.
**V10 suites:** [`V10_PROMPT_SUITES.md`](V10_PROMPT_SUITES.md).

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

## V10 — Evaluation suite hardening and divergence forensics (active)

| ID | Item | Status | V10 phase | Success criteria |
|---|---|---|---|---|
| D26 | **`core_v2` + category benchmark suites** | **Complete** | Phase 1 ✅ | 128 prompts; [`V10_PROMPT_SUITES.md`](V10_PROMPT_SUITES.md); validator + tests |
| D27 | **Draft length sensitivity (2/4/8)** | **Planned** | Phase 3 (Exp 013) | Documented acceptance/divergence vs `draft_len` |
| D28 | **Generation length sensitivity (16/32/64)** | **Planned** | Phase 3 (Exp 013) | Phased sweeps; no performance claims |
| D29 | **Category-stratified divergence forensics** | **Planned** | Phase 3 (Exp 013) | Supersede 006A proxy where feasible |
| — | **Per-category leaderboards + prompt win/loss** | **Planned** | Phase 2 (Exp 012) | Category tables; regression ID tracking |
| D6 | **Sparse V dequantization** | Deferred (V10 research) | Phase 3+ | Acceptance-only evaluation |
| D7 | **True attention logging** | Deferred (V10 optional) | Phase 3+ | Small subset; no fabricated weights |
| D8 | **Per-layer/head/token divergence forensics** | **Planned** | Phase 3 | Layer/head where weights exist |
| D9 | **Pre-RoPE key quantization experiments** | Deferred | — | Compare vs post-RoPE baselines |
| D10 | **Boundary / layer-policy extensions** | Deferred | — | N>4 only with explicit approval |

**Phase 1 (complete):** suites + validator — [`V10_PROMPT_SUITES.md`](V10_PROMPT_SUITES.md).
**Phase 2 (next):** Experiment 012.

---

## V11 — Scale and serving gauntlet

| ID | Item | Status | Success criteria |
|---|---|---|---|
| D11 | **Direct vLLM integration** | No-go (Phase A) | Re-approval only if safe full-KV export path demonstrated |
| D12 | **LMCache integration** | No-go (Phase A) | Re-approval only if ownership + verify isolation proven |
| D13 | **vLLM / LMCache sidecar probe** | Deferred | Metadata-only or isolated sidecar evaluation |
| D14 | **Active GPU memory profiling** | Deferred | Approved methodology; distinct from `total_kv_footprint_bytes` |
| D16 | **PagedAttention kernel integration** | Deferred | Local harness remains default |

V11 covers serving/profiling/optional production-path work — **not** V10 evaluation-suite scope.

---

## V12 / v1.0.0 — Final public launch package

| ID | Item | Status | Success criteria |
|---|---|---|---|
| D17 | **Raw report bundle** | Deferred | Curated archive for experiments 001–011+ with manifests |
| D18 | **Final public launch narrative** | Deferred | Reviewed post/docs; explicit negation of performance claims |
| D19 | **Project status v1.0.0** | Deferred | Supersedes [`PROJECT_STATUS_V0.9.0.md`](PROJECT_STATUS_V0.9.0.md) |
| D20 | **Git tag `v1.0.0`** | Deferred | V10–V11 exit criteria met |

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

- [`V10_SCOPE_STATEMENT.md`](V10_SCOPE_STATEMENT.md) — active V10 formal scope
- [`V10_SCOPE_DRAFT.md`](V10_SCOPE_DRAFT.md) — superseded draft
- [`RELEASE_NOTES_V0.9.0.md`](RELEASE_NOTES_V0.9.0.md) — what shipped in v0.9.0
- [`ROADMAP.md`](ROADMAP.md) — version planning
- [`RESEARCH_BACKLOG.md`](RESEARCH_BACKLOG.md) — experiment ideas
