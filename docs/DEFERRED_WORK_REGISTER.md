# ExactKV Deferred Work Register

**Status:** Living register of major deferred items. **Nothing listed here is
implemented** unless a linked experiment or release note says otherwise.

> Guardrails: `exactkv_failures == 0` on every published experiment; no throughput,
> latency, speedup, runtime, or production-serving claims; `_sim` ≠ packed-bit storage;
> external paper results are **not** ExactKV results.

**Version path:** V9 (active) → V10 → V11 → V12/v1.0.0 (public launch).

**V9 scope:** [`V9_SCOPE_STATEMENT.md`](V9_SCOPE_STATEMENT.md) — Phase 0 complete.

---

## V9 — Real backend integration gauntlet (active / planned)

| ID | Item | Status | V9 phase | Success criteria |
|---|---|---|---|---|
| D1 | **TurboQuant full integration** | **Evaluated (Phase C)** | D+ / Exp 009 anchor | Exp 008: 272 cells, `exactkv_failures == 0`; accept **0.435** vs `int8` **0.961**; see [`EXPERIMENT_008_TURBOQUANT_PYTHON.md`](EXPERIMENT_008_TURBOQUANT_PYTHON.md) |
| D2 | **TurboQuant+ full integration** | **Evaluated (Python path only)** | D+ | Production llama.cpp/MLX formats still deferred; Python adapter evaluated in Exp 008 |
| D3 | **KIVI adapter** | **Evaluated (Phase D3)** | D+ / Phase E | Exp 009: 272 cells, `exactkv_failures == 0`; accept **0.012** vs `int8` **0.961**; offline simulate only — see [`EXPERIMENT_009_KIVI_OFFLINE.md`](EXPERIMENT_009_KIVI_OFFLINE.md) |
| D4 | **KVQuant-style adapter** | **RunPod-validated go (provisional, D4)** | D5 / Exp 010 | Phase D4: Qwen2.5 module-compatible; faithful path = draft clone + `_compresses_via_full_state` replay; GPU calibration scripted — see [`KVQUANT_RUNPOD_VALIDATION.md`](KVQUANT_RUNPOD_VALIDATION.md) §15 |
| D5 | **KVTC / Palu feasibility** | Planned (V9 optional) | D | Written feasibility + optional thin PoC; V5 metadata honesty |
| D15 | **Larger-model RunPod validation** | **Planned (V9)** | E | ≥1.5B Qwen2.5; `exactkv_failures == 0`; hardware manifest |

---

## V10 — Compression research gauntlet

| ID | Item | Status | Success criteria |
|---|---|---|---|
| D6 | **Sparse V dequantization** | Deferred | Attention-gated or policy-gated sparse V materialization evaluated by acceptance only |
| D7 | **True attention logging** | Deferred | Log real attention weights during prefill/decode; no fabricated weights in reports |
| D8 | **Per-layer/head/token divergence forensics** | Deferred | Upgrade 006A proxy analysis to weight-aware forensics where weights exist |
| D9 | **Pre-RoPE key quantization experiments** | Deferred | Compare acceptance vs post-RoPE `int8` / `_sim` baselines |
| D10 | **Boundary / layer-policy extensions** | Deferred | Optional N&gt;4 or attention-informed boundaries — only with explicit scope approval |

---

## V11 — Scale and serving gauntlet

| ID | Item | Status | Success criteria |
|---|---|---|---|
| D11 | **Direct vLLM integration** | No-go (Phase A) | Re-approval only if safe full-KV export path demonstrated; else remain deferred |
| D12 | **LMCache integration** | No-go (Phase A) | Re-approval only if ownership + verify isolation proven; else remain deferred |
| D13 | **vLLM / LMCache sidecar probe** | Deferred | Metadata-only or isolated sidecar evaluation without breaking exactness gate |
| D14 | **Active GPU memory profiling** | Deferred | Approved methodology; distinct from `total_kv_footprint_bytes`; optional field only |
| D16 | **PagedAttention kernel integration** | Deferred | Not planned; local harness remains default unless V8 Phase C re-approved |

_Larger-model RunPod validation (formerly D15) is **active in V9 Phase E** — see V9 table above._

---

## V12 / v1.0.0 — Final public launch package

| ID | Item | Status | Success criteria |
|---|---|---|---|
| D17 | **Raw report bundle** | Deferred | Curated, reproducible JSON/CSV archive for experiments 001–007+ with manifests |
| D18 | **Final public launch narrative** | Deferred | Reviewed post/docs; explicit negation of performance and serving claims |
| D19 | **Project status v1.0.0** | Deferred | Supersedes [`PROJECT_STATUS_V0.8.0.md`](PROJECT_STATUS_V0.8.0.md) |
| D20 | **Git tag `v1.0.0`** | Deferred | All V9–V11 exit criteria met or explicitly scoped down with documented honesty |

---

## Cross-cutting (ongoing)

| ID | Item | Status | Notes |
|---|---|---|---|
| D21 | **Sampling / parallel verify / bonus tokens** | Deferred | Out of scope until explicit future version |
| D22 | **Multi-request batching** | Deferred | Serving-scale feature; not v0.8.0 |
| D23 | **CPU offload / CUDA kernels** | Deferred | No custom kernels in ExactKV today |
| D24 | **Broader kvpress** | Deferred | KnormPress only (V6); no expansion without approval |
| D25 | **RESEARCH_BACKLOG sync** | Ongoing | Keep aligned with this register |

---

## Phase C reminder (V8)

vLLM and LMCache **direct integration** were judged **no-go** in
[`SERVING_CONTEXT_FEASIBILITY.md`](SERVING_CONTEXT_FEASIBILITY.md). Items D11–D12
remain **deferred, not forgotten** — sidecar probes (D13) are the approved
re-entry path if stack integration is revisited.

---

## Related

- [`V9_SCOPE_STATEMENT.md`](V9_SCOPE_STATEMENT.md) — active V9 scope (Phase 0 complete)
- [`ROADMAP.md`](ROADMAP.md) — version planning
- [`RESEARCH_BACKLOG.md`](RESEARCH_BACKLOG.md) — experiment ideas
- [`RELEASE_NOTES_V0.8.0.md`](RELEASE_NOTES_V0.8.0.md) — what shipped in v0.8.0
