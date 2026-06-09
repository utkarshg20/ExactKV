# ExactKV Deferred Work Register

**Status:** Living register of major deferred items. **Nothing listed here is
implemented** unless a linked experiment or release note says otherwise.

> Guardrails: `exactkv_failures == 0` on every published experiment; no throughput,
> latency, speedup, runtime, or production-serving claims; `_sim` ≠ packed-bit storage;
> external paper results are **not** ExactKV results.

**Version path:** V9 → V10 → V11 → V12/v1.0.0 (public launch).

---

## V9 — Real backend integration gauntlet

| ID | Item | Status | Success criteria |
|---|---|---|---|
| D1 | **TurboQuant full integration** | Deferred | `BackendAdapter` wrapping real TurboQuant path; acceptance + V5 memory on core suite; `exactkv_failures == 0` |
| D2 | **TurboQuant+ full integration** | Deferred | Same as D1 for TurboQuant+; honest `supports_real_bytes_claim`; no external throughput claims |
| D3 | **KIVI adapter** | Deferred | Real per-channel K / per-token V + residual; acceptance vs `int8` at comparable budget |
| D4 | **KVQuant-style adapter** | Deferred | Pre-RoPE key quant + outlier handling; verify hook safety; acceptance report |
| D5 | **KVTC / Palu feasibility** | Deferred | Written feasibility + optional thin adapter PoC; metadata accounting under V5 schema |

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
| D15 | **Larger-model RunPod validation** | Deferred | At least one model &gt;0.5B on documented GPU hardware; exactness gate preserved |
| D16 | **PagedAttention kernel integration** | Deferred | Not planned; local harness remains default unless Phase C re-approved |

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

- [`ROADMAP.md`](ROADMAP.md) — version planning
- [`RESEARCH_BACKLOG.md`](RESEARCH_BACKLOG.md) — experiment ideas
- [`RELEASE_NOTES_V0.8.0.md`](RELEASE_NOTES_V0.8.0.md) — what shipped in v0.8.0
