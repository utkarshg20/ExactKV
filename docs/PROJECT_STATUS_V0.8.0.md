# ExactKV Project Status (v0.8.0)

**As of:** v0.8.0 (V8 complete). **Not public-launch final.**

ExactKV is a correctness-first, compressor-agnostic research platform for
evaluating lossy KV-cache compression under ExactKV's draft-verify-commit loop.
Through V8 it has published nine experiment reports (001–007, including 006A and
006C), a real-backend adapter boundary (V6), simulated layer-aware V policies (V7),
and a local serving-context lifecycle harness (V8) — all with `exactkv_failures ==
0` on every published sweep cell. The project is **ready to tag v0.8.0** but
**not ready for public launch**; substantial backend, research, and scale work
remains before v1.0.0.

---

## 1. Version timeline (v0.1.0 → v0.8.0)

| Version | Tag | Theme |
|---|---|---|
| V1 | — | Draft-verify-commit prototype; exactness gate |
| v0.2.0 | V2 | Compressor registry; JSON/CSV reports |
| v0.3.0 | V3 | Named prompt suites; Markdown reports |
| v0.4.0 | V4 | Asymmetric K/V compressors; Experiment 003 |
| v0.5.0 | V5 | Workspace-aware memory accounting; Experiment 004 |
| v0.6.0 | V6 | `BackendAdapter`; restricted kvpress; Experiment 005 |
| v0.7.0 | V7 | Layer-aware V policies; Experiments 006 / 006C |
| **v0.8.0** | **V8** | Serving-context harness; Experiment 007 |

---

## 2. What ExactKV is

- A **verification and evaluation framework** for lossy KV-cache compression.
- Built on the VeriCache draft-then-verify idea: compressors draft on lossy KV;
  verification uses **full-precision KV**; output matches `generate_full_greedy`.
- Measures **exactness, acceptance, rejection, correction, divergence**, and
  **honest workspace-memory accounting**.
- Hugging Face–centric local runtime; CPU-first sweeps on `Qwen/Qwen2.5-0.5B`.

---

## 3. What ExactKV is not

- **Not** a production serving system (no vLLM/LMCache integration in v0.8.0).
- **Not** a throughput or latency benchmark (no tokens/sec, speedup, or runtime claims).
- **Not** a real packed-bit quantization library for all compressors (`_sim` = int8 containers).
- **Not** TurboQuant, KIVI, KVQuant, Sparse V, or attention-gated materialization (deferred).
- **Not** public-launch final at v0.8.0.

---

## 4. Strongest results so far

| Finding | Source |
|---|---|
| Exactness gate on all published cells | Experiments 001–007 |
| Keys far more fragile than values | Experiment 003 |
| `k8_v4_boundary4_v8_sim` accept **0.954** | Experiment 006C |
| Real INT8 asymmetric `k_full_v8` accept **~0.99** | Experiments 003, 006, 007 |
| Restricted kvpress KnormPress exactness preserved | Experiment 005 |
| Serving harness invariants pass on 238 cells | Experiment 007 |

---

## 5. Current best compressors / policies

| Class | Best in panel | Accept (approx.) | Notes |
|---|---|---:|---|
| Lossless | `noop`, `backend_passthrough` | 1.000 | Correctness baselines |
| Real INT8 | `k_full_v8` | 0.990 (Exp 007) | Real asymmetric storage |
| Real symmetric | `int8` | 0.961 (Exp 007) | Genuine INT8 storage |
| Simulated layer-aware | `k8_v4_boundary4_v8_sim` | 0.954 (Exp 006C/007) | int8 containers; N=4 boundary |
| Simulated uniform | `k8_v4_sim` | 0.891 (Exp 007) | Baseline asymmetric sim |
| Real pruned (isolated) | `kvpress_knorm_restricted` | varies (Exp 005) | Not default registry |

---

## 6. Memory-honesty story

- **V5 workspace fields:** `stored_kv_bytes`, `materialized_working_kv_bytes`,
  `metadata_bytes`, `temporary_workspace_bytes`, `total_kv_footprint_bytes`.
- **`total_kv_footprint_bytes`** = conservative accounting sum — **not** measured peak GPU memory.
- **`is_simulated`** and **`supports_real_bytes_claim`** labelling on every report row.
- **`_sim` compressors** use int8 containers — not packed 4-bit/2-bit storage.
- **Active GPU memory** not reported (deferred to V11).

---

## 7. Real-backend story

- **V6 `BackendAdapter`** sealed interface in `exactkv/compressors/backend_adapter.py`.
- **`backend_passthrough`** — zero-dependency PoC (lossless).
- **`kvpress_knorm_restricted`** — isolated optional extra; real pruned-cache bytes;
  hook-safety gates documented in Experiment 005.
- **TurboQuant+, KIVI, KVQuant** — not implemented; planned V9.

---

## 8. Serving-context story

- **Phase A:** vLLM/LMCache direct integration **no-go/deferred**.
- **Phase B:** `ServingCacheLifecycleHarness` models ownership, logical/physical
  mapping, blocks, append lifecycle.
- **Phase D:** Experiment 007 (238 cells, Mode B harness-only) — all invariants pass.
- **Phase C:** deferred; sidecar probes planned V11.

---

## 9. Remaining limitations

- Single small default model; CPU-first evaluation.
- No multi-request batching or GPU serving integration.
- Simulated compressors dominate research panels; few real backends integrated.
- No true attention weights in divergence analysis (proxy only in 006A).
- No public launch narrative or curated artifact bundle yet.

---

## 10. Why the project is not public-launch final yet

v0.8.0 completes a coherent **research arc** (V1–V8) but the deferred-work register
shows major gaps before a credible public story:

- Real backend gauntlet (TurboQuant+, KIVI, KVQuant-style) not done.
- Sparse V, attention logging, and per-layer forensics not done.
- Scale validation (larger models, RunPod, GPU memory profiling) not done.
- Serving-stack sidecar probes not done.
- Launch narrative and raw report bundle intentionally **deferred**.

Public posting is delayed until the project is **substantially more impressive**
— see §11.

---

## 11. What would make it “insane enough” for v1.0.0

A credible v1.0.0 would combine:

| Pillar | Target version | Bar |
|---|---|---|
| Real backends evaluated honestly | V9 | TurboQuant+ and/or KIVI/KVQuant adapters behind `BackendAdapter`; acceptance + memory honesty |
| Deep compression research | V10 | Sparse V, true attention logging, per-layer/head divergence forensics |
| Scale + serving probes | V11 | Larger-model validation; active GPU profiling methodology; vLLM/LMCache sidecar feasibility |
| Launch package | V12 / v1.0.0 | Experiment index + raw bundles + reviewed narrative; still no performance claims |

See [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md) and [`ROADMAP.md`](ROADMAP.md).

---

## Related documents

| Document | Purpose |
|---|---|
| [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md) | All experiments at a glance |
| [`RELEASE_NOTES_V0.8.0.md`](RELEASE_NOTES_V0.8.0.md) | v0.8.0 changelog |
| [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md) | Deferred work tracker |
| [`V8_SCOPE_STATEMENT.md`](V8_SCOPE_STATEMENT.md) | V8 scope and exit criteria |
