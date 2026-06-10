# ExactKV Project Status (v0.9.0)

**As of:** v0.9.0 (V9 complete). **Not public-launch final.**

ExactKV is a correctness-first, compressor-agnostic research platform for evaluating
lossy KV-cache compression under ExactKV's draft-verify-commit loop. Through V9 it
has published eleven experiment reports (001–011, including 006A and 006C), three
restricted real-backend adapter families evaluated behind `BackendAdapter` (TurboQuant
Python, KIVI offline, KVQuant simquant), simulated layer-aware V policies (V7), a
local serving-context lifecycle harness (V8), and RunPod larger-model validation on
Qwen2.5-1.5B — all with `exactkv_failures == 0` on every published sweep cell. The
project is **ready to tag v0.9.0** but **not ready for public launch**; V10 must
harden the evaluation suite and divergence forensics before v1.0.0.

---

## 1. Version timeline (v0.1.0 → v0.9.0)

| Version | Tag | Theme |
|---|---|---|
| V1 | — | Draft-verify-commit prototype; exactness gate |
| v0.2.0 | V2 | Compressor registry; JSON/CSV reports |
| v0.3.0 | V3 | Named prompt suites; Markdown reports |
| v0.4.0 | V4 | Asymmetric K/V compressors; Experiment 003 |
| v0.5.0 | V5 | Workspace-aware memory accounting; Experiment 004 |
| v0.6.0 | V6 | `BackendAdapter`; restricted kvpress; Experiment 005 |
| v0.7.0 | V7 | Layer-aware V policies; Experiments 006 / 006C |
| v0.8.0 | V8 | Serving-context harness; Experiment 007 |
| **v0.9.0** | **V9** | Real backend gauntlet (Exp 008–010); 1.5B validation (Exp 011) |

---

## 2. What ExactKV is

- A **verification and evaluation framework** for lossy KV-cache compression.
- Built on the VeriCache draft-then-verify idea: compressors draft on lossy KV;
  verification uses **full-precision KV**; output matches `generate_full_greedy`.
- Measures **exactness, acceptance, rejection, correction, divergence**, and
  **honest workspace-memory accounting**.
- Hugging Face–centric runtime; primary sweeps on `Qwen/Qwen2.5-0.5B`; 1.5B validated on RunPod.

---

## 3. What ExactKV is not

- **Not** a production serving system (no vLLM/LMCache integration).
- **Not** a throughput or latency benchmark (no tokens/sec, speedup, or runtime claims).
- **Not** a packed-bit quantization library for all compressors (`_sim` = int8 containers).
- **Not** TurboQuant production, KIVI CUDA/Triton, or KVQuant deployment CUDA (restricted adapters only).
- **Not** public-launch final at v0.9.0.

---

## 4. Strongest findings so far

| Finding | Source |
|---|---|
| Exactness gate on all published cells | Experiments 001–011 |
| Keys far more fragile than values | Experiment 003 |
| `k8_v4_boundary4_v8_sim` accept **~0.95** (0.5B and 1.5B) | Experiments 006C, 010, 011 |
| Real INT8 / K-full policies accept **~0.97–0.99** | Experiments 003–011 |
| Three external backends preserve exactness when wrapped | Experiments 008–010 |
| KVQuant simquant accept **0.792** >> TurboQuant Python **0.435** >> KIVI offline **0.012** | Exp 008–010 (0.5B) |
| Serving harness invariants pass | Experiment 007 |
| 1.5B exactness gate passes on RunPod | Experiment 011 |

---

## 5. Current leaderboard (core suite, 0.5B reference)

_Approximate mean acceptance; `draft_len=4`, `max_new_tokens=16`. Not a throughput ranking._

| Rank | Compressor | Accept (approx.) | Class |
|---:|---|---:|---|
| 1 | `noop`, `backend_passthrough` | 1.000 | Lossless |
| 2 | `k_full_v8` | 0.990 | Real INT8 asymmetric |
| 3 | `k8_v_full`, `int8` | 0.96–0.97 | Real INT8 |
| 4 | `k8_v4_boundary4_v8_sim` | 0.950 | Simulated layer-aware |
| 5 | `k8_v4_sim` | 0.898 | Simulated uniform |
| 6 | `kvquant_sim_qwen05b` | 0.792 | Restricted real (KVQuant simquant) |
| 7 | `turboquant_python_k3_v3` | 0.435 | Restricted real (TurboQuant Python) |
| 8 | `kivi_offline_k2_v2` | 0.012 | Restricted real (KIVI offline) |

On **1.5B** (Exp 011): `int8` **0.980**; boundary4 **0.954** > k8_v4_sim **0.945**.

---

## 6. Simulated-policy story

- **`k8_v4_sim`** — uniform asymmetric K8/V4 in int8 containers (`is_simulated=True`).
- **`k8_v4_boundary4_v8_sim`** — layer-aware boundary V (N=4); best simulated policy.
- Experiment 006C: boundary depth N=4 beats N=1/2 and uniform sim on 0.5B.
- Experiment 011: advantage **transfers to 1.5B** but margin shrinks (+0.009 vs +0.051).
- Simulated policies are **not** real packed-bit backends.

---

## 7. Real-backend story

| Backend | Adapter | Experiment | Accept (0.5B) | Registry |
|---|---|---|---:|---|
| TurboQuant Python | `TurboQuantPythonAdapter` | 008 | 0.435 | Factory-only |
| KIVI offline | `KIVIOfflineAdapter` | 009 | 0.012 | Factory-only |
| KVQuant simquant | `KVQuantSimAdapter` | 010 | 0.792 | Factory-only |
| kvpress Knorm | restricted | 005 | varies | Isolated extra |

All preserve `exactkv_failures == 0`. None claim upstream paper results as ExactKV results.

---

## 8. Larger-model story

- **Experiment 011:** Qwen2.5-1.5B, RunPod L40S, float16 CUDA, 238 cells, `exactkv_failures == 0`.
- Exactness gate holds at larger scale without code changes to generation/verification.
- Acceptance ordering broadly stable; boundary4 still beats k8_v4_sim.
- Qwen2.5-3B stretch **not run**. KVQuant 1.5B quantizer artifact generated separately.

---

## 9. Memory-honesty story

- **V5 workspace fields** on every experiment row.
- **`total_kv_footprint_bytes`** = conservative accounting sum — **not** measured peak GPU memory.
- **`supports_real_bytes_claim`** and **`is_simulated`** labelling enforced in reports.
- Real-backend adapters honestly report `supports_real_bytes_claim=False` where payload is not packed-bit storage.
- **Active GPU memory** not reported (deferred to V11).

---

## 10. Serving-context story

- **V8 Phase A:** vLLM/LMCache direct integration **no-go/deferred**.
- **V8 Phase B–D:** local `ServingCacheLifecycleHarness`; Experiment 007 — all invariants pass.
- **V9:** no change to serving integration scope; real-backend work is orthogonal to vLLM/LMCache.

---

## 11. Remaining weaknesses

- **Narrow evaluation suite** — 34 core prompts; not category-comprehensive.
- **Proxy divergence analysis** — Experiment 006A lacks true attention weights.
- **No production backend paths** — CUDA/Triton, llama.cpp, KVQuant deployment deferred.
- **Factory-only real backends** — reproducibility requires isolated environments.
- **No public launch bundle** — raw JSON/CSV gitignored; narrative deferred.
- **Single-model focus** for backend sweeps; limited multi-model matrix.

---

## 12. Why the current prompt suite is controlled but not comprehensive

The **core** suite (34 prompts) was designed for **controlled engineering evaluation**:
repeatable, fast, and sufficient to catch exactness regressions and compare compressor
ordering. It spans natural language, code-ish, JSON, QA, and long-ish prompts — but it
is **not** stratified for broad public claims about code, math, multilingual, retrieval,
or tool-use robustness.

**Implication:** v0.9.0 results are **valid for ExactKV's stated gate** but **insufficient**
for a comprehensive public benchmark narrative. V10 addresses this explicitly.

---

## 13. What V10 must fix before public launch

| Requirement | V10 direction |
|---|---|
| Broader prompt taxonomy | `core_v2`, code, long-context, math, multilingual, retrieval, tool/JSON |
| Category-stratified leaderboards | Per-category acceptance and divergence tables |
| Sensitivity analysis | `draft_len` 2/4/8; `max_new_tokens` 16/32/64 |
| Divergence forensics | First divergence position, category, token type; layer/head where feasible |
| True attention logging | Small-subset feasibility (not fabricated weights) |
| Multi-model matrix | 0.5B, 1.5B, optional 3B |
| Preserve exactness gate | `exactkv_failures == 0` on all published V10 experiments |

V10 is **not** a throughput benchmark. See [`V10_SCOPE_DRAFT.md`](V10_SCOPE_DRAFT.md).

---

## 14. Launch readiness assessment

| Criterion | v0.9.0 | v1.0.0 target |
|---|---|---|
| Exactness gate on published experiments | ✅ | ✅ |
| Real backends evaluated honestly | ✅ (restricted) | Broader panels |
| Layer-aware V validated | ✅ (0.5B + 1.5B) | Category-stratified |
| Larger-model validation | ✅ (1.5B) | Multi-model matrix |
| Evaluation suite breadth | ❌ Narrow | V10 |
| Divergence forensics | ❌ Proxy only | V10 |
| Serving/scale probes | ❌ Deferred | V11 |
| Public narrative + raw bundle | ❌ Deferred | v1.0.0 |

**Verdict:** Ready for **`v0.9.0` tag**. **Not** ready for public launch or **v1.0.0**.
Proceed to **V10** before launch narrative.

---

## Related documents

- [`RELEASE_NOTES_V0.9.0.md`](RELEASE_NOTES_V0.9.0.md) — V9 changelog
- [`V9_SCOPE_STATEMENT.md`](V9_SCOPE_STATEMENT.md) — V9 phases
- [`V10_SCOPE_DRAFT.md`](V10_SCOPE_DRAFT.md) — next version draft
- [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md) — experiments 001–011
- [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md) — V10–v1.0.0 tracker
- [`ROADMAP.md`](ROADMAP.md) — version planning
