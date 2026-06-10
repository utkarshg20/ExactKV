# V10 Scope Draft — Evaluation Suite Hardening and Divergence Forensics

**Status:** Draft scope only — **no code, no experiments** until explicit approval.
**Builds on:** `v0.9.0` — V9 complete (Experiments 008–011, real-backend gauntlet).
**Not public launch.** V10 precedes credible v1.0.0 narrative.

> The current **core** suite (34 prompts) is **valid for controlled engineering evaluation**.
> It is **not comprehensive enough** for broad public claims. V10 hardens the
> evaluation story before launch.
> V10 is **not** a performance benchmark. V10 preserves the exactness gate:
> `exactkv_failures == 0` on every published experiment.
> No throughput, latency, tokens/sec, speedup, `runtime_seconds`,
> `active_gpu_kv_bytes`, or production-serving claims.

---

## 1. Why V10 is needed after V9

V9 established that ExactKV can wrap restricted real backends and preserve exactness
on a controlled 34-prompt panel at 0.5B and 1.5B. That is necessary but not sufficient
for public launch:

- **Suite breadth** — core prompts do not cover code, structured tool output, long
  context stress, multilingual, retrieval-copy, or reasoning/math categories in a
  stratified way.
- **Divergence understanding** — Experiment 006A used proxy fields only; no true
  attention weights or per-category forensics.
- **Sensitivity unknowns** — most published cells use `draft_len=4` and
  `max_new_tokens=16` only.
- **Leaderboard interpretability** — a single global accept rate hides category failure modes.

V10 answers: **where** compressors fail, **under what prompt types**, and **whether
findings survive suite expansion** — still under the exactness gate.

---

## 2. Current eval-suite limitations

| Limitation | Evidence |
|---|---|
| 34 prompts, single `core` suite | Experiments 002, 008–011 |
| One draft length in backend sweeps | `draft_len=4` |
| One generation length in backend sweeps | `max_new_tokens=16` |
| Primary model 0.5B for backend comparisons | Exp 008–010 |
| Proxy divergence only | Experiment 006A |
| No category-stratified leaderboards | All published sweeps |
| Real backends factory-only | Not default-registry stress |

---

## 3. Proposed benchmark suites

| Suite ID | Purpose | Initial size target |
|---|---|---|
| **`core_v2`** | Superset/refinement of current core; fixed IDs for regression | ~40–50 prompts |
| **`code_structured`** | Code, indentation, brackets, syntax-sensitive tokens | ~20–30 |
| **`long_context`** | Long prefill stress (within model limits) | ~15–20 |
| **`reasoning_math`** | Arithmetic, step-wise reasoning, symbol tokens | ~15–20 |
| **`multilingual`** | Non-English prefill + generation | ~15–20 |
| **`retrieval_copy`** | Near-verbatim copy / entity repetition | ~10–15 |
| **`tool_json`** | JSON/tool-call shaped prompts | ~10–15 |

Suites may overlap in taxonomy tags but each prompt has a **primary category** for leaderboard stratification.

---

## 4. Prompt taxonomy

Proposed primary tags (one per prompt):

- `natural_language`
- `code`
- `structured_json`
- `long_context`
- `reasoning_math`
- `multilingual`
- `retrieval_copy`
- `qa_factual`
- `tool_schema`

Secondary tags (optional): `short_prefill`, `medium_prefill`, `repetition_heavy`, `symbol_heavy`.

---

## 5. Dataset construction principles

1. **Deterministic, committed prompts** — versioned in-repo; no live web fetch at sweep time.
2. **No benchmark laundering** — do not claim external benchmark scores as ExactKV results.
3. **Category balance** — each suite has documented intent; leaderboards reported per category.
4. **Exactness first** — prompts must be compatible with greedy decoding and the exactness gate.
5. **Honest labelling** — `_sim` and `supports_real_bytes_claim` on every row.
6. **Reproducibility** — manifest records suite version, model, `draft_len`, `max_new_tokens`.
7. **No performance claims** — acceptance and divergence only.

---

## 6. Per-category leaderboard plan

For each published V10 experiment report:

| Table | Contents |
|---|---|
| **Global** | Mean acceptance across all cells (legacy compatibility) |
| **Per-category** | Mean acceptance, rejection, correction per taxonomy tag |
| **Per-compressor × category** | Heatmap-style rows in Markdown |
| **Divergence rate** | Lossy divergence cells / total per category |
| **Exactness** | `exactkv_failures` always zero or experiment is unpublished |

Real backends compared **within category** where sample size permits; sparse categories
flagged with low-n warnings.

---

## 7. Draft length sensitivity

| `draft_len` | Rationale |
|---:|---|
| **2** | Conservative drafting; higher verify rounds |
| **4** | Current default (V9 anchor) |
| **8** | Aggressive drafting; stress rejection/correction |

**Plan:** Phase 1 sensitivity on `core_v2` × built-in panel + `k8_v4_boundary4_v8_sim`
+ `int8` at 0.5B. Phase 2 optional on 1.5B for winning configs only.

---

## 8. Generation length sensitivity

| `max_new_tokens` | Rationale |
|---:|---|
| **16** | V9 anchor |
| **32** | Medium decode horizon |
| **64** | Longer divergence accumulation |

Same phasing as draft-length sensitivity — avoid combinatorial explosion in first V10 experiment.

---

## 9. Model matrix

| Model | Priority | Notes |
|---|---|---|
| `Qwen/Qwen2.5-0.5B` | **Required** | Primary regression anchor |
| `Qwen/Qwen2.5-1.5B` | **Required** | V9 Exp 011 baseline |
| `Qwen/Qwen2.5-3B` | **Optional stretch** | RunPod L40S if 1.5B sweeps clean |

CPU-first acceptable for 0.5B built-ins; GPU (RunPod) for 1.5B+ and KVQuant rows.

---

## 10. Divergence forensics

Upgrade beyond Experiment 006A proxy analysis:

| Signal | Source | Priority |
|---|---|---|
| **First divergence position** | Already in reports | Required |
| **Rejection position** | ExactKV traces | Required |
| **Prompt category** | Suite taxonomy | Required |
| **Token type** | Tokenizer class heuristics (symbol, whitespace, numeric) | Required |
| **Layer/head analysis** | Attention weights where logged | Optional Phase 2 |
| **True attention logging** | Small-subset prefill/decode capture | Optional Phase 2 |

**Rule:** Never fabricate attention weights. If logging is infeasible, document blocker.

---

## 11. Real backend comparison plan

Restricted adapters only (factory-only, not default registry):

| Backend | Experiment anchor | V10 role |
|---|---|---|
| TurboQuant Python | Exp 008 | Re-run on `core_v2` subset + 1–2 categories |
| KIVI offline | Exp 009 | Same |
| KVQuant simquant | Exp 010 | Same (RunPod; per-model quantizer) |
| kvpress Knorm | Exp 005 | Category spot-check for token-dropping |

Cross-experiment anchors remain labelled when not same-run.

---

## 12. Experiment 012 plan

**Name (proposed):** Evaluation Suite Expansion — `core_v2` + category suites.

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` |
| Suites | `core_v2` + all category suites in §3 |
| Compressors | Built-in panel: `noop`, `int8`, `k8_v4_sim`, `k8_v4_boundary4_v8_sim`, `k_full_v8`, `k8_v_full`, `backend_passthrough` |
| `draft_len` | 4 (anchor) |
| `max_new_tokens` | 16 (anchor) |
| Deliverables | `EXPERIMENT_012_EVAL_SUITE_EXPANSION.md`; gitignored JSON/CSV |

**Success:** `exactkv_failures == 0`; per-category leaderboards published; suite version documented.

---

## 13. Experiment 013 plan

**Name (proposed):** Sensitivity + Divergence Forensics.

| Parameter | Value |
|---|---|
| Models | 0.5B required; 1.5B optional |
| Suite | `core_v2` (or stratified subset) |
| `draft_len` sweep | 2, 4, 8 |
| `max_new_tokens` sweep | 16, 32, 64 (phased) |
| Compressors | `int8`, `k8_v4_sim`, `k8_v4_boundary4_v8_sim` + optional one real backend |
| Analysis | Category-stratified divergence tables; 006A superseded where weights exist |
| Deliverables | `EXPERIMENT_013_SENSITIVITY_FORENSICS.md`; gitignored JSON/CSV |

**Success:** `exactkv_failures == 0`; documented sensitivity findings; no performance claims.

---

## 14. What V10 does not claim

- Throughput, latency, speedup, tokens/sec, or runtime superiority.
- Active GPU memory peaks or production memory savings.
- That simulated `_sim` compressors are packed-bit backends.
- Upstream TurboQuant, KIVI, or KVQuant paper results.
- Production serving readiness or vLLM/LMCache integration.
- Comprehensive AGI-benchmark coverage (V10 expands but does not infinite-expand).

---

## 15. Exit criteria for v1.0.0 readiness

v1.0.0 public launch requires **V10 + V11 substance** (minimum):

| Gate | Owner |
|---|---|
| Expanded suites + per-category leaderboards published | V10 |
| Sensitivity sweeps documented | V10 |
| Divergence forensics beyond 006A proxy | V10 |
| Sparse V / attention logging (or documented deferral with blocker) | V10 research |
| Larger-model / multi-model matrix extended | V10–V11 |
| Serving sidecar probe or documented no-go refresh | V11 |
| `RELEASE_NOTES` + `PROJECT_STATUS` v1.0.0 | v1.0.0 |
| Curated raw report bundle policy | v1.0.0 |
| Public narrative with explicit negation of performance claims | v1.0.0 |

**v0.9.0 is not v1.0.0.** Tag v0.9.0 now; plan V10 before launch.

---

## Related

- [`RELEASE_NOTES_V0.9.0.md`](RELEASE_NOTES_V0.9.0.md) — what V9 shipped
- [`PROJECT_STATUS_V0.9.0.md`](PROJECT_STATUS_V0.9.0.md) — launch readiness assessment
- [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md) — V10 item tracker
- [`EXPERIMENT_006A_DIVERGENCE_ANALYSIS.md`](EXPERIMENT_006A_DIVERGENCE_ANALYSIS.md) — proxy baseline
- [`ROADMAP.md`](ROADMAP.md) — version path
