# V10 Scope Statement: Evaluation Suite Hardening and Divergence Forensics

**Status:** **Phase 2 complete** — Experiment 012 published (`exactkv_failures == 0`).
Phase 3 (Experiment 013) is next.
**Builds on:** `v0.9.0` — V9 complete (Experiments 008–011; real-backend gauntlet).
**Not public launch.** v1.0.0 deferred until V10 (and V11 substance) exit criteria are met.

> The current **core** suite (34 prompts) is **valid for controlled engineering evaluation**.
> It is **not comprehensive enough** for broad public claims. V10 hardens the
> evaluation story before launch.
> V10 is **not** a performance benchmark. V10 preserves the exactness gate:
> `exactkv_failures == 0` on every published experiment.
> V10 must **not overfit** to the current 34-prompt core suite.
> No throughput, latency, tokens/sec, speedup, `runtime_seconds`,
> `active_gpu_kv_bytes`, or production-serving claims.

**Supersedes:** [`V10_SCOPE_DRAFT.md`](V10_SCOPE_DRAFT.md) (draft retained for history).

---

## 1. Status

| Phase | Focus | Status |
|---|---|---|
| **0** | Formal scope statement (this document) | **Complete** |
| **1** | Suite authoring (`core_v2` + category suites); validator + tests | **Complete** |
| **2** | Experiment 012 — suite expansion + per-category leaderboards | **Complete** |
| **3** | Experiment 013 — draft/generation sensitivity + divergence forensics | **Next** |
| **4** | Optional real-backend category spot-checks (factory-only) | Planned |
| **5** | v1.0.0 readiness assessment (not launch) | Planned |

**Latest release:** `v0.9.0`. **Phase 1 deliverables:**
[`V10_PROMPT_SUITES.md`](V10_PROMPT_SUITES.md),
`scripts/validate_v10_prompt_suites.py`, seven suite files under `benchmarks/prompts/`.

---

## 2. V10 goal

Harden ExactKV's evaluation suite and divergence forensics so that **public claims
about compressor behaviour are defensible** — without changing the exactness gate,
generation logic, verification logic, or report schemas.

V10 must answer:

> **Where do compressors fail, under what prompt types, and do findings survive
> suite expansion and sensitivity sweeps — still with `exactkv_failures == 0`?**

---

## 3. Why V10 is needed after V9

V9 established that ExactKV can wrap restricted real backends and preserve exactness
on a controlled 34-prompt panel at 0.5B and 1.5B. That is necessary but not sufficient
for public launch:

| Gap | Evidence |
|---|---|
| **Suite breadth** | Core prompts do not stratify code, structured tool output, long context, multilingual, retrieval-copy, or reasoning/math |
| **Divergence understanding** | Experiment 006A used proxy fields only; no category-stratified forensics |
| **Sensitivity unknowns** | Backend sweeps anchor on `draft_len=4`, `max_new_tokens=16` only |
| **Leaderboard interpretability** | Single global accept rate hides category failure modes |
| **Overfitting risk** | Layer-aware and asymmetric policies were tuned on the same 34 prompts |

V10 makes ExactKV's claims **more defensible** before v1.0.0 — not faster or more
production-ready.

---

## 4. Current evaluation-suite limitations

| Limitation | Evidence |
|---|---|
| 34 prompts, single `core` suite | Experiments 002, 008–011 |
| One draft length in backend sweeps | `draft_len=4` |
| One generation length in backend sweeps | `max_new_tokens=16` |
| Primary model 0.5B for backend comparisons | Experiments 008–010 |
| Proxy divergence only | Experiment 006A |
| No category-stratified leaderboards | All published sweeps |
| Real backends factory-only | Not default-registry stress on expanded suites |
| Existing named suites (`code`, `structured`, `stress`) not used in published experiments | Experiments 001–011 use `core` or `smoke` |

---

## 5. What V10 should add

1. **Versioned prompt suites** — `core_v2` plus six category suites (§7).
2. **Prompt taxonomy and metadata** — primary category, secondary tags, suite version (§8).
3. **Per-category leaderboards** — global + stratified tables in Experiment 012 reports (§15).
4. **Prompt-level win/loss analysis** — which prompts drive compressor ranking changes (§16).
5. **Draft-length sensitivity** — `draft_len` 2, 4, 8 (§13).
6. **Generation-length sensitivity** — `max_new_tokens` 16, 32, 64 (§14).
7. **Divergence forensics** — first divergence, rejection, correction positions by category (§17).
8. **Optional real-backend spot-checks** — factory-only subset on expanded suites (§18).
9. **Larger-model extension** — 1.5B on expanded suites where RunPod budget permits (§12).
10. **Documentation** — Experiment 012/013 reports; updated experiment index; no schema changes.

---

## 6. What V10 explicitly should not add

- **No new default-registry compressors** beyond what exists at v0.9.0 unless a phase
  explicitly approves a restricted adapter re-run panel.
- **No generation or verification logic changes.**
- **No report schema changes** (JSON/CSV field additions require separate approval).
- **No CLI behaviour changes** unless required for suite loading (deferred to Phase 1).
- **No vLLM, LMCache, llama.cpp, MLX, TurboQuant production runtime, KIVI CUDA/Triton,
  KVQuant deployment CUDA, Sparse V production, KVTC, or Palu integration.**
- **No throughput, latency, tokens/sec, speedup, `runtime_seconds`,
  `active_gpu_kv_bytes`, or production-serving claims.**
- **No implication** that `_sim` compressors are real packed-bit backends.
- **No implication** that external paper results are ExactKV results.
- **No infinite benchmark expansion** — V10 expands deliberately, not comprehensively.

---

## 7. Proposed prompt suites

| Suite ID | Purpose | Initial size target |
|---|---|---|
| **`core_v2`** | Superset/refinement of current `core`; fixed IDs for regression | ~40–50 prompts |
| **`code_structured`** | Code, indentation, brackets, syntax-sensitive tokens | ~20–30 |
| **`long_context`** | Long prefill stress (within model limits) | ~15–20 |
| **`reasoning_math`** | Arithmetic, step-wise reasoning, symbol tokens | ~15–20 |
| **`multilingual`** | Non-English prefill + generation | ~15–20 |
| **`retrieval_copy`** | Near-verbatim copy / entity repetition | ~10–15 |
| **`tool_json`** | JSON/tool-call shaped prompts | ~10–15 |

Suites may share taxonomy tags; each prompt has exactly one **primary category** for
leaderboard stratification. Existing `benchmarks/prompts/*.jsonl` files may inform
authoring but V10 suites are **versioned separately** (`core_v2.jsonl`, etc.).

---

## 8. Prompt taxonomy and required metadata

### Primary tags (exactly one per prompt)

| Tag | Typical suite |
|---|---|
| `natural_language` | `core_v2` |
| `code` | `code_structured` |
| `structured_json` | `tool_json` |
| `long_context` | `long_context` |
| `reasoning_math` | `reasoning_math` |
| `multilingual` | `multilingual` |
| `retrieval_copy` | `retrieval_copy` |
| `qa_factual` | `core_v2` |
| `tool_schema` | `tool_json` |

### Secondary tags (optional, multi-valued)

`short_prefill`, `medium_prefill`, `long_prefill`, `repetition_heavy`, `symbol_heavy`,
`whitespace_sensitive`, `numeric_heavy`.

### Required JSONL fields (per prompt row)

| Field | Required | Notes |
|---|---|---|
| `id` | yes | Stable string; never reused across suite versions |
| `prompt` | yes | Deterministic text |
| `primary_category` | yes | One of primary tags above |
| `secondary_tags` | no | List of strings |
| `suite_version` | yes | e.g. `core_v2_v1` |
| `source_note` | no | Provenance; no live-web dependency at sweep time |

---

## 9. Dataset construction principles

1. **Deterministic, committed prompts** — versioned in-repo; no live web fetch at sweep time.
2. **No benchmark laundering** — do not claim external benchmark scores as ExactKV results.
3. **Category balance** — each suite has documented intent; leaderboards reported per category.
4. **Exactness first** — prompts compatible with greedy decoding and the exactness gate.
5. **Honest labelling** — `_sim` and `supports_real_bytes_claim` on every compressor row.
6. **Reproducibility** — manifest records suite version, model, `draft_len`, `max_new_tokens`.
7. **No performance claims** — acceptance and divergence only.
8. **Regression anchors** — retain overlap with original `core` IDs where possible for
   `core_v2` without duplicating the entire old suite unchanged.
9. **Anti-overfitting** — new prompts must not be selected solely to maximize boundary4 margin.

---

## 10. Suite size targets

| Suite | Target prompts | V10 Phase 1 minimum |
|---|---:|---:|
| `core_v2` | 40–50 | 40 |
| `code_structured` | 20–30 | 20 |
| `long_context` | 15–20 | 15 |
| `reasoning_math` | 15–20 | 15 |
| `multilingual` | 15–20 | 15 |
| `retrieval_copy` | 10–15 | 10 |
| `tool_json` | 10–15 | 10 |
| **Total (all suites)** | **~125–170** | **~115 minimum** |

Experiment 012 cell budget scales with suite size × compressor panel (§11). Phased
rollout permitted: `core_v2` + two category suites first if full matrix exceeds budget.

---

## 11. Compressor panel

### Experiment 012 (primary panel — built-in)

| Compressor | Role |
|---|---|
| `noop` | Lossless upper bound |
| `int8` | Real INT8 symmetric baseline |
| `k8_v4_sim` | Uniform simulated asymmetric |
| `k8_v4_boundary4_v8_sim` | Best simulated layer-aware policy (V7/V9 anchor) |
| `k_full_v8` | Real INT8 V-only compression |
| `k8_v_full` | Real INT8 K-only compression |
| `backend_passthrough` | Adapter identity control |

**7 compressors** × suite prompts × anchor `draft_len=4`, `max_new_tokens=16`.

### Experiment 013 (sensitivity panel — subset)

| Compressor | Role |
|---|---|
| `noop` | Control |
| `int8` | Real baseline |
| `k8_v4_sim` | Simulated uniform |
| `k8_v4_boundary4_v8_sim` | Simulated layer-aware |
| `kvquant_sim_qwen05b` | Restricted real (if quantizer available) |

### Optional real-backend subset (Experiment 012, runtime permitting)

Factory-only; not default registry:

| Compressor | Prerequisite |
|---|---|
| `kvquant_sim_qwen05b` | KVQuant venv + quantizer pickle (RunPod) |
| `turboquant_python_k3_v3` | TurboQuant venv |
| `kivi_offline_k2_v2` | KIVI offline path |

Include only if sweep runtime remains manageable; omit rather than rush incomplete rows.

---

## 12. Model matrix

| Model | Priority | Environment | Notes |
|---|---|---|---|
| `Qwen/Qwen2.5-0.5B` | **Required** | CPU-first acceptable | Primary regression anchor; Exp 012 first |
| `Qwen/Qwen2.5-1.5B` | **Required** | RunPod L40S, float16 CUDA | V9 Exp 011 baseline; Exp 012/013 extension |
| `Qwen/Qwen2.5-3B` | **Optional stretch** | RunPod L40S | Only if 1.5B sweeps clean with `exactkv_failures == 0` |

Per-model KVQuant quantizer pickles are **separate artifacts** (not committed);
cross-model comparison uses clearly labelled anchors.

---

## 13. Draft-length sensitivity

| `draft_len` | Rationale |
|---:|---|
| **2** | Conservative drafting; higher verify rounds |
| **4** | Current default (V9 anchor) |
| **8** | Aggressive drafting; stress rejection/correction |

**Experiment 013** sweeps all three on `core_v2` (or stratified subset) with the
§11 sensitivity panel. Phase 1 at 0.5B; Phase 2 optional on 1.5B for winning configs only.

---

## 14. Generation-length sensitivity

| `max_new_tokens` | Rationale |
|---:|---|
| **16** | V9 anchor |
| **32** | Medium decode horizon |
| **64** | Longer divergence accumulation |

**Experiment 013** sweeps all three (phased to avoid combinatorial explosion with
draft-length grid). Full 3×3 grid optional; minimum is anchor cross + one non-anchor
per axis documented with rationale.

---

## 15. Per-category leaderboard plan

For each published V10 experiment report:

| Table | Contents |
|---|---|
| **Global** | Mean acceptance across all cells (legacy compatibility) |
| **Per-category** | Mean acceptance, rejection, correction per `primary_category` |
| **Per-compressor × category** | Rows in Markdown; low-n categories flagged |
| **Divergence rate** | Lossy divergence cells / total per category |
| **Exactness** | `exactkv_failures` always zero or experiment is unpublished |

Real backends compared **within category** where sample size permits. Sparse categories
carry explicit low-n warnings. Category tables are the **primary** V10 deliverable;
global mean is secondary.

---

## 16. Prompt-level win/loss analysis

Beyond aggregate acceptance, V10 reports must include:

| Analysis | Description |
|---|---|
| **Per-prompt acceptance** | Compressor × prompt heatmap or ranked table |
| **Win/loss vs anchor** | Prompts where `k8_v4_boundary4_v8_sim` beats `k8_v4_sim` (and vice versa) |
| **Category drivers** | Which categories explain global ranking changes vs Experiment 002/006C |
| **Regression prompts** | `core_v2` IDs shared with original `core` — track acceptance drift |
| **Failure-adjacent prompts** | Lowest-accept prompts per compressor (still `exactkv_failures == 0`) |

Purpose: detect **suite overfitting** and **category-specific fragility** without
claiming production readiness.

---

## 17. Divergence forensics plan

Upgrade beyond Experiment 006A proxy analysis:

| Signal | Source | Priority |
|---|---|---|
| **First divergence position** | `first_divergence_idx` in reports | Required |
| **Rejection position** | ExactKV traces / `rejection_position_summary` | Required |
| **Correction position** | Trace correction events | Required |
| **Prompt category** | Suite `primary_category` | Required |
| **Token type** | Tokenizer heuristics (symbol, whitespace, numeric, punctuation) | Required |
| **Structured-output breakage** | JSON/bracket mismatch flags on `tool_json` / `code_structured` | Required |
| **Layer/head analysis** | Attention weights where logged | Optional Phase 3 |
| **True attention logging** | Small-subset prefill/decode capture | Optional Phase 3 |

**Rule:** Never fabricate attention weights. If logging is infeasible, document blocker
in deferred register (D7). Experiment 013 deliverable supersedes 006A where new weights exist.

---

## 18. Real-backend comparison policy

Restricted adapters only (**factory-only**, not default registry):

| Backend | Experiment anchor | V10 role |
|---|---|---|
| kvpress Knorm | Exp 005 | Category spot-check for token-dropping |
| TurboQuant Python | Exp 008 | Re-run on `core_v2` subset + 1–2 categories |
| KIVI offline | Exp 009 | Same |
| KVQuant simquant | Exp 010 | Same (RunPod; per-model quantizer) |

**Cross-experiment anchors** must be labelled when not same-run (model, suite version,
`draft_len`, `max_new_tokens` may differ). V10 does not claim V9 numbers apply unchanged
to expanded suites.

---

## 19. Experiment 012 plan

**Name:** Evaluation Suite Expansion — `core_v2` + category suites.

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` first; 1.5B extension if approved |
| Suites | `core_v2` + all category suites (§7) |
| Compressors | Built-in panel (§11): 7 compressors |
| Optional | Real-backend subset if runtime manageable |
| `draft_len` | 4 (anchor) |
| `max_new_tokens` | 16 (anchor) |
| Deliverables | `EXPERIMENT_012_EVAL_SUITE_EXPANSION.md`; gitignored JSON/CSV |
| Script | `scripts/run_experiment_012_eval_suite_expansion.py` (Phase 2) |

**Success criteria:**

- `exactkv_failures == 0`
- Per-category leaderboards published
- Prompt-level win/loss tables published
- Suite version documented in manifest
- No performance claims

**Estimated cells (0.5B, built-in only):** ~115 prompts × 7 compressors ≈ **805 cells**
(minimum suite targets); upper bound ~170 × 7 ≈ **1,190 cells**.

---

## 20. Experiment 013 plan

**Name:** Sensitivity + Divergence Forensics.

| Parameter | Value |
|---|---|
| Models | 0.5B required; 1.5B optional |
| Suite | `core_v2` (or stratified subset if full grid too large) |
| `draft_len` sweep | 2, 4, 8 |
| `max_new_tokens` sweep | 16, 32, 64 (phased) |
| Compressors | Sensitivity panel (§11): 4–5 compressors |
| Analysis | Category-stratified divergence; correction/rejection positions; token-type breakdown |
| Deliverables | `EXPERIMENT_013_SENSITIVITY_FORENSICS.md`; gitignored JSON/CSV |
| Script | `scripts/run_experiment_013_sensitivity_forensics.py` (Phase 3) |

**Success criteria:**

- `exactkv_failures == 0`
- Documented sensitivity findings (draft_len and max_new_tokens)
- Divergence forensics beyond 006A proxy where feasible
- No performance claims

---

## 21. RunPod plan

| Use | Model | Environment |
|---|---|---|
| Exp 012 extension | Qwen2.5-1.5B | L40S, float16 CUDA |
| Exp 013 extension | Qwen2.5-1.5B | L40S, float16 CUDA |
| KVQuant rows | 0.5B / 1.5B | KVQuant venv; per-model quantizer pickle |
| Optional 3B stretch | Qwen2.5-3B | L40S; only after 1.5B clean |

**Constraints:**

- `HF_HUB_ENABLE_HF_TRANSFER=0` for model downloads (V9 lesson).
- Raw JSON/CSV remain **gitignored**; Markdown reports committed.
- SSH pattern documented in experiment reports; not a production deployment claim.
- CPU-first for 0.5B built-in sweeps is acceptable.

---

## 22. Tests and gates

| Gate | When | Requirement |
|---|---|---|
| **Suite validation** | Phase 1 | New prompts load; metadata schema tests; `exactkv_failures == 0` smoke on `smoke` subset |
| **Full pytest** | Before each phase commit | Default env: all tests pass (D5 `model_runtime.py` change at v0.9.0) |
| **Exactness gate** | Every published experiment | `exactkv_failures == 0` |
| **No-performance-claim audit** | Every doc commit | Forbidden terms only in negation/guardrails |
| **`git diff --check`** | Every doc commit | Clean |
| **Report schema freeze** | V10 | No new JSON/CSV fields without separate approval |

Docs-only Phase 0: **no full pytest required**; `git diff --check` and
no-performance-claim audit only.

---

## 23. Exit criteria for v1.0.0 readiness

v1.0.0 public launch requires **V10 + V11 substance** (minimum):

| Gate | Owner | V10 contribution |
|---|---|---|
| Expanded suites + per-category leaderboards published | V10 | Experiment 012 |
| Draft/generation sensitivity documented | V10 | Experiment 013 |
| Divergence forensics beyond 006A proxy | V10 | Experiment 013 |
| Sparse V / attention logging or documented deferral | V10 research | D6–D8 |
| Multi-model matrix on expanded suites | V10–V11 | 0.5B + 1.5B minimum |
| Serving sidecar probe or documented no-go refresh | V11 | D11–D13 |
| `RELEASE_NOTES` + `PROJECT_STATUS` v1.0.0 | v1.0.0 | — |
| Curated raw report bundle policy | v1.0.0 | D17 |
| Public narrative with explicit negation of performance claims | v1.0.0 | D18 |

**v0.9.0 is not v1.0.0.** V10 exit does not automatically trigger v1.0.0 tag.

---

## 24. Risks and unknowns

| Risk | Mitigation |
|---|---|
| Combinatorial cell explosion (suites × compressors × draft_len × max_new_tokens) | Phased experiments; subset sweeps with documented rationale |
| Category suites too small for stable means | Low-n warnings; minimum size targets (§10) |
| Overfitting new suites to boundary4 | Hold-out prompts; regression IDs; prompt-level analysis |
| KVQuant/TurboQuant/KIVI runtime on expanded suites | Optional subset only; omit if unmanageable |
| Attention logging infeasible on HF path | Document blocker; do not fabricate weights |
| 1.5B/3B RunPod cost | 0.5B first; 3B stretch optional |
| Report schema pressure | Freeze schema for V10; analysis in Markdown only |
| Global accept rate hides category regressions | Per-category tables mandatory |

---

## 25. No-performance-claim policy

V10 documents, experiment reports, and any updated README sections must **not**:

- Add or imply `tokens_per_second`, `throughput`, `latency`, `speedup`,
  `runtime_seconds`, or `active_gpu_kv_bytes` as ExactKV results.
- Claim production serving readiness or vLLM/LMCache integration.
- Present `_sim` compressors as real packed-bit backends.
- Cite TurboQuant, KIVI, or KVQuant **paper** results as ExactKV experiment results.
- Imply TurboQuant production, KIVI CUDA/Triton, or KVQuant deployment CUDA was integrated.

Forbidden terms may appear **only** in explicit negation or guardrail prose (as in this
section). Acceptance, rejection, correction, divergence, and honest workspace-memory
accounting remain the permitted metrics.

---

## Related

- [`V10_SCOPE_DRAFT.md`](V10_SCOPE_DRAFT.md) — superseded draft
- [`RELEASE_NOTES_V0.9.0.md`](RELEASE_NOTES_V0.9.0.md) — what V9 shipped
- [`PROJECT_STATUS_V0.9.0.md`](PROJECT_STATUS_V0.9.0.md) — launch readiness at v0.9.0
- [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md) — D26–D29 tracker
- [`EXPERIMENT_006A_DIVERGENCE_ANALYSIS.md`](EXPERIMENT_006A_DIVERGENCE_ANALYSIS.md) — proxy baseline
- [`ROADMAP.md`](ROADMAP.md) — version path
