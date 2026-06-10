# V10 Readiness Assessment

**Status:** V10 Phase 5 complete — evaluation-suite hardening goal **met** for v0.10.0.
**v1.0.0 public launch:** **not ready** — V11 substance still required per
[`V10_SCOPE_STATEMENT.md`](V10_SCOPE_STATEMENT.md) §23.

> This is a **readiness assessment**, not a performance benchmark or launch announcement.
> V10 suites are **stronger than the old 34-prompt `core` suite** but are **not universal benchmarks**.
> Simulated compressors (`_sim`) are **not** real packed-bit backends.
> `total_kv_footprint_bytes` is a conservative accounting sum, **not** measured peak GPU memory.
> **Active GPU memory is not reported.**
> ExactKV does **not** claim speedup, throughput, latency, runtime, tokens/sec, active GPU memory, or production readiness.
> ExactKV does **not** claim external-paper results as ExactKV results.

**Supersedes:** informal Phase 4 handoff notes. **Builds on:** `v0.9.0`.

---

## 1. Purpose

Decide whether V10 completed its evaluation-suite hardening and divergence-forensics
goals, summarize Experiments 012–014, and determine whether ExactKV is ready for
**v1.0.0** or still needs **V11** substance before public launch.

---

## 2. V10 summary

| Phase | Deliverable | Status |
|---|---|---|
| 0 | [`V10_SCOPE_STATEMENT.md`](V10_SCOPE_STATEMENT.md) | ✅ |
| 1 | Seven versioned suites (128 prompts); validator + tests | ✅ |
| 2 | Experiment 012 — suite expansion + per-category leaderboards | ✅ |
| 3 | Experiment 013 — sensitivity + divergence forensics | ✅ |
| 4 | Experiment 014 — real-backend category spot-checks | ✅ |
| 5 | This assessment + v0.10.0 documentation package | ✅ |

**Hard gate:** `exactkv_failures == 0` on Experiments 012, 013, and 014.

**Unchanged:** generation logic, verification logic, default compressor registry,
report JSON/CSV schema.

---

## 3. Experiment 012 summary

| Parameter | Value |
|---|---|
| Prompts | **128** (seven V10 suites) |
| Cells | **896** (128 × 7 built-in compressors) |
| ExactKV failures | **0** |
| Model | `Qwen/Qwen2.5-0.5B`, float32, CPU-first |

**Key findings:**

- **Per-category leaderboards** published — acceptance is no longer a single global number.
- **`long_context` hardest** — suite mean accept **0.896**; category mean **0.902** (lowest among primary categories).
- **boundary4 still beats k8_v4_sim** globally (**0.923** vs **0.914**) but **margin shrinks** vs V9 core: **+0.009** on V10 suites vs **+0.052** on the old 34-prompt core (0.950 vs 0.898).
- Prompt-level win/loss: boundary4 wins **20** / **128** vs k8_v4_sim (**11** losses, **97** ties).
- **`int8` remains strongest simple lossy baseline** at **0.957** global accept.

Full report: [`EXPERIMENT_012_EVAL_SUITE_EXPANSION.md`](EXPERIMENT_012_EVAL_SUITE_EXPANSION.md).

---

## 4. Experiment 013 summary

| Parameter | Value |
|---|---|
| Prompts | **60** (`core_v2` 40 + stress subset 20) |
| Cells | **2160** (60 × 4 compressors × 3×3 grid) |
| ExactKV failures | **0** |
| Model | `Qwen/Qwen2.5-0.5B`, float16, RunPod A5000 |

**Key findings:**

- **draft_len sensitivity:** accept falls with longer drafts — **0.983** (2) → **0.958** (4) → **0.929** (8).
- **max_new_tokens sensitivity:** accept rises slightly with longer generation — **0.947** (16) → **0.959** (32) → **0.963** (64) on the pooled grid.
- **boundary4 > k8_v4_sim** at all draft lengths; margin widens at `draft_len=8` (**+0.024** on `core_v2`).
- **Divergence token-type forensics** (heuristic, no attention weights): **624** wordpiece/other, **159** punctuation, **30** numeric first-divergence tokens.
- **Structured-output flags** on code/tool_json cells (n=756): **300** unmatched-bracket (lossy), **219** quote-imbalance; malformed-JSON-prefix heuristic **0**.
- **`int8` strongest lossy baseline** across grid: **0.970** global mean.

Full report: [`EXPERIMENT_013_SENSITIVITY_FORENSICS.md`](EXPERIMENT_013_SENSITIVITY_FORENSICS.md).

---

## 5. Experiment 014 summary

| Parameter | Value |
|---|---|
| Prompts | **40** (10 per harder suite: `long_context`, `retrieval_copy`, `tool_json`, `code_structured`) |
| Cells | **280** merged unique (40 × 7 compressors; cross-panel) |
| ExactKV failures | **0** |
| Model | `Qwen/Qwen2.5-0.5B`, anchor `draft_len=4`, `max_new_tokens=16` |

**Key findings:**

- **Restricted real-backend ranking:** KVQuant **0.634** > TurboQuant **0.309** > KIVI **0.019** — KVQuant remains strongest restricted real backend; all preserve exactness.
- Accept drops vs V9 **core** anchors on harder mix (KVQuant 0.792 → **0.634**; TurboQuant 0.435 → **0.309**).
- **`long_context` hardest** category: pooled mean accept **0.538** across all compressors.
- **boundary4 > k8_v4_sim** on subset: **0.876** vs **0.864** (+0.012); per-prompt wins **7** / **2** / **31** ties.
- Real backends run in **isolated panels** (factory-only; not co-installed).

Full report: [`EXPERIMENT_014_REAL_BACKEND_SPOTCHECKS.md`](EXPERIMENT_014_REAL_BACKEND_SPOTCHECKS.md).

---

## 6. What V10 fixed from v0.9.0

| Gap at v0.9.0 | V10 resolution |
|---|---|
| Single 34-prompt `core` suite | **128-prompt** versioned panel + taxonomy ([`V10_PROMPT_SUITES.md`](V10_PROMPT_SUITES.md)) |
| No per-category leaderboards | Experiment **012** stratified tables |
| Single `draft_len` / `max_new_tokens` in sweeps | Experiment **013** 3×3 sensitivity grid |
| Proxy-only divergence (006A) | Experiment **013** token-type + structured-output forensics |
| Real backends only on old `core` | Experiment **014** harder-category spot-check |
| Suite overfitting risk undocumented | Prompt-level win/loss + category heatmaps (Exp 012) |
| Evaluation story not launch-defensible | Documented limitations; no universal-benchmark claims |

---

## 7. What V10 did not fix

| Item | Status | Owner |
|---|---|---|
| **1.5B on expanded V10 suites** | Not run (Exp 011 remains 1.5B on legacy `core`) | V11 |
| **True attention logging** | Deferred (D7); no fabricated weights | V11 / research |
| **Sparse V evaluation** (D6) | Deferred | Research |
| **Per-layer/head forensics** (D8) | Deferred without weights | V11 |
| **Serving sidecar / vLLM-LMCache probe** (D13) | Not started | V11 |
| **Active GPU memory methodology** (D14) | Not started | V11 |
| **Production TurboQuant / KIVI CUDA / KVQuant deployment CUDA** | Out of scope | — |
| **Curated raw report bundle** (D17) | Not packaged | v1.0.0 |
| **Public launch narrative** (D18) | Not written (no launch post) | v1.0.0 |
| **Universal benchmark coverage** | Explicitly not claimed | — |

---

## 8. Updated strongest findings

1. **ExactKV preserves exact output** across broader suites (896 + 2160 + 280 published V10 cells; all `exactkv_failures == 0`).
2. **`int8` remains the strongest simple lossy baseline** on V10 panels (0.957–0.970 depending on sweep).
3. **`k8_v4_boundary4_v8_sim` advantage transfers** but is **category-sensitive and smaller** on expanded/harder prompts (+0.009 global in Exp 012; +0.012 on Exp 014 subset).
4. **Restricted real backends are exact-safe** but **less acceptance-efficient** on hard V10 categories (KVQuant 0.634 on Exp 014 vs 0.792 on V9 core).
5. **`long_context`, retrieval-copy, tool-JSON, and code** categories expose the most fragility — especially long context.

---

## 9. Updated leaderboard summary

_Reference: `Qwen/Qwen2.5-0.5B`, greedy, `draft_len=4`, `max_new_tokens=16` unless noted. Not a throughput ranking._

| Context | Top lossy compressors (accept) | Notes |
|---|---|---|
| V9 core (34 prompts) | `int8` ~0.97; boundary4 ~0.95; k8_v4_sim ~0.90 | Pre-V10 anchor |
| V10 full suites (Exp 012) | `int8` **0.957**; boundary4 **0.923**; k8_v4_sim **0.914** | 128 prompts |
| V10 sensitivity grid (Exp 013) | `int8` **0.970**; boundary4 **0.932**; k8_v4_sim **0.923** | Pooled 3×3 grid |
| V10 hard categories (Exp 014) | `int8` **0.958**; boundary4 **0.876**; KVQuant **0.634** | 40-prompt spot-check |
| Restricted real (V9 core) | KVQuant **0.792** > TurboQuant **0.435** > KIVI **0.012** | Factory-only |

---

## 10. Evaluation-suite credibility assessment

**Verdict: materially improved, bounded.**

- Seven versioned suites with metadata, validator, and tests provide a **repeatable, documented** evaluation surface.
- 128 prompts stratify categories the old `core` suite could not.
- Per-category and per-suite tables prevent silent global-mean masking.
- Low-n categories are flagged; suites are **not** claimed universal.
- **Gap:** expanded suites exercised primarily at **0.5B**; 1.5B+ on V10 suites remains open.

---

## 11. Divergence-forensics assessment

**Verdict: upgraded beyond 006A proxy; not complete.**

- Experiment 013 adds first-divergence position, rejection/correction counts, token-type heuristics, and structured-output flags.
- **No true attention weights** were logged — by design and documented deferral (D7).
- Findings support qualitative claims (punctuation/wordpiece-heavy divergence; bracket stress on tool/code) but **not** causal layer/head attribution.

---

## 12. Real-backend credibility assessment

**Verdict: strengthened on harder categories; still factory-only.**

- Experiment 014 confirms exactness + acceptance ordering on long-context / tool / code / retrieval prompts.
- KVQuant remains strongest restricted adapter; accept degrades on hard categories as expected.
- Panels require **isolated environments** — credible for research, not production co-deployment.
- TurboQuant Python, KIVI offline, and KVQuant simquant are **not** production runtimes.

---

## 13. Memory-honesty status

**Unchanged from v0.9.0 — still honest, still limited.**

- Five-field workspace accounting (`stored_kv_bytes`, `materialized_working_kv_bytes`, etc.) remains the published model.
- `total_kv_footprint_bytes` is a **conservative accounting sum**, not measured peak GPU memory.
- **`active_gpu_kv_bytes` is not reported** (D14 deferred to V11).
- Restricted adapters keep `supports_real_bytes_claim=False`.

---

## 14. Remaining blockers for v1.0.0

Per [`V10_SCOPE_STATEMENT.md`](V10_SCOPE_STATEMENT.md) §23, v1.0.0 requires **V10 + V11** minimum:

| Blocker | V10 status | Remaining |
|---|---|---|
| Expanded suites + per-category leaderboards | ✅ Exp 012 | — |
| Draft/generation sensitivity | ✅ Exp 013 | — |
| Divergence forensics beyond 006A | ✅ Exp 013 (heuristic) | Attention logging optional/deferred |
| Multi-model on expanded suites | ❌ | **1.5B+ on V10 suites** (V11) |
| Serving sidecar probe / no-go refresh | ❌ | **V11** (D13) |
| Active GPU memory methodology | ❌ | **V11** (D14) |
| Raw report bundle (D17) | ❌ | **v1.0.0** package |
| Public launch narrative (D18) | ❌ | **v1.0.0** package |
| `PROJECT_STATUS` + `RELEASE_NOTES` v1.0.0 | ❌ | After V11 |

---

## 15. V11 recommendation

**Proceed with V11** as the **scale, serving, and launch-hardening** phase:

1. **1.5B (optional 3B) on V10 suites** — close the multi-model gap on expanded evaluation.
2. **Serving sidecar probe** (D13) — metadata-only or isolated; refresh vLLM/LMCache no-go if needed.
3. **Active GPU memory profiling methodology** (D14) — distinct from footprint accounting.
4. **Optional:** true attention logging on a **small subset** (D7) if feasible without fabrication.
5. **Prepare v1.0.0 package:** raw report bundle (D17), launch narrative draft (D18), `PROJECT_STATUS` / `RELEASE_NOTES` v1.0.0.

V11 is **not** a license to add throughput/latency claims or production-serving claims.

---

## 16. v0.10.0 tag readiness

**Ready to tag `v0.10.0`** when this documentation package is committed:

- V10 Phases 0–5 complete.
- Experiments 012–014 published with `exactkv_failures == 0`.
- [`PROJECT_STATUS_V0.10.0.md`](PROJECT_STATUS_V0.10.0.md) and [`RELEASE_NOTES_V0.10.0.md`](RELEASE_NOTES_V0.10.0.md) written.
- No-performance-claim audit clean on changed docs.

**`v0.10.0` is a research milestone — not public launch.**

---

## 17. Why public launch is still deferred

**No launch gate changed.** [`V10_SCOPE_STATEMENT.md`](V10_SCOPE_STATEMENT.md) explicitly requires **V11 substance** before v1.0.0. V10 hardened evaluation and forensics; it did **not**:

- validate expanded suites on larger models,
- complete serving/profiling gauntlet,
- ship curated report bundle or public narrative,
- integrate production external backends.

ExactKV is **more defensible** as a research platform after V10, but **not ready** for a broad public v1.0.0 launch claim.

---

## Related

- [`PROJECT_STATUS_V0.10.0.md`](PROJECT_STATUS_V0.10.0.md)
- [`RELEASE_NOTES_V0.10.0.md`](RELEASE_NOTES_V0.10.0.md)
- [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md)
- [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md)
