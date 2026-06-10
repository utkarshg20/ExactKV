# ExactKV v0.10.0 Release Notes

**Status:** V10 complete (Phases 0–5). **Tag:** `v0.10.0` — **research milestone, not public launch.**
**Base:** Builds on `v0.9.0` (real-backend gauntlet, Experiments 008–011).

> **V10 is an evaluation-suite hardening release, not a performance or production release.**
> V10 suites are **stronger than the old 34-prompt `core` suite** but are **not universal benchmarks**.
> Simulated compressors (`_sim`) are **not** real packed-bit backends.
> Restricted real backends remain **factory-only** — not in the default registry.
> `total_kv_footprint_bytes` is a conservative accounting sum, not measured peak GPU memory.
> Active GPU memory is **not** reported.
> ExactKV does **not** claim speedup, throughput, latency, runtime, tokens/sec, or production readiness.
> External paper results are **not** ExactKV results.

---

## 1. V10 summary

V10 hardens ExactKV's evaluation suite and divergence forensics so compressor
behaviour claims are **more defensible** before any public v1.0.0 launch — without
changing the exactness gate, generation logic, verification logic, or report schemas.

V10 delivers:

- **Phase 0:** Formal [`V10_SCOPE_STATEMENT.md`](V10_SCOPE_STATEMENT.md).
- **Phase 1:** Seven versioned prompt suites (128 prompts); validator + tests.
- **Phase 2:** Experiment **012** — suite expansion + per-category leaderboards.
- **Phase 3:** Experiment **013** — draft/generation sensitivity + forensics.
- **Phase 4:** Experiment **014** — real-backend category spot-checks.
- **Phase 5:** Readiness assessment + this release package.

**Hard gate:** `exactkv_failures == 0` on Experiments 012, 013, and 014.

---

## 2. New prompt suites

| Suite | Prompts | Focus |
|---|---:|---|
| `core_v2` | 40 | Expanded core |
| `code_structured` | 21 | Code / structured output |
| `long_context` | 15 | Long-context stress |
| `reasoning_math` | 15 | Reasoning / math |
| `multilingual` | 15 | Multilingual |
| `retrieval_copy` | 11 | Retrieval / copy |
| `tool_json` | 11 | Tool / JSON schema |
| **Total** | **128** | |

Documented in [`V10_PROMPT_SUITES.md`](V10_PROMPT_SUITES.md).

---

## 3. Experiment 012

| | |
|---|---|
| Cells | **896** (128 prompts × 7 built-in compressors) |
| Failures | **0** |
| Headline | `int8` **0.957**; boundary4 **0.923** > k8_v4_sim **0.914** |
| Hardest | `long_context` (suite mean **0.896**) |

[`EXPERIMENT_012_EVAL_SUITE_EXPANSION.md`](EXPERIMENT_012_EVAL_SUITE_EXPANSION.md)

---

## 4. Experiment 013

| | |
|---|---|
| Prompts | **60** (`core_v2` + stress subset) |
| Cells | **2160** (3×3 `draft_len` × `max_new_tokens` grid) |
| Failures | **0** |
| Headline | boundary4 **0.932** > k8_v4_sim **0.923**; accept falls with `draft_len` 8 |

[`EXPERIMENT_013_SENSITIVITY_FORENSICS.md`](EXPERIMENT_013_SENSITIVITY_FORENSICS.md)

---

## 5. Experiment 014

| | |
|---|---|
| Prompts | **40** (harder categories, 10 per suite) |
| Cells | **280** merged (cross-panel) |
| Failures | **0** |
| Headline | KVQuant **0.634** > TurboQuant **0.309** > KIVI **0.019**; `long_context` hardest (**0.538**) |

[`EXPERIMENT_014_REAL_BACKEND_SPOTCHECKS.md`](EXPERIMENT_014_REAL_BACKEND_SPOTCHECKS.md)

---

## 6. What changed since v0.9.0

| Area | v0.9.0 | v0.10.0 |
|---|---|---|
| Prompt suites | 34-prompt `core` primary | **128-prompt** V10 panel + metadata |
| Leaderboards | Global means dominant | **Per-suite / per-category** tables |
| Sensitivity | `draft_len=4`, `max_new_tokens=16` only | **3×3 grid** documented (Exp 013) |
| Divergence | 006A proxy | **Token-type + structured-output** forensics |
| Real backends on V10 | Not run | **Exp 014** spot-check on hard categories |
| Experiments published | 001–011 (+ 006A, 006C) | **+ 012, 013, 014** |
| Public launch | Deferred | **Still deferred** (V11 required) |

**Unchanged:** generation logic, verification logic, default compressor registry (15 built-in),
report JSON/CSV schema.

---

## 7. What V10 proves

- The exactness gate holds on **3,336** new V10 cells (896 + 2160 + 280) at 0.5B.
- Compressor rankings and boundary-layer findings **partially survive** suite expansion and sensitivity sweeps.
- Category-stratified analysis reveals **`long_context`** as the primary fragility driver.
- Restricted real backends remain **exact-safe** on harder V10 categories; KVQuant leads on accept.
- Evaluation claims are **more defensible** than at v0.9.0.

---

## 8. What V10 does not prove

- Universal benchmark coverage or production serving readiness.
- Throughput, latency, speedup, or active GPU memory behaviour.
- That `_sim` compressors are real packed-bit backends.
- That TurboQuant, KIVI, or KVQuant **paper** results apply unchanged.
- That 1.5B+ behaviour matches 0.5B on expanded suites (not run).
- Causal attention-head importance (no weights logged).

---

## 9. Known limitations

- **Single model** (0.5B) for Experiments 012–014.
- **Low-n** categories in some suites; Exp 014 is 40 prompts only.
- **Heuristic** token-type and structured-output flags — not full JSON validation.
- **Cross-panel** real-backend runs — not single-environment co-installation.
- **No** active GPU memory profiling (D14 deferred).
- **No** serving sidecar probe (D13 deferred).

---

## 10. Deferred work

Moved to **V11** or **v1.0.0**:

| ID | Item |
|---|---|
| D7 | True attention logging (optional) |
| D13 | vLLM / LMCache sidecar probe |
| D14 | Active GPU memory profiling |
| D17 | Raw report bundle |
| D18 | Public launch narrative |
| D19–D20 | `PROJECT_STATUS` / tag v1.0.0 |

See [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md).

---

## 11. Tag readiness

| Check | Status |
|---|---|
| V10 Phases 0–5 complete | ✅ |
| Experiments 012–014 published | ✅ |
| `exactkv_failures == 0` on V10 experiments | ✅ |
| Readiness assessment | ✅ [`V10_READINESS_ASSESSMENT.md`](V10_READINESS_ASSESSMENT.md) |
| v1.0.0 launch | ❌ Deferred — V11 required |

**Ready to tag `v0.10.0`.** Not ready for public v1.0.0 launch.

---

## Related

- [`PROJECT_STATUS_V0.10.0.md`](PROJECT_STATUS_V0.10.0.md)
- [`V10_SCOPE_STATEMENT.md`](V10_SCOPE_STATEMENT.md)
- [`RELEASE_NOTES_V0.9.0.md`](RELEASE_NOTES_V0.9.0.md)
- [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md)
