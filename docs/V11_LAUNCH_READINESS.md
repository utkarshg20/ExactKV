# V11 Launch Readiness Assessment

**Status:** V11 Phase 6 complete — launch package prepared. **Not a public launch announcement.**
**Tag readiness:** **`v0.11.0`** — research milestone with V11 substance complete.
**v1.0.0 public launch:** **not yet** — see §20.

> This is a **readiness assessment**, not a performance benchmark or production-serving claim.
> V10/V11 suites are **stronger than the legacy `core` suite** but are **not universal benchmarks**.
> Simulated compressors (`_sim`) are **not** real packed-bit backends.
> `total_kv_footprint_bytes` is a conservative accounting sum, **not** measured peak GPU memory.
> Active GPU memory is **not** a standard schema metric (pilot only, Exp 018).
> ExactKV does **not** claim speedup, throughput, latency, runtime, tokens/sec, active GPU memory savings, or production readiness.
> ExactKV does **not** claim final model accuracy improvement.
> External paper results are **not** ExactKV results.

**Supersedes:** informal V11 phase handoff notes. **Builds on:** `v0.10.0`.

---

## 1. Purpose

Decide whether V11 completed its launch-hardening goals (Experiments 015–020, serving/profiling
refresh, divergence forensics, repair-policy pilot, documentation package), summarize evidence,
and determine readiness for **`v0.11.0`** tag vs **v1.0.0** public launch.

---

## 2. V11 summary

| Phase | Deliverable | Status |
|---|---|---|
| 0 | [`V11_SCOPE_STATEMENT.md`](V11_SCOPE_STATEMENT.md) | ✅ |
| 1 | Experiment 015 — 1.5B on V10 suites | ✅ |
| 2 | Experiment 016 — 3B built-in stretch | ✅ |
| 3 | Experiment 017 — serving sidecar / no-go refresh | ✅ |
| 4 | Experiment 018 — GPU memory methodology pilot | ✅ |
| 5 | Experiment 019 — divergence autopsy | ✅ |
| 5b | Experiment 020 — repair-policy pilot | ✅ |
| 6 | Launch package (this doc + status + release notes + artifact policy + narrative draft) | ✅ |

**Hard gate:** `exactkv_failures == 0` on Experiments 015–020.

**Unchanged:** generation logic, verification logic, default compressor registry, standard report JSON/CSV schema.

---

## 3. Experiment 015 summary

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B`, float16, CUDA |
| Prompts | **128** (full V10 suites) |
| Cells | **896** |
| ExactKV failures | **0** |

**Key findings:**

- V10 compressor rankings **transfer** from 0.5B (Exp 012) to 1.5B.
- `int8` accept **0.978**; boundary4 **0.951** > k8_v4_sim **0.942**; margin **+0.009** (matches Exp 012).
- `long_context` remains among the hardest categories.

Report: [`EXPERIMENT_015_QWEN15B_V10_SUITES.md`](EXPERIMENT_015_QWEN15B_V10_SUITES.md).

---

## 4. Experiment 016 summary

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen2.5-3B`, float16, CUDA |
| Prompts | **128** (full V10 suites) |
| Cells | **896** |
| ExactKV failures | **0** |

**Key findings:**

- Exactness gate holds at **3B** scale on full V10 suites.
- `int8` accept **0.991**; boundary4 **0.952** > k8_v4_sim **0.951**; margin **+0.001** (shrinks vs 0.5B/1.5B).
- Scale story strengthened; layer-aware margin is category- and scale-sensitive.

Report: [`EXPERIMENT_016_QWEN3B_V10_SUITES.md`](EXPERIMENT_016_QWEN3B_V10_SUITES.md).

---

## 5. Experiment 017 summary

| Parameter | Value |
|---|---|
| Focus | Serving sidecar/probe feasibility refresh |
| Cells | **32** |
| ExactKV failures | **0** |

**Key findings:**

- Metadata-only **sidecar probe passes** ownership/invariant checks.
- **Direct vLLM and LMCache integration reaffirmed no-go** for authoritative full-KV verifier.
- Not production serving; harness/sidecar observational layer only.

Report: [`EXPERIMENT_017_SERVING_SIDECAR_PROBE.md`](EXPERIMENT_017_SERVING_SIDECAR_PROBE.md).

---

## 6. Experiment 018 summary

| Parameter | Value |
|---|---|
| Focus | Active GPU memory methodology + pilot |
| Cells | **100** (0.5B + 1.5B × 10 prompts × 5 compressors) |
| ExactKV failures | **0** |
| Decision | `pilot_success` |

**Key findings:**

- PyTorch CUDA allocation observations documented separately from V5 `total_kv_footprint_bytes`.
- **`active_gpu_kv_bytes` not added** to standard report schema.
- V5 accounting remains the stable memory story for launch docs.

Reports: [`GPU_MEMORY_METHODOLOGY.md`](GPU_MEMORY_METHODOLOGY.md), [`EXPERIMENT_018_GPU_MEMORY_PILOT.md`](EXPERIMENT_018_GPU_MEMORY_PILOT.md).

---

## 7. Experiment 019 summary

| Parameter | Value |
|---|---|
| Focus | Deep divergence autopsy + repair hypotheses |
| Cells | **400** (0.5B + 1.5B × 25 prompts × 4 compressors × draft_len {4,8}) |
| ExactKV failures | **0** |

**Key findings:**

- **173** ExactKV rejection events analyzed with logit margins and top-k overlap.
- **long_context** dominates first-divergence cells; K cosine >> V perturbation on simulated policies.
- **Attention logging deferred** (sdpa backend); no fabricated weights.
- Repair hypotheses proposed (not implemented in core).

Report: [`EXPERIMENT_019_DIVERGENCE_AUTOPSY.md`](EXPERIMENT_019_DIVERGENCE_AUTOPSY.md).

---

## 8. Experiment 020 summary

| Parameter | Value |
|---|---|
| Focus | Autopsy-guided repair-policy pilot |
| Cells | **300** (0.5B + 1.5B × 25 prompts × 6 policies) |
| ExactKV failures | **0** |

**Key findings:**

- `fallback_int8_for_hard_categories` accept **0.979** vs `baseline_boundary4` **0.932**.
- `category_adaptive_policy` accept **0.973**; best prompt-level wins vs baselines.
- Policies are **experiment-layer only** — not enabled in core ExactKV.
- `structured_safe_mode` inconclusive (mixed structured-suite deltas).

Report: [`EXPERIMENT_020_REPAIR_POLICY_PILOT.md`](EXPERIMENT_020_REPAIR_POLICY_PILOT.md).

---

## 9. Strongest findings so far

| Finding | Source |
|---|---|
| Exactness gate on all published cells (001–020) | All experiments |
| `int8` strongest simple built-in baseline across scales | Exp 012–016, 019–020 |
| boundary4 > k8_v4_sim in many panels; margin shrinks at 3B and on hard categories | Exp 012–016, 014 |
| `long_context` / `retrieval_copy` expose compressor fragility | Exp 012–014, 019–020 |
| KVQuant simquant strongest restricted real backend; KIVI weakest | Exp 008–010, 014 |
| Larger-model validation through **3B** preserves exactness | Exp 015–016 |
| Serving sidecar feasible; direct vLLM/LMCache **no-go** | Exp 017 |
| V5 memory accounting stable; GPU pilot separate | Exp 018 |
| Autopsy-guided policies improve acceptance while preserving exactness | Exp 020 |

---

## 10. What ExactKV proves

- Lossy KV can serve as **draft state** under a full-KV verifier while preserving **exact greedy output**.
- **Acceptance, rejection, correction, and divergence** quantify how useful compressed drafts are.
- Built-in baselines (`int8`, layer-aware boundary policies) and restricted real backends can be compared **fairly** under the same exactness gate.
- Multi-model validation (0.5B → 3B) and expanded V10 suites make compressor claims **more defensible**.
- Category-aware **policy selection** (pilot) can improve draft acceptance without changing verification.

---

## 11. What ExactKV does not prove

- Speed, throughput, latency, runtime, or tokens/sec improvement.
- Active GPU memory savings or production memory reduction.
- Production serving readiness or vLLM/LMCache integration.
- Final model accuracy improvement from compression.
- Universal benchmark coverage.
- That `_sim` compressors are real packed-bit backends.
- That upstream TurboQuant/KIVI/KVQuant **paper** results are ExactKV results.

---

## 12. Claims allowed for launch

Evidence-backed, narrow claims only:

- ExactKV preserves exact greedy output in published experiments (`exactkv_failures == 0`).
- ExactKV evaluates compressed KV as draft state under full-KV verification.
- Acceptance, rejection, correction, and divergence expose draft usefulness.
- Simple baselines such as `int8` remain very strong.
- Layer-aware V policies help in many settings but are category- and scale-sensitive.
- Restricted real backends are exact-safe but not automatically better draft sources.
- KVQuant simquant is the strongest restricted real backend tested so far.
- Larger-model validation up to Qwen2.5-3B preserves exactness on V10 suites.
- Long-context and structured/retrieval categories expose compressor fragility.
- Autopsy-guided policy selection improved draft acceptance in a pilot while preserving exactness.

---

## 13. Claims forbidden for launch

- No speedup, throughput, latency, runtime, or tokens/sec claims.
- No active GPU memory saving claims.
- No production serving, vLLM, LMCache, or PagedAttention implementation claims.
- No model accuracy improvement claims.
- No universal benchmark claims.
- No simulated-as-real-packed-bit claims.
- No external-paper-results-as-ExactKV claims.

---

## 14. Memory-honesty status

| Item | Status |
|---|---|
| V5 `total_kv_footprint_bytes` | Stable launch memory story |
| Workspace accounting (Exp 004) | Validated |
| Active GPU memory in standard schema | **Not added** |
| Exp 018 pilot | Documented; isolated artifact only |
| Claim active GPU savings | **Forbidden** |

---

## 15. Serving-honesty status

| Item | Status |
|---|---|
| Experiment 007 harness | Reference lifecycle implementation |
| Exp 017 sidecar probe | Pass; metadata-only |
| Direct vLLM integration | **No-go** |
| Direct LMCache integration | **No-go** |
| Production serving claim | **Forbidden** |

---

## 16. Real-backend status

| Backend | Exact-safe | Draft usefulness | Default registry |
|---|---|---|---|
| KVQuant simquant | Yes | Strongest restricted | No (factory-only) |
| TurboQuant Python | Yes | Moderate | No |
| KIVI offline | Yes | Very weak accept | No |
| kvpress Knorm | Yes | Evaluated (Exp 005) | No |

Production CUDA/Triton paths remain **deferred**.

---

## 17. Evaluation-suite status

- **128 prompts**, seven versioned V10 suites — validated, not universal benchmarks.
- Per-category leaderboards (Exp 012+) — standard for published sweeps.
- Sensitivity grid (Exp 013) — draft_len and max_new_tokens documented.
- Hard-category spot-checks (Exp 014) — real backends on 40-prompt panel.

---

## 18. Repair-policy status

- Exp 019 hypotheses → Exp 020 pilot validation.
- Best pilots: `fallback_int8_for_hard_categories`, `category_adaptive_policy`.
- **Not in core ExactKV**; not enabled by default.
- `structured_safe_mode` — inconclusive; not recommended without further work.

---

## 19. Remaining limitations

- Attention logging deferred (D7); per-head forensics partial (D8).
- No sampling, batching, or bonus-token acceptance.
- Single-request, greedy, Hugging Face–centric runtime.
- Repair policies validated on 25-prompt pilot panel only.
- Physical curated tarball optional until release attach (policy documented).
- Public launch narrative is **draft** — requires review before v1.0.0 post.

---

## 20. v1.0.0 readiness decision

| Decision | Outcome |
|---|---|
| **Tag `v0.11.0`** | **Ready** — V11 substance and documentation package complete |
| **Public `v1.0.0` launch** | **Not ready** — deliberate v1.0.0 review still required |

**Rationale for v0.11.0:** All V11 phases complete; Experiments 015–020 pass exactness gate;
launch docs, artifact policy, and narrative draft prepared; no schema or core-logic changes.

**Rationale against immediate v1.0.0 public launch:**

- `PROJECT_STATUS` / `RELEASE_NOTES` **v1.0.0** not yet published (D19/D20).
- Launch narrative is **draft only** — not approved for external posting.
- Curated artifact **bundle** policy documented; physical bundle optional pre-tag.
- Full pytest sweep recommended before v1.0.0 (optional for v0.11.0 tag).

**Recommended next step:** Tag **`v0.11.0`**, then v1.0.0 launch polish (narrative review, optional bundle build, full pytest, v1.0.0 status/release notes).

---

## Related

- [`PROJECT_STATUS_V0.11.0.md`](PROJECT_STATUS_V0.11.0.md)
- [`RELEASE_NOTES_V0.11.0.md`](RELEASE_NOTES_V0.11.0.md)
- [`RAW_ARTIFACT_POLICY.md`](RAW_ARTIFACT_POLICY.md)
- [`LAUNCH_NARRATIVE_DRAFT.md`](LAUNCH_NARRATIVE_DRAFT.md)
- [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md)
