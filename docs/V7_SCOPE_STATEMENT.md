# V7 Scope Statement: Attention-Aware and V-Specific Experiments

**Status:** Phase B complete (simulated layer-aware V policy). Phase A analysis:
[`docs/EXPERIMENT_006A_DIVERGENCE_ANALYSIS.md`](EXPERIMENT_006A_DIVERGENCE_ANALYSIS.md).
**Builds on:** `v0.6.0` — `BackendAdapter` boundary complete; restricted
`KVPressKnormAdapter` (KnormPress only); Experiment 005 (272 runs,
`exactkv_failures == 0`).
**Expands:** [`docs/FUTURE_ROADMAP_V6_V8.md`](FUTURE_ROADMAP_V6_V8.md) §V7 into an
approvable, phased scope.

> No attention-aware compressors, Sparse V dequantization, layer-aware V policies,
> pre-RoPE key quantization, KIVI, KVQuant, TurboQuant, TurboQuant+, LMCache,
> vLLM, or PagedAttention are implemented in this document.
> No performance, throughput, latency, speedup, or production-readiness claims.
> External systems named below are **research motivation and related work**,
> not current ExactKV capabilities.

---

## 1. V7 goal

Design the next research layer after V6 by evaluating **attention-aware and
V-specific compression ideas** through ExactKV's existing framework:

- exact token equality (`exactkv_output_ids == full_output_ids`)
- acceptance rate and average accepted length
- first-divergence position
- rejection and correction counts
- V5 workspace-memory fields (`stored_kv_bytes`, `materialized_working_kv_bytes`,
  `metadata_bytes`, `temporary_workspace_bytes`, `total_kv_footprint_bytes`)
- honest `supports_real_bytes_claim` and `is_simulated` labelling

V7 is a **research and evaluation** version, not a serving-stack benchmark and
not a performance release. It asks whether policies that go beyond uniform
per-tensor bit-width — attention-gated V handling, layer-specific V budgets,
pre-RoPE key ideas, and real-vs-simulated asymmetric comparisons — show up in
**ExactKV acceptance behaviour** under full-KV verification.

If analysis alone is sufficient, **V7 may validly deliver analysis-only results**
without new compressor code (see §7 and §14).

---

## 2. Why V7 matters after V6

V6 established that a **real backend** can plug into ExactKV behind
`BackendAdapter` and pass the exactness gate. Experiment 005 showed:

- **Keys and values remain asymmetric in acceptance impact** — `k_full_v8` (~99%
  draft acceptance) vs aggressive key compression (`k4_v8_sim` ~56%) and token
  dropping (`kvpress_knorm_restricted` ~41%).
- **Real pruned-cache bytes differ from simulated int8-container bytes** — kvpress
  reports real pruned `DynamicCache` storage; `_sim` compressors do not.
- **Low draft acceptance does not imply ExactKV failure** — verification corrects;
  `exactkv_failures == 0` holds.

V6 answered: *Can a real backend integrate?* V7 asks deeper questions:

> **Where and why do lossy drafts diverge, and can attention-aware or V-specific
> policies improve acceptance without breaking exactness?**

Related work (TurboQuant+, KV-AdaQuant, KIVI, KVQuant, PyramidKV, eviction methods)
motivates policies that are **not** captured by ExactKV's current uniform
per-tensor quantizers or V6's restricted KnormPress token-dropping. V7 evaluates
those ideas **through ExactKV's own metrics**, not by reproducing external-paper
accuracy or speed numbers.

**Critical nuance from V4:** ExactKV's `k8_v2_sim` (~33% acceptance) does **not**
refute TurboQuant+'s "aggressive V is nearly free" claim — `k8_v2_sim` is naive
INT2 in int8 containers with no rotation, no sparse V, and no layer-aware policy.
V7 must test **policy-shaped hypotheses**, not conflate naive simulators with
external backends.

---

## 3. What V7 should add

### 3.1 Attention-weighted divergence analysis (analysis-only, preferred first)

A pure analysis layer that correlates existing experiment traces (Experiments 003,
004, 005) with attention statistics:

- first-divergence position vs attention entropy / weight concentration
- rejection position vs high-attention vs low-attention tokens
- compressor-family patterns (key-fragile vs value-fragile vs eviction-style)

**No generation-logic change.** May require a one-pass attention-logging utility
on existing prompts; must not introduce timing or throughput metrics.

### 3.2 Simulated V-specific / layer-aware policy experiments (conditional)

If Phase A analysis reveals a clear direction, a **small simulated policy**
compressor (new `_sim` or policy-labelled variant) that models:

- **Layer-aware V precision** — e.g. higher V precision on first/last *N* layers,
  lower elsewhere (PyramidKV-motivated, not PyramidKV implementation).
- **Boundary-layer policies** — asymmetric treatment of early vs late layers.

All such compressors must set `is_simulated=True` and
`supports_real_bytes_claim=False` unless they use genuine packed storage (unlikely
in V7 Phase B/C simulated work).

### 3.3 Sparse V dequantization evaluation (conditional, separate approval)

An adapter or compressor extension that **skips or approximates** dequantisation
of value positions below an attention-weight threshold during draft materialize.

**Hard gate:** materialize must remain **deterministic and reproducible** so draft
and verify semantics are not corrupted. Full correctness re-validation before any
Experiment 006 cell runs.

ExactKV does **not** implement TurboQuant+ Sparse V dequantization in Phase 0;
this is a candidate direction only.

### 3.4 Pre-RoPE key quantization ideas (conditional, separate approval)

A KVQuant-**motivated** adapter that quantises keys before rotary embedding.
Deferred from V6 because it requires deeper forward-pass hooking.

ExactKV does **not** implement KVQuant in Phase 0. Any future adapter must pass the
same exactness and hook-safety gates as V6 kvpress work.

### 3.5 Real asymmetric compressor comparison (conditional on approved real adapter)

Side-by-side core-suite comparison of an approved real backend (e.g. KIVI-style
per-channel K + per-token V) against ExactKV's simulated asymmetric compressors
(`k8_v4_sim`, `k4_v8_sim`, `k_full_v8`, etc.) with explicit simulated/real
labelling.

V6's restricted KnormPress adapter is **token-dropping**, not asymmetric
quantization — it is a baseline, not a substitute for KIVI/KVQuant/TurboQuant
evaluation.

### 3.6 Experiment 006 report and documentation

- `docs/EXPERIMENT_006_ATTENTION_V.md` (or equivalent name at implementation time)
- V7 release notes when the phase completes
- Clear separation of analysis-only vs simulated-policy vs real-backend results

### 3.7 No changes to V6 kvpress scope

V7 does **not** broaden kvpress beyond existing V6 restricted KnormPress work
(no DecodingPress, AdaKVPress, ComposedPress, default registry, or default deps).

---

## 4. What V7 explicitly does not add

* ❌ **No throughput, latency, tokens/sec, speedup, or `runtime_seconds`.**
* ❌ **No production-serving or production-readiness claims.**
* ❌ **No vLLM, LMCache, or PagedAttention integration** (V8 evaluation context
  at most; not promised in V7).
* ❌ **No CUDA/Triton kernels written by ExactKV.**
* ❌ **No CPU offload, batching, sampling, parallel verification, or bonus-token
  acceptance.**
* ❌ **No changes to generation logic or the draft-verify-commit loop** (analysis
  utilities may read model activations; they must not alter decode semantics).
* ❌ **No changes to existing report schemas** beyond additive analysis fields or
  capability metadata if a later approved phase genuinely needs them (additive and
  backward-compatible only).
* ❌ **No active GPU memory profiling** (`torch.cuda.memory_reserved`, etc.) —
  deferred to V8 at the earliest.
* ❌ **No implementation claims** for Sparse V dequantization, layer-aware V
  compression, pre-RoPE key quantization, KVQuant, TurboQuant, TurboQuant+,
  KIVI, PyramidKV, SnapKV, H2O, or StreamingLLM unless a **later approved phase**
  explicitly implements a scoped adapter or simulator.
* ❌ **No new default-registry kvpress compressors** or default kvpress dependencies.
* ❌ **No presentation of external-paper perplexity, accuracy, or speed numbers
  as ExactKV results.**

---

## 5. Candidate research directions

> All items below are **candidates for evaluation**, not V7 deliverables in
> Phase 0. Each requires separate phase approval before code.

| Direction | ExactKV question | Likely form in V7 |
|---|---|---|
| **Sparse V dequantization** | Does attention-gated V materialize change acceptance vs dense V at the same stored budget? | Real or simulated adapter extension; determinism gate |
| **Layer-aware V policies** | Does higher V precision on sensitive layers improve acceptance at matched average budget? | Simulated policy compressor |
| **Attention-weighted divergence analysis** | Do divergences/rejections cluster at high-entropy positions? | Analysis-only on existing reports |
| **Pre-RoPE key quantization ideas** | Does pre-RoPE K quant preserve acceptance better than post-RoPE per-tensor K? | KVQuant-motivated adapter (separate approval) |
| **Boundary-layer policies** | Are first/last layers more acceptance-sensitive for V compression? | Layer-tagged analysis + optional simulated policy |
| **Real asymmetric compressor comparisons** | Does real per-channel K / per-token V beat simulated asymmetric at comparable budget? | Experiment 006b with approved real adapter |
| **Real vs simulated K/V policy comparison** | Where do simulated `_sim` policies mis-rank vs a real backend on acceptance? | Side-by-side tables with explicit labels |

**Token eviction (orthogonal):** PyramidKV, SnapKV, H2O, and StreamingLLM motivate
*which tokens exist*, not just precision. Evaluating eviction under ExactKV
verification requires a **separate scope review** defining how dropped tokens
interact with `logical_seq_len` and the exactness gate. Not in V7 Phase 0.

---

## 6. Relationship to related work

> ExactKV **does not implement** these systems in Phase 0. Citations are for
> motivation and honest attribution only.

| External work | V7 relation | ExactKV stance |
|---|---|---|
| **TurboQuant+** | Rotation-based V, Sparse V dequant, layer-aware V themes | Evaluate *whether* policy-shaped ideas change acceptance; **do not** cite TurboQuant+ speed or quality numbers as ExactKV results |
| **KV-AdaQuant** | Key–value norm disparity; key-prioritized quantization theory | V4 Experiment 003 acceptance aligns with "keys need more bits"; V7 extends with attention-aware analysis — **not** a reproduction of KV-AdaQuant accuracy claims |
| **KIVI** | Per-channel K, per-token V, residual | Candidate for real asymmetric comparison (006b) if a scoped adapter is approved in a later phase |
| **KVQuant** | Pre-RoPE key quant, per-channel scales, dense-sparse outliers | Motivates pre-RoPE direction; **not** implemented in Phase 0 |
| **TurboQuant** | Rotation + Lloyd-Max / PolarQuant | Motivates rotation-based V adapter for V7+; deferred unless separately approved |
| **PyramidKV** | Layer-aware cache budgets | Motivates layer-aware V **policy experiments**; ExactKV measures acceptance, not PyramidKV perplexity |
| **SnapKV** | Observation-window KV selection | Eviction candidate; separate scope review |
| **H2O** | Heavy-hitter token retention | Eviction candidate; separate scope review |
| **StreamingLLM** | Attention sinks + sliding window | Eviction candidate; separate scope review |

**Simulated compressors (`_sim`):** store sub-INT8 values in int8 containers.
Their `stored_kv_bytes` reflects int8 container reality, **not** packed-bit
savings. Never present `_sim` figures alongside real-backend figures without
explicit labelling.

---

## 7. Recommended first V7 direction and why

**Recommendation: analysis-first, then a small simulated policy experiment only
if the analysis reveals a clear direction.**

**Phase A — Attention-weighted divergence analysis** using existing Experiment 003,
004, and 005 JSON/CSV reports (and optional one-pass attention logging on the
core suite) because:

1. **Lowest risk** — no generation-logic change, no new compressor registry entries.
2. **High information** — V4/V6 already show key-fragility and real-vs-sim
   separation; V7 should explain *where* divergences occur before building new
   compressors.
3. **Avoids premature implementation** — Sparse V, layer-aware V, and pre-RoPE
   adapters have high correctness risk; analysis may show they are unnecessary or
   may narrow the policy design space.
4. **Reuses V6 investment** — Experiment 005 traces are fresh; combining with
   Experiment 003 asymmetric data is natural.

**If Phase A warrants Phase B:** prefer a **simulated layer-aware V policy**
(e.g. `k_full_v4_sim` on outer layers, `k_full_v8` on inner layers) over jumping
to a real TurboQuant/KVQuant/KIVI adapter. Real backends remain **separate
approval per adapter**.

**If Phase A does not reveal a actionable direction:** V7 may complete with an
analysis report only (valid design-only outcome per §15).

---

## 8. Proposed V7 phases

| Phase | Scope | Deliverable | Status |
|---|---|---|---|
| **Phase 0** (this document) | Scope statement only; no code | `docs/V7_SCOPE_STATEMENT.md` committed and reviewed | ✅ Complete |
| **Phase A** | Proxy divergence analysis on Experiments 003/004/005; no generation change | [`docs/EXPERIMENT_006A_DIVERGENCE_ANALYSIS.md`](EXPERIMENT_006A_DIVERGENCE_ANALYSIS.md); `exactkv/analysis/attention_weighted.py` | ✅ Complete |
| **Phase B** | Simulated layer-aware V policy (`k8_v4_boundary_v8_sim`); no attention weights | `exactkv/compressors/layer_aware_sim.py` + tests | ✅ Complete |
| **Phase C** | Optional real asymmetric adapter (KIVI / KVQuant / TurboQuant-style) — **each requires separate approval** | Scoped adapter + exactness gate on core suite | Pending scope |
| **Phase D** | Experiment 006 (attention-aware or V-specific sweep) + report | `docs/EXPERIMENT_006_*.md`; gitignored JSON/CSV | Pending B or A |
| **Phase E** | V7 release notes; README/ROADMAP updates; audit; tag | `docs/RELEASE_NOTES_V0.7.0.md` | Pending D |

> Phases B–E require **separate explicit approval** before any code is written.
> Phase 0 (this document) is design-only and introduces no code and no behaviour
> change.

**Experiment numbering:**

- **Experiment 006** — primary V7 sweep (analysis-informed simulated policy and/or
  attention-aware compressor candidates vs baselines).
- **Experiment 006b** (conditional) — real vs simulated asymmetric comparison when
  an approved real adapter exists beyond V6 KnormPress.

---

## 9. Required metrics

Every V7 experiment cell must report (unchanged from V1–V6):

| Metric | Requirement |
|---|---|
| `exactkv_failures` | **Must be 0** (hard gate) |
| `exactkv_output_ids == full_output_ids` | Must hold under greedy decoding |
| Acceptance rate | Per compressor / policy |
| Average accepted length | Per compressor / policy |
| First divergence position | Distribution or per-run value |
| Rejection count | Per run; must reconcile with draft bookkeeping |
| Correction count | Per run; must reconcile |
| `stored_kv_bytes` | V5 workspace field |
| `materialized_working_kv_bytes` | V5 workspace field |
| `metadata_bytes` | V5 workspace field |
| `temporary_workspace_bytes` | V5 workspace field |
| `total_kv_footprint_bytes` | Conservative accounting sum, **not** measured peak GPU memory |
| `supports_real_bytes_claim` | Honest per compressor |
| `is_simulated` | Honest per compressor |
| `backend_name` / `backend_version` | When a real adapter is used (additive) |

**Not reported:** tokens/second, throughput, latency, speedup, wall-clock runtime.

---

## 10. Experiment 006 plan

**Name:** Attention-Aware or V-Specific Compression Sweep

**Goal:** Characterise whether attention-aware or V-specific policies change
acceptance behaviour relative to V4/V6 baselines, with `exactkv_failures == 0`.

**Illustrative configuration (finalised at implementation time):**

| Parameter | Illustrative value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` (float32; CPU for simulated policies) |
| Suite | `core` (34 prompts) |
| Baselines | `noop`, `int8`, `k_full_v8`, `k8_v_full`, `k8_v4_sim`, `k_full_v4_sim` |
| V7 candidates | Phase A/B outputs only (e.g. layer-aware `_sim` policy) |
| Optional real row | Approved real asymmetric adapter only (006b) |
| `draft_len` | Small set (e.g. 4) |
| `max_new_tokens` | Small but meaningful (e.g. 16) |

**Reported (per compressor / policy):**

- All metrics in §9
- Explicit **experiment class** label: `analysis-only`, `simulated-policy`, or
  `real-backend`
- Lossy divergence count (expected for lossy policies; not an ExactKV failure)

**Required wording in the report:**

- "Simulated compressors store sub-INT8 values in int8 containers;
  `stored_kv_bytes` reflects int8 storage, not packed-bit savings."
- "`total_kv_footprint_bytes` is a conservative accounting sum, not a measured
  peak GPU memory value. Active GPU measurement is deferred."
- "ExactKV evaluates policies by acceptance behaviour and memory honesty. It
  does not claim speedup, throughput, latency, runtime, or production readiness,
  and does not cite external-paper numbers as ExactKV results."
- VeriCache attribution (unchanged).

**Artifacts:** `reports/experiment_006_*.json` and `.csv` — **gitignored** (same
policy as Experiment 005).

---

## 11. Separating simulated policy, real backend, and analysis-only experiments

| Class | `is_simulated` | `supports_real_bytes_claim` | Report labelling |
|---|---|---|---|
| **Analysis-only** | N/A (no new compressor) | N/A | Section titled "Analysis only — no new compressor"; no memory comparison to real backends |
| **Simulated policy** | `True` | `False` (unless genuine packed storage, unlikely) | ⚠️ Simulated; int8-container bytes |
| **Real backend** | `False` | `True` only when storage is genuinely packed or pruned real cache bytes | Backend name + version; separate table from `_sim` |

**Hard rules:**

- Never rank compressors by "memory savings" across classes without matching
  `supports_real_bytes_claim` semantics.
- Never imply a `_sim` policy implements TurboQuant+, KVQuant, or KIVI.
- V6 `kvpress_knorm_restricted` is **real token-dropping**, not asymmetric
  quantization — label separately from KIVI-style adapters in any 006b table.

---

## 12. Avoiding overclaiming external-paper results

1. **Attribute externally** — "KIVI reports X perplexity (Liu et al., 2024)" is
   fine in related-work prose; "ExactKV achieves X perplexity" is not.
2. **ExactKV metrics only in results tables** — acceptance, divergence,
   rejection, correction, workspace fields, `exactkv_failures`.
3. **No speed adoption** — wrapping or evaluating a backend does not authorize
   citing that backend's throughput or serving benchmarks.
4. **Theory alignment ≠ reproduction** — "Experiment 003 acceptance aligns with
   KV-AdaQuant's key-prioritization direction" is acceptable; "ExactKV confirms
   KV-AdaQuant's 75% accuracy" is not.
5. **Naive sim ≠ named backend** — `k8_v2_sim` is not TurboQuant `turbo2`; say so
   explicitly in any V7 report comparing to TurboQuant+ themes.

---

## 13. GPU requirements

- **Phase A analysis on existing CPU experiment reports:** No GPU required.
- **Optional attention logging on core suite:** CPU sufficient for
  `Qwen/Qwen2.5-0.5B` float32 (same as V6 Experiment 005).
- **Simulated layer-aware / V-specific policies:** CPU sufficient for exactness
  gate on small model.
- **Real asymmetric adapters (KIVI, KVQuant, TurboQuant-style):** CUDA likely
  required for reference implementations; each adapter documents device
  requirements. CPU smoke correctness where a CPU path exists.
- **CI:** Default env remains kvpress-free; analysis-only phases need no kvpress.
- **No GPU memory profiling in V7** — deferred to V8.

---

## 14. Risks and unknowns

| Risk | Severity | Mitigation |
|---|---|---|
| Analysis phase does not reveal actionable policy direction | Medium | Valid V7 outcome = analysis report only (§15) |
| Sparse V materialize breaks determinism / exactness | High | Determinism gate before Experiment 006; reject policy if unmet |
| Pre-RoPE / KVQuant-style hooking expands verification risk | High | Separate approval; hook-safety gate like V6 |
| Scope drifts toward "best compressor" or serving benchmarks | High | No-performance-claim policy (§16); acceptance framing only |
| Conflating `_sim` with real TurboQuant/KIVI in reports | High | §11 labelling rules; separate tables |
| Attention logging reintroduces timing metrics | Medium | Log activations only; forbid `runtime_seconds` |
| Real adapter approval expands V7 into V6-repeat integration work | Medium | 006b optional; KnormPress baseline already in V6 |
| Eviction policies break `logical_seq_len` invariants | High | Separate scope review; not in Phase 0 |

---

## 15. Exit criteria

V7 is complete when **either** the full path or the analysis-only fallback is met.

**Phase B layer-aware simulated compressor (2026-06-09):**

- `k8_v4_boundary_v8_sim`: K=INT8 all layers; V=INT8 on first/last layer,
  V=INT4-sim on interior (`boundary_layers=1`).
- `is_simulated=True`, `supports_real_bytes_claim=False`, `value_bit_width=None`
  (mixed per-layer V precision).
- No true attention weights, Sparse V, TurboQuant+, or pre-RoPE logic.
- ExactKV gate: 2 prompts × 2 draft lengths × 2 max_new_tokens →
  `exactkv_failures == 0`.

**Phase A proxy divergence analysis (2026-06-09, `docs/EXPERIMENT_006A_DIVERGENCE_ANALYSIS.md`):**

- Experiments 003 (612), 004 (340), 005 (272) JSON reports analysed locally.
- No attention weights in reports; analysis labelled **proxy divergence** only.
- Key finding: aggressive compressors (`kvpress_knorm_restricted`, `int4_sim`,
  `k8_v2_sim`) show ~97% divergence rate with mean first-divergence index ~1.9–2.1;
  `k_full_v8` remains conservative (~6–9% divergence).
- Module: `exactkv/analysis/attention_weighted.py`; tests in
  `tests/test_analysis_attention_weighted.py`.

**Full path:**

- [x] Phase A attention-weighted divergence analysis completed and documented.
- [ ] At least one V7 experiment (Experiment 006 or justified sub-experiment)
      completes with `exactkv_failures == 0` on the core suite.
- [ ] All new compressors/policies populate V5 workspace fields honestly.
- [x] Simulated vs real vs analysis-only results are clearly separated in reports (Phase A).
- [ ] No forbidden performance fields anywhere in V7 code, tests, or docs.
- [ ] Full prior test suite remains green.
- [ ] V7 release notes published.

**Analysis-only fallback (acceptable V7 outcome):**

- [x] Phase A analysis report completed with actionable findings (§11 of 006A).
- [x] Written recommendation for Phase B: simulated layer-aware V policy scoped to
      early-divergence cluster; defer Sparse V / pre-RoPE / real adapters.
- [x] No ExactKV failure regression; no generation-logic change.
- [x] No forbidden performance fields in V7 Phase A docs/code.

---

## 16. No-performance-claim policy (unchanged from V1)

The following may **never** appear as data fields, table columns, or key-value
pairs in any ExactKV output — code, tests, CLI, or Markdown:

```
tokens_per_second
throughput
latency
speedup
runtime_seconds
```

They may appear **only in explicit negation prose or methodology caveats**, e.g.
"ExactKV does not measure tokens/second", "No latency claim is made", or the V8
methodology checklist describing what *would* be required before any such
measurement.

V7 evaluates policies by **acceptance behaviour and memory honesty only**, and
never presents an external system's speedup, throughput, latency, or perplexity
as an ExactKV result.

---

## 17. Relationship to V8

- **V8 — serving-stack context and optional GPU profiling.** After V7,
  ExactKV will have real-backend integration (V6), attention/V-specific
  characterisation (V7), and honest workspace accounting (V5). V8 may use
  vLLM/LMCache **only as an evaluation context** for caches ExactKV verifies —
  never as a performance-claim source. Active GPU memory profiling is deferred to
  V8 at the earliest.
- **V7 does not block V8** if V7 ends at analysis-only, but V8 assumes V5–V7
  correctness and acceptance framing are stable.
- **Eviction and serving integrations** that V7 lists as related work but does
  not implement are natural V8 scope items **only** with explicit protocol
  definitions for token existence vs exactness.

See [`docs/FUTURE_ROADMAP_V6_V8.md`](FUTURE_ROADMAP_V6_V8.md) and
[`docs/RESEARCH_BACKLOG.md`](RESEARCH_BACKLOG.md).

---

## V7 positioning summary

| Statement | True in V7 Phase 0? |
|---|---|
| V7 is a serving-stack benchmark | **No** |
| V7 claims speedup / throughput / latency / production readiness | **No** |
| TurboQuant+, KVQuant, KIVI, LMCache, vLLM, or PagedAttention integrated by ExactKV today | **No** (unless a later approved phase says otherwise) |
| `_sim` compressors are real packed-bit backends | **No** |
| External-paper numbers are ExactKV results | **No** |
| `exactkv_output_ids == full_output_ids` must hold | **Yes** (hard gate) |
| Analysis-first is the recommended entry | **Yes** |

---

## Attribution

The draft-then-verify compressed-KV algorithm is from:

> **VeriCache: Turning Lossy KV Cache into Lossless LLM Inference.**
> Yao et al., arXiv:2605.17613, 2026.

ExactKV does not claim to have invented this algorithm. V7 extends evaluation with
attention-aware and V-specific research questions — measured only by acceptance
behaviour and honest memory accounting, never by performance.

External systems cited (none implemented by ExactKV in Phase 0): TurboQuant+
(community); KV-AdaQuant (arXiv:2502.15075); KIVI (arXiv:2402.02750); KVQuant
(arXiv:2401.18079); TurboQuant (arXiv:2504.19874); PyramidKV; SnapKV; H2O;
StreamingLLM. See [`docs/RELATED_WORK_KV_CACHE_COMPRESSION.md`](RELATED_WORK_KV_CACHE_COMPRESSION.md).
