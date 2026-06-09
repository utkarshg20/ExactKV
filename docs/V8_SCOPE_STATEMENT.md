# V8 Scope Statement: Serving-Context Evaluation

**Status:** Phase D complete (Experiment 007 harness evaluation). Phase B complete.
Phase A complete. Phase 0 complete. See
[`docs/EXPERIMENT_007_SERVING_CONTEXT.md`](EXPERIMENT_007_SERVING_CONTEXT.md),
[`docs/SERVING_CACHE_LIFECYCLE_HARNESS.md`](SERVING_CACHE_LIFECYCLE_HARNESS.md), and
[`docs/SERVING_CONTEXT_FEASIBILITY.md`](SERVING_CONTEXT_FEASIBILITY.md).
Phase C remains **no-go/deferred** for vLLM/LMCache integration.
**Builds on:** `v0.7.0` — V7 complete; simulated layer-aware V policies
(`k8_v4_boundary*_v8_sim`); Experiments 006 and 006C (`exactkv_failures == 0`);
proxy divergence analysis (006A).
**Expands:** [`docs/FUTURE_ROADMAP_V6_V8.md`](FUTURE_ROADMAP_V6_V8.md) §V8 into an
approvable, phased scope.

> ExactKV does **not** implement vLLM, LMCache, PagedAttention, CUDA/Triton
> kernels, KIVI, KVQuant, TurboQuant, TurboQuant+, KVTC, Palu, or Sparse V in
> this document or in Phase 0.
> No throughput, latency, tokens/sec, speedup, `runtime_seconds`, or
> production-readiness claims as ExactKV results.
> External systems named below are **evaluation-context candidates and related
> work**, not current ExactKV capabilities.
> Simulated (`_sim`) compressors remain int8-container simulations; real packed-bit
> savings must not be implied.

---

## 1. V8 goal

Evaluate ExactKV in a **realistic serving/cache context**, focusing on:

- **Compatibility** — can ExactKV's verified compressed-KV workflow coexist with
  serving-style cache ownership and lifecycle?
- **Cache lifecycle** — prefill, append, block/page allocation, eviction, and
  external cache tiers without corrupting verification semantics.
- **Serving-style KV ownership** — who holds authoritative full KV vs compressed
  draft KV during draft-verify-commit?
- **Memory honesty** — V5 workspace fields remain accurate; optional active GPU
  profiling only under approved methodology.
- **Verification correctness** — `exactkv_output_ids == full_output_ids` and
  `exactkv_failures == 0` remain the hard gate.

V8 is a **serving-context evaluation layer**, not a production-serving claim.
Default V8 success is **compatibility and correctness**, not performance.

---

## 2. Why V8 matters after V7

After V1–V7, ExactKV has:

| Layer | What is established |
|---|---|
| V1–V3 | Draft-verify-commit loop; exactness gate; acceptance metrics |
| V4–V5 | Asymmetric K/V evaluation; workspace-aware memory accounting |
| V6 | `BackendAdapter`; restricted kvpress KnormPress; real pruned-cache bytes |
| V7 | Simulated layer-aware V policies; boundary-depth ablation; mixed-V reporting honesty |

All evaluation to date runs in ExactKV's **Hugging Face–centric local runtime** —
`ModelRuntime`, `DynamicCache` / layer tuples, CPU-first sweeps. Production LLM
systems use **serving stacks** (vLLM, LMCache, PagedAttention-backed allocators)
with different cache ownership, block mapping, and lifecycle rules.

V8 asks:

> **Can ExactKV's verification and acceptance evaluation remain correct when KV
> caches are produced or managed in a serving-style context — and what must be
> true for that to work?**

If integration is infeasible without breaking exactness, V8 must **document
incompatibility** and deliver a harness-based evaluation story — still a valid
V8 outcome.

---

## 3. What V8 should add

### 3.1 Serving-context feasibility research (Phase A)

A written feasibility study comparing:

- vLLM cache semantics (PagedAttention blocks, `KVCache` layout)
- LMCache tiering / disaggregated cache semantics
- PagedAttention as a **context** (block tables, physical vs logical sequence)
- ExactKV's current HF `FullKVState` / `CompressedKVState` model
- [`docs/BACKEND_ADAPTER_INTERFACE.md`](BACKEND_ADAPTER_INTERFACE.md) adapter boundary

**Output:** decision document — integrate, simulate lifecycle, or document
incompatibility.

### 3.2 Restricted serving-context harness (Phase B)

A **local cache-lifecycle simulator** or minimal harness that models serving
concerns without full vLLM/LMCache dependency if Phase A deems direct integration
too risky:

- Logical vs physical sequence length
- Block/page mapping to prompt positions
- Separate compressed vs authoritative full KV stores
- Cache append after commit without mutating verification state

Must preserve `exactkv_failures == 0` on a smoke/core subset before any optional
real-stack PoC.

### 3.3 Optional serving-stack proof-of-concept (Phase C)

**Only if Phase A proves feasibility** with an explicit per-stack approval:

- vLLM **or** LMCache as evaluation harness (not as ExactKV's runtime)
- ExactKV evaluates caches the stack produces; verification still uses full
  authoritative KV or an equivalent approved path
- No promise of full integration unless demonstrated

If infeasible: Phase C documents **why** and Phase B harness becomes the V8
evaluation path.

### 3.4 Experiment 007 (Phase D)

Serving-context compatibility / cache-lifecycle evaluation report. See §14.

### 3.5 Final release package (Phase E)

Release notes, README/ROADMAP cleanup, experiment index, project status update,
launch narrative draft, tag `v0.8.0` or `v1.0.0` depending on readiness. See §18.

### 3.6 Optional active GPU memory profiling

Deferred from V5; **optional in V8** only if methodology in §11 is approved.
Conservative `total_kv_footprint_bytes` accounting remains the default honest figure.

---

## 4. What V8 explicitly does not add

Unless a **new scope document** with separate explicit approval is written:

- ❌ **vLLM, LMCache, or PagedAttention integration** in Phase 0 or by default in V8
- ❌ **Throughput, latency, tokens/sec, speedup, `runtime_seconds`** as ExactKV results
- ❌ **Production-serving or production-readiness claims**
- ❌ **CUDA/Triton kernels, CPU offload, batching, sampling**
- ❌ **Parallel verification, bonus-token acceptance**
- ❌ **Broader kvpress** beyond V6 restricted KnormPress
- ❌ **KIVI, KVQuant, TurboQuant, TurboQuant+, KVTC, Palu** adapters
- ❌ **Sparse V dequantization, attention-gated materialization, fabricated attention weights**
- ❌ **Changes to generation logic, verification logic, or report schema** (except
  additive fields approved in a later phase)
- ❌ **New experiments beyond Experiment 007** without scope update
- ❌ **CLI behaviour changes** in Phase 0

---

## 5. Serving-context candidates

| Candidate | Role in V8 | Notes |
|---|---|---|
| **vLLM** | Optional evaluation harness (Phase C) | PagedAttention-backed KV; block tables; strong ecosystem; HF architecture fork |
| **LMCache** | Optional evaluation harness (Phase C) | Disaggregated / tiered KV cache; different ownership model |
| **PagedAttention context** | Feasibility study (Phase A) | Block/page mapping, physical vs logical seq; may be simulated without vLLM |
| **Restricted local serving harness** | Preferred Phase B path | Cache-lifecycle simulator; lowest integration risk |
| **Existing Hugging Face runtime** | **Baseline** | Current `ModelRuntime`; all V1–V7 experiments; control for compatibility studies |

**Positioning:** Candidates are **not implemented** in ExactKV today. Naming them
does not imply integration, performance parity, or reproduction of external-paper
results.

---

## 6. Recommended first V8 direction

**Start with a serving-context feasibility and compatibility study before any
integration code.**

Recommended sequence:

1. **Phase A** — Research document: vLLM vs LMCache vs PagedAttention context vs
   ExactKV's cache model; answer §7 questions for each candidate.
2. **Phase B** — Restricted local harness or cache-lifecycle simulator (always
   planned; not contingent on vLLM/LMCache success).
3. **Phase C** — Optional vLLM or LMCache PoC **only if** Phase A proves feasible
   and explicit approval is granted per stack.
4. **Phase D** — Experiment 007 report (compatibility + exactness + acceptance +
   workspace memory).
5. **Phase E** — Release notes, audit, public launch package.

**Why:** V6 kvpress integration showed that real external stacks can break hook
safety, cache semantics, and exactness unless isolated. Serving stacks add block
mapping, shared ownership, and GPU residency — higher risk than HF `DynamicCache`.
A harness-first path preserves the exactness gate while still answering serving
lifecycle questions.

---

## 7. Serving-context questions

Every candidate (including the Phase B harness) must answer:

| # | Question | Rejection criterion |
|---|---|---|
| 1 | **Who owns the KV cache?** | If ownership is ambiguous and verification cannot pin authoritative state, reject or document incompatible |
| 2 | **Can ExactKV access full authoritative KV?** | If verify path cannot use full-precision KV, **incompatible** |
| 3 | **Can ExactKV store compressed KV separately?** | Draft path must not overwrite authoritative full KV used for verify/commit |
| 4 | **Can verification run without corrupting the serving cache?** | Any mutation of serving-owned state during verify → reject or isolate (separate model/copy) |
| 5 | **Can logical vs physical sequence length be preserved?** | Required for acceptance bookkeeping and V6 kvpress-style alignment |
| 6 | **Can cache blocks/pages be mapped back to prompt positions?** | Required for divergence analysis and honest `logical_seq_len` |
| 7 | **Can workspace-memory accounting remain honest?** | All five V5 fields populated; `supports_real_bytes_claim` / `is_simulated` correct |

If **full authoritative KV cannot be accessed safely**, the integration must be
**rejected or documented as incompatible** — not worked around by weakening
verification.

---

## 8. Compatibility requirements

1. **`exactkv_output_ids == full_output_ids`** under greedy decoding — hard gate.
2. **`exactkv_failures == 0`** on every Experiment 007 cell.
3. **Verification engine unchanged** — full-precision verify path; serving context
   affects draft/materialize only (same principle as `BackendAdapter`).
4. **`KVCompressor` protocol preserved** — serving integration via adapter/harness
   boundary, not by forking `ExactKVGenerator` verification semantics.
5. **Hook safety** — verification model hooks remain zero (V6 kvpress lesson).
6. **Deterministic materialize** — draft cache materialization must be reproducible.
7. **Report compatibility** — additive fields only; old reports remain valid.
8. **Simulated vs real labelling** — `_sim` and layer-aware compressors keep
   `is_simulated=True`; serving-stack byte counts are not conflated with simulated
   int8-container figures.
9. **Optional stack isolation** — vLLM/LMCache PoC may require separate venv/pin
   (as with `.venv-kvpress`); must not pollute default install.

---

## 9. Memory-honesty requirements

All V8 experiment cells must report (unchanged from V1–V7):

| Field | Requirement |
|---|---|
| `stored_kv_bytes` | Persistent cache representation bytes |
| `materialized_working_kv_bytes` | Working copy during attention |
| `metadata_bytes` | Scales, codebooks, block tables (if applicable) |
| `temporary_workspace_bytes` | Transient scratch |
| `total_kv_footprint_bytes` | Conservative accounting sum — **not** measured peak GPU memory |
| `supports_real_bytes_claim` | Honest per compressor / harness |
| `is_simulated` | Honest per compressor |

**Serving-context additions (additive, if approved):**

- `cache_owner` — e.g. `exactkv_hf`, `harness_sim`, `vllm_poc` (label only)
- `logical_seq_len` / `physical_seq_len` — when block mapping is evaluated

**Rules:**

- Do not rank simulated int8-container compressors against real serving-stack
  bytes without matching `supports_real_bytes_claim` semantics.
- Do not cite vLLM/LMCache external memory or throughput numbers as ExactKV results.

---

## 10. Active GPU memory profiling requirements (if included)

Active GPU measurement is **optional** in V8. If pursued:

| Requirement | Detail |
|---|---|
| Approval | Separate sub-phase approval; not part of Phase 0 |
| Hardware | Documented GPU model, driver, CUDA version |
| Isolation | KV memory separated from activation memory where possible |
| Method | e.g. `torch.cuda.memory_reserved` at defined checkpoint; methodology in repo |
| Labelling | Named field (e.g. `active_gpu_kv_bytes`); distinct from `total_kv_footprint_bytes` |
| Caveats | Single-device, single-sequence default; not production workload |
| Fallback | If profiling omitted, V5 accounting sum remains the honest figure |

**Active GPU memory is not reported in Phase 0 and is not required for V8 exit.**

---

## 11. Performance-claim methodology requirements (if ever included)

**Phase 0 does not add performance metrics.** If a future approved sub-phase ever
reports throughput, latency, or speedup, **all** of the following must hold before
any number appears in an ExactKV report:

1. **Fixed hardware** — documented GPU/CPU, driver, software versions pinned
2. **Fixed model** — same checkpoint, dtype, device policy
3. **Fixed prompt suite** — named suite ID and cell count
4. **Fixed `max_new_tokens`** and draft parameters
5. **Warmup policy** — documented warm-up runs discarded
6. **Repeated trials** — multiple independent runs; variance reported
7. **Baseline comparison** — defined baseline on same hardware (not external-paper numbers)
8. **Cache state definition** — what KV state exists at measurement start
9. **Measurement code checked into repo** — reproducible instrumentation
10. **Raw results stored separately** — gitignored artifacts; methodology in docs
11. **Methodology caveats** — single-sequence default, no batching claim, etc.
12. **Explicit disclaimer** — not production serving under real scheduling/load
13. **Exactness gate** — `exactkv_failures == 0` on the same experimental cells
14. **No external-paper comparison as ExactKV results** — attribution separated

**Default V8 success does not require any performance measurement.**

---

## 12. Tests and gates

### Per-phase gates

| Phase | Gate |
|---|---|
| **A** | Feasibility document reviewed; §7 questions answered per candidate; go/no-go for Phase C stacks |
| **B** | Harness smoke: 2+ prompts, 2 draft lengths, `exactkv_failures == 0`; logical/physical seq invariant tests |
| **C** | If pursued: isolated env; verify hooks zero; core-suite exactness on approved compressor subset |
| **D** | Experiment 007: `exactkv_failures == 0`; full metric set; no forbidden fields |
| **E** | Release audit; launch narrative reviewed against §17 |

### Hard global gates (unchanged)

- `exactkv_output_ids == full_output_ids`
- Acceptance bookkeeping reconciles (`drafted == accepted + rejected`)
- `validate_report` passes on Experiment 007 JSON
- `_assert_no_forbidden_fields` — no `tokens_per_second`, `throughput`, `latency`,
  `speedup`, `runtime_seconds` as data fields

### Test scope

- Phase B/C code changes: targeted harness tests + existing exactness gates
- Full pytest when shared runtime, reporting, or registry behaviour changes
- Docs-only phases: `git diff --check` + prose audit only

---

## 13. GPU requirements

| Phase / activity | GPU |
|---|---|
| Phase 0 (this document) | None |
| Phase A feasibility research | None (literature + API review) |
| Phase B harness (default) | Optional; CPU sufficient for exactness smoke |
| Phase C vLLM/LMCache PoC | **Likely required** for realistic serving-context evaluation |
| Phase D Experiment 007 | CPU path acceptable if harness-based; GPU if stack PoC |
| Active GPU profiling (optional) | **Required** if that sub-phase is approved |

CI default remains CPU-first; GPU sweeps are manual/local with documented hardware.

---

## 14. Experiment 007 plan

**Name:** Serving-Context Compatibility / Cache-Lifecycle Evaluation

**Goal:** Characterise whether ExactKV's verified compressed-KV workflow remains
correct and measurable in a serving context (real stack PoC or Phase B harness).

### Illustrative configuration (finalised at implementation time)

| Parameter | Illustrative value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` (baseline; others optional) |
| Suite | `core` (34 prompts) minimum |
| Compressors | Subset spanning lossless, real INT8, simulated, layer-aware; harness/stack-labelled |
| `draft_len` | Small set (e.g. 4) |
| `max_new_tokens` | Small but meaningful (e.g. 16) |
| Context | `exactkv_hf` baseline + harness and/or optional stack PoC row |

### Two valid Experiment 007 modes

**Mode A — Serving-stack PoC (if Phase C approved):**

- Evaluate caches produced/managed by approved vLLM or LMCache harness
- Report compatibility findings, exactness, acceptance, workspace memory
- Document cache ownership and any isolation requirements

**Mode B — Harness-only (if stack integration infeasible):**

- Run core suite through Phase B cache-lifecycle harness
- Report lifecycle invariants, exactness, acceptance, workspace memory
- Document why stack integration was not pursued

Either mode must produce:

- `exactkv_failures == 0`
- Per-compressor acceptance, divergence, rejection, correction
- V5 workspace-memory table with honest simulated/real labelling
- Explicit **experiment class** label: `harness_sim`, `vllm_poc`, or `lmcache_poc`

**Artifacts:** `reports/experiment_007_serving_context.{json,csv}` — **gitignored**.

**Report:** `docs/EXPERIMENT_007_SERVING_CONTEXT.md`

---

## 15. Risks and unknowns

| Risk | Severity | Mitigation |
|---|---|---|
| vLLM/LMCache APIs incompatible with ExactKV exactness model | **High** | Phase A go/no-go; default to harness Mode B |
| Serving cache mutates during verify | **High** | Isolate compression model / separate authoritative copy (V6 lesson) |
| Block/page mapping breaks `logical_seq_len` | **High** | Explicit invariant tests in Phase B |
| Scope drifts to throughput engineering | **High** | §11 methodology gate; §17 policy |
| GPU hardware unavailable | Medium | CPU harness path; optional profiling skipped |
| transformers / vLLM version fork | Medium | Pin versions; document in feasibility report |
| Community expects production-serving story | Medium | Launch narrative review; correctness-first framing |
| Simulated compressors misread as serving-stack savings | Medium | `is_simulated` + report labelling; no cross-class memory ranking |
| Phase E tag version (`v0.8.0` vs `v1.0.0`) unclear until Experiment 007 | Low | Decide at Phase E based on launch readiness |

---

## 16. Exit criteria

V8 is complete when:

- [ ] Phase A feasibility document exists and is reviewed
- [x] Phase B harness passes exactness smoke gate (or Phase C PoC passes equivalent gate)
- [x] Experiment 007 completes with **`exactkv_failures == 0`**
- [x] `docs/EXPERIMENT_007_SERVING_CONTEXT.md` written with required disclaimers
- [ ] Final documentation package drafted (release notes, experiment index, project status)
- [ ] Launch narrative draft reviewed against no-performance-claim policy
- [ ] No forbidden performance fields in code, tests, or docs
- [ ] `docs/RELEASE_NOTES_V0.8.0.md` or `docs/RELEASE_NOTES_V1.0.0.md` written
- [ ] Git tag assigned (`v0.8.0` or `v1.0.0`)

**Valid partial success:** Harness-based Experiment 007 with documented
incompatibility of vLLM/LMCache — still satisfies V8 if exactness and honesty
requirements are met.

---

## 17. No-performance-claim policy

V8 inherits the global ExactKV policy:

**Forbidden as data fields, table columns, or metric keys:**

```
tokens_per_second
throughput
latency
speedup
runtime_seconds
```

These terms may appear **only** in explicit negation prose or in §11 future
methodology requirements.

**V8 additionally states:**

- ExactKV does **not** claim speedup, throughput, latency, runtime, or production
  readiness as V8 deliverables by default.
- vLLM/LMCache/PagedAttention external benchmarks are **not** ExactKV results.
- `total_kv_footprint_bytes` is a conservative accounting sum, not measured peak
  GPU memory unless §10 profiling is separately approved and executed.
- Active GPU memory is **not** reported unless §10 requirements are met.

---

## 18. Relationship to final public launch

V8 is the **last planned research version** before a consolidated public launch
package. Phase E may produce:

| Deliverable | Purpose |
|---|---|
| `docs/RELEASE_NOTES_V0.8.0.md` or `docs/RELEASE_NOTES_V1.0.0.md` | Full changelog V1–V8 |
| `docs/EXPERIMENT_INDEX.md` | One-line summary per experiment 001–007 |
| `docs/PROJECT_STATUS_V1.0.md` | Public project status |
| `docs/LAUNCH_NARRATIVE_DRAFT.md` | Private launch post draft (reviewed for honesty) |
| Updated `README.md` | Public-facing overview |

**Default public story (even without serving-stack integration):**

> ExactKV is a correctness-first verification and evaluation framework for lossy
> KV-cache compression. It proves exact greedy outputs under verification,
> measures draft acceptance and correction behaviour, and reports honest
> workspace-memory accounting — including real backends and simulated policies.
> It is not a production serving system and does not claim throughput or latency
> advantages.

Serving-context evaluation **strengthens** the story if feasible; it is **not
required** for an honest launch if incompatibility is documented.

**Tag decision:** `v0.8.0` if V8 delivers serving-context evaluation without full
launch package; `v1.0.0` if Phase E completes the full public launch criteria in §16.

---

## 19. Proposed V8 phases

| Phase | Scope | Deliverable | Status |
|---|---|---|---|
| **Phase 0** (this document) | Scope statement only; no code | `docs/V8_SCOPE_STATEMENT.md` | ✅ Complete |
| **Phase A** | Serving-stack feasibility research | [`docs/SERVING_CONTEXT_FEASIBILITY.md`](SERVING_CONTEXT_FEASIBILITY.md) | ✅ Complete |
| **Phase B** | Restricted serving-context / cache-lifecycle harness (**primary path**) | [`SERVING_CACHE_LIFECYCLE_HARNESS.md`](SERVING_CACHE_LIFECYCLE_HARNESS.md) + `exactkv/serving/` | ✅ Complete |
| **Phase C** | Optional vLLM or LMCache PoC | **Deferred (no-go)** per Phase A; re-approval only for metadata-only export study | ❌ No-go default |
| **Phase D** | Experiment 007 + report (Mode B harness) | [`EXPERIMENT_007_SERVING_CONTEXT.md`](EXPERIMENT_007_SERVING_CONTEXT.md); gitignored JSON/CSV | ✅ Complete |
| **Phase E** | Release notes, audit, launch package, tag | `RELEASE_NOTES_V0.8.0.md` or `V1.0.0.md` | Pending D |

> Phases A–E require **separate explicit approval** before any code is written.
> Phase 0 (this document) is design-only and introduces no code and no behaviour
> change.

---

## 20. Related documents

| Document | Relevance |
|---|---|
| [`FUTURE_ROADMAP_V6_V8.md`](FUTURE_ROADMAP_V6_V8.md) | Planning source for §V8 |
| [`RELEASE_NOTES_V0.7.0.md`](RELEASE_NOTES_V0.7.0.md) | V7 baseline |
| [`BACKEND_ADAPTER_INTERFACE.md`](BACKEND_ADAPTER_INTERFACE.md) | Adapter boundary for stack integration |
| [`EXPERIMENT_005_KVPRESS_KNORM.md`](EXPERIMENT_005_KVPRESS_KNORM.md) | External stack isolation precedent |
| [`SERVING_CACHE_LIFECYCLE_HARNESS.md`](SERVING_CACHE_LIFECYCLE_HARNESS.md) | Phase B harness design and API |
| [`EXPERIMENT_007_SERVING_CONTEXT.md`](EXPERIMENT_007_SERVING_CONTEXT.md) | Phase D harness evaluation report |
| [`EXPERIMENT_004_WORKSPACE_MEMORY.md`](EXPERIMENT_004_WORKSPACE_MEMORY.md) | V5 accounting reference |
| [`RESEARCH_BACKLOG.md`](RESEARCH_BACKLOG.md) | B10 serving-stack backlog item |
| [`RELATED_WORK_KV_CACHE_COMPRESSION.md`](RELATED_WORK_KV_CACHE_COMPRESSION.md) | vLLM, LMCache survey |

---

## Attribution

**VeriCache** (draft-then-verify algorithm): Yao et al., arXiv:2605.17613, 2026.

**Serving-stack references** (evaluation context only; not implemented):

- vLLM / PagedAttention: Kwon et al., SOSP 2023, arXiv:2309.06180
- LMCache: Liu et al., 2025, arXiv:2510.09665

ExactKV does not reproduce or claim external-system performance or accuracy results.
