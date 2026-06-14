# V13 Scope Statement — Practicality Proof

**Status:** **Phase 10C complete** — parallel work integrated ([`PARALLEL_WORK_INTEGRATION_REPORT.md`](PARALLEL_WORK_INTEGRATION_REPORT.md)). **Public launch NOT approved.** Clean-clone validation and should-fix items remain.
**Builds on:** V12 Phases 0–7 complete (Experiments 021–027); V12 Phase 8 release package may proceed in parallel but **does not authorize public launch**.
**Not public launch.** v1.0.0 deferred until Phase 9B must-fix blockers resolved and launch decision explicitly approved.

> V13 is a **practicality proof** — not a public launch, not production serving
> integration by default, and not a license to overclaim.
> V13 preserves the exactness gate: `exactkv_failures == 0` on every published
> experiment that produces ExactKV outputs.
> V13 **builds and measures** the missing systems pieces identified in
> Experiment 027 before any speed, latency, throughput, active GPU memory savings,
> or production-serving headline is allowed.
> Until V13 Phase 9 explicitly approves a claim under documented methodology,
> **forbidden claims remain forbidden** (see §22).

**Phase 9C (launch validation) may proceed** — clean-clone repro, demo recording, should-fix items. **Do not** tag v1.0.0 or post public launch until [`PRELAUNCH_HARDENING_REPORT.md`](PRELAUNCH_HARDENING_REPORT.md) gates pass.

---

## 1. Status

| Phase | Focus | Status |
|---|---|---|
| **0** | Formal scope statement (this document) | **Complete** |
| **1** | Span/parallel verification design (Exp 028 design) | **Complete** — [`SPAN_VERIFICATION_DESIGN.md`](SPAN_VERIFICATION_DESIGN.md) |
| **2** | Span/parallel verification implementation (Exp 028–029) | **Complete** — [`EXPERIMENT_028_SPAN_VERIFICATION_SMOKE.md`](EXPERIMENT_028_SPAN_VERIFICATION_SMOKE.md), [`EXPERIMENT_029_SPAN_VERIFICATION_GRID.md`](EXPERIMENT_029_SPAN_VERIFICATION_GRID.md) |
| **3** | Diagnostic timing harness (Exp 030) | **Complete** — [`EXPERIMENT_030_DIAGNOSTIC_TIMING.md`](EXPERIMENT_030_DIAGNOSTIC_TIMING.md) |
| **3b** | Batched span GPU/fp16 parity (Exp 030b) | **Complete** — [`EXPERIMENT_030B_SPAN_PARITY_INVESTIGATION.md`](EXPERIMENT_030B_SPAN_PARITY_INVESTIGATION.md) |
| **4** | Active GPU memory isolation (Exp 031) | **Complete** — [`EXPERIMENT_031_GPU_MEMORY_ISOLATION.md`](EXPERIMENT_031_GPU_MEMORY_ISOLATION.md) |
| **5** | Hot adapter feasibility — SnapKV / Shard / SpectralQuant (Exp 032 + addendum) | **Complete** — [`EXPERIMENT_032_SNAPKV_SHARDKV_FEASIBILITY.md`](EXPERIMENT_032_SNAPKV_SHARDKV_FEASIBILITY.md), [`EXPERIMENT_032_ADDENDUM_SHARD_SPECTRALQUANT.md`](EXPERIMENT_032_ADDENDUM_SHARD_SPECTRALQUANT.md) |
| **5b** | SnapKV experimental adapter MVP | **Complete** — [`EXPERIMENT_032B_SNAPKV_EXPERIMENTAL_SMOKE.md`](EXPERIMENT_032B_SNAPKV_EXPERIMENTAL_SMOKE.md); factory-only; `exactkv_failures == 0` |
| **6** | Llama-3.1-8B small-suite validation (Exp 033) | **Complete** — [`EXPERIMENT_033_LLAMA31_8B_SMALL_SUITE.md`](EXPERIMENT_033_LLAMA31_8B_SMALL_SUITE.md); 48 cells, 0 failures; span ≡ sequential |
| **7** | Killer correction demo (Exp 034) | **Complete** — [`EXPERIMENT_034_KILLER_CORRECTION_DEMO.md`](EXPERIMENT_034_KILLER_CORRECTION_DEMO.md); `tj_002` × `int4_sim`; rejection/correction trace; exact match |
| **7b** | Live correction terminal demo | **Complete** — [`DEMO_EXACTKV_LIVE_CORRECTION.md`](DEMO_EXACTKV_LIVE_CORRECTION.md); recordable terminal replay of Exp 034 trace |
| **8** | Visual plot package (Exp 035) | **Complete** — [`EXPERIMENT_035_VISUAL_PLOTS_AND_LEADERBOARD.md`](EXPERIMENT_035_VISUAL_PLOTS_AND_LEADERBOARD.md); internal research figures |
| **8b** | Public visual polish (Exp 036) | **Complete** — [`PUBLIC_VISUAL_PACKAGE.md`](PUBLIC_VISUAL_PACKAGE.md); launch-ready `public_*.png` cards |
| **8c** | Cinematic crash-test video | **Complete** — [`EXACTKV_CRASH_TEST_VIDEO.md`](EXACTKV_CRASH_TEST_VIDEO.md); `exactkv_crash_test_demo.mp4` (120s) |
| **8d** | Leaderboard tiering cleanup | **Complete** — tiered [`leaderboard.md`](leaderboard.md); TurboQuant/KIVI/KVQuant/SnapKV/Shard/SpectralQuant |
| **8e** | Terminal-native crash-test demo (Exp 034b) | **Complete** — [`EXACTKV_TERMINAL_CRASH_TEST.md`](EXACTKV_TERMINAL_CRASH_TEST.md); `pharm_001` semantic trace; recordable live dashboard |
| **8f** | Terminal + HTML crash-test leaderboard | **Complete** — `scripts/exactkv_leaderboard.py`; [`leaderboard.md`](leaderboard.md) · [`leaderboard.html`](leaderboard.html) |
| **9A** | Launch readiness gap audit | **Complete** — [`LAUNCH_READINESS_GAP_AUDIT.md`](LAUNCH_READINESS_GAP_AUDIT.md); **launch NOT approved** |
| **9B** | Prelaunch hardening infrastructure | **Complete** — [`PRELAUNCH_HARDENING_REPORT.md`](PRELAUNCH_HARDENING_REPORT.md); smoke + audits |
| **9C** | Launch validation & should-fix | Planned |
| **9** | V13 completion / launch decision | **Deferred** — until 9C validation |
| **10A** | LongBench-style drift demo (Exp 037) | **Complete** — [`EXPERIMENT_037_LONGBENCH_STYLE_DRIFT_DEMO.md`](EXPERIMENT_037_LONGBENCH_STYLE_DRIFT_DEMO.md); **secondary** terminal demo; bounded CPU search |
| **10B** | Shard external-drafter probe (Exp 038) | **Complete (restricted_go)** — RunPod L40S pass; 4-prompt panel — [`EXPERIMENT_038_SHARD_EXTERNAL_DRAFTER_PROBE.md`](EXPERIMENT_038_SHARD_EXTERNAL_DRAFTER_PROBE.md) |
| **10B2–10B4** | Shard stress / ablation / combined (Exp 039–041) | **Complete** — bounded probe complete; max 56.25% draft divergence, 0 ExactKV failures |
| **10C** | Shard leaderboard integration | **Complete** — RESTRICTED BACKEND tier |
| **10D–10G** | SpectralQuant probe → restricted panel (Exp 042–045) | **Complete** — RESTRICTED BACKEND adapter row (Exp 045) |
| **10H** | External methods consolidation | **Complete** — [`PHASE_10_EXTERNAL_METHODS_SUMMARY.md`](PHASE_10_EXTERNAL_METHODS_SUMMARY.md) |
| **10I** | Benchmark gap analysis | **Complete** — [`BENCHMARK_GAP_ANALYSIS.md`](BENCHMARK_GAP_ANALYSIS.md) |
| **9C** | Launch validation | **Complete** — [`LAUNCH_VALIDATION_REPORT.md`](LAUNCH_VALIDATION_REPORT.md) |
| **9D** | RC blocker fixes | **Complete** — verdict: `research-preview-rc-ready` (commit + optional MP4) |

**Latest tagged release:** `v0.11.0`. `v0.13.0-rc` research preview is a **future possibility only** — not approved. V12 substance (Exp 021–027) may ship as `v0.12.0` after Phase 8 without public v1.0.0.

---

## 2. Why V13 is needed

V12 closed many deferred research tracks (production-fidelity backend feasibility,
full-suite repair policies, attention logging feasibility, claim-boundary review).
Experiment 027 documented that ExactKV remains a **correctness-first evaluation
framework** — not yet a practical-systems prototype with evidence for:

| Gap | Exp 027 finding |
|---|---|
| Speed / latency / throughput proof | **Not measured**; sequential verify adds overhead |
| Active GPU memory savings proof | **Not proven**; Exp 018 peak dominated by weights |
| Parallel / span verification | **Not implemented** (D21 deferred) |
| Compressed-active-KV memory path | Full materialization + recompress still default |
| Hot public-legibility adapter | No SnapKV / ShardKV; factory-only niche adapters |
| Llama-3.1-8B validation | Qwen-only published suites |
| Visual plot package | Markdown tables only |
| Killer correction demo | Not scripted |
| Approved headline beyond exactness/acceptance | Speed/VRAM/production **forbidden** |

V13 turns ExactKV from “exactness-safe evaluation” into a **measured practicality
prototype** — or documents exact blockers — before public launch.

Companion gap table: [`PRACTICALITY_GAP_ANALYSIS.md`](PRACTICALITY_GAP_ANALYSIS.md).

---

## 3. What V12 proved (baseline for V13)

- **Exactness gate** through Qwen2.5-3B on full V10 suites; `exactkv_failures == 0` on Exp 021–026 published cells.
- **Repair policies** validated at full-suite scale (Exp 025); `int8_all` remains acceptance ceiling.
- **Attention logging:** eager prefill-only `restricted_go` (Exp 026); diagnostic only.
- **Claim boundary:** speed and active GPU memory savings **forbidden** (Exp 027).
- **Serving:** sidecar probe pass; direct vLLM/LMCache **no-go** (Exp 017).
- **GPU memory methodology pilot** documented (Exp 018); V5 `total_kv_footprint_bytes` remains stable accounting story.

---

## 4. What V12 did not prove

- That ExactKV is **faster** or **slower** than full greedy — no controlled timing.
- That ExactKV saves **active GPU memory** — no isolated KV-only device proof.
- That **span verification** reduces verifier overhead — not built.
- That ExactKV works on **Llama-3.1-8B** or other public benchmark models.
- That a **SnapKV/ShardKV-class** adapter preserves exactness.
- **Production serving readiness** — sidecar observational only.
- **Model accuracy improvement** from compression or policies.

---

## 5. V13 goal

Build and measure the missing systems pieces needed to determine whether ExactKV
is **practically useful**, not only exactness-safe — with reproducible evidence
or documented no-go at each step.

V13 must answer:

| Question | V13 phase |
|---|---|
| Does ExactKV slow things down today? | **Yes in Exp 030 diagnostic setup** (~2.67× vs full greedy, fp16 A5000); not a general speed claim |
| Can span verification reduce verifier overhead? | **Forward count yes (~4× fewer est. forwards); wall-clock no** — span ~10% slower than sequential post–030b (Exp 030) |
| Can ExactKV ever claim speed, latency, throughput, or tokens/sec? | 3, 9 — only if methodology + exactness pass |
| Can ExactKV show active GPU memory savings? | **No in Exp 031** — peak allocated indistinguishable from full greedy on 0.5B A5000; V5 accounting ~1.3 MiB vs ~1.2 GiB CUDA peak; savings claim **forbidden** |
| Can ExactKV run on Llama-3.1-8B? | 6 (Exp 033) |
| Can ExactKV show a killer correction demo on JSON/tool/code? | 7 (Exp 034) |
| Can ExactKV produce launch-quality visual plots? | 8 (Exp 035) |
| Is ExactKV production-serving ready? | 9 — expected **no** unless later phase explicitly scopes serving |

---

## 6. V13 non-goals

- **No public launch during Phase 0** or before Phase 9 gate review.
- **No production serving** unless a later V13 phase explicitly scopes it (default: **no**).
- **No direct vLLM, LMCache, PagedAttention, llama.cpp, MLX, TurboQuant production,
  KIVI production CUDA, KVQuant deployment CUDA, Sparse V production, KVTC, or Palu**
  integration unless scope explicitly changes.
- **No new default-registry compressors** unless a phase explicitly approves a restricted adapter.
- **No report schema changes** (JSON/CSV field additions require separate approval).
- **No positive speed, throughput, latency, runtime, tokens/sec claim** without Phase 3
  methodology and Phase 9 approval.
- **No active GPU memory savings claim** without Phase 4 robust isolation and Phase 9 approval.
- **No model accuracy improvement claim.**
- **No universal benchmark claim** — V10/V11/V12 suites remain internal evaluation panels.
- **No implication** that `_sim` compressors are real packed-bit backends.
- **No implication** that upstream paper results are ExactKV results.

---

## 7. Phase plan

### Phase 0 — Scope statement (this document)

Formal V13 scope, experiment plans 028–035, methodology requirements, policies, gate criteria.
**No code. No experiments.**

### Phase 1 — Span/parallel verification design

**Deliverable:** design document (part of Exp 028).

**Algorithm (target):**

1. Draft *k* tokens on compressed KV.
2. Verify span in **one** (or fixed small number of) full-KV forward pass(es).
3. Accept **longest matching prefix** of draft vs verifier predictions.
4. On first mismatch: emit **correction token**; truncate accepted span.
5. Commit accepted span (+ correction if any) into authoritative `FullKVState`.
6. Realign compressed KV from full state (same invariant as today).

**Affected files (future implementation):**

| File | Role |
|---|---|
| `exactkv/verification/engine.py` | Add span verify; keep `verify_sequential` as baseline |
| `exactkv/runtime/exactkv_generator.py` | Optional span path behind explicit flag or separate generator |
| `exactkv/verification/acceptance.py` | Span-level accept/reject semantics |
| Tests under `tests/` | All-match, first mismatch, middle mismatch, all reject, EOS, cache alignment |

**Invariants (unchanged):**

- `full_state.seq_len == compressed_state.logical_seq_len` every round.
- Authoritative full KV never corrupted by draft path.
- Rejected tokens never committed.
- `exactkv_output_ids == full_output_ids` when `exactkv_failure == False`.

### Phase 2 — Span/parallel verification implementation

- Implement behind **explicit flag** or **separate generator path** — sequential verifier remains default baseline.
- **Hard gate:** `exactkv_failures == 0` on smoke and grid (Exp 028–029).
- **No performance claims** in this phase — correctness only.

### Phase 3 — Diagnostic timing harness (Exp 030)

Compare on **fixed hardware**, **warmup**, **repeated trials**, **CUDA sync**:

| Arm | Description |
|---|---|
| Full greedy | `generate_full_greedy` baseline |
| Lossy-only | Draft on compressed KV without verify/commit (if implemented as benchmark arm) |
| ExactKV sequential | Current `ExactKVGenerator` + `verify_sequential` |
| ExactKV span | Span verification path from Phase 2 |

**Prerequisite:** Phase 2 exactness passes on same cells.

**Output:** Diagnostic timing artifacts (gitignored raw data); methodology doc.
**Claims:** No positive performance headline unless measured, variance reported, and Phase 9 approves wording.

### Phase 4 — Active GPU memory isolation (Exp 031)

Extend Exp 018 protocol to separate:

- Model-loaded baseline
- Prefill allocation
- Decode / ExactKV loop allocation
- V5 KV accounting vs device allocation
- Allocator effects (repeated trials where feasible)

Compare: full greedy, lossy-only, ExactKV sequential, ExactKV span.

**Claims:** No active GPU memory **savings** headline unless robust and attributable to KV compression — not weights or temporaries.

### Phase 5 — Hot adapter feasibility (Exp 032)

**Primary candidate:** SnapKV (unless research shows ShardKV/Shard is easier to integrate).

**Alternative:** Shard / ShardKV.

**Process:** Feasibility memo first → restricted factory adapter only if approved → **not** default registry.

**Gate:** `exactkv_failures == 0` on probe panel if cells run.

### Phase 6 — Llama-3.1-8B small-suite validation (Exp 033)

| Parameter | Planned value |
|---|---|
| Model | `meta-llama/Llama-3.1-8B-Instruct` or base |
| Prompts | 10–20 stratified (long_context, retrieval_copy, tool_json, code_structured) |
| Arms | Full greedy; `int8`; best repair policy (e.g. category_adaptive from Exp 025); optional span verify once Phase 2 complete |
| Environment | RunPod GPU, documented dtype/device |
| Claims | Exactness + accept only — **not** speed or VRAM |

### Phase 7 — Killer correction demo (Exp 034)

Reproducible script or notebook:

1. JSON / tool / code_structured V10-style prompt.
2. Lossy draft (`k8_v4_sim` or boundary4) drifts from full greedy.
3. ExactKV catches mismatch; verifier emits correction.
4. Final output **matches** `generate_full_greedy`.

**No accuracy improvement claim.** Optional sidecar metadata (Exp 017 lineage) — observational only.

### Phase 8 — Visual plot package (Exp 035)

Generate static **PNG/SVG** from gitignored JSON/CSV:

| Plot | Source experiments |
|---|---|
| Acceptance by compressor / policy | 012–016, 020, 025 |
| First divergence histogram | 013, 019 |
| Category leaderboard | 012 |
| Rejection / correction counts | 007, 017, 025 |
| K vs V error | 003, 019 |
| Exactness summary (`failures == 0`) | All published |

Output: `docs/assets/` or `reports/figures/` (gitignore raw regeneration inputs as policy dictates).

**No misleading axes or unsupported claims.**

### Phase 9 — V13 completion / launch decision

- `V13_READINESS_ASSESSMENT.md` (or equivalent)
- Decide: which claims (if any) move from **forbidden** to **methodology-gated allowed**
- Prepare `v0.13.0` release package
- **Public v1.0.0 only if claims are evidence-backed** — otherwise defer with honest blocker doc

---

## 8. Experiment plan (028–035)

| Exp | Phase | Focus | Success criteria |
|---|---|---|---|
| **028** | 1–2 | Span verification design + implementation smoke | Design doc; smoke cells `exactkv_failures == 0`; sequential baseline unchanged |
| **029** | 2 | Span verification exactness grid | Grid over prompts × compressors × mismatch cases; `exactkv_failures == 0` |
| **030** | 3 | Diagnostic timing harness | Four-arm comparison on fixed hardware; methodology doc; raw data gitignored; **no launch headline without Phase 9** |
| **031** | 4 | Active GPU memory isolation | Extended Exp 018 protocol; weight/KV/temp separation documented; savings claim only if robust |
| **032** | 5 | SnapKV / ShardKV feasibility | Feasibility memo + optional probe; factory-only; `exactkv_failures == 0` if cells run |
| **033** | 6 | Llama-3.1-8B small suite | 10–20 prompts; exactness gate; accept metrics only |
| **034** | 7 | Killer correction demo | Reproducible trace; exact match to full greedy |
| **035** | 8–9 | Visual plot package + headline audit | Figure bundle; approved headline candidates only from allowed set |

---

## 9. Required exactness gate

Every published V13 experiment that produces ExactKV outputs must satisfy:

| Requirement | Check |
|---|---|
| `exactkv_failures == 0` | Aggregate across published cells |
| `exactkv_output_ids == full_output_ids` | Per cell when not failure |
| Rejected tokens never committed | Trace / invariant tests |
| Cache alignment preserved | `full_state.seq_len == compressed_state.logical_seq_len` every round |

Span verification must pass the **same gate** as sequential verification before timing or memory arms use it.

---

## 10. Performance methodology requirements

Before reporting **any** speed, latency, throughput, tokens/sec, or runtime numbers as ExactKV results:

| Requirement | Detail |
|---|---|
| Fixed hardware | GPU model, driver, torch/CUDA version documented |
| Fixed model | Same checkpoint across arms |
| Fixed prompt suite | Named panel; cell count reported |
| Fixed `max_new_tokens` | Same generation budget |
| Warmup | Discarded runs before timed trials |
| Repeated trials | Multiple trials per cell; report variance or CI |
| CUDA synchronization | `torch.cuda.synchronize()` before/after timed regions |
| Baseline comparison | Same hardware session for all arms |
| Raw data | Gitignored; manifest in report |
| Methodology | Document in experiment report |
| Exactness gate | Passed on **same cells** before timing claims |

**Default:** Timing numbers are **diagnostic** until Phase 9 approves public wording.

---

## 11. Memory methodology requirements

Before reporting **active GPU memory savings**:

| Requirement | Detail |
|---|---|
| Model-loaded baseline | Record after load + sync |
| Prefill isolation | Post-prefill allocated minus baseline (heuristic) |
| Decode isolation | Peak during generation vs post-prefill |
| V5 vs device | Never conflate `total_kv_footprint_bytes` with `memory_allocated` |
| Allocator effects | Note caching; repeated trials if feasible |
| Attribution | Savings must be attributable to KV compression path, not weights/temporaries |
| Robustness | Cross-compressor ordering meaningful at target scale (e.g. 8B, longer context) |

**Default:** Active GPU memory savings remain **forbidden** until Exp 031 + Phase 9 approve.

Reference: [`GPU_MEMORY_METHODOLOGY.md`](GPU_MEMORY_METHODOLOGY.md), [`EXPERIMENT_018_GPU_MEMORY_PILOT.md`](EXPERIMENT_018_GPU_MEMORY_PILOT.md).

---

## 12. Sequential vs span verification (current baseline)

Today (`exactkv/verification/engine.py`):

- `verify_sequential` checks one draft token at a time.
- Cost: up to **`(draft_len − 1)`** full-model forwards when all match.
- Each round: draft deep-copy, verify deep-copy, recompress after commit.

Span verification (Phase 1–2 target) amortizes verifier work across the draft span.
Whether wall-clock improves depends on implementation and hardware — **measured in Phase 3**, not assumed.

---

## 13. Compressed-active-KV memory path

V13 Phase 4 investigates whether full materialization + recompress (V1 default) blocks
VRAM savings claims. Longer-term options (may remain deferred after V13):

- Packed-bit draft path (KIVI CUDA/Triton lineage — Exp 024 `B_restricted_go` only)
- Backend keeping compressed KV active during attention
- Reduced copy semantics in span verify path

Valid V13 outcome: **document blocker** if savings cannot be isolated.

---

## 14. Hot adapter policy

| Rule | Detail |
|---|---|
| SnapKV first | Preferred unless ShardKV feasibility is clearly easier |
| Factory-only | Restricted adapter; not default registry |
| Labeling | `supports_real_bytes_claim` per adapter honesty |
| Exactness | `exactkv_failures == 0` on probe panel |
| No paper results | External SnapKV/Shard numbers are **not** ExactKV results |

---

## 15. Llama-3.1-8B policy

Qwen2.5 results (0.5B–3B) remain valid **within family**. Llama-3.1-8B is the
**public-legibility** anchor for V13 — not a replacement for V10 Qwen suites.

Exp 033 is a **small suite** (10–20 prompts), not a full 128-prompt migration.

---

## 16. Demo and visualization policy

- **Killer demo (Phase 7):** Markdown trace — [`EXPERIMENT_034_KILLER_CORRECTION_DEMO.md`](EXPERIMENT_034_KILLER_CORRECTION_DEMO.md).
- **Live demo (Phase 7b):** Recordable terminal UI — `python3 scripts/demo_exactkv_live_correction.py`.
- **Plots (Phase 8):** Derived from existing experiment JSON; axes labeled; no speedup implied without Exp 030 data.
- **Headline audit (Phase 9):** Choose from allowed candidates (exactness, acceptance, divergence caught) or newly methodology-gated metrics — **not** fabricated numbers.

---

## 17. Serving readiness expectation

Exp 017: direct vLLM/LMCache **no-go**; sidecar probe **pass**.

V13 **default expectation:** ExactKV remains **not production-serving ready** at Phase 9 unless a future phase explicitly scopes multi-request serving — out of V13 Phase 0–8 plan.

Serving-ish observational demos (sidecar timeline) remain **P2** per [`PRACTICALITY_GAP_ANALYSIS.md`](PRACTICALITY_GAP_ANALYSIS.md).

---

## 18. Launch criteria after V13

Public v1.0.0 launch requires **all** of:

| Criterion | Required |
|---|---|
| V13 Phases 0–9 complete or honestly deferred with documented blockers |
| `exactkv_failures == 0` on all published V13 experiments |
| Span verification exactness grid passed (Exp 029) or documented no-go |
| Diagnostic timing (Exp 030) completed or deferred with blocker |
| GPU memory isolation (Exp 031) completed or deferral documented |
| Phase 9 claim decision documented — which headlines allowed vs forbidden |
| Llama-3.1-8B small suite (Exp 033) or documented blocker |
| Killer demo + plot package or explicit deferral |
| V12 Phase 8 release package (optional `v0.12.0`) does **not** substitute for V13 |

**`v0.13.0` tag** may ship after V13 substance even if public v1.0.0 remains deferred.

---

## 19. Stop/revise criteria

**Stop phase and document blocker** if:

- Span verification cannot preserve `exactkv_failures == 0` on grid.
- Timing harness variance too high for any comparison (document; do not headline).
- GPU memory isolation cannot separate KV from weights at target scale.
- SnapKV/ShardKV interop requires core generation or verification changes not approved.
- Llama-3.1-8B license or environment blocks reproducible runs.

**Revise scope** (requires explicit approval) if:

- Phase expands into production vLLM/LMCache integration.
- Report schema changes requested.
- Positive performance claim proposed without Exp 030 + Phase 9 approval.

---

## 20. Risks and unknowns

| Risk | Mitigation |
|---|---|
| Span verify harder than sequential for exactness | Keep sequential default; extensive grid (Exp 029) |
| Timing shows ExactKV slower than full greedy | Publish honestly; practicality ≠ speed claim |
| VRAM savings invisible at 0.5B scale | Target 8B + longer context in Exp 031/033 |
| SnapKV API mismatch | Feasibility-first (Exp 032); ShardKV fallback |
| Llama 8B RunPod cost | Small suite only |
| Plot generation misleading | Phase 9 headline audit |
| Scope creep into serving | D11/D12 remain no-go unless explicit approval |

---

## 21. Relationship to V12 Phase 8

V12 Phase 8 (release package, `v0.12.0`, launch narrative review) may proceed in parallel with V13 Phase 0–1.

**V12 Phase 8 must not imply public v1.0.0 launch** — Exp 027 deferred public launch until practicality evidence exists. V13 is that path.

---

## 22. No-performance-claim policy (V13)

V13 documents, experiment reports, and updated README/ROADMAP sections must **not**:

- Add or imply `tokens_per_second`, `throughput`, `latency`, `speedup`,
  `runtime_seconds`, or `active_gpu_kv_bytes` as ExactKV **results** until Phase 9
  approves under §10–§11 methodology — **default remains forbidden**.
- Claim production serving readiness or vLLM/LMCache **integration**.
- Present `_sim` compressors as real packed-bit backends.
- Cite external paper results as ExactKV experiment results.
- Claim model accuracy improvement from compression.
- Imply V10/V11/V12/V13 panels are **universal** public benchmarks.

Forbidden terms may appear **only** in explicit negation, future-methodology guardrails, or Phase 9 approved caveated wording.

---

## Related

- [`EXPERIMENT_027_PERFORMANCE_MEMORY_TRUTH_BOUNDARY.md`](EXPERIMENT_027_PERFORMANCE_MEMORY_TRUTH_BOUNDARY.md) — V12 claim boundary
- [`PRACTICALITY_GAP_ANALYSIS.md`](PRACTICALITY_GAP_ANALYSIS.md) — gap table
- [`V12_SCOPE_STATEMENT.md`](V12_SCOPE_STATEMENT.md) — prior gauntlet
- [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md) — D21 and V13 tracks
- [`ROADMAP.md`](ROADMAP.md) — version path
- `exactkv/runtime/exactkv_generator.py` — current generation loop
- `exactkv/verification/engine.py` — current sequential verify
- `exactkv/metrics/gpu_memory_pilot.py` — Exp 018 pilot helpers
