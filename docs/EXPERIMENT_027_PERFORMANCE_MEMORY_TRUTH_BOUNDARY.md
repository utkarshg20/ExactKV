# Experiment 027: Performance / Memory Truth Boundary Review

**Status:** V12 Phase 7 — claim-boundary review only. **Not a benchmark.**
**Generated:** Manual synthesis from V1–V12 evidence (Experiments 001–026, scope docs, runtime code).

> This is a **performance/memory truth boundary review**, not a performance benchmark.
> ExactKV does **not** claim speedup, throughput, latency, runtime improvement,
> tokens/sec improvement, active GPU memory savings, production serving, or model
> accuracy improvement.
> Forbidden metrics may appear only in explicit negation or future-methodology guardrails.
> External paper results are **not** ExactKV results.
> Simulated compressors (`_sim`) are **not** real packed-bit backends.

---

## 1. Purpose

Define the honest **performance and active GPU memory claim boundary** after V1–V12,
and determine what practical-systems work remains before ExactKV may make speed or
VRAM-savings headlines. Experiment 027 does **not** fake missing pieces — it separates
what is proven, what is forbidden, and what must be built next.

Companion table: [`PRACTICALITY_GAP_ANALYSIS.md`](PRACTICALITY_GAP_ANALYSIS.md).

---

## 2. Why Experiment 027 is needed

External feedback after V11 identified gaps that block a practical-systems narrative:

- performance proof
- active GPU memory proof
- parallel verification
- a hot adapter (SnapKV / Shard-style)
- Llama-3.1-8B validation
- visual plots
- killer demo
- serving-ish demo
- one headline number

V12 Phases 1–6 closed backend feasibility, full-suite repair policies, and attention
logging — but **did not** close speed or device-memory practicality. Phase 7 records
that boundary explicitly so Phase 8 (release package) does not overclaim.

---

## 3. Evidence reviewed

| Source | Relevance |
|---|---|
| [`V12_SCOPE_STATEMENT.md`](V12_SCOPE_STATEMENT.md) | V12 goals, forbidden claims, Exp 027 plan |
| [`EXPERIMENT_018_GPU_MEMORY_PILOT.md`](EXPERIMENT_018_GPU_MEMORY_PILOT.md) | GPU peak dominated by weights; V5 ≠ device peak |
| [`GPU_MEMORY_METHODOLOGY.md`](GPU_MEMORY_METHODOLOGY.md) | Pilot protocol; `active_gpu_kv_bytes` not in schema |
| [`EXPERIMENT_017_SERVING_SIDECAR_PROBE.md`](EXPERIMENT_017_SERVING_SIDECAR_PROBE.md) | Sidecar pass; vLLM/LMCache no-go |
| [`EXPERIMENT_020_REPAIR_POLICY_PILOT.md`](EXPERIMENT_020_REPAIR_POLICY_PILOT.md) | Pilot acceptance gains; not speed |
| [`EXPERIMENT_025_FULL_SUITE_REPAIR_POLICY.md`](EXPERIMENT_025_FULL_SUITE_REPAIR_POLICY.md) | Full-suite policy validation; `int8_all` ceiling |
| [`EXPERIMENT_026_ATTENTION_LOGGING_FEASIBILITY.md`](EXPERIMENT_026_ATTENTION_LOGGING_FEASIBILITY.md) | Diagnostic only; no verification change |
| [`V11_LAUNCH_READINESS.md`](V11_LAUNCH_READINESS.md) | v0.11.0 ready; v1.0.0 not |
| [`PROJECT_STATUS_V0.11.0.md`](PROJECT_STATUS_V0.11.0.md) | Strongest findings through Exp 020 |
| `exactkv/runtime/exactkv_generator.py` | Sequential verify; recompress after commit |
| `exactkv/verification/engine.py` | `verify_sequential` cost model |
| `exactkv/metrics/gpu_memory_pilot.py` | Pilot-only GPU fields; forbidden-field guard |

Diagnostic inspect (no benchmark): `scripts/research/performance_memory_boundary_inspect.py`.

---

## 4. Current exactness claim boundary

**Allowed and proven:**

- `exactkv_token_match == True` when `exactkv_failure == False` on published cells.
- `exactkv_failures == 0` across Experiments 001–026 published sweeps (thousands of cells).
- Output matches `generate_full_greedy` under the ExactKV draft-verify-commit loop.
- Gate holds through **Qwen2.5-3B** on full 128-prompt V10 suites (Exp 015–016).

**Boundary:**

- Exactness is **greedy, single-request, HF-centric** — not sampling, not batched serving.
- Restricted factory adapters preserve exactness on their panels; not all compressors are production backends.
- Repair policies (Exp 020/025) change **draft compressor selection only** — verification unchanged.

---

## 5. Current acceptance / draft-usefulness claim boundary

**Allowed:**

- Mean **accept rate** (fraction of draft tokens accepted before correction).
- Rejection, correction, and lossy-divergence counts.
- Per-category and per-compressor leaderboards on V10 suites (not universal benchmarks).
- Repair-policy comparisons: e.g. Exp 025 `int8_all` **0.957** vs `baseline_boundary4` **0.923** on 0.5B full suite; Exp 020 pilot gains **shrank** at full-suite scale.

**Forbidden:**

- Model accuracy improvement from compression.
- Implication that higher accept rate means better downstream task quality.

**Ceiling:** `int8_all` remains the global acceptance ceiling on full V10 suite (Exp 025); category-adaptive policies help hard categories but do not beat int8 globally.

---

## 6. Current speed claim boundary

### Does ExactKV currently measure tokens/sec?

**No.** Standard JSON/CSV reports and `validate_report` schema contain no `tokens_per_second`, `throughput`, `latency`, `speedup`, or `runtime_seconds`.

### Does ExactKV currently prove lower runtime?

**No.** No controlled timing study compares ExactKV sequential vs full greedy vs lossy-only draft end-to-end.

### Does sequential verification likely add overhead?

**Yes — by design in V1.** From `exactkv/verification/engine.py`:

- Verification is **one position at a time** against authoritative full KV.
- Cost: up to **`(draft_len − 1)` full-model forward passes** when all draft tokens match; fewer on early mismatch.
- Each round also: draft generation on compressed KV, **deep-copy** of cache for verify safety, **recompress** from full KV after commit (`exactkv_generator.py`).

ExactKV today is a **correctness-first evaluation framework**, not an optimized inference path. Sequential verify + recompress is expected to be **slower** than naive full-KV greedy generation until parallel span verification and reduced materialization exist.

### What would be needed for a future (methodology-gated) speed story

| Requirement | Status |
|---|---|
| **Parallel / span verification** | Not implemented (D21 deferred) |
| **Accepted tokens per verifier call** | Not reported |
| **Warmup-controlled benchmark** | Not built |
| **Baseline panel:** full greedy | Reference exists (`generate_full_greedy`) but not timed in reports |
| **Baseline panel:** lossy-only draft (no verify) | Not implemented as benchmark arm |
| **ExactKV sequential** | Current default — overhead expected |
| **ExactKV span-verify** | Future design only |

**Decision:** Speed claims remain **forbidden**. Exp 027 does **not** approve a caveated positive speed methodology — evidence is insufficient and sequential verify is not a practical speed story.

---

## 7. Current active GPU memory claim boundary

### Does ExactKV currently prove active GPU memory savings?

**No.**

### What ExactKV does measure (memory)

**V5 workspace accounting** (`total_kv_footprint_bytes` and related fields): conservative **tensor-shape sum** for KV-related storage — hardware-independent, **does not include model weights**, **not measured device peak**.

**Exp 018 pilot** (isolated artifact only): `torch.cuda.memory_allocated` / `max_memory_allocated` at lifecycle points. Key findings:

- Mean peak ~**1969 MiB** per cell (0.5B/1.5B, fp16, A5000) — **dominated by model weights**.
- Mean prefill→peak delta ~**12 MiB** — heuristic, not isolated KV.
- Compressor ordering on V5 footprint (**~1.5–1.9 MiB**) **does not** match GPU peak ordering.
- `active_gpu_kv_bytes` **not** added to standard schema (pilot-success with caveats).

### What would be needed for a future (methodology-gated) memory story

| Requirement | Status |
|---|---|
| Avoid full materialization of compressed KV where possible | Not done — V1 recompresses from full state |
| True packed attention path (KIVI CUDA/Triton, etc.) | Exp 024 `B_restricted_go` feasibility only; no Qwen roundtrip |
| Backend keeping **compressed KV active** on device during draft | Simulated int8 containers; not production packed path |
| Isolated methodology: weights vs KV vs temporaries vs allocator | Exp 018 protocol exists; **KV-only isolation not achieved** |

**Decision:** Active GPU memory **savings** claims remain **forbidden**. V5 `total_kv_footprint_bytes` remains the **stable honest memory story** (accounting, not VRAM proof). Pilot GPU fields stay outside standard schema.

---

## 8. Parallel verification gap

### Why one-token-at-a-time verification is not a practical speed story

Current loop (`ExactKVGenerator.generate`):

1. Draft `draft_len` tokens on compressed KV.
2. `VerificationEngine.verify_sequential` — up to `draft_len−1` extra full forwards.
3. Commit accepted prefix + optional correction; recompress compressed state from full KV.

Verifier work scales with **draft_len per round**, not with **accepted span length per forward**. A practical system amortizes verification across a span in **one** full-KV forward (or batched verify kernel).

### Future design (not implemented in V12)

```
Draft k tokens on compressed KV
  → Verify whole span in ONE full-KV forward pass (or fixed small number of passes)
  → Accept longest matching prefix
  → On mismatch: emit correction token, truncate span
  → Commit accepted span into authoritative FullKVState
  → Realign compressed KV
```

### Files likely affected (future V13+)

| File | Change |
|---|---|
| `exactkv/verification/engine.py` | Add `verify_span` (or equivalent); keep `verify_sequential` for regression |
| `exactkv/runtime/exactkv_generator.py` | Round loop calls span verify; acceptance bookkeeping |
| `exactkv/verification/acceptance.py` | Span-level accept/reject semantics |
| `tests/test_verification*.py`, generator tests | Span mismatch at positions 0…k−1; correction token; exactness |

**D21** in [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md): sampling / parallel verify / bonus tokens — deferred until explicit future version.

---

## 9. Compressed-active-KV memory gap

Today:

- Draft uses **materialized** compressed KV (often int8 containers for `_sim`).
- After every commit, compressor **recompresses from authoritative full KV** — correct but not memory-minimal.
- Exp 018 shows device peak is **weight-dominated** at 0.5B/1.5B on short generations — compressor footprint differences are **swamped** in pilot peaks.

Before any VRAM headline:

- Packed-bit or fused dequant attention must keep **compressed representation live** during draft.
- Measurement must subtract baseline model load and isolate KV growth (extend Exp 018 protocol with empty-KV baselines, longer contexts, 8B scale).
- `_sim` compressors must **not** be presented as proof of packed-bit VRAM savings.

---

## 10. Hot-adapter gap

**Current adapters (factory-only, restricted):**

| Adapter | Exp | Accept (indicative) | Legibility |
|---|---|---:|---|
| KVQuant simquant | 010, 023 | 0.79 (0.5B), 0.61 (1.5B hard panel) | Research niche |
| TurboQuant Python | 008 | 0.44 | Not production TurboQuant |
| KIVI offline | 009 | 0.01 | Not CUDA/Triton |
| kvpress KnormPress | 005 | varies | Narrow |

**Missing for public legibility:** SnapKV, Shard/ShardKV, or similarly cited KV compression methods — names reviewers and practitioners recognize from recent literature.

**Why legibility matters:** ExactKV’s technical story (exact greedy preservation + accept rate) is strong on Qwen `_sim` panels but reads as **internal evaluation** without a recognizable “hot” comparator. A restricted factory adapter with clear labeling (`supports_real_bytes_claim=False` until proven) would anchor external comparisons without implying production integration.

**V12 action:** Document gap only — **no new adapter in Phase 7.**

---

## 11. Llama-3.1-8B public-legibility gap

**Why Qwen results are valid but insufficient for public narrative:**

- Qwen2.5-0.5B / 1.5B / 3B full V10 suites prove **scale transfer within one model family** with `exactkv_failures == 0`.
- Blog posts, HF leaderboards, and serving benchmarks overwhelmingly cite **Llama-3.x** (especially 8B).
- Reviewers may dismiss Qwen-only evidence as non-representative even when methodologically sound.

**Proposed future validation (not run in Exp 027):**

| Parameter | Value |
|---|---|
| Model | `meta-llama/Llama-3.1-8B-Instruct` (or base) |
| Prompts | 10–20 stratified (long_context, retrieval_copy, tool_json) |
| Arms | full greedy; int8; best repair policy (e.g. category_adaptive); optional span-verify once built |
| Gate | `exactkv_failures == 0` |
| Claims | Exactness + accept only — **not** speed or VRAM |

---

## 12. Visual / demo gap

### Plot specifications (not generated in Exp 027)

| Plot | Data source | Purpose |
|---|---|---|
| Acceptance by compressor / policy | Exp 012–016, 025 JSON | Bar chart; error bars optional |
| First divergence position histogram | Exp 013, 019 | Where lossy draft first fails |
| Category leaderboard | Exp 012 per-suite tables | Heatmap or grouped bars |
| Rejection / correction flow | Exp 007/017 traces | Sankey or round timeline |
| K error vs V error | Exp 003, 019 autopsy | Asymmetric fragility |
| Exactness failures = 0 | All experiments | Single summary stat panel |

Generate from gitignored `reports/*.json` in **V13** — not launch graphics in Phase 7.

### Killer demo (scripted, not built)

1. **Prompt:** JSON/tool or code_structured V10 prompt.
2. **Lossy draft** (`k8_v4_sim` or boundary4) drifts from full greedy mid-span.
3. **ExactKV** detects mismatch at verify step.
4. **Verifier** emits correction token; commit preserves exact greedy output.
5. **Final output** byte-identical to `generate_full_greedy`.

Optional: sidecar metadata overlay (Exp 017) showing authoritative vs draft cache lengths — **observational only**.

### Serving-ish demo

Exp 017 sidecar probe passes on 32 cells — sufficient for **harness** demo, not production serving. Multi-request batching and paged KV remain out of scope (D11/D12 no-go).

---

## 13. Headline-number candidates

### Allowed (with careful wording)

| Candidate | Source | Wording guardrail |
|---|---|---|
| **`exactkv_failures == 0`** | All published exps | Count cells and experiments; not “never fails in production” |
| **Lossy divergence cells caught** | Exp 012–016 | ExactKV caught and corrected — not “prevented all errors in the wild” |
| **Rejection reduction (pilot)** | Exp 020 | Pilot panel only; gains **shrank** at full suite (Exp 025) |
| **Full-suite repair policy** | Exp 025 | e.g. `category_adaptive` **0.948** vs boundary4 **0.923** on 128 prompts — **draft usefulness**, not accuracy |
| **int8_all ceiling** | Exp 025 | **0.957** accept — not “best compressor for all tasks” |

### Forbidden as headlines

- speedup, throughput, latency, runtime improvement, tokens/sec
- active GPU memory savings, VRAM reduction %
- production readiness, “ships with vLLM”
- model accuracy improvement
- External paper numbers presented as ExactKV results

---

## 14. Claims allowed after V12 (Phases 0–7)

1. ExactKV is a **correctness-first** KV-compression evaluation framework with draft-verify-commit exact greedy output.
2. **`exactkv_failures == 0`** on published experiment cells through Exp 026.
3. **Acceptance, rejection, correction, divergence** metrics on V10 suites (128 prompts, 7 categories) — not universal benchmarks.
4. **V5 `total_kv_footprint_bytes`** as conservative workspace accounting — not device peak.
5. **Repair policies** (experiment-layer) improve draft usefulness on some categories; `int8_all` remains ceiling on full suite.
6. **Backend feasibility** documented: TurboQuant llama.cpp Mode B restricted go; KVQuant 1.5B; KIVI Triton pack restricted go; vLLM/LMCache direct integration **no-go**.
7. **Attention logging:** eager prefill-only **restricted_go** (Exp 026); diagnostic only.
8. **Sidecar probe** passes; not production serving.

---

## 15. Claims still forbidden

- Any **positive** speed, throughput, latency, speedup, runtime, or tokens/sec claim.
- **Active GPU memory savings** or VRAM reduction vs baseline.
- **Production serving** readiness or vLLM/LMCache integration.
- **Model accuracy improvement** from compression or policies.
- `_sim` compressors as **real packed-bit** storage or attention paths.
- TurboQuant llama.cpp / KIVI CUDA / KVQuant deployment CUDA as **fully integrated** ExactKV backends.
- External paper results as ExactKV experiment results.
- Universal benchmark coverage.

---

## 16. Recommended next phase: V13 Practicality Proof

V13 should prioritize **P0** items from [`PRACTICALITY_GAP_ANALYSIS.md`](PRACTICALITY_GAP_ANALYSIS.md):

1. Implement **span / parallel verification** (design in §8).
2. Build **warmup-controlled diagnostic benchmark** — compare full greedy, lossy-only, ExactKV sequential, ExactKV span — report **only** under explicit “diagnostic, not headline” policy until baselines stable.
3. Extend **GPU memory isolation** (Exp 018 lineage) — longer context, 8B when available; still no savings claim until compressed-active path exists.
4. Add **one hot adapter** + **Llama-3.1-8B small suite** for public legibility.
5. Produce **plots + killer demo** from existing JSON artifacts.

V12 Phase 8 (release package) should **explicitly defer public v1.0.0 launch** if P0 practicality items remain open.

---

## 17. What this proves

- ExactKV has a **documented, enforceable claim boundary** after V1–V12.
- Speed and active GPU memory **savings** headlines are **not supported** by existing evidence.
- Sequential verification and recompress semantics **explain** why ExactKV is not currently a speed story.
- Exp 018 methodology is **pilot-success** but **insufficient** for VRAM savings claims at tested scale.
- A concrete **V13 Practicality Proof** backlog exists with prioritized gaps.

---

## 18. What this does not prove

- That ExactKV **cannot** ever show speed or memory benefits — only that it **has not yet**.
- That Qwen results are invalid — they are **family-valid** but **public-legibility-limited**.
- That repair policies are production-ready — experiment-layer only.
- That sidecar probe implies serving readiness.
- Any timing or GPU number from this document — **no new measurements** were taken in Exp 027.

---

## 19. Launch recommendation

| Decision | Recommendation |
|---|---|
| **Public v1.0.0 launch** | **Defer** — P0 practicality gaps (performance proof, active GPU memory proof, parallel verify) remain open |
| **`v0.12.0` research tag** | **Eligible** after Phase 8 release package if `exactkv_failures == 0` on V12 published work |
| **Phase 8 scope** | Release assessment + updated narrative + explicit “not public launch” unless gates met |
| **Next major phase** | **V13 Practicality Proof** (not Phase 8 alone) |

Phase 8 should document: *ExactKV closes deferred-work and claim-boundary gates in V12; public launch waits for V13 practicality evidence.*

---

## VeriCache attribution

ExactKV’s draft-verify-commit loop is inspired by [VeriCache](https://arxiv.org/abs/2605.17613). Experiment 027 reviews claim boundaries only; it does not modify the verification algorithm.
