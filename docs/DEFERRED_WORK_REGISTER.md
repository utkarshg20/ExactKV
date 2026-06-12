# ExactKV Deferred Work Register

**Status:** Living register of major deferred items. **Nothing listed here is
implemented** unless a linked experiment or release note says otherwise.

> Guardrails: `exactkv_failures == 0` on every published experiment; no throughput,
> latency, speedup, runtime, or production-serving claims; `_sim` ≠ packed-bit storage;
> external paper results are **not** ExactKV results.

**Version path:** V9 ✅ → **V10 ✅ (`v0.10.0`)** → **V11 ✅ (`v0.11.0`)** → **V12 (Phases 0–7 ✅)** → **V13 (Phase 1 ✅)** → v1.0.0 (public launch).

**V9 scope:** [`V9_SCOPE_STATEMENT.md`](V9_SCOPE_STATEMENT.md) — **complete** (`v0.9.0`).
**V10 scope:** [`V10_SCOPE_STATEMENT.md`](V10_SCOPE_STATEMENT.md) — **complete** (`v0.10.0`).
**V10 readiness:** [`V10_READINESS_ASSESSMENT.md`](V10_READINESS_ASSESSMENT.md).
**V11 scope:** [`V11_SCOPE_STATEMENT.md`](V11_SCOPE_STATEMENT.md) — **complete** (`v0.11.0`).
**V11 readiness:** [`V11_LAUNCH_READINESS.md`](V11_LAUNCH_READINESS.md).
**V12 scope:** [`V12_SCOPE_STATEMENT.md`](V12_SCOPE_STATEMENT.md) — Phases 0–7 complete; Phase 8 release package planned.
**V13 scope:** [`V13_SCOPE_STATEMENT.md`](V13_SCOPE_STATEMENT.md) — Phase 1 complete.
**Experiment 014:** [`EXPERIMENT_014_REAL_BACKEND_SPOTCHECKS.md`](EXPERIMENT_014_REAL_BACKEND_SPOTCHECKS.md).
**V10 suites:** [`V10_PROMPT_SUITES.md`](V10_PROMPT_SUITES.md).
**Experiment 012:** [`EXPERIMENT_012_EVAL_SUITE_EXPANSION.md`](EXPERIMENT_012_EVAL_SUITE_EXPANSION.md).
**Experiment 013:** [`EXPERIMENT_013_SENSITIVITY_FORENSICS.md`](EXPERIMENT_013_SENSITIVITY_FORENSICS.md).

---

## V9 — Real backend integration gauntlet (complete)

| ID | Item | Status | V9 phase | Success criteria |
|---|---|---|---|---|
| D1 | **TurboQuant full integration** | **Evaluated (Phase C)** | ✅ | Exp 008: 272 cells, `exactkv_failures == 0`; accept **0.435** — [`EXPERIMENT_008_TURBOQUANT_PYTHON.md`](EXPERIMENT_008_TURBOQUANT_PYTHON.md) |
| D2 | **TurboQuant+ full integration** | **Evaluated (Python path only)** | ✅ | Production llama.cpp/MLX deferred — [`RELEASE_NOTES_V0.9.0.md`](RELEASE_NOTES_V0.9.0.md) |
| D3 | **KIVI adapter** | **Evaluated (Phase D3)** | ✅ | Exp 009: accept **0.012** — [`EXPERIMENT_009_KIVI_OFFLINE.md`](EXPERIMENT_009_KIVI_OFFLINE.md) |
| D4 | **KVQuant-style adapter** | **Evaluated (Phase D6)** | ✅ | Exp 010: accept **0.792** — [`EXPERIMENT_010_KVQUANT_SIM.md`](EXPERIMENT_010_KVQUANT_SIM.md) |
| D5 | **KVTC / Palu feasibility** | Deferred (optional) | — | Not in V9 scope |
| D15 | **Larger-model RunPod validation** | **Complete (Phase E)** | ✅ | Exp 011: 1.5B, 238 cells — [`EXPERIMENT_011_LARGER_MODEL_VALIDATION.md`](EXPERIMENT_011_LARGER_MODEL_VALIDATION.md) |

---

## V10 — Evaluation suite hardening and divergence forensics (complete)

| ID | Item | Status | V10 phase | Success criteria |
|---|---|---|---|---|
| D26 | **`core_v2` + category benchmark suites** | **Complete** | Phase 1 ✅ | 128 prompts; [`V10_PROMPT_SUITES.md`](V10_PROMPT_SUITES.md); validator + tests |
| D27 | **Draft length sensitivity (2/4/8)** | **Complete** | Phase 3 ✅ | Exp 013: 2160 cells, `exactkv_failures == 0` |
| D28 | **Generation length sensitivity (16/32/64)** | **Complete** | Phase 3 ✅ | Exp 013 full 3×3 grid |
| D29 | **Category-stratified divergence forensics** | **Complete** | Phase 3 ✅ | Token-type + structured-output heuristics; no attention weights |
| — | **Per-category leaderboards + prompt win/loss** | **Complete** | Phase 2 ✅ | Exp 012: 896 cells, `exactkv_failures == 0` |
| — | **Real-backend category spot-checks** | **Complete** | Phase 4 ✅ | Exp 014: 280 cells — [`EXPERIMENT_014_REAL_BACKEND_SPOTCHECKS.md`](EXPERIMENT_014_REAL_BACKEND_SPOTCHECKS.md) |
| — | **V10 readiness assessment** | **Complete** | Phase 5 ✅ | [`V10_READINESS_ASSESSMENT.md`](V10_READINESS_ASSESSMENT.md) |
| D6 | **Sparse V dequantization** | **Moved to V11 / research** | — | Acceptance-only evaluation |
| D7 | **True attention logging** | **Moved to V11** | — | Small subset; no fabricated weights |
| D8 | **Per-layer/head/token divergence forensics** | **Moved to V11** | — | Requires attention weights or documented blocker |
| D9 | **Pre-RoPE key quantization experiments** | Deferred | — | Compare vs post-RoPE baselines |
| D10 | **Boundary / layer-policy extensions** | Deferred | — | N>4 only with explicit approval |
| — | **1.5B on expanded V10 suites** | **Moved to V11** | — | Exp 011 is legacy `core` only |

**V10 exit:** tag **`v0.10.0`** — not v1.0.0. See [`RELEASE_NOTES_V0.10.0.md`](RELEASE_NOTES_V0.10.0.md).

---

## V11 — Final launch hardening (complete — `v0.11.0`)

| ID | Item | Status | V11 phase | Success criteria |
|---|---|---|---|---|
| — | **V11 scope statement** | **Complete** | Phase 0 | [`V11_SCOPE_STATEMENT.md`](V11_SCOPE_STATEMENT.md) |
| — | **1.5B on V10 expanded suites** | **Complete** | Phase 1 / Exp 015 | [`EXPERIMENT_015_QWEN15B_V10_SUITES.md`](EXPERIMENT_015_QWEN15B_V10_SUITES.md); `exactkv_failures == 0` |
| — | **Optional 3B built-in stretch** | **Complete** | Phase 2 / Exp 016 | [`EXPERIMENT_016_QWEN3B_V10_SUITES.md`](EXPERIMENT_016_QWEN3B_V10_SUITES.md); `exactkv_failures == 0` |
| — | **Optional 1.5B real-backend panel** | Deferred | — | Not run in Phase 2; may revisit post-Exp 017 |
| D13 | **vLLM / LMCache sidecar probe** | **Complete** | Phase 3 / Exp 017 | [`EXPERIMENT_017_SERVING_SIDECAR_PROBE.md`](EXPERIMENT_017_SERVING_SIDECAR_PROBE.md); sidecar pass; direct integration **no-go reaffirmed** |
| D14 | **Active GPU memory methodology** | **Complete** | Phase 4 / Exp 018 | [`GPU_MEMORY_METHODOLOGY.md`](GPU_MEMORY_METHODOLOGY.md); pilot **success** — not added to standard schema |
| D7 | **True attention logging** | **Deferred (partial)** | Phase 5 / Exp 019 | sdpa backend lacks `output_attentions`; no fabricated weights — [`EXPERIMENT_019_DIVERGENCE_AUTOPSY.md`](EXPERIMENT_019_DIVERGENCE_AUTOPSY.md) |
| D8 | **Per-layer/head divergence forensics** | **Partial complete** | Phase 5 / Exp 019 | Per-layer KV error in Exp 019; per-head deferred without attention weights |
| — | **Repair-policy pilot (Exp 019 hypotheses)** | **Complete** | Phase 5b / Exp 020 | [`EXPERIMENT_020_REPAIR_POLICY_PILOT.md`](EXPERIMENT_020_REPAIR_POLICY_PILOT.md); policies **not** in core ExactKV |
| D17 | **Raw report bundle** | **Policy complete** | Phase 6 | [`RAW_ARTIFACT_POLICY.md`](RAW_ARTIFACT_POLICY.md); physical zip optional until v1.0.0 |
| D18 | **Launch narrative** | **Draft complete** | Phase 6 | [`LAUNCH_NARRATIVE_DRAFT.md`](LAUNCH_NARRATIVE_DRAFT.md); not for public posting until v1.0.0 review |
| D6 | **Sparse V dequantization** | Deferred (out of V11 scope) | — | Not in V11 unless explicitly approved |
| D11 | **Direct vLLM integration** | No-go (Phase A) | — | Out of V11 scope |
| D12 | **LMCache integration** | No-go (Phase A) | — | Out of V11 scope |
| D16 | **PagedAttention kernel integration** | Deferred | — | Out of V11 scope |
| D1–D4 | **Production TurboQuant / KIVI CUDA / KVQuant CUDA** | Evaluated (V9) | — | **Not** in V11 unless factory-only re-panel in Exp 016 |

V11 covers multi-model validation, serving/profiling probes, and launch package prep — **not** a performance benchmark or production integration release.

---

## V12 — Deferred Work Completion Gauntlet (Phases 0–7 complete)

| ID | Item | Status | V12 phase | Success criteria |
|---|---|---|---|---|
| — | **V12 scope statement** | **Complete** | Phase 0 ✅ | [`V12_SCOPE_STATEMENT.md`](V12_SCOPE_STATEMENT.md) |
| D2 | **TurboQuant llama.cpp / GGUF / production-fidelity** | **External probe complete (Mode B)** | Phase 1–2 ✅ | [`EXPERIMENT_022_TURBOQUANT_LLAMACPP_PROBE.md`](EXPERIMENT_022_TURBOQUANT_LLAMACPP_PROBE.md); BackendAdapter **no-go**; Mode B **go with restrictions** |
| D4 | **KVQuant 1.5B/3B real-backend validation** | **1.5B complete** | Phase 3 ✅ / Exp 023 | [`EXPERIMENT_023_KVQUANT_LARGER_MODEL.md`](EXPERIMENT_023_KVQUANT_LARGER_MODEL.md); 1.5B accept **0.609**; 3B stretch not run |
| D3 | **KIVI CUDA/Triton packed path** | **Feasibility complete (`B_restricted_go`)** | Phase 4 ✅ / Exp 024 | [`EXPERIMENT_024_KIVI_CUDA_TRITON_FEASIBILITY.md`](EXPERIMENT_024_KIVI_CUDA_TRITON_FEASIBILITY.md); Triton pack OK; `dequant_cuda` missing; no Qwen model; BackendAdapter **restricted_future_only** |
| — | **Full-suite repair-policy validation** | **Complete** | Phase 5 ✅ / Exp 025 | [`EXPERIMENT_025_FULL_SUITE_REPAIR_POLICY.md`](EXPERIMENT_025_FULL_SUITE_REPAIR_POLICY.md); 768 cells 0.5B; `exactkv_failures == 0`; 1.5B optional not run |
| D7 | **True attention logging** | **Restricted go (eager prefill-only)** | Phase 6 ✅ / Exp 026 | [`EXPERIMENT_026_ATTENTION_LOGGING_FEASIBILITY.md`](EXPERIMENT_026_ATTENTION_LOGGING_FEASIBILITY.md); sdpa blocked; eager prefill OK on Qwen2.5-0.5B |
| D8 | **Per-head divergence forensics** | **Partial — prefill-only path** | Phase 6 ✅ / Exp 026 | Per-layer KV in Exp 019; per-head via eager prefill snapshots only; decode-step/default runtime blocked |
| — | **Performance/memory truth boundary** | **Complete** | Phase 7 ✅ / Exp 027 | [`EXPERIMENT_027_PERFORMANCE_MEMORY_TRUTH_BOUNDARY.md`](EXPERIMENT_027_PERFORMANCE_MEMORY_TRUTH_BOUNDARY.md); speed/VRAM savings **forbidden**; **V13 recommended** |
| D17 | **Physical raw report bundle** | Planned | Phase 8 | Optional until v1.0.0; policy in [`RAW_ARTIFACT_POLICY.md`](RAW_ARTIFACT_POLICY.md) |
| D18 | **Launch narrative (final)** | Planned | Phase 8 | Review/approve [`LAUNCH_NARRATIVE_DRAFT.md`](LAUNCH_NARRATIVE_DRAFT.md) |
| D11 | **Direct vLLM integration** | No-go (Phase A) | — | Out of V12/V13 unless scope changes |
| D12 | **LMCache integration** | No-go (Phase A) | — | Out of V12/V13 unless scope changes |
| D16 | **PagedAttention kernel integration** | Deferred | — | Out of V12/V13 scope unless explicitly approved |
| D6 | **Sparse V dequantization** | Deferred | — | Out of V12/V13 unless explicitly approved |
| D9/D10 | **Pre-RoPE / boundary N>4** | Deferred | — | Out of V12/V13 scope |

V12 closed deferred backend, policy, forensics, and claim-boundary tracks — **not** practicality proof.

---

## V13 — Practicality Proof (Phase 1 complete)

| ID / track | Item | Status | V13 phase | Success criteria |
|---|---|---|---|---|
| — | **V13 scope statement** | **Complete** | Phase 0 ✅ | [`V13_SCOPE_STATEMENT.md`](V13_SCOPE_STATEMENT.md) |
| D21 | **Parallel / span verification** | **Design complete** | Phases 1 ✅ / Exp 028 design | [`SPAN_VERIFICATION_DESIGN.md`](SPAN_VERIFICATION_DESIGN.md); implementation Phase 2 |
| — | **Performance proof (diagnostic timing)** | Planned | Phase 3 / Exp 030 | Four-arm harness; fixed hardware; warmup + variance; no headline without Phase 9 |
| — | **Active GPU memory isolation** | Planned | Phase 4 / Exp 031 | Extend Exp 018; weight/KV/temp separation; no savings claim unless robust |
| — | **Hot adapter (SnapKV / ShardKV)** | Planned | Phase 5 / Exp 032 | Feasibility first; factory-only; SnapKV preferred |
| — | **Llama-3.1-8B small suite** | Planned | Phase 6 / Exp 033 | 10–20 prompts; RunPod GPU; `exactkv_failures == 0` |
| — | **Killer correction demo** | Planned | Phase 7 / Exp 034 | Reproducible JSON/tool/code trace; exact match to full greedy |
| — | **Visual plot package** | Planned | Phase 8 / Exp 035 | PNG/SVG from reports; no misleading axes |
| — | **Headline number audit** | Planned | Phase 9 / Exp 035 | Allowed headlines only; speed/VRAM forbidden unless evidence-backed |
| — | **Compressed-active-KV memory path** | Investigate in Phase 4 | Phase 4 | Document blocker if full materialization prevents savings |
| D11/D12 | **Production serving / vLLM / LMCache** | No-go | — | Remain deferred unless later V13 phase explicitly scopes serving |
| D22 | **Multi-request batching** | Deferred | — | Serving-scale; out of V13 Phase 0–8 |

V13 builds and measures missing practicality pieces — **not** public launch by default.

---

## v1.0.0 — Public launch tag (after V13 exit)

| ID | Item | Status | Success criteria |
|---|---|---|---|
| D19 | **Project status v1.0.0** | Deferred | Supersedes [`PROJECT_STATUS_V0.11.0.md`](PROJECT_STATUS_V0.11.0.md); after V13 Phase 9 |
| D20 | **Git tag `v1.0.0`** | Deferred | [`V13_SCOPE_STATEMENT.md`](V13_SCOPE_STATEMENT.md) §18 gates met |

_D17 and D18 may finalize in V12 Phase 8; **public posting** waits for V13 Phase 9 claim decision._

---

## Cross-cutting (ongoing)

| ID | Item | Status | Notes |
|---|---|---|---|
| D21 | **Sampling / parallel verify / bonus tokens** | **Span design complete (V13 Phase 1)** | Phases 1 ✅ / Exp 028 design | Implementation Phase 2; sampling/bonus still deferred |
| D22 | **Multi-request batching** | Deferred | Serving-scale feature |
| D23 | **CPU offload / CUDA kernels** | Deferred | No custom kernels in ExactKV today |
| D24 | **Broader kvpress** | Deferred | KnormPress only (V6) |
| D25 | **RESEARCH_BACKLOG sync** | Ongoing | Keep aligned with this register |

---

## Phase C reminder (V8)

vLLM and LMCache **direct integration** were judged **no-go** in
[`SERVING_CONTEXT_FEASIBILITY.md`](SERVING_CONTEXT_FEASIBILITY.md). Items D11–D12
remain **deferred, not forgotten**.

---

## Related

- [`V13_SCOPE_STATEMENT.md`](V13_SCOPE_STATEMENT.md) — V13 formal scope (Phase 1 complete)
- [`SPAN_VERIFICATION_DESIGN.md`](SPAN_VERIFICATION_DESIGN.md) — span verification design (Phase 1)
- [`V12_SCOPE_STATEMENT.md`](V12_SCOPE_STATEMENT.md) — V12 formal scope (Phases 0–7 complete)
- [`V11_SCOPE_STATEMENT.md`](V11_SCOPE_STATEMENT.md) — V11 formal scope (complete)
- [`V11_LAUNCH_READINESS.md`](V11_LAUNCH_READINESS.md) — v0.11.0 gate decision
- [`V10_SCOPE_STATEMENT.md`](V10_SCOPE_STATEMENT.md) — V10 formal scope (complete)
- [`V10_READINESS_ASSESSMENT.md`](V10_READINESS_ASSESSMENT.md) — Phase 5 readiness
- [`RELEASE_NOTES_V0.10.0.md`](RELEASE_NOTES_V0.10.0.md) — v0.10.0 changelog
- [`V10_SCOPE_DRAFT.md`](V10_SCOPE_DRAFT.md) — superseded draft
- [`RELEASE_NOTES_V0.9.0.md`](RELEASE_NOTES_V0.9.0.md) — what shipped in v0.9.0
- [`ROADMAP.md`](ROADMAP.md) — version planning
- [`RESEARCH_BACKLOG.md`](RESEARCH_BACKLOG.md) — experiment ideas
