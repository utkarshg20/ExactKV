# ExactKV Version Lineage (V1–V21)

Generated: 2026-06-26T14:58:00.590506+00:00

The **version arc** (V1–V21) spans pre-formal-release research milestones. It is **distinct** from formal release phases A–K and from the authoritative **1500-cell** benchmark (`reports/scale_7b/raw.json`).

- **Verified (scope statements):** 13
- **Partial (phase/experiment docs):** 8
- **Source pending:** 0

## Version table

| Version | Status | Theme | Sample evidence |
|---------|--------|-------|-----------------|
| V1 | verified | Correctness prototype | `docs/V1_SCOPE_STATEMENT.md` |
| V2 | verified | Framework generalization | `docs/RELEASE_NOTES_V0.2.0.md` |
| V3 | verified | Benchmark seriously | `docs/RELEASE_NOTES_V0.3.0.md` |
| V4 | verified | Asymmetric K/V compression | `docs/RELEASE_NOTES_V0.4.0.md` |
| V5 | verified | Workspace-aware memory accounting | `docs/RELEASE_NOTES_V0.5.0.md` |
| V6 | verified | Backend adapter interface | `docs/RELEASE_NOTES_V0.6.0.md` |
| V7 | verified | Layer-aware V policies | `docs/RELEASE_NOTES_V0.7.0.md` |
| V8 | verified | Serving harness | `docs/RELEASE_NOTES_V0.8.0.md` |
| V9 | verified | Real backend gauntlet | `docs/RELEASE_NOTES_V0.9.0.md` |
| V10 | verified | Suite hardening | `docs/RELEASE_NOTES_V0.10.0.md` |
| V11 | verified | Launch hardening | `docs/RELEASE_NOTES_V0.11.0.md` |
| V12 | verified | Deferred work completion gauntlet | `docs/V12_SCOPE_STATEMENT.md` |
| V13 | verified | Practicality proof | `docs/V13_SCOPE_STATEMENT.md` |
| V14 | partial | CUDA restored verifier & GPU memory diagnostics | `docs/EXPERIMENT_055_EXPERIMENTAL_RESTORED_VERIFIER_CLI.md` |
| V15 | partial | vLLM feasibility probes | `docs/EXPERIMENT_059_VLLM_FEASIBILITY_PROBE.md` |
| V16 | partial | Shadow observers & streaming quant feasibility | `docs/EXPERIMENT_066_STREAMING_QUANT_ATTENTION_FEASIBILITY.md` |
| V17 | partial | Claim-safe demo & broader validation | `docs/EXPERIMENT_086_CLAIM_SAFE_DEMO_PACKAGING.md` |
| V18 | partial | Integration safety & L3 draft-shadow no-commit | `docs/EXPERIMENT_090_INTEGRATION_SAFETY_SPEC.md` |
| V19 | partial | L3 round-log proposal source | `docs/EXPERIMENT_095_ROUND_LOG_DRAFT_PROPOSAL_SOURCE.md` |
| V20 | partial | L4 pre-gate & verifier-mediated design | `docs/EXPERIMENT_098_PRE_L4_SAFETY_GATE_REVIEW.md` |
| V21 | partial | L4 scaffolds & trace-only dry-run | `docs/EXPERIMENT_102_L4_NOOP_OPT_IN_SCAFFOLD.md` |

## Per-version detail

### V1 — Correctness prototype

- **Evidence status:** `verified` (high)
- **Purpose:** Pre-formal-release milestone V1: Correctness prototype.
- **Key contribution:** verifier-first core
- **Caveats:** Historical prototype milestone; not the 1500-cell public headline.
- **Evidence files:**
  - `docs/V1_SCOPE_STATEMENT.md`

### V2 — Framework generalization

- **Evidence status:** `verified` (high)
- **Purpose:** Pre-formal-release milestone V2: Framework generalization.
- **Key contribution:** registry, CLI
- **Caveats:** Historical prototype milestone; not the 1500-cell public headline.
- **Evidence files:**
  - `docs/RELEASE_NOTES_V0.2.0.md`
  - `docs/V2_SCOPE_STATEMENT.md`

### V3 — Benchmark seriously

- **Evidence status:** `verified` (high)
- **Purpose:** Pre-formal-release milestone V3: Benchmark seriously.
- **Key contribution:** sweeps, markdown reports
- **Caveats:** Historical prototype milestone; not the 1500-cell public headline.
- **Evidence files:**
  - `docs/RELEASE_NOTES_V0.3.0.md`
  - `docs/V3_SCOPE_STATEMENT.md`

### V4 — Asymmetric K/V compression

- **Evidence status:** `verified` (high)
- **Purpose:** Pre-formal-release milestone V4: Asymmetric K/V compression.
- **Key contribution:** Experiment 003; git tag `v0.4.0`
- **Caveats:** Historical prototype milestone; not the 1500-cell public headline.
- **Evidence files:**
  - `docs/RELEASE_NOTES_V0.4.0.md`
  - `docs/V4_SCOPE_STATEMENT.md`

### V5 — Workspace-aware memory accounting

- **Evidence status:** `verified` (high)
- **Purpose:** Pre-formal-release milestone V5: Workspace-aware memory accounting.
- **Key contribution:** Experiment 004; git tag `v0.5.0`
- **Caveats:** Historical prototype milestone; not the 1500-cell public headline.
- **Evidence files:**
  - `docs/RELEASE_NOTES_V0.5.0.md`
  - `docs/V5_SCOPE_DRAFT.md`
  - `docs/V5_SCOPE_STATEMENT.md`

### V6 — Backend adapter interface

- **Evidence status:** `verified` (high)
- **Purpose:** Pre-formal-release milestone V6: Backend adapter interface.
- **Key contribution:** kvpress KnormPress; git tag `v0.6.0`
- **Caveats:** Historical prototype milestone; not the 1500-cell public headline.
- **Evidence files:**
  - `docs/RELEASE_NOTES_V0.6.0.md`
  - `docs/V6_SCOPE_STATEMENT.md`

### V7 — Layer-aware V policies

- **Evidence status:** `verified` (high)
- **Purpose:** Pre-formal-release milestone V7: Layer-aware V policies.
- **Key contribution:** Experiments 006/006C; git tag `v0.7.0`
- **Caveats:** Historical prototype milestone; not the 1500-cell public headline.
- **Evidence files:**
  - `docs/RELEASE_NOTES_V0.7.0.md`
  - `docs/V7_SCOPE_STATEMENT.md`

### V8 — Serving harness

- **Evidence status:** `verified` (high)
- **Purpose:** Pre-formal-release milestone V8: Serving harness.
- **Key contribution:** Experiment 007; git tag `v0.8.0`
- **Caveats:** Historical prototype milestone; not the 1500-cell public headline.
- **Evidence files:**
  - `docs/RELEASE_NOTES_V0.8.0.md`
  - `docs/V8_SCOPE_STATEMENT.md`

### V9 — Real backend gauntlet

- **Evidence status:** `verified` (high)
- **Purpose:** Pre-formal-release milestone V9: Real backend gauntlet.
- **Key contribution:** Exp 008–011; git tag `v0.9.0`
- **Caveats:** Historical prototype milestone; not the 1500-cell public headline.
- **Evidence files:**
  - `docs/RELEASE_NOTES_V0.9.0.md`
  - `docs/V9_SCOPE_STATEMENT.md`

### V10 — Suite hardening

- **Evidence status:** `verified` (high)
- **Purpose:** Pre-formal-release milestone V10: Suite hardening.
- **Key contribution:** Exp 012–014; git tag `v0.10.0`
- **Caveats:** Historical prototype milestone; not the 1500-cell public headline.
- **Evidence files:**
  - `docs/RELEASE_NOTES_V0.10.0.md`
  - `docs/V10_SCOPE_DRAFT.md`
  - `docs/V10_SCOPE_STATEMENT.md`

### V11 — Launch hardening

- **Evidence status:** `verified` (high)
- **Purpose:** Pre-formal-release milestone V11: Launch hardening.
- **Key contribution:** Exp 015–020; git tag `v0.11.0`
- **Caveats:** Historical prototype milestone; not the 1500-cell public headline.
- **Evidence files:**
  - `docs/RELEASE_NOTES_V0.11.0.md`
  - `docs/V11_SCOPE_STATEMENT.md`

### V12 — Deferred work completion gauntlet

- **Evidence status:** `verified` (high)
- **Purpose:** Pre-formal-release milestone V12: Deferred work completion gauntlet.
- **Key contribution:** Exp 021–027
- **Caveats:** Historical prototype milestone; not the 1500-cell public headline.
- **Evidence files:**
  - `docs/V12_SCOPE_STATEMENT.md`

### V13 — Practicality proof

- **Evidence status:** `verified` (high)
- **Purpose:** Pre-formal-release milestone V13: Practicality proof.
- **Key contribution:** span verification, demos, external methods
- **Caveats:** Historical prototype milestone; not the 1500-cell public headline.
- **Evidence files:**
  - `docs/V13_SCOPE_STATEMENT.md`

### V14 — CUDA restored verifier & GPU memory diagnostics

- **Evidence status:** `partial` (medium)
- **Purpose:** Pre-formal-release milestone V14: CUDA restored verifier & GPU memory diagnostics. Aligns with Phase 14 safety/runtime ladder work (distinct from formal A–K release phases).
- **Key contribution:** Phase 14A–14C; Exp 055–058
- **Caveats:** No V14_SCOPE_STATEMENT.md; version inferred from Phase 14 / experiment docs. Historical lineage only — not release benchmark evidence.
- **Evidence files:**
  - `docs/EXPERIMENT_055_EXPERIMENTAL_RESTORED_VERIFIER_CLI.md`
  - `docs/EXPERIMENT_056_CUDA_RESTORED_VERIFIER_RUNTIME_GATE.md`
  - `docs/EXPERIMENT_057_GPU_MEMORY_ACCOUNTING.md`
  - `docs/EXPERIMENT_058_EXPANDED_GPU_MEMORY_PANEL.md`
  - `scripts/research/run_exp055_experimental_restored_verifier_cli.py`
  - `scripts/research/run_exp056_cuda_restored_verifier_runtime_gate.py`
  - `scripts/research/run_exp057_gpu_memory_accounting.py`
  - `scripts/research/run_exp058_expanded_gpu_memory_panel.py`
  - `tests/test_exp055_experimental_restored_verifier_cli.py`
  - `tests/test_exp056_cuda_restored_verifier_runtime_gate.py`

### V15 — vLLM feasibility probes

- **Evidence status:** `partial` (medium)
- **Purpose:** Pre-formal-release milestone V15: vLLM feasibility probes. Aligns with Phase 15 safety/runtime ladder work (distinct from formal A–K release phases).
- **Key contribution:** Phase 15A–15E; Exp 059–065
- **Caveats:** No V15_SCOPE_STATEMENT.md; version inferred from Phase 15 / experiment docs. Historical lineage only — not release benchmark evidence.
- **Evidence files:**
  - `docs/EXPERIMENT_059_VLLM_FEASIBILITY_PROBE.md`
  - `docs/EXPERIMENT_060_VLLM_VENV_FEASIBILITY.md`
  - `docs/EXPERIMENT_061_VLLM_VERSION_SWEEP.md`
  - `docs/EXPERIMENT_062_VLLM_CONTAINER_FEASIBILITY.md`
  - `docs/EXPERIMENT_063_VLLM_API_SURFACE_RECON.md`
  - `docs/EXPERIMENT_064_VLLM_KV_VISIBILITY_PROBE.md`
  - `docs/EXPERIMENT_065_IDLE_VLLM_OBJECT_KV_PROBE.md`
  - `run_exp065_remote.sh`
  - `scripts/research/run_exp059_vllm_feasibility_probe.py`
  - `scripts/research/run_exp060_vllm_venv_feasibility.py`

### V16 — Shadow observers & streaming quant feasibility

- **Evidence status:** `partial` (medium)
- **Purpose:** Pre-formal-release milestone V16: Shadow observers & streaming quant feasibility. Aligns with Phase 16 safety/runtime ladder work (distinct from formal A–K release phases).
- **Key contribution:** Phase 16A–16T; Exp 066–085
- **Caveats:** No V16_SCOPE_STATEMENT.md; version inferred from Phase 16 / experiment docs. Historical lineage only — not release benchmark evidence.
- **Evidence files:**
  - `docs/EXPERIMENT_066_STREAMING_QUANT_ATTENTION_FEASIBILITY.md`
  - `docs/EXPERIMENT_067_HF_SINGLE_LAYER_ATTENTION_DRIFT.md`
  - `docs/EXPERIMENT_068_QWEN_ROPE_LONG_CONTEXT_ATTENTION_PROBE.md`
  - `docs/EXPERIMENT_069_MULTILAYER_ATTENTION_DRIFT_ACCUMULATION.md`
  - `docs/EXPERIMENT_070_STREAMING_MULTILAYER_NUMERICS_AUDIT.md`
  - `docs/EXPERIMENT_071_FULL_PREFIX_LOGIT_DRIFT_SMOKE.md`
  - `docs/EXPERIMENT_072_FULL_DEPTH_DIVERGENCE_TRACE.md`
  - `docs/EXPERIMENT_073_QWEN_FAMILY_DIVERGENCE_PANEL.md`
  - `docs/EXPERIMENT_074_ATTENTION_TOLERANCE_POLICY_PANEL.md`
  - `docs/EXPERIMENT_075_GENERATION_SHADOW_WIRING_REVIEW.md`

### V17 — Claim-safe demo & broader validation

- **Evidence status:** `partial` (medium)
- **Purpose:** Pre-formal-release milestone V17: Claim-safe demo & broader validation. Aligns with Phase 17 safety/runtime ladder work (distinct from formal A–K release phases).
- **Key contribution:** Phase 17A–17D
- **Caveats:** No V17_SCOPE_STATEMENT.md; version inferred from Phase 17 / experiment docs. Historical lineage only — not release benchmark evidence.
- **Evidence files:**
  - `docs/EXPERIMENT_086_CLAIM_SAFE_DEMO_PACKAGING.md`
  - `docs/EXPERIMENT_087_BROADER_MODEL_VALIDATION_PANEL.md`
  - `docs/EXPERIMENT_088_LONG_CONTEXT_VALIDATION_PANEL.md`
  - `docs/EXPERIMENT_089_INTEGRATION_DESIGN_REVIEW.md`
  - `docs/PHASE_17B_BROADER_MODEL_VALIDATION.md`
  - `docs/PHASE_17C_LONG_CONTEXT_VALIDATION.md`
  - `docs/PHASE_17D_INTEGRATION_DESIGN_REVIEW.md`
  - `docs/PHASE_17_CLAIM_SAFE_DEMO.md`
  - `docs/PHASE_17_DEMO_SCRIPT.md`
  - `scripts/research/run_exp086_claim_safe_demo_packaging.py`

### V18 — Integration safety & L3 draft-shadow no-commit

- **Evidence status:** `partial` (medium)
- **Purpose:** Pre-formal-release milestone V18: Integration safety & L3 draft-shadow no-commit. Aligns with Phase 18 safety/runtime ladder work (distinct from formal A–K release phases).
- **Key contribution:** Phase 18A–18E; Exp 090–091
- **Caveats:** No V18_SCOPE_STATEMENT.md; version inferred from Phase 18 / experiment docs. Historical lineage only — not release benchmark evidence.
- **Evidence files:**
  - `docs/EXPERIMENT_090_INTEGRATION_SAFETY_SPEC.md`
  - `docs/EXPERIMENT_091_GUARDED_DRAFT_SHADOW_NO_COMMIT_SCAFFOLD.md`
  - `docs/EXPERIMENT_092_GUARDED_DRAFT_SHADOW_PANEL_VALIDATION.md`
  - `docs/EXPERIMENT_093_SHADOW_TOP1_EXTRACTION_HARDENING.md`
  - `docs/EXPERIMENT_094_SHADOW_PROPOSAL_PROVENANCE_AUDIT.md`
  - `docs/PHASE_18A_INTEGRATION_SAFETY_SPEC.md`
  - `docs/PHASE_18B_GUARDED_DRAFT_SHADOW_NO_COMMIT.md`
  - `docs/PHASE_18C_GUARDED_DRAFT_SHADOW_PANEL_VALIDATION.md`
  - `docs/PHASE_18D_SHADOW_TOP1_EXTRACTION_HARDENING.md`
  - `docs/PHASE_18E_SHADOW_PROPOSAL_PROVENANCE_AUDIT.md`

### V19 — L3 round-log proposal source

- **Evidence status:** `partial` (medium)
- **Purpose:** Pre-formal-release milestone V19: L3 round-log proposal source. Aligns with Phase 19 safety/runtime ladder work (distinct from formal A–K release phases).
- **Key contribution:** Phase 19A–19C
- **Caveats:** No V19_SCOPE_STATEMENT.md; version inferred from Phase 19 / experiment docs. Historical lineage only — not release benchmark evidence.
- **Evidence files:**
  - `docs/EXPERIMENT_095_ROUND_LOG_DRAFT_PROPOSAL_SOURCE.md`
  - `docs/EXPERIMENT_096_ROUND_LOG_PROPOSAL_SOURCE_COMPARISON_PANEL.md`
  - `docs/EXPERIMENT_097_L3_PROMOTED_SOURCE_VALIDATION.md`
  - `docs/PHASE_19A_ROUND_LOG_DRAFT_PROPOSAL_SOURCE.md`
  - `docs/PHASE_19B_ROUND_LOG_PROPOSAL_SOURCE_COMPARISON.md`
  - `docs/PHASE_19C_L3_PROMOTED_SOURCE_VALIDATION.md`
  - `scripts/research/run_exp095_round_log_draft_proposal_source.py`
  - `scripts/research/run_exp096_round_log_proposal_source_comparison_panel.py`
  - `scripts/research/run_exp097_l3_promoted_source_validation.py`
  - `tests/test_exp095_round_log_draft_proposal_source.py`

### V20 — L4 pre-gate & verifier-mediated design

- **Evidence status:** `partial` (medium)
- **Purpose:** Pre-formal-release milestone V20: L4 pre-gate & verifier-mediated design. Aligns with Phase 20 safety/runtime ladder work (distinct from formal A–K release phases).
- **Key contribution:** Phase 20A–20D
- **Caveats:** No V20_SCOPE_STATEMENT.md; version inferred from Phase 20 / experiment docs. Historical lineage only — not release benchmark evidence.
- **Evidence files:**
  - `docs/EXPERIMENT_098_PRE_L4_SAFETY_GATE_REVIEW.md`
  - `docs/EXPERIMENT_099_L4_VERIFIER_MEDIATED_DESIGN_SPEC.md`
  - `docs/EXPERIMENT_100_L4_CONTRACT_TESTS_NO_RUNTIME.md`
  - `docs/EXPERIMENT_101_L4_INTEGRATION_PLAN_REVIEW.md`
  - `docs/PHASE_20A_PRE_L4_SAFETY_GATE_REVIEW.md`
  - `docs/PHASE_20B_L4_VERIFIER_MEDIATED_DESIGN_SPEC.md`
  - `docs/PHASE_20C_L4_CONTRACT_TESTS_NO_RUNTIME.md`
  - `docs/PHASE_20D_L4_INTEGRATION_PLAN_REVIEW.md`
  - `scripts/research/run_exp098_pre_l4_safety_gate_review.py`
  - `scripts/research/run_exp099_l4_verifier_mediated_design_spec.py`

### V21 — L4 scaffolds & trace-only dry-run

- **Evidence status:** `partial` (medium)
- **Purpose:** Pre-formal-release milestone V21: L4 scaffolds & trace-only dry-run. Aligns with Phase 21 safety/runtime ladder work (distinct from formal A–K release phases).
- **Key contribution:** Phase 21A–21L
- **Caveats:** No V21_SCOPE_STATEMENT.md; version inferred from Phase 21 / experiment docs. Historical lineage only — not release benchmark evidence.
- **Evidence files:**
  - `docs/EXPERIMENT_102_L4_NOOP_OPT_IN_SCAFFOLD.md`
  - `docs/EXPERIMENT_103_L4_NOOP_SCAFFOLD_PANEL_VALIDATION.md`
  - `docs/EXPERIMENT_104_L4_TRACE_ONLY_DRY_RUN_DESIGN.md`
  - `docs/EXPERIMENT_105_L4_TRACE_ONLY_DRY_RUN_SCAFFOLD.md`
  - `docs/EXPERIMENT_106_L4_TRACE_ONLY_DRY_RUN_PANEL_VALIDATION.md`
  - `docs/EXPERIMENT_107_L4_VERIFIER_EVIDENCE_TRACE_SCHEMA_DESIGN.md`
  - `docs/EXPERIMENT_108_L4_VERIFIER_EVIDENCE_TRACE_SCHEMA_SCAFFOLD.md`
  - `docs/EXPERIMENT_109_L4_VERIFIER_TRACE_SCHEMA_EXAMPLE_VALIDATION.md`
  - `docs/EXPERIMENT_110_L4_TRACE_SCHEMA_ADVERSARIAL_INJECTION_PANEL.md`
  - `docs/EXPERIMENT_111_L4_VERIFIER_RUNTIME_INSTRUMENTATION_DESIGN.md`

## Version-lineage entries requiring manual source attachment

V14–V21 lack dedicated `V{N}_SCOPE_STATEMENT.md` files. Evidence is drawn from Phase N / experiment documentation. Do not cite as verified benchmark evidence.

- **V14** (partial): No V14_SCOPE_STATEMENT.md; version inferred from Phase 14 / experiment docs. Historical lineage only — not release benchmark evidence.
- **V15** (partial): No V15_SCOPE_STATEMENT.md; version inferred from Phase 15 / experiment docs. Historical lineage only — not release benchmark evidence.
- **V16** (partial): No V16_SCOPE_STATEMENT.md; version inferred from Phase 16 / experiment docs. Historical lineage only — not release benchmark evidence.
- **V17** (partial): No V17_SCOPE_STATEMENT.md; version inferred from Phase 17 / experiment docs. Historical lineage only — not release benchmark evidence.
- **V18** (partial): No V18_SCOPE_STATEMENT.md; version inferred from Phase 18 / experiment docs. Historical lineage only — not release benchmark evidence.
- **V19** (partial): No V19_SCOPE_STATEMENT.md; version inferred from Phase 19 / experiment docs. Historical lineage only — not release benchmark evidence.
- **V20** (partial): No V20_SCOPE_STATEMENT.md; version inferred from Phase 20 / experiment docs. Historical lineage only — not release benchmark evidence.
- **V21** (partial): No V21_SCOPE_STATEMENT.md; version inferred from Phase 21 / experiment docs. Historical lineage only — not release benchmark evidence.

## Technical Report Versions (paper release arc)

| Version | Date | Changes | Cells |
|---------|------|---------|------:|
| v2.5.3 | 2026-06-26 | Abstract typo fix, int8/Mistral leaderboard contradiction resolved | 3,844 |
| v2.5.4 | 2026-06-27 | Paper hygiene: 360→1560 totals, BFCL repro command, Case P formatting, All Panels appendix | 3,844 |
| **v2.6** | **2026-06-28** | **720-cell real HF LongBench drift panel (both models), 600-cell BFCL validity v2.7 Llama. Key finding: task type dominates int4_sim divergence (6% code → 90% reading).** | **5,164+** |
