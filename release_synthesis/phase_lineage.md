# ExactKV Phase Lineage A–K + Gates R0/R1/R2 (Release Synthesis — Part 2)

The **formal release arc** uses a **lettered** numbering (Phase A–K) plus
evidence gates (R0–R2). This is a different numbering system from the **V1–V21**
version arc and from the **numbered** safety-ladder phases (Phase 11–21, which
belong to the version arc).

For each phase: purpose, representative files, outputs, validation status, and
relation to the version arc.

## Phase A — Cross-model scale benchmark

- **Purpose:** Formal multi-model, multi-compressor token-drift benchmark.
- **Files:** `scripts/run_phase_a_scale_benchmark.py`, `scripts/run_scale_7b_benchmark.py`
- **Outputs:** `reports/phaseA_benchmark.json` (336 cells, internal/historical); `reports/scale_7b/raw.json` (1500 cells, **public headline**, `phase_id="phaseA_scale_benchmark"`).
- **Validation:** PASS — `exactkv_failures=0`, `deterministic_mode=false`, both 7B/8B models present.
- **Relation to V-arc:** Formalizes the sweep methodology from V3/V10.

## Phase B — Leaderboard

- **Purpose:** Locked composite scoring of compressor×model cells.
- **Files:** `scripts/run_leaderboard.py`, `scripts/exactkv_leaderboard.py`
- **Outputs:** `reports/leaderboard.json`, `reports/scale_7b/leaderboard.json`
- **Validation:** PASS — `validation_result.valid=true`.
- **Relation to V-arc:** Generalizes V11 analysis layers.

## Phase C — Publication visuals

- **Purpose:** Public visual package + plots.
- **Files:** `scripts/run_phase_c_publication.py`, `scripts/render_public_visuals_036.py`
- **Outputs:** `docs/PUBLIC_VISUAL_PACKAGE.md`, `reports/phaseG_divergence_map.png`
- **Validation:** PASS (artifacts present).
- **Relation to V-arc:** Builds on V-series visual demos (Exp 035/036).

## Phase D — Runtime probe

- **Purpose:** Llama runtime probe + layer drift / memory profile.
- **Files:** `scripts/run_phase_d_runtime_probe.py`
- **Outputs:** `reports/phaseD_runtime_probe.json`, `phaseD_layer_drift.json`, `phaseD_memory_profile.json`
- **Validation:** PASS (evidence-complete per R0).
- **Relation to V-arc:** Extends V14 CUDA restored-verifier diagnostics.

## Phase E–F — KV compression kernels

- **Purpose:** CUDA/Triton KV compression kernel + microbenchmark.
- **Files:** `scripts/run_phase_e_kernel_demo.py`, `scripts/run_phase_f_kernel_benchmark.py`
- **Outputs:** `reports/phaseF_kernel_benchmark.json` (int8 1.63×, int4 1.54×, block_sparse 0.98×).
- **Validation:** PASS — Triton+CUDA available; **block_sparse uses torch backend**.
- **Caveat:** **Kernel microbenchmark only — NOT end-to-end inference speedup.**
- **Relation to V-arc:** Hardens V12 KIVI CUDA/Triton feasibility (Exp 024).

## Phase G — Unified truth engine

- **Purpose:** Canonical `FirstDivergenceAuthority` across cells.
- **Files:** `scripts/run_phase_g_unified_truth_engine.py`
- **Outputs:** `reports/phaseG_unified_truth.json`, `phaseG_kernel_consistency.json`, `phaseG_divergence_map.png`
- **Validation:** PASS.
- **Relation to V-arc:** Consolidates V10/V11 trace-correctness work into a single authority.

## Phase H / H+ — Platform + 7B/8B scale

- **Purpose:** Public leaderboard platform; real 7B/8B scale panel.
- **Outputs:** `reports/benchmark.json` (platform), `reports/scale_7b/*` (**authoritative 1500-cell**).
- **Validation:** PASS — 1500 cells, 0 failures, both models present, non-deterministic.
- **Relation to V-arc:** Scales V13 practicality proof to real 7B/8B models.

## Phase I — Novelty audit + claim lock

- **Purpose:** Prior-art catalogue + allowed/qualified/forbidden claim sets.
- **Outputs:** `docs/NOVELTY_AUDIT.md`, `reports/novelty_audit.json`, `reports/novelty_audit_matrix.csv`
- **Validation:** PASS — 14 prior-art systems catalogued, 7 verified, 5 source-pending.

## Phase J — Public release freeze + reproducibility

- **Purpose:** Freeze metric definitions, claim boundaries, repro path.
- **Outputs:** `docs/CLAIM_BOUNDARIES.md`, `docs/METRIC_DEFINITIONS.md`, `docs/REPRODUCIBILITY.md`
- **Validation:** PASS — enforced by `scripts/audit_public_claims.py`, `scripts/check_public_release.py`.

## Phase K — Launch pack + technical report

- **Purpose:** Public bundle, demo cards, technical report, launch posts.
- **Files:** `scripts/build_launch_pack.py`, `scripts/check_launch_pack.py`
- **Outputs:** `reports/public_release/*`, `docs/EXACTKV_TECHNICAL_REPORT.md`, `docs/launch_*_final.md`
- **Validation:** PASS — `reports/launch_pack_validation.json`.

## Gate R0 — Evidence integrity

- **Output:** `reports/release_evidence_status.json` (status **PASS**).
- **Checks:** scale raw exists, 1500 cells, both models present, 0 failures, non-deterministic, Phase F shapes, SpectralQuant fallback disclosed, Shard probe disclosed, public claim safety clean.

## Gate R1 — Mistral leaderboard repair

- **Output:** `reports/public_release/leaderboard_final.json` (Mistral numeric rows restored, 750 raw Mistral cells).

## Gate R2 — Lineage archaeology

- **Output:** `docs/PROJECT_LINEAGE.md`, `docs/HISTORICAL_ARTIFACT_INVENTORY.md` (1,176 catalogued historical artifacts).

## Validation summary

| Phase/Gate | Status | Source |
|------------|--------|--------|
| A–C | PASS | scale_7b, leaderboard, visuals present |
| D | PASS | phaseD reports present |
| E–F | PASS (microbench caveat) | phaseF_kernel_benchmark.json |
| G | PASS | phaseG_unified_truth.json |
| H/H+ | PASS | scale_7b 1500 cells, 0 failures |
| I | PASS | novelty_audit.json |
| J | PASS | claim/metric freeze docs |
| K | PASS | public_release bundle |
| R0 | PASS | release_evidence_status.json |
| R1 | PASS | leaderboard_final.json |
| R2 | PASS | project_lineage / historical inventory |
