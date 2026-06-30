# ExactKV Project Lineage (Release Synthesis — Part 2)

> **ExactKV did not start at Phase A.** Two distinct timelines run through this
> repository and must not be conflated:
>
> 1. a **version arc V1–V21** (pre-formal-release research prototypes), and
> 2. a **formal release arc Phase A–K plus gates R0/R1/R2** (formalization,
>    scaling, packaging, validation).
>
> These use **different numbering systems**. The version arc is *not* the same as
> the phase arc. The 1500-cell public benchmark belongs to the **Phase A–K**
> formal arc; the verifier-first ideas, demos, and no-go boundaries originate in
> the **V1–V21** arc.

This synthesis reconciles both timelines from repository evidence only. It is a
distilled, cross-checked companion to the repo's own
[`docs/PROJECT_LINEAGE.md`](../docs/PROJECT_LINEAGE.md) and
[`docs/VERSION_LINEAGE.md`](../docs/VERSION_LINEAGE.md), which remain the
authoritative lineage documents.

## 1. How the lineage was reconstructed

Evidence sources (no memory, no invention):

- `git ls-files` — 1,524 tracked artifacts (see [`artifact_inventory.md`](artifact_inventory.md)).
- `git log --oneline --all` — 178 commits, 12 release tags (`v0.1.0-phase1` … `v0.13.0-rc1`).
- Git tags trace the V-arc: `v0.2.0`→V2, `v0.4.0`→V4 … `v0.11.0`→V11.
- Report metadata (`phase_id`, `total_cells`, `exactkv_failures`) read from JSON.
- 121 `EXPERIMENT_*.md` docs, 31 `PHASE_*.md` docs, 13 `V*_SCOPE_STATEMENT.md` docs.

## 2. Two timelines at a glance

| Arc | Numbering | Nature | Evidence quality |
|-----|-----------|--------|------------------|
| **Version arc** | V1 → V21 | Pre-release research prototypes, demos, safety ladder, no-go probes | V1–V13 verified (scope statements + tags); V14–V21 partial (phase/experiment docs only) |
| **Formal release arc** | Phase A → K, gates R0/R1/R2 | Cross-model benchmark, kernels, truth engine, scale, novelty audit, release freeze | Release-grade (curated authoritative set) |

The two arcs **overlap in time** but serve different purposes. The numbered
"Phase 11–21" safety-ladder documents (`PHASE_16_*`, `PHASE_18B_*`, etc.) belong
to the **version arc** (mapped to V14–V21), and are **distinct** from the
lettered **Phase A–K** formal release pipeline.

## 3. Version arc V1–V21 (summary)

| Range | Theme | Status |
|-------|-------|--------|
| V1–V3 | Correctness prototype, framework generalization, serious benchmarking | verified |
| V4–V9 | Asymmetric K/V, workspace memory, backend adapters, layer-aware V, serving harness, real-backend gauntlet | verified |
| V10–V11 | Eval-suite hardening, launch hardening (`v0.10.0`, `v0.11.0` tags) | verified |
| V12–V13 | Deferred-work gauntlet, practicality proof, demos, external-method probes | verified |
| V14–V15 | CUDA restored-verifier gate, GPU-memory diagnostics, vLLM **no-go** probes | partial |
| V16 | Shadow observers, streaming-quant attention feasibility | partial |
| V17 | Claim-safe demo packaging, broader-model & long-context validation | partial |
| V18–V19 | Integration safety, L3 guarded draft-shadow **no-commit**, round-log sources | partial |
| V20–V21 | L4 pre-gate, verifier-mediated **dry-run** scaffolds (no runtime commit) | partial |

Full per-version detail with evidence files and caveats:
[`version_lineage.md`](version_lineage.md).

## 4. Formal release arc Phase A–K + gates

| Phase / Gate | Role | Primary artifact |
|--------------|------|------------------|
| A–C | Cross-model benchmark + leaderboard + visuals | `reports/phaseA_benchmark.json`, `scripts/run_phase_a_scale_benchmark.py` |
| D | Runtime probe layer | `reports/phaseD_runtime_probe.json` |
| E–F | KV compression kernels (**F = microbenchmark only**) | `reports/phaseF_kernel_benchmark.json` |
| G | Canonical divergence truth engine | `reports/phaseG_unified_truth.json` |
| H / H+ | Public leaderboard platform + **7B/8B scale** | `reports/benchmark.json`, `reports/scale_7b/raw.json` |
| I | Novelty audit + claim lock | `reports/novelty_audit.json`, `docs/NOVELTY_AUDIT.md` |
| J | Public release freeze + reproducibility | `docs/CLAIM_BOUNDARIES.md`, `docs/METRIC_DEFINITIONS.md` |
| R0 | Evidence integrity gate | `reports/release_evidence_status.json` |
| R1 | Mistral leaderboard aggregate repair | `reports/public_release/leaderboard_final.json` |
| R2 | Full lineage archaeology | `docs/PROJECT_LINEAGE.md`, `docs/HISTORICAL_ARTIFACT_INVENTORY.md` |
| K | Final launch pack + technical report | `reports/public_release/*`, `docs/EXACTKV_TECHNICAL_REPORT.md` |

Full per-phase detail: [`phase_lineage.md`](phase_lineage.md).

## 5. What each arc still supports vs. what is superseded

**Still supported (cite as methodology / historical):**
- Verifier/draft/commit semantics and trace metrics (V-arc origin).
- Demo narratives (terminal crash-test, structured-output drift).
- Claim boundaries and adapter-honesty disclosures.

**Superseded (do not cite as public headline):**
- Phase A 336-cell panel → replaced by `scale_7b` 1500-cell panel.
- Legacy live-correction demo → terminal crash-test demo.

**Intentionally not claimed (no-go):**
- vLLM / LMCache integration (probes only).
- Production serving, throughput, end-to-end speedups.
- Active GPU memory / VRAM savings (forbidden unless directly measured).
- Real SpectralQuant / full Shard integration (fallback/probe only).
- L4 runtime commit paths (dry-run / scaffold only).

## 6. Source-of-truth pointer

The single authoritative benchmark source of truth is
**`reports/scale_7b/raw.json`**. The full hierarchy (authoritative vs.
supporting vs. exploratory vs. launch) is in
[`source_of_truth_map.md`](source_of_truth_map.md).
