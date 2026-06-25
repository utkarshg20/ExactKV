# ExactKV Artifact Index (Phase J)

Maps source artifacts to generated outputs, regeneration commands, and claim relevance.

| Source artifact | Generated / related | Regenerate command | Claim relevance | Public-safe |
|-----------------|---------------------|--------------------|-----------------|-------------|
| `reports/scale_7b/raw.json` | leaderboard, public bundle | `exactkv.py run full-scale-7b` (GPU, expensive) | **Authoritative** 1500-cell exactness evidence | Yes |
| `reports/scale_7b/leaderboard.json` | `public_release/leaderboard_final.json` | `exactkv.py run publish` | Compressor rankings | Yes |
| `reports/scale_7b/scale_summary.json` | README_PUBLIC, benchmark_summary | `exactkv.py run publish` | Cell count, failure gate | Yes |
| `reports/public_release/leaderboard_final.json` | — | `exactkv_repro.py --reports-only` | Public leaderboard | Yes |
| `reports/public_release/benchmark_summary.md` | — | `exactkv.py run publish` | Headline metrics | Yes (with caveats) |
| `reports/public_release/methodology.md` | — | `exactkv.py run publish` | Metric definitions + boundaries | Yes |
| `reports/public_release/release_manifest.json` | — | `exactkv.py run publish` | Source pointers | Yes |
| `reports/phaseF_kernel_benchmark.json` | — | Phase F kernel script (CUDA) | Kernel microbenchmark speedups only | Qualified |
| `reports/phaseG_unified_truth.json` | divergence authority | Phase G pipeline | First-divergence canonical | Yes |
| `reports/release_evidence_status.json` | `docs/RELEASE_EVIDENCE_STATUS.md` | `check_release_evidence.py` | Gate R0 integrity | Yes |
| `reports/novelty_audit.json` | `docs/NOVELTY_AUDIT.md` | `run_novelty_audit.py` | Prior-art / claim lock | Yes |
| `docs/NOVELTY_AUDIT.md` | — | `run_novelty_audit.py` | Human-readable novelty audit | Yes |
| `docs/RELEASE_EVIDENCE_STATUS.md` | — | `check_release_evidence.py` | Launch evidence status | Yes |
| `reports/phaseA_benchmark.json` | historical only | Phase A benchmark | Internal 336-cell panel | Yes (not headline) |
| `reports/repro_manifest.json` | — | `exactkv_repro.py` | Repro audit trail | Yes |

---

## Public release bundle layout

```
reports/public_release/
  README_PUBLIC.md
  benchmark_summary.md
  methodology.md
  leaderboard_final.json
  release_manifest.json
  repro_command.sh
```

---

## Validation commands

```bash
python3 scripts/check_public_release.py
python3 scripts/check_release_evidence.py
python3 scripts/audit_public_claims.py
python3 scripts/check_no_secrets.py
python3 scripts/check_project_lineage.py
```

---

## Historical, Pre-Release, and Exploratory Artifacts

> Full inventory: [`HISTORICAL_ARTIFACT_INVENTORY.md`](HISTORICAL_ARTIFACT_INVENTORY.md) · Lineage: [`PROJECT_LINEAGE.md`](PROJECT_LINEAGE.md) · Version arc: [`VERSION_LINEAGE.md`](VERSION_LINEAGE.md)

Regenerate inventory: `python3 scripts/build_project_lineage.py`

| Version lineage artifact | Role |
|--------------------------|------|
| `docs/VERSION_LINEAGE.md` | Human-readable V1–V21 table |
| `reports/version_lineage.json` | Machine-readable version records |
| `reports/version_lineage.csv` | Spreadsheet export |

**Caveat:** Version lineage is historical/project context — **not** release benchmark evidence. Authoritative benchmark: `reports/scale_7b/raw.json` (1500 cells).

### Early foundation & verifier core (V1–V9, Exp 001–007)

| Representative files | Role | Public status | Release relevance |
|---------------------|------|---------------|-------------------|
| `docs/V1_SCOPE_STATEMENT.md` … `docs/V9_SCOPE_STATEMENT.md` | Version arc problem framing | Historical | Methodology context |
| `docs/EXPERIMENT_001_SMOKE_SWEEP.md`, `002`, `003` | First compressor sweeps | Internal evidence | Exactness gate origin |
| `exactkv/` verifier/generator modules | Draft/verify/commit core | Supporting | Architecture claims |

**Caveats:** Pre–Phase A experiments are panel-scoped; not the 1500-cell public headline.

### V10–V13 evaluation suites & demos (Exp 012–037)

| Representative files | Role | Public status | Release relevance |
|---------------------|------|---------------|-------------------|
| `docs/EXPERIMENT_012_EVAL_SUITE_EXPANSION.md` | V10 suite hardening | Historical | Cross-model methodology |
| `docs/EXACTKV_TERMINAL_CRASH_TEST.md` | Primary public demo replay | Public illustrative | Demo narrative |
| `docs/EXPERIMENT_034B_SEMANTIC_CORRECTION_SEARCH.md` | Structured-output drift case | Public illustrative | First-divergence storytelling |

**Superseded by:** Phase H+ scale panel for headline leaderboard; demos remain illustrative.

### External compressor investigations (Exp 008–045)

| Representative files | Role | Caveats |
|---------------------|------|---------|
| `docs/EXPERIMENT_010_KVQUANT_SIM.md` | KVQuant adapter panel | Simquant; not deployment CUDA |
| `docs/EXPERIMENT_038_SHARD_EXTERNAL_DRAFTER_PROBE.md` | Shard Mode B probe | Probe-only in release |
| `docs/EXPERIMENT_045_SPECTRALQUANT_RESTRICTED_PANEL.md` | SpectralQuant panel | Fallback/proxy in current env |

### Safety ladder, shadow observers, L3/L4 (Phase 16–21, Exp 076–113)

| Representative files | Role | Caveats |
|---------------------|------|---------|
| `docs/PHASE_18B_GUARDED_DRAFT_SHADOW_NO_COMMIT.md` | L3 no-commit scaffold | Diagnostic only |
| `docs/PHASE_20B_L4_VERIFIER_MEDIATED_DESIGN_SPEC.md` | L4 dry-run design | Not runtime commit |
| `docs/EXPERIMENT_083_GUARDED_DECODE_TIME_SHADOW_SMOKE.md` | Decode-time shadow | Not streaming-attention integration |

### Serving / vLLM / LMCache no-go probes (Exp 017, 059–065)

| Representative files | Role | Public framing |
|---------------------|------|----------------|
| `docs/EXPERIMENT_059_VLLM_FEASIBILITY_PROBE.md` | vLLM import probe | Claim-boundary evidence |
| `docs/EXPERIMENT_027_PERFORMANCE_MEMORY_TRUTH_BOUNDARY.md` | Speed/memory forbidden list | **Required** claim caveats |
| `docs/VERICACHE_PARITY_CLAIM_GATE.md` | VeriCache relationship | Does not reproduce VeriCache |

### V14–V21 safety/runtime ladder (partial evidence)

| Representative files | Role | Evidence status |
|---------------------|------|-----------------|
| `docs/EXPERIMENT_056_CUDA_RESTORED_VERIFIER_RUNTIME_GATE.md` | V14 CUDA verifier gate | partial (Phase 14) |
| `docs/EXPERIMENT_059_VLLM_FEASIBILITY_PROBE.md` … `065` | V15 vLLM no-go probes | partial |
| `docs/PHASE_16_CLOSEOUT.md`, Exp 076–085 | V16 shadow observers | partial |
| `docs/PHASE_17_CLAIM_SAFE_DEMO.md` | V17 demo packaging | partial |
| `docs/PHASE_18B_GUARDED_DRAFT_SHADOW_NO_COMMIT.md` | V18 L3 no-commit | partial |
| `docs/PHASE_19A_ROUND_LOG_DRAFT_PROPOSAL_SOURCE.md` | V19 round-log source | partial |
| `docs/PHASE_20B_L4_VERIFIER_MEDIATED_DESIGN_SPEC.md` | V20 L4 design | partial |
| `docs/PHASE_21L_L4_STAGE3_VERIFIER_MEDIATED_DRY_RUN_SCAFFOLD.md` | V21 L4 dry-run | partial |

**Caveats:** No `V14_SCOPE_STATEMENT.md` … `V21_SCOPE_STATEMENT.md` in-repo. See [`VERSION_LINEAGE.md`](VERSION_LINEAGE.md). Not benchmark evidence.

### Formal Phase A–J + release gates

| Phase | Key artifacts | Authoritative? |
|-------|---------------|--------------|
| A–C | `reports/phaseA_benchmark.json`, leaderboard platform | Phase A historical; scale_7b authoritative |
| E–F | `reports/phaseF_kernel_benchmark.json` | Kernel microbenchmark only |
| G | `reports/phaseG_unified_truth.json` | Divergence authority |
| H+ | `reports/scale_7b/raw.json` | **Yes — public headline** |
| I–J, R0–R2 | novelty audit, repro manifest, lineage inventory | Release packaging |
| R2.1 | `docs/VERSION_LINEAGE.md`, `reports/version_lineage.json` | V1–V21 version arc correction |
