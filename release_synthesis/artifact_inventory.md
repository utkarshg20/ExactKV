# ExactKV Artifact Inventory (Release Synthesis — Part 1)

Generated: 2026-06-26T14:57:49.141483+00:00

> Heuristic-assisted forensic inventory of **every tracked artifact** (`git ls-files`). Full per-artifact rows are in [`artifact_inventory.json`](artifact_inventory.json) and [`artifact_inventory.csv`](artifact_inventory.csv). This document summarizes counts and the curated release-grade evidence set.

**Total tracked artifacts:** 1524

## Counts by type

| Type | Count |
|------|------:|
| `plot` | 327 |
| `report_md` | 273 |
| `test` | 199 |
| `unknown` | 192 |
| `code` | 179 |
| `script` | 176 |
| `report_json` | 147 |
| `doc` | 13 |
| `launch_artifact` | 13 |
| `config` | 4 |
| `benchmark_output` | 1 |

## Counts by likely timeline

| Timeline | Count |
|----------|------:|
| unknown | 830 |
| V-lineage | 623 |
| Phase-lineage | 42 |
| final launch | 25 |
| release gate | 4 |

## Evidence tiers

| Tier | Count |
|------|------:|
| Release-grade (curated authoritative) | 19 |
| Historical / exploratory only | 659 |
| Superseded | 1 |

## Release-grade evidence set (authoritative)

| Path | Type | Key evidence | Caveat |
|------|------|--------------|--------|
| `docs/CLAIM_BOUNDARIES.md` | report_md | — | — |
| `docs/EXACTKV_TECHNICAL_REPORT.md` | report_md | — | — |
| `docs/METRIC_DEFINITIONS.md` | report_md | — | — |
| `docs/NOVELTY_AUDIT.md` | report_md | — | — |
| `docs/PROJECT_LINEAGE.md` | report_md | — | — |
| `docs/RELEASE_EVIDENCE_STATUS.md` | report_md | — | — |
| `docs/VERSION_LINEAGE.md` | report_md | — | — |
| `reports/public_release/README_PUBLIC.md` | doc | — | — |
| `reports/public_release/benchmark_summary.md` | doc | — | — |
| `reports/public_release/demo_cards.json` | report_json | — | — |
| `reports/public_release/demo_cards.md` | doc | — | — |
| `reports/public_release/leaderboard_final.json` | report_json | deterministic_mode=False; status=leaderboard_complete | — |
| `reports/public_release/methodology.md` | doc | — | — |
| `reports/public_release/release_manifest.json` | report_json | total_cells=1500 | — |
| `reports/scale_7b/leaderboard.csv` | benchmark_output | — | — |
| `reports/scale_7b/leaderboard.json` | report_json | deterministic_mode=False; status=leaderboard_complete | — |
| `reports/scale_7b/leaderboard.md` | doc | — | — |
| `reports/scale_7b/raw.json` | report_json | total_cells=1500; exactkv_failures=0; deterministic_mode=False; status=benchmark_complete | — |
| `reports/scale_7b/scale_summary.json` | report_json | total_cells=1500; exactkv_failures=0; deterministic_mode=False; status=scale_benchmark_complete | — |

## Notes
- The single benchmark **source of truth** for public claims is `reports/scale_7b/raw.json` (1500 cells, `exactkv_failures=0`, `deterministic_mode=false`).
- `reports/phaseA_benchmark.json` (336 cells) is **superseded** as the public headline but retained as supporting cross-model evidence.
- `reports/phaseF_kernel_benchmark.json` is a **kernel microbenchmark** — not an end-to-end speedup.
- See [`source_of_truth_map.md`](source_of_truth_map.md) for the full hierarchy.

