# ExactKV Source-of-Truth Map (Release Synthesis — Part 2)

This map states, unambiguously, which artifacts may back which kinds of claims.

## Tier 1 — Authoritative release benchmark (the single source of truth)

| Artifact | Role |
|----------|------|
| **`reports/scale_7b/raw.json`** | **THE** benchmark source of truth: 1500 cells, Llama-3.1-8B + Mistral-7B-Instruct-v0.3, `exactkv_failures=0`, `deterministic_mode=false`. |
| `reports/scale_7b/leaderboard.json` / `.md` / `.csv` | Ranked leaderboard derived from the source of truth. |
| `reports/scale_7b/scale_summary.json` | Headline summary (cells, failures, models). |
| `reports/public_release/leaderboard_final.json` | Public leaderboard bundle (R1 Mistral repair applied). |
| `reports/public_release/release_manifest.json` | Pointers to authoritative source artifacts. |

**Any public performance/exactness claim must trace to Tier 1.**

## Tier 2 — Release methodology / claim documents (govern wording)

| Artifact | Role |
|----------|------|
| `docs/CLAIM_BOUNDARIES.md` | Allowed / qualified / forbidden claim sets (authoritative). |
| `docs/NOVELTY_AUDIT.md` + `reports/novelty_audit.json` | Prior-art positioning; uniqueness limits. |
| `docs/METRIC_DEFINITIONS.md` | Canonical metric definitions + limitations. |
| `reports/public_release/methodology.md` | Scoring formula + claim boundaries. |
| `reports/release_evidence_status.json` | Gate R0 evidence-integrity ledger. |
| `reports/phaseF_kernel_benchmark.json` | Kernel microbenchmark (qualified; **not** end-to-end). |
| `reports/phaseG_unified_truth.json` | Canonical first-divergence authority. |

## Tier 3 — Historical / supporting evidence (cite as history, not headline)

| Artifact | Role |
|----------|------|
| `reports/phaseA_benchmark.json` | 336-cell cross-model panel — **superseded** as public headline by scale_7b. |
| `reports/experiment_0*.json` / `.csv` | V-series experiment outputs (research arc). |
| `docs/EXPERIMENT_*.md`, `docs/V*_SCOPE_STATEMENT.md` | Version-arc narrative + scope. |
| `docs/HISTORICAL_ARTIFACT_INVENTORY.md` | 1,176 historical artifacts catalogue. |
| `reports/public_release/demo_cards.*` | Release + historical demo cards (illustrative). |

## Tier 4 — Exploratory / no-go probes (claim-boundary evidence ONLY)

| Artifact | Role |
|----------|------|
| `docs/EXPERIMENT_059…065_*VLLM*.md` | vLLM feasibility — **no integration claim**. |
| `docs/LMCACHE_PROTOTYPE_PATH.md`, `docs/VLLM_PROTOTYPE_PATH.md` | Prototype paths — not shipped. |
| Serving sidecar / shadow-observer / L4 dry-run docs | Runtime probes — **no production / no-commit** claims. |
| GPU-memory / timing pilots (Exp 018, 027, 030, 031, 057–058) | Establish forbidden memory/speed claims. |

**These artifacts must NOT be used as public performance claims.** They exist to
define what cannot be claimed.

## Tier 5 — Launch materials (derived, not evidence)

| Artifact | Role |
|----------|------|
| `paper/`, `site/`, `launch/`, `release_synthesis/` | This synthesis package (derived from Tiers 1–2). |
| `docs/launch_*_final.md`, `docs/EXACTKV_TECHNICAL_REPORT.md` | Public narrative (claim-safe, evidence-mapped). |
| `RELEASE.md`, `reports/public_release/README_PUBLIC.md` | Release entry points. |

## One-line rules

- **Benchmark source of truth:** `reports/scale_7b/raw.json`.
- **Technical report support:** Tier 1 + Tier 2.
- **Historical only:** Tier 3.
- **Never a public performance claim:** Tier 4.
