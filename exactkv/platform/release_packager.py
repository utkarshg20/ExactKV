"""Public release package builder (Phase H+ / Phase J)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_RELEASE_DIR = Path("reports/public_release")

MANIFEST_SOURCE_ARTIFACTS = (
    "reports/scale_7b/raw.json",
    "reports/scale_7b/leaderboard.json",
    "reports/scale_7b/scale_summary.json",
    "reports/phaseF_kernel_benchmark.json",
    "reports/phaseG_unified_truth.json",
    "reports/novelty_audit.json",
    "reports/release_evidence_status.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_release_package(
    *,
    release_dir: Path | str = DEFAULT_RELEASE_DIR,
    benchmark_path: Path | str = Path("reports/benchmark.json"),
    leaderboard_path: Path | str = Path("reports/leaderboard.json"),
    phase_a_path: Path | str = Path("reports/phaseA_benchmark.json"),
    scale_path: Path | str = Path("reports/scale_7b/scale_summary.json"),
) -> dict[str, Any]:
    """Generate public launch artifacts from existing reports."""
    out_dir = Path(release_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_a = _read_json(Path(phase_a_path))
    scale = _read_json(Path(scale_path))
    scale_lb_path = Path("reports/scale_7b/leaderboard.json")
    scale_raw_path = Path("reports/scale_7b/raw.json")

    scale_authoritative = bool(scale.get("total_cells")) and scale.get("deterministic_mode") is False

    # Release Gate R1: rebuild scale leaderboard from raw cells when present.
    if scale_authoritative and scale_raw_path.is_file():
        from exactkv.platform.leaderboard_aggregates import rebuild_scale_leaderboard_from_raw  # noqa: PLC0415

        rebuild_scale_leaderboard_from_raw(scale_raw_path, write_raw_repairs=True)

    leaderboard_final = _read_json(scale_lb_path) if scale_authoritative else _read_json(Path(leaderboard_path))
    if leaderboard_final:
        (out_dir / "leaderboard_final.json").write_text(
            json.dumps(leaderboard_final, indent=2) + "\n",
        )

    phase_a_cells = phase_a.get("total_cells") or 0
    if scale_authoritative:
        total_cells = scale["total_cells"]
        models = scale.get("models") or []
        deterministic = scale.get("deterministic_mode", False)
        failures = scale.get("exactkv_failures", 0)
        evidence_label = "Phase H+ scale_7b (authoritative public release)"
    else:
        benchmark = _read_json(Path(benchmark_path))
        total_cells = benchmark.get("benchmark_run", {}).get("total_cells") or phase_a_cells or 0
        models = phase_a.get("models_evaluated") or []
        deterministic = phase_a.get("deterministic_mode")
        failures = phase_a.get("exactkv_failures", 0)
        evidence_label = "Phase A (fallback — scale_7b not available)"

    historical_note = ""
    if scale_authoritative and phase_a_cells:
        historical_note = (
            f"\n- **Historical Phase A panel (internal):** {phase_a_cells} cells — "
            "supporting cross-model evidence, not the final public release benchmark.\n"
        )

    readme = f"""# ExactKV Public Benchmark Release

ExactKV is a compressor-agnostic crash-test and leaderboard framework for LLM
KV-cache compression. It measures token-level drift, first divergence, acceptance
rate, verifier agreement, and exactness failures across compressors and models.

ExactKV grew through a long verifier-first research arc (**V1–V21**) before the formal release
phases. See [`docs/PROJECT_LINEAGE.md`](../../docs/PROJECT_LINEAGE.md),
[`docs/VERSION_LINEAGE.md`](../../docs/VERSION_LINEAGE.md), and
[`docs/HISTORICAL_ARTIFACT_INVENTORY.md`](../../docs/HISTORICAL_ARTIFACT_INVENTORY.md).

**Not a production serving system.** ExactKV does not reproduce VeriCache serving throughput.

## Quick start

```bash
python3 scripts/exactkv_repro.py --reports-only
python3 scripts/exactkv_repro.py --release-check
python3 scripts/build_launch_pack.py
python3 scripts/exactkv.py run publish
```

## Public release benchmark (authoritative)

- **Evidence track:** {evidence_label}
- **Total benchmark cells:** {total_cells}
- **Models evaluated:** {', '.join(models) if models else 'see reports/scale_7b/scale_summary.json'}
- **ExactKV failures:** {failures}
- **Deterministic mode:** {deterministic}
- **Divergence authority:** Phase G `FirstDivergenceAuthority` (canonical)
{historical_note}
## Artifacts

| File | Description |
|------|-------------|
| `leaderboard_final.json` | Ranked compressor × model scores (from scale_7b when available) |
| `benchmark_summary.md` | Aggregate metrics |
| `methodology.md` | Evaluation methodology + claim boundaries |
| `demo_cards.json` / `demo_cards.md` | Release + historical demo cards |
| `launch_manifest.json` | Phase K launch pack manifest |
| `repro_command.sh` | Reports-only reproduction |
| `release_manifest.json` | Source artifact pointers |

## Documentation links

- Technical report: [`docs/EXACTKV_TECHNICAL_REPORT.md`](../../docs/EXACTKV_TECHNICAL_REPORT.md)
- Project lineage: [`docs/PROJECT_LINEAGE.md`](../../docs/PROJECT_LINEAGE.md)
- Version lineage (V1–V21): [`docs/VERSION_LINEAGE.md`](../../docs/VERSION_LINEAGE.md)
- Novelty audit: [`docs/NOVELTY_AUDIT.md`](../../docs/NOVELTY_AUDIT.md)
- Claim boundaries: [`docs/CLAIM_BOUNDARIES.md`](../../docs/CLAIM_BOUNDARIES.md)
- Metric definitions: [`docs/METRIC_DEFINITIONS.md`](../../docs/METRIC_DEFINITIONS.md)
- Reproducibility: [`docs/REPRODUCIBILITY.md`](../../docs/REPRODUCIBILITY.md)
- Demo cards: [`demo_cards.md`](demo_cards.md)

## Claims policy

No end-to-end speedup, latency, or active GPU memory savings claims. Phase F results (when cited) are kernel microbenchmark only. Compression ratios are stored tensor byte ratios unless active GPU memory is explicitly measured. SpectralQuant: fallback/proxy when dependency unavailable. Shard: probe-first analysis only. Scale run used sequential model execution (volume constraint).

Generated: {datetime.now(timezone.utc).isoformat()}
"""
    (out_dir / "README_PUBLIC.md").write_text(readme)

    summary_md = f"""# Benchmark Summary

**Public release cells:** {total_cells} (Phase H+ scale_7b real GPU benchmark)
**ExactKV failures:** {failures}
"""
    if phase_a_cells and scale_authoritative:
        summary_md += f"**Historical Phase A cells (internal):** {phase_a_cells}\n"

    summary_md += "\n## Top leaderboard entries\n\n"
    for row in (leaderboard_final.get("entries") or [])[:10]:
        if row.get("availability") == "unavailable":
            continue
        summary_md += (
            f"- `{row.get('compressor')}` on {row.get('model_short')} — "
            f"score {row.get('score')} acceptance {row.get('acceptance_rate')}\n"
        )

    summary_md += """
## Claim boundaries

SpectralQuant rows use **fallback/proxy** mode when the real dependency is unavailable. Shard rows are **probe-first** heuristic analysis, not a full Shard integration. Compression ratios in source reports are **stored tensor byte ratios** unless active GPU memory is explicitly measured.
"""
    (out_dir / "benchmark_summary.md").write_text(summary_md)

    methodology = """# ExactKV Evaluation Methodology

## Public release evidence

The authoritative public release benchmark is **Phase H+ scale_7b** (`reports/scale_7b/raw.json`, 1500 cells, real GPU, `deterministic_mode=false`). Phase A (336 cells) remains internal/historical supporting evidence.

## Divergence (canonical)

All divergence metrics use Phase G `FirstDivergenceAuthority`:
- `canonical_first_divergence_index`
- Types: `token_mismatch`, `length_drift`, `kernel_inconsistency`, `verifier_disagreement`, `none`

## Acceptance

Token-level ExactKV speculative decoding acceptance rate from benchmark cells.

## Leaderboard scoring (locked)

```
score = 0.35 * acceptance_rate
      + 0.25 * verifier_agreement
      + 0.20 * (1 - normalized_first_divergence)
      + 0.10 * (1 - failure_rate)
      + 0.10 * stability_score
```

## Compressors

Built-in: noop, int8, int4_sim, k8_v4_sim
Phase H+ scale panel: noop, int8, int4_sim, spectralquant (fallback/proxy), shard (probe-first)
External adapters: spectralquant_real (fallback mode when dependency missing), shard_real (probe-first heuristic)

## Claim boundaries (Phase I/J)

- ExactKV is **not a production serving system**.
- Phase F speedups (when cited) are **kernel microbenchmark** results only — **not end-to-end** inference speedups.
- Compression ratios are **stored tensor byte ratios** unless active GPU memory is explicitly measured.
- **SpectralQuant** uses **fallback/proxy** mode when the real dependency is unavailable.
- **Shard** (`shard_real`) is **probe-first** heuristic analysis, not a full Shard integration.
- ExactKV is inspired by verifier-mediated compressed-KV ideas; it does **not reproduce VeriCache** serving throughput.

## Reproducibility

Regenerate from existing reports without re-inference:

```bash
python3 scripts/exactkv_repro.py --reports-only
```
"""
    (out_dir / "methodology.md").write_text(methodology)

    repro = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 scripts/exactkv_repro.py --reports-only
echo "ExactKV public release bundle complete (reports-only, no inference)."
"""
    repro_path = out_dir / "repro_command.sh"
    repro_path.write_text(repro)
    repro_path.chmod(0o755)

    manifest = {
        "release_dir": str(out_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "authoritative_evidence": "reports/scale_7b/raw.json" if scale_authoritative else "reports/phaseA_benchmark.json",
        "total_cells": total_cells,
        "source_artifacts": list(MANIFEST_SOURCE_ARTIFACTS),
        "files": [
            "README_PUBLIC.md",
            "benchmark_summary.md",
            "methodology.md",
            "repro_command.sh",
            "leaderboard_final.json",
            "release_manifest.json",
        ],
    }
    (out_dir / "release_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    try:
        from exactkv.platform.launch_pack_builder import write_launch_pack  # noqa: PLC0415

        write_launch_pack(out_dir.parent.parent)
        manifest["files"].extend(["demo_cards.json", "demo_cards.md", "launch_manifest.json"])
    except OSError:
        pass

    return manifest
