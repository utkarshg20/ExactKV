"""Public release package builder (Phase H+)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

DEFAULT_RELEASE_DIR = Path("reports/public_release")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


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

    benchmark = _read_json(Path(benchmark_path))
    leaderboard = _read_json(Path(leaderboard_path))
    phase_a = _read_json(Path(phase_a_path))
    scale = _read_json(Path(scale_path))
    scale_lb_path = Path("reports/scale_7b/leaderboard.json")
    scale_raw_path = Path("reports/scale_7b/raw.json")

    # Prefer Phase H+ scale artifacts when present (real run).
    if scale.get("total_cells") and scale.get("deterministic_mode") is False:
        leaderboard_final = _read_json(scale_lb_path) or leaderboard
    else:
        leaderboard_final = leaderboard or _read_json(scale_lb_path)
    if leaderboard_final:
        (out_dir / "leaderboard_final.json").write_text(
            json.dumps(leaderboard_final, indent=2) + "\n",
        )

    total_cells = (
        benchmark.get("benchmark_run", {}).get("total_cells")
        or phase_a.get("total_cells")
        or scale.get("total_cells")
        or 0
    )
    models = phase_a.get("models_evaluated") or scale.get("models") or []
    deterministic = phase_a.get("deterministic_mode", benchmark.get("benchmark_run", {}).get("config", {}).get(
        "deterministic_mode",
    ))
    failures = scale.get("exactkv_failures")
    if failures is None:
        failures = phase_a.get("exactkv_failures", 0)

    readme = f"""# ExactKV Public Benchmark Release

ExactKV is a reproducible KV compression benchmarking platform with plugin-based
compressors, standardized evaluation, and public leaderboard generation.

## Quick start

```bash
python scripts/exactkv.py run full --deterministic
python scripts/exactkv.py run publish
```

## Current snapshot

- **Total benchmark cells:** {total_cells}
- **Models evaluated:** {', '.join(models) if models else 'see reports'}
- **Deterministic mode:** {deterministic}
- **Divergence authority:** Phase G `FirstDivergenceAuthority` (canonical)

## Artifacts

| File | Description |
|------|-------------|
| `leaderboard_final.json` | Ranked compressor × model scores |
| `benchmark_summary.md` | Aggregate metrics |
| `methodology.md` | Evaluation methodology |
| `repro_command.sh` | One-command reproduction |

## Claims policy

No speedup, latency, or memory savings claims unless directly measured in Phase F.
Token-level acceptance and divergence metrics only.

Generated: {datetime.now(timezone.utc).isoformat()}
"""
    (out_dir / "README_PUBLIC.md").write_text(readme)

    summary_md = f"""# Benchmark Summary

**Cells:** {total_cells}
**Failure rate:** {failures}

## Top leaderboard entries

"""
    for row in (leaderboard_final.get("entries") or [])[:10]:
        summary_md += (
            f"- `{row.get('compressor')}` on {row.get('model_short')} — "
            f"score {row.get('score')} acceptance {row.get('acceptance_rate')}\n"
        )
    (out_dir / "benchmark_summary.md").write_text(summary_md)

    methodology = """# ExactKV Evaluation Methodology

## Divergence (canonical)

All divergence metrics use Phase G `FirstDivergenceAuthority`:
- `canonical_first_divergence_index`
- Types: `token_mismatch`, `length_drift`, `kernel_inconsistency`, `verifier_disagreement`, `none`

## Acceptance

Token-level ExactKV speculative decoding acceptance rate from Phase A cells.

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
External (adapter/mock): spectralquant (fallback if dependency unavailable), kvquant, shard (probe-first), turboquant
Phase H+ adapters: spectralquant_real (fallback mode when dependency missing), shard_real (probe-first heuristic)

## Reproducibility

All benchmarks are reproducible from disk reports without re-inference when using `--deterministic`.
"""
    (out_dir / "methodology.md").write_text(methodology)

    repro = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python scripts/exactkv.py run benchmark --deterministic
python scripts/exactkv.py run leaderboard
python scripts/exactkv.py run publish
python scripts/exactkv.py plot all
echo "ExactKV public release bundle complete."
"""
    repro_path = out_dir / "repro_command.sh"
    repro_path.write_text(repro)
    repro_path.chmod(0o755)

    manifest = {
        "release_dir": str(out_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_artifacts": [
            str(scale_raw_path),
            str(scale_lb_path),
            str(scale_path),
        ],
        "files": [
            "README_PUBLIC.md",
            "benchmark_summary.md",
            "methodology.md",
            "repro_command.sh",
            "leaderboard_final.json",
        ],
    }
    (out_dir / "release_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
