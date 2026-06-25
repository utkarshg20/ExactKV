#!/usr/bin/env python3
"""Gate R0 — release evidence integrity checker."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.platform.evidence_integrity import (  # noqa: E402
    validate_release_evidence,
)

DEFAULT_JSON_OUT = _ROOT / "reports" / "release_evidence_status.json"
DEFAULT_MD_OUT = _ROOT / "docs" / "RELEASE_EVIDENCE_STATUS.md"


def _render_markdown(report_dict: dict) -> str:
    scale = report_dict.get("scale_summary") or {}
    pf = report_dict.get("phase_f_summary") or {}
    adapter = report_dict.get("adapter_honesty") or {}
    lines = [
        "# Release Evidence Status (Gate R0)",
        "",
        f"**Status:** {report_dict.get('status', 'unknown').upper()}",
        "",
        "## Evidence-complete",
        "",
        "- Phase A 336-cell multi-model benchmark (`reports/phaseA_benchmark.json`)",
        "- Phase D Llama runtime probe (`reports/phaseD_runtime_probe.json`)",
        "- Phase F CUDA/Triton kernel microbenchmark (`reports/phaseF_kernel_benchmark.json`)",
        "- Phase G unified truth engine (`reports/phaseG_unified_truth.json`)",
        "- Phase H platform layer (`reports/benchmark.json`)",
        "- Phase H+ real 7B/8B scale benchmark (`reports/scale_7b/raw.json`)",
        "",
        "## Qualified / disclosure required",
        "",
        f"- SpectralQuant: `{adapter.get('spectralquant_real', {}).get('mode', 'unknown')}`",
        f"- Shard: `{adapter.get('shard_real', {}).get('mode', 'unknown')}` (probe-only)",
        "- Phase F speedups are **kernel microbenchmarks only** (not end-to-end inference)",
        "- Compression ratios are stored tensor byte ratios unless active GPU memory was measured",
        "- ExactKV is a research-grade evaluation framework, not a production serving runtime",
        "",
        "## Current headline evidence",
        "",
        f"- Scale cells: {scale.get('total_cells', 'n/a')}",
        f"- Models: {', '.join(scale.get('models') or [])}",
        f"- ExactKV failures: {scale.get('exactkv_failures', scale.get('exactkv_failures', 'n/a'))}",
        f"- Deterministic mode: {scale.get('deterministic_mode')}",
        f"- Phase F INT8 kernel speedup: {pf.get('int8_speedup_x')}x",
        f"- Phase F INT4 kernel speedup: {pf.get('int4_speedup_x')}x",
        f"- block_sparse execution backend: {pf.get('block_sparse_execution_backend')}",
        "",
        "## Validation checks",
        "",
        "| Check | Pass | Detail |",
        "|-------|------|--------|",
    ]
    for c in report_dict.get("checks") or []:
        mark = "yes" if c.get("passed") else "no"
        lines.append(f"| {c.get('name')} | {mark} | {c.get('detail', '')[:80]} |")
    if report_dict.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for w in report_dict["warnings"]:
            lines.append(f"- {w}")
    lines.extend([
        "",
        "## Remaining release blockers",
        "",
        "- Full `pytest` must pass",
        "- Novelty audit must be complete before public claims are finalized",
        "- Token/secret scan must pass before publishing",
        "- Final README/public posts must use claim-safe language",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate release evidence artifacts")
    parser.add_argument("--root", type=Path, default=_ROOT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    report = validate_release_evidence(args.root)
    data = report.to_dict()

    print(f"Release evidence status: {data['status'].upper()}")
    for c in report.checks:
        mark = "PASS" if c.passed else c.severity.upper()
        detail = f" — {c.detail}" if c.detail else ""
        print(f"  [{mark}] {c.name}{detail}")
    for w in report.warnings:
        print(f"  [WARN] {w}")

    if not args.no_write:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        args.md_out.write_text(_render_markdown(data), encoding="utf-8")
        print(f"\nWrote {args.json_out}")
        print(f"Wrote {args.md_out}")

    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
