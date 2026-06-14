#!/usr/bin/env python3
"""Experiment 042: SpectralQuant external probe feasibility (Phase 10D).

Restricted feasibility only — NOT default registry, NOT vendored SpectralQuant.
Tensor smoke is not ExactKV generation. No speed/memory/serving claims.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.external.spectralquant_probe import (  # noqa: E402
    assess_model_probe_feasibility,
    blocked_report,
    build_report,
    classify_api,
    inspect_repo_layout,
    resolve_spectralquant_repo_path,
    run_tensor_smoke,
    try_import_spectralquant,
)

DEFAULT_JSON = _ROOT / "reports" / "experiment_042_spectralquant_probe.json"


def run_probe_job(
    *,
    try_import: bool,
    try_tensor_smoke: bool,
    try_model_probe: bool,
) -> dict[str, Any]:
    repo_path = resolve_spectralquant_repo_path()
    generated_at = datetime.now(timezone.utc).isoformat()

    if repo_path is None:
        report = blocked_report(
            reason="blocked: SpectralQuant repo not provided (set SPECTRALQUANT_REPO_PATH)",
            repo_path_present=False,
        )
        report["generated_at"] = generated_at
        return report

    layout = inspect_repo_layout(repo_path)
    discovered = {
        "repo_layout": layout,
        "public_repo": "https://github.com/Dynamis-Labs/spectralquant",
    }

    if not try_import:
        report = build_report(
            probe_status="blocked",
            blocked_reason="blocked: import check not executed (pass --try-import)",
            repo_path_present=True,
            import_success=False,
            dependency_blocker="",
            discovered_api_summary=discovered,
            classification={},
            tensor_smoke_result=None,
            model_probe_result=None,
            exactkv_failures=None,
            limitations=[
                "Static layout inspection only without --try-import.",
                "SpectralQuant requires torch, transformers, scipy, scikit-learn.",
            ],
            notes=[
                f"Discovered src modules: {layout.get('src_modules', [])}",
                "Pass --try-import to attempt package import.",
            ],
            recommendation="blocked",
            repo_path=str(repo_path),
        )
        report["generated_at"] = generated_at
        return report

    import_result = try_import_spectralquant(repo_path)
    discovered["import_modules"] = list(import_result.modules)
    discovered["public_symbols"] = list(import_result.public_symbols)

    if not import_result.success:
        report = build_report(
            probe_status="import_only" if layout.get("src_modules") else "restricted_no_go",
            blocked_reason=f"blocked: {import_result.reason}",
            repo_path_present=True,
            import_success=False,
            dependency_blocker=import_result.reason,
            discovered_api_summary=discovered,
            classification=classify_api(layout=layout, import_result=import_result),
            tensor_smoke_result=None,
            model_probe_result=None,
            exactkv_failures=None,
            limitations=[
                "Install SpectralQuant deps: pip install -e SPECTRALQUANT_REPO_PATH",
                "Optional: clone baseline/turboquant_cutile for kernel engine only.",
            ],
            notes=[import_result.reason],
            recommendation="blocked",
            repo_path=str(repo_path),
        )
        report["generated_at"] = generated_at
        return report

    classification = classify_api(layout=layout, import_result=import_result)

    tensor_result: dict[str, Any] | None = None
    if try_tensor_smoke:
        tensor_result = run_tensor_smoke(repo_path)

    model_result: dict[str, Any] | None = None
    exactkv_failures: int | None = None
    if try_model_probe:
        model_result = assess_model_probe_feasibility(
            classification=classification,
            import_result=import_result,
        )
        # No model probe ran — do not fabricate exactkv_failures.
    else:
        model_result = assess_model_probe_feasibility(
            classification=classification,
            import_result=import_result,
        )

    if tensor_result and tensor_result.get("status") == "pass":
        probe_status = "tensor_smoke_only"
        recommendation = "tensor_smoke_only"
        blocked_reason = ""
    elif import_result.success:
        probe_status = "import_only"
        recommendation = "import_only"
        blocked_reason = ""
    else:
        probe_status = "restricted_no_go"
        recommendation = "restricted_no_go"
        blocked_reason = import_result.reason

    if model_result and model_result.get("status") == "restricted_no_go":
        recommendation = "tensor_smoke_only" if probe_status == "tensor_smoke_only" else "restricted_no_go"

    limitations = [
        "External clone only — SpectralQuant is not vendored into ExactKV.",
        "Tensor smoke uses synthetic K/V tensors — not HF model generation.",
        "Full SpectralQuant evaluation requires calibration on real model forwards.",
        "KernelSpectralQuantEngine needs optional turboquant_cutile baseline.",
        "Paper/results/ JSON under SpectralQuant repo are external — not ExactKV results.",
    ]

    notes = [
        "SpectralQuant exposes pure-Python SpectralQuantEngine for per-layer K/V tensors.",
        "EigenspectralCalibrator collects statistics via model forward hooks.",
        "No HF past_key_values drop-in for ExactKV draft loop without new adapter work.",
    ]
    if tensor_result and tensor_result.get("status") == "pass":
        notes.append(
            f"Tensor smoke: key max abs err={tensor_result.get('key_max_abs_error'):.4f}, "
            f"value max abs err={tensor_result.get('value_max_abs_error'):.4f}"
        )

    report = build_report(
        probe_status=probe_status,
        blocked_reason=blocked_reason,
        repo_path_present=True,
        import_success=True,
        dependency_blocker="",
        discovered_api_summary=discovered,
        classification=classification,
        tensor_smoke_result=tensor_result,
        model_probe_result=model_result,
        exactkv_failures=exactkv_failures,
        limitations=limitations,
        notes=notes,
        recommendation=recommendation,
        repo_path=str(repo_path),
    )
    report["generated_at"] = generated_at
    return report


def write_json_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 042 — SpectralQuant probe")
    parser.add_argument("--try-import", action="store_true", help="Attempt spectralquant import")
    parser.add_argument("--try-tensor-smoke", action="store_true", help="Run synthetic K/V tensor smoke")
    parser.add_argument("--try-model-probe", action="store_true", help="Assess model probe feasibility")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args()

    try_tensor = args.try_tensor_smoke or args.try_model_probe
    try_import = args.try_import or try_tensor

    report = run_probe_job(
        try_import=try_import,
        try_tensor_smoke=args.try_tensor_smoke,
        try_model_probe=args.try_model_probe or try_tensor,
    )
    write_json_report(report, args.json_out)

    status = report["probe_status"]
    print(f"SpectralQuant probe: {status}")
    reason = report.get("blocked_reason") or ""
    if reason:
        print(reason)
    if report.get("import_success"):
        cats = (report.get("classification") or {}).get("categories") or []
        print(f"api_categories={cats}")
    ts = report.get("tensor_smoke_result") or {}
    if ts.get("status") == "pass":
        print(f"tensor_smoke=pass key_max_err={ts.get('key_max_abs_error')}")
    mp = report.get("model_probe_result") or {}
    if mp:
        print(f"model_probe={mp.get('status')} attempted={mp.get('attempted')}")
    print(f"recommendation={report.get('recommendation')}")
    print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
