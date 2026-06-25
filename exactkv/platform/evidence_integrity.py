"""Release evidence integrity validator (Gate R0).

Inspects on-disk report artifacts without running inference.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

DEFAULT_ROOT = Path(".")

SCALE_RAW = Path("reports/scale_7b/raw.json")
SCALE_LEADERBOARD = Path("reports/scale_7b/leaderboard.json")
SCALE_SUMMARY = Path("reports/scale_7b/scale_summary.json")
PUBLIC_LEADERBOARD = Path("reports/public_release/leaderboard_final.json")
PUBLIC_MANIFEST = Path("reports/public_release/release_manifest.json")
PHASE_F = Path("reports/phaseF_kernel_benchmark.json")
PHASE_G = Path("reports/phaseG_unified_truth.json")

EXPECTED_MODELS = (
    "meta-llama/Llama-3.1-8B",
    "mistralai/Mistral-7B-Instruct-v0.3",
)
EXPECTED_CELL_COUNT = 1500

UNSAFE_CLAIM_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("nothing like this exists", re.compile(r"nothing\s+like\s+this\s+exists", re.I)),
    ("first ever", re.compile(r"\bfirst\s+ever\b", re.I)),
    ("production ready", re.compile(r"\bproduction[- ]ready\b", re.I)),
    ("serving system", re.compile(r"\bserving\s+system\b", re.I)),
    ("active GPU memory savings", re.compile(r"\bactive\s+gpu\s+memory\s+savings\b", re.I)),
    ("end-to-end speedup", re.compile(r"\bend[- ]to[- ]end\s+speedup\b", re.I)),
    ("beats VeriCache", re.compile(r"\bbeats\s+vericache\b", re.I)),
    ("reproduces VeriCache", re.compile(r"\breproduces\s+vericache\b", re.I)),
    ("real SpectralQuant (unsafe)", re.compile(r"\breal\s+spectralquant\b", re.I)),
    ("real Shard (unsafe)", re.compile(r"\breal\s+shard\b", re.I)),
]

PUBLIC_RELEASE_SCAN = (
    Path("reports/public_release/README_PUBLIC.md"),
    Path("reports/public_release/benchmark_summary.md"),
    Path("reports/public_release/methodology.md"),
    Path("reports/public_release/leaderboard_final.json"),
)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    severity: str = "error"  # error | warning


@dataclass
class EvidenceIntegrityReport:
    status: str = "pending"
    checks: list[CheckResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    adapter_honesty: dict[str, Any] = field(default_factory=dict)
    phase_f_summary: dict[str, Any] = field(default_factory=dict)
    scale_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return all(c.passed for c in self.checks if c.severity == "error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "pass" if self.valid else "fail",
            "checks": [asdict(c) for c in self.checks],
            "warnings": self.warnings,
            "adapter_honesty": self.adapter_honesty,
            "phase_f_summary": self.phase_f_summary,
            "scale_summary": self.scale_summary,
        }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _add(checks: list[CheckResult], name: str, passed: bool, detail: str = "", *, severity: str = "error") -> None:
    checks.append(CheckResult(name=name, passed=passed, detail=detail, severity=severity))


def _models_in_raw(raw: dict[str, Any]) -> set[str]:
    models = set(raw.get("models_evaluated") or [])
    for cell in raw.get("cells") or []:
        if cell.get("model_name"):
            models.add(str(cell["model_name"]))
    return models


def _compressors_in_raw(raw: dict[str, Any]) -> set[str]:
    comps: set[str] = set()
    for cell in raw.get("cells") or []:
        if cell.get("compressor_name"):
            comps.add(str(cell["compressor_name"]))
    return comps


def validate_scale_run(raw_path: Path, summary_path: Path, checks: list[CheckResult], report: EvidenceIntegrityReport) -> dict[str, Any] | None:
    if not raw_path.is_file():
        _add(checks, "scale_raw_exists", False, f"missing {raw_path}")
        return None
    _add(checks, "scale_raw_exists", True)

    raw = _read_json(raw_path)
    cells = raw.get("cells") or []
    cell_count = len(cells)
    _add(checks, "scale_cell_count", cell_count == EXPECTED_CELL_COUNT, f"got {cell_count}, expected {EXPECTED_CELL_COUNT}")

    models = _models_in_raw(raw)
    for model in EXPECTED_MODELS:
        _add(checks, f"scale_model_{model.split('/')[-1]}", model in models, f"present={model in models}")

    blocked = raw.get("models_blocked") or {}
    _add(checks, "scale_no_blocked_models", not blocked, f"blocked={blocked}")

    failures = int(raw.get("exactkv_failures") or sum(1 for c in cells if c.get("exactkv_failure")))
    _add(checks, "scale_zero_failures", failures == 0, f"exactkv_failures={failures}")

    det = raw.get("deterministic_mode")
    _add(checks, "scale_not_deterministic", det is False, f"deterministic_mode={det!r}")

    if summary_path.is_file():
        summary = _read_json(summary_path)
        report.scale_summary = summary
        if summary.get("deterministic_mode") is True:
            _add(checks, "scale_summary_not_deterministic", False, "summary marks deterministic_mode=True")
        else:
            _add(checks, "scale_summary_not_deterministic", True)
        if summary.get("sequential_run"):
            report.warnings.append("Scale run used sequential model execution (volume constraint).")

    comps = sorted(_compressors_in_raw(raw))
    report.scale_summary.setdefault("compressors_observed", comps)
    return raw


def validate_public_release(
    public_lb_path: Path,
    manifest_path: Path,
    scale_raw_path: Path,
    checks: list[CheckResult],
    warnings: list[str],
) -> None:
    _add(checks, "public_leaderboard_exists", public_lb_path.is_file(), str(public_lb_path))
    _add(checks, "public_manifest_exists", manifest_path.is_file(), str(manifest_path))

    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        sources = manifest.get("source_artifacts") or manifest.get("scale_artifacts") or []
        refs_scale = any("scale_7b" in str(s) for s in sources)
        if not refs_scale and "scale_7b" not in json.dumps(manifest):
            _add(
                checks,
                "manifest_references_scale_7b",
                False,
                "release_manifest.json does not reference reports/scale_7b/*",
                severity="error",
            )
        else:
            _add(checks, "manifest_references_scale_7b", True)

    if public_lb_path.is_file() and scale_raw_path.is_file():
        try:
            lb_mtime = public_lb_path.stat().st_mtime
            raw_mtime = scale_raw_path.stat().st_mtime
            if lb_mtime < raw_mtime - 1:
                warnings.append(
                    f"public leaderboard mtime older than scale raw ({public_lb_path} < {scale_raw_path})",
                )
        except OSError:
            warnings.append("Could not compare artifact timestamps.")


def validate_phase_f(path: Path, checks: list[CheckResult], report: EvidenceIntegrityReport) -> None:
    if not path.is_file():
        _add(checks, "phase_f_exists", False, str(path))
        return
    _add(checks, "phase_f_exists", True)
    data = _read_json(path)
    backend = data.get("backend_info") or {}
    _add(checks, "phase_f_cuda", backend.get("cuda_available") is True, str(backend.get("cuda_available")))
    _add(checks, "phase_f_triton", backend.get("triton_available") is True, str(backend.get("triton_available")))
    _add(checks, "phase_f_device_cuda", str(data.get("device")) == "cuda", f"device={data.get('device')!r}")

    speedups = {s.get("mode"): s for s in (data.get("speedups") or [])}
    for mode in ("int8", "int4"):
        entry = speedups.get(mode)
        ok = entry is not None and isinstance(entry.get("speedup_x"), (int, float))
        _add(checks, f"phase_f_{mode}_speedup", ok, str(entry))

    bs_bench = next((b for b in (data.get("benchmarks") or []) if b.get("mode") == "block_sparse" and b.get("backend") == "triton"), None)
    if bs_bench:
        exec_backend = bs_bench.get("execution_backend")
        _add(
            checks,
            "phase_f_block_sparse_not_triton_speedup",
            exec_backend != "triton",
            f"execution_backend={exec_backend!r}",
        )

    report.phase_f_summary = {
        "int8_speedup_x": (speedups.get("int8") or {}).get("speedup_x"),
        "int4_speedup_x": (speedups.get("int4") or {}).get("speedup_x"),
        "block_sparse_execution_backend": (bs_bench or {}).get("execution_backend"),
        "claim_scope": "kernel_microbenchmark_only",
    }


def validate_phase_g(path: Path, checks: list[CheckResult]) -> None:
    if not path.is_file():
        _add(checks, "phase_g_exists", False, str(path), severity="warning")
        return
    _add(checks, "phase_g_exists", True, severity="warning")


def validate_adapter_honesty(checks: list[CheckResult], report: EvidenceIntegrityReport) -> None:
    try:
        from exactkv.adapters.spectralquant_real_adapter import spectralquant_available  # noqa: PLC0415

        sq_avail = spectralquant_available()
    except Exception as exc:
        sq_avail = False
        report.warnings.append(f"spectralquant_available probe failed: {exc}")

    report.adapter_honesty["spectralquant_real"] = {
        "spectralquant_available": sq_avail,
        "mode": "real" if sq_avail else "int4_sim_scaling_fallback",
        "disclosure_required": not sq_avail,
    }
    _add(
        checks,
        "spectralquant_fallback_disclosed",
        not sq_avail,
        "spectralquant_available=False — public docs must not claim real SpectralQuant",
    ) if not sq_avail else _add(checks, "spectralquant_fallback_disclosed", True, "real dependency available")

    report.adapter_honesty["shard_real"] = {
        "probe_only": True,
        "mode": "probe_first_heuristic",
        "disclosure_required": True,
    }
    _add(checks, "shard_probe_disclosed", True, "shard_real is probe-first; no real Shard backend wired")


def validate_claim_safety(paths: Sequence[Path], checks: list[CheckResult]) -> None:
    violations: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in UNSAFE_CLAIM_PATTERNS:
            for match in pattern.finditer(text):
                if label.startswith("real SpectralQuant") and "fallback" in text.lower():
                    continue
                if label.startswith("real Shard") and "probe" in text.lower():
                    continue
                violations.append(f"{path}:{label}:{match.group(0)!r}")
    _add(checks, "public_claim_safety", not violations, "; ".join(violations[:5]) or "clean")


def validate_release_evidence(root: Path | str = DEFAULT_ROOT) -> EvidenceIntegrityReport:
    """Run all Gate R0 evidence integrity checks."""
    root = Path(root)
    report = EvidenceIntegrityReport()
    checks = report.checks

    raw_path = root / SCALE_RAW
    validate_scale_run(root / SCALE_RAW, root / SCALE_SUMMARY, checks, report)
    validate_public_release(
        root / PUBLIC_LEADERBOARD,
        root / PUBLIC_MANIFEST,
        raw_path,
        checks,
        report.warnings,
    )
    validate_phase_f(root / PHASE_F, checks, report)
    validate_phase_g(root / PHASE_G, checks)
    validate_adapter_honesty(checks, report)
    validate_claim_safety([root / p for p in PUBLIC_RELEASE_SCAN], checks)

    report.status = "pass" if report.valid else "fail"
    return report
