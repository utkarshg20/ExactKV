"""Public release artifact consistency validator (Phase J)."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

FORBIDDEN_PHRASES = (
    r"nothing\s+like\s+this\s+exists",
    r"\bfirst\s+ever\b",
    r"\bproduction[- ]ready\b",
    r"\bactive\s+gpu\s+memory\s+savings\b",
    r"\bend[- ]to[- ]end\s+speedup\b",
    r"\bbeats\s+vericache\b",
    r"\breproduces\s+vericache\b",
)

REQUIRED_MANIFEST_SOURCES = (
    "reports/scale_7b/raw.json",
    "reports/scale_7b/leaderboard.json",
    "reports/scale_7b/scale_summary.json",
    "reports/phaseF_kernel_benchmark.json",
    "reports/phaseG_unified_truth.json",
    "reports/novelty_audit.json",
    "reports/release_evidence_status.json",
)

STALE_HEADLINE = re.compile(
    r"(?:total\s+benchmark\s+cells|^\*\*cells:\*\*)\s*[:*]*\s*336\b",
    re.I | re.M,
)
SCALE_HEADLINE = re.compile(r"(\b1500\b.*\bcell|\bcell.*\b1500\b)", re.I)


@dataclass
class ValidationIssue:
    severity: str  # error | warning
    check: str
    detail: str = ""


@dataclass
class PublicReleaseValidationReport:
    status: str = "pending"
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "pass" if self.valid else "fail",
            "issues": [asdict(i) for i in self.issues],
        }


def _add(report: PublicReleaseValidationReport, check: str, ok: bool, detail: str = "", *, warning: bool = False) -> None:
    if not ok:
        report.issues.append(
            ValidationIssue(
                severity="warning" if warning else "error",
                check=check,
                detail=detail,
            )
        )


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _line_negated(line: str, match_start: int) -> bool:
    window = line[max(0, match_start - 60) : match_start]
    return bool(re.search(r"\b(not|no|never|forbidden|without)\b", window, re.I))


def validate_public_release(root: Path | str = ".") -> PublicReleaseValidationReport:
    root = Path(root)
    report = PublicReleaseValidationReport()
    release = root / "reports/public_release"
    readme = _read_text(release / "README_PUBLIC.md")
    summary = _read_text(release / "benchmark_summary.md")
    methodology = _read_text(release / "methodology.md")
    manifest_path = release / "release_manifest.json"
    lb_path = release / "leaderboard_final.json"

    # 1500-cell authoritative evidence
    _add(report, "readme_mentions_1500_cells", bool(SCALE_HEADLINE.search(readme)), "README_PUBLIC.md")
    _add(report, "summary_mentions_1500_cells", bool(SCALE_HEADLINE.search(summary)), "benchmark_summary.md")
    if STALE_HEADLINE.search(readme) and not re.search(r"phase\s*a|historical|internal", readme, re.I):
        _add(report, "readme_no_stale_336_headline", False, "336 cells presented as final public benchmark")
    if re.search(r"^\*\*cells:\*\*\s*336\b", summary, re.I | re.M):
        _add(report, "summary_no_stale_336_headline", False, "benchmark_summary.md Cells: 336")

    # Manifest sources
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = manifest.get("source_artifacts") or []
    for src in REQUIRED_MANIFEST_SOURCES:
        _add(report, f"manifest_source_{Path(src).name}", src in sources, src)
        _add(report, f"source_exists_{Path(src).name}", (root / src).is_file(), src, warning=True)

    # Methodology caveats
    meth_lower = methodology.lower()
    _add(report, "methodology_spectralquant_fallback", "fallback" in meth_lower or "proxy" in meth_lower)
    _add(report, "methodology_shard_probe", "probe" in meth_lower)
    if re.search(r"speedup|phase\s*f", methodology, re.I):
        _add(report, "methodology_kernel_microbenchmark_caveat", "kernel microbenchmark" in meth_lower)
    if re.search(r"compression[_ ]ratio", methodology, re.I):
        _add(report, "methodology_byte_ratio_caveat", "stored tensor" in meth_lower or "byte ratio" in meth_lower)
    if re.search(r"vericache", methodology, re.I):
        _add(report, "methodology_vericache_caveat", "not reproduce" in meth_lower or "does not reproduce" in meth_lower)

    # Forbidden phrases (line-level negation)
    for rel in ("README_PUBLIC.md", "benchmark_summary.md", "methodology.md"):
        text = _read_text(release / rel)
        for line in text.splitlines():
            for pattern in FORBIDDEN_PHRASES:
                for match in re.finditer(pattern, line, re.I):
                    if not _line_negated(line, match.start()):
                        _add(report, f"forbidden_phrase_{rel}", False, f"{match.group(0)!r} in {rel}")
                        break

    # Leaderboard JSON
    _add(report, "leaderboard_exists", lb_path.is_file())
    if lb_path.is_file():
        try:
            lb = json.loads(lb_path.read_text(encoding="utf-8"))
            entries = lb.get("entries") or []
            _add(report, "leaderboard_parseable", True)
            _add(report, "leaderboard_has_entries", bool(entries))
            if entries:
                sample = entries[0]
                _add(report, "leaderboard_compressor_field", "compressor" in sample)
                _add(report, "leaderboard_model_field", "model" in sample or "model_short" in sample)
        except json.JSONDecodeError as exc:
            _add(report, "leaderboard_parseable", False, str(exc))

    # Release Gate R1: raw cells must appear as scored leaderboard rows.
    raw_path = root / "reports/scale_7b/raw.json"
    if raw_path.is_file() and lb_path.is_file():
        try:
            from exactkv.platform.leaderboard_aggregates import validate_leaderboard_against_raw  # noqa: PLC0415

            phase_a = json.loads(raw_path.read_text(encoding="utf-8"))
            lb = json.loads(lb_path.read_text(encoding="utf-8"))
            lb_errors = validate_leaderboard_against_raw(phase_a, lb)
            _add(report, "leaderboard_covers_raw_models", not lb_errors, "; ".join(lb_errors[:3]))
            mistral_cells = sum(
                1 for c in (phase_a.get("cells") or []) if "mistral" in str(c.get("model_name", "")).lower()
            )
            if mistral_cells > 0:
                mistral_scored = [
                    e for e in (lb.get("entries") or [])
                    if "mistral" in str(e.get("model", "")).lower()
                    and e.get("score") is not None
                    and e.get("availability") != "unavailable"
                ]
                _add(
                    report,
                    "mistral_numeric_leaderboard_rows",
                    bool(mistral_scored),
                    f"{mistral_cells} raw Mistral cells",
                )
        except (json.JSONDecodeError, OSError) as exc:
            _add(report, "leaderboard_raw_consistency", False, str(exc))

    # Stale Mistral unavailable wording in public summary
    if re.search(r"mistral.*unavailable|unavailable.*mistral", summary, re.I):
        if not re.search(r"historical|phase\s*a|previously|fixed", summary, re.I):
            _add(report, "no_stale_mistral_unavailable_wording", False, "benchmark_summary.md")

    gen_at = manifest.get("generated_at")
    if gen_at and (root / "reports/scale_7b/raw.json").is_file():
        raw_mtime = (root / "reports/scale_7b/raw.json").stat().st_mtime
        try:
            from datetime import datetime

            gen_ts = datetime.fromisoformat(str(gen_at).replace("Z", "+00:00")).timestamp()
            if gen_ts < raw_mtime - 60:
                _add(
                    report,
                    "release_newer_than_scale_raw",
                    False,
                    "public release older than scale_7b/raw.json",
                    warning=True,
                )
        except (ValueError, OSError):
            _add(report, "timestamp_comparison", True, "timestamps unavailable", warning=True)

    report.status = "pass" if report.valid else "fail"
    return report
