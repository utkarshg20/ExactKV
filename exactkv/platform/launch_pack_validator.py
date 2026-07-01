"""Phase K launch pack validator."""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

FORBIDDEN_PHRASES = (
    r"\bfirst\s+ever\b",
    r"\bnothing\s+like\s+this\s+exists\b",
    r"\bproduction[- ]ready\b",
    r"(?<!\bnot )(?<!\bnon-)\bserving\s+system\b",
    r"\bactive\s+gpu\s+memory\s+savings\b",
    r"\bend[- ]to[- ]end\s+speedup\b",
    r"\bbeats\s+vericache\b",
    r"\breproduces\s+vericache\b",
    r"\breal\s+spectralquant\b",
    r"\breal\s+shard\b",
    r"\bfastest\b",
    r"\bsota\b",
)

LAUNCH_POST_FILES: tuple[str, ...] = ()

REQUIRED_CAVEATS = (
    ("vericache", ("not reproduce", "does not reproduce")),
    ("serving", ("not a production", "not production", "research-grade")),
    ("kernel microbenchmark", ("kernel microbenchmark", "not end-to-end")),
    ("spectralquant", ("fallback", "proxy")),
    ("shard", ("probe", "probe-first", "heuristic")),
)


@dataclass
class LaunchPackIssue:
    severity: str
    check: str
    detail: str = ""


@dataclass
class LaunchPackValidationReport:
    status: str = "pending"
    issues: list[LaunchPackIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {"status": "pass" if self.valid else "fail", "issues": [asdict(i) for i in self.issues]}


def _add(report: LaunchPackValidationReport, check: str, ok: bool, detail: str = "") -> None:
    if not ok:
        report.issues.append(LaunchPackIssue("error", check, detail))


def _read(root: Path, rel: str) -> str:
    path = root / rel
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _check_forbidden_phrases(report: LaunchPackValidationReport, path: Path, text: str) -> None:
    for line in text.splitlines():
        lower = line.lower()
        if any(x in lower for x in ("forbidden", "not claim", "does not", "no-go", "caveat", "not a production", "not production", "microbenchmark", "not end-to-end", "not active", "does not reproduce", "h2o serving", "vericache [", "published h2o", "throughput-oriented serving", "algorithmic overlap")):
            continue
        for pat in FORBIDDEN_PHRASES:
            m = re.search(pat, line, re.I)
            if not m:
                continue
            before = line[: m.start()]
            if re.search(r"\b(not|no|never|forbidden|without)\b", before[-80:], re.I):
                continue
            _add(report, f"forbidden_phrase_{path.name}", False, f"{m.group()}: {line.strip()[:100]}")
            return


def _check_caveats(report: LaunchPackValidationReport, rel: str, text: str) -> None:
    lower = re.sub(r"\*+", "", text.lower())
    for trigger, required in REQUIRED_CAVEATS:
        if trigger in lower:
            if not any(r in lower for r in required):
                _add(report, f"caveat_{rel}_{trigger}", False, f"missing {trigger} caveat")
    if "vericache" in lower and not any(r in lower for r in ("does not reproduce", "not reproduce", "inspired by")):
        _add(report, f"caveat_{rel}_vericache", False, "missing vericache caveat")


def validate_launch_pack(root: Path | str = ".") -> LaunchPackValidationReport:
    root = Path(root)
    report = LaunchPackValidationReport()

    required = {
        "technical_report": root / "paper/ExactKV_Technical_Report.md",
        "project_lineage": root / "docs/PROJECT_LINEAGE.md",
        "historical_inventory_md": root / "docs/HISTORICAL_ARTIFACT_INVENTORY.md",
        "site": root / "site/index.html",
        "demo_cards_json": root / "reports/public_release/demo_cards.json",
        "demo_cards_md": root / "reports/public_release/demo_cards.md",
        "launch_manifest": root / "reports/public_release/launch_manifest.json",
    }
    for key, path in required.items():
        _add(report, f"exists_{key}", path.is_file(), str(path))

    checklist = _read(root, "docs/RELEASE_CHECKLIST.md").lower()
    _add(report, "checklist_final_signoff", "final launch sign-off" in checklist)

    readme = _read(root, "README.md").lower()
    pub_readme = _read(root, "reports/public_release/README_PUBLIC.md").lower()
    for needle, label in (
        ("exactkv_technical_report.md", "readme_technical_report"),
        ("project_lineage.md", "readme_project_lineage"),
        ("version_lineage.md", "readme_version_lineage"),
        ("historical_artifact_inventory.md", "readme_historical_inventory"),
        ("novelty_audit.md", "readme_novelty_audit"),
        ("claim_boundaries.md", "readme_claim_boundaries"),
        ("metric_definitions.md", "readme_metric_definitions"),
    ):
        _add(report, label, needle in readme, needle)

    for needle in ("exactkv_technical_report.md", "demo_cards", "launch_manifest"):
        _add(report, f"public_readme_{needle}", needle in pub_readme, needle)

    tech = _read(root, "paper/ExactKV_Technical_Report.md").lower()
    if tech:
        _add(report, "tech_lineage_section", "lineage" in tech or "phase a" in tech)
        _add(report, "tech_vericache", "vericache" in tech and ("does not reproduce" in tech or "not reproduce" in tech))
        _add(report, "tech_serving", "not a production serving" in tech or "not production serving" in tech)
        _add(report, "tech_kernel", "kernel microbenchmark" in tech or "microbenchmark" in tech)
        _add(report, "tech_spectralquant", ("fallback" in tech or "proxy" in tech) and "spectralquant" in tech)
        _add(report, "tech_shard", "probe" in tech and "shard" in tech)
        _check_forbidden_phrases(report, required["technical_report"], tech)

    for rel in LAUNCH_POST_FILES:
        text = _read(root, rel)
        if not text:
            continue
        _check_caveats(report, rel, text)
        _check_forbidden_phrases(report, root / rel, text)
        lower = re.sub(r"\*+", "", text.lower())
        has_cells = any(x in lower for x in ("8132", "8,132", "1500", "1,500"))
        _add(report, f"{rel}_cell_count", has_cells, "need 8132 or 1500 headline")
        has_failures = "exactkv_failures" in lower or "exactkv failures" in lower or "exactness failures" in lower
        _add(report, f"{rel}_exactkv_failures", has_failures, "exactkv_failures")
        if "linkedin" in rel or "short_announcement" in rel:
            has_models = any(x in lower for x in ("llama", "mistral", "7b", "8b", "both model"))
            _add(report, f"{rel}_models", has_models, "models")
        if rel.endswith("x_thread.md") or "linkedin" in rel:
            _add(report, f"{rel}_verifier", "verifier" in lower or "exactness" in lower, "verifier")

    demo_path = root / "reports/public_release/demo_cards.json"
    if demo_path.is_file():
        cards = json.loads(demo_path.read_text(encoding="utf-8")).get("demo_cards") or []
        _add(report, "demo_cards_min_count", len(cards) >= 5, f"count={len(cards)}")
        for card in cards:
            note = str(card.get("decoded_output_note", "")).lower()
            if "unavailable" in note:
                continue
            # historical cards may have decoded text from demo_pack
            if "full_reference:" in note or "compressed_draft:" in note:
                continue
            _add(report, "demo_decoded_honesty", False, card.get("demo_id", ""))

    manifest_path = root / "reports/public_release/launch_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _add(
            report,
            "manifest_source_of_truth",
            manifest.get("source_of_truth_artifact") == "reports/scale_7b/raw.json",
        )
        _add(
            report,
            "manifest_cell_count",
            int(manifest.get("benchmark_cell_count") or 0) >= 1500,
            f"count={manifest.get('benchmark_cell_count')}",
        )
        _add(report, "manifest_failures", manifest.get("exactkv_failures") == 0)
        _add(report, "manifest_claim_boundaries", bool(manifest.get("claim_boundaries_path")))

    lb_path = root / "reports/public_release/leaderboard_final.json"
    if lb_path.is_file():
        lb = json.loads(lb_path.read_text(encoding="utf-8"))
        entries = lb.get("entries") or []
        llama = [
            e
            for e in entries
            if "llama" in str(e.get("model", "")).lower() and isinstance(e.get("score"), (int, float))
        ]
        mistral = [
            e
            for e in entries
            if "mistral" in str(e.get("model", "")).lower() and isinstance(e.get("score"), (int, float))
        ]
        _add(report, "leaderboard_llama_numeric", bool(llama), f"rows={len(llama)}")
        _add(report, "leaderboard_mistral_numeric", bool(mistral), f"rows={len(mistral)}")

    # Delegate to existing validators (non-fatal subprocess capture for report detail only).
    for script in (
        "scripts/check_no_secrets.py",
        "scripts/audit_public_claims.py",
        "scripts/check_public_release.py",
        "scripts/check_release_evidence.py",
        "scripts/check_project_lineage.py",
    ):
        try:
            proc = subprocess.run(
                [__import__("sys").executable, script],
                cwd=root,
                capture_output=True,
                text=True,
            )
            _add(report, f"delegate_{script.split('/')[-1]}", proc.returncode == 0, proc.stdout[-200:])
        except OSError as exc:
            _add(report, f"delegate_{script}", False, str(exc))

    report.status = "pass" if report.valid else "fail"
    return report
