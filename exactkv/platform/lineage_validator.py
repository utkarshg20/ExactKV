"""Project lineage validator (Release Gate R2)."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

FORBIDDEN_PHRASES = (
    r"\bproduction[- ]ready\b",
    r"\bend[- ]to[- ]end\s+speedup\b",
    r"\bactive\s+gpu\s+memory\s+savings\b",
    r"\bnothing\s+like\s+this\s+exists\b",
    r"\bfirst\s+ever\b",
)

# Undercounting full version arc as V1–V13 only (Release Gate R2.1).
V13_UNDERCOUNT_RE = re.compile(r"v1[–\-]v13\b", re.I)

PUBLIC_LAUNCH_VERSION_SCAN = (
    "README.md",
    "RELEASE.md",
    "reports/public_release/README_PUBLIC.md",
    "paper/ExactKV_Technical_Report.md",
    "site/index.html",
    "docs/PROJECT_LINEAGE.md",
    "docs/HISTORICAL_ARTIFACT_INVENTORY.md",
    "docs/ARTIFACT_INDEX.md",
    "docs/RELEASE_CHECKLIST.md",
)


@dataclass
class LineageIssue:
    severity: str
    check: str
    detail: str = ""


@dataclass
class LineageValidationReport:
    status: str = "pending"
    issues: list[LineageIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {"status": "pass" if self.valid else "fail", "issues": [asdict(i) for i in self.issues]}


def _add(report: LineageValidationReport, check: str, ok: bool, detail: str = "", *, warning: bool = False) -> None:
    if not ok:
        report.issues.append(LineageIssue("warning" if warning else "error", check, detail))


def validate_project_lineage(root: Path | str = ".") -> LineageValidationReport:
    root = Path(root)
    report = LineageValidationReport()

    required = {
        "inventory_md": root / "docs/HISTORICAL_ARTIFACT_INVENTORY.md",
        "lineage_md": root / "docs/PROJECT_LINEAGE.md",
        "version_lineage_md": root / "docs/VERSION_LINEAGE.md",
        "inventory_json": root / "reports/historical_artifact_inventory.json",
        "inventory_csv": root / "reports/historical_artifact_inventory.csv",
        "graph_json": root / "reports/project_lineage_graph.json",
        "version_lineage_json": root / "reports/version_lineage.json",
        "version_lineage_csv": root / "reports/version_lineage.csv",
    }
    for key, path in required.items():
        _add(report, f"exists_{key}", path.is_file(), str(path))

    lineage_text = ""
    if required["lineage_md"].is_file():
        lineage_text = required["lineage_md"].read_text(encoding="utf-8", errors="replace").lower()
        _add(report, "lineage_not_start_at_phase_a", "did not start at phase a" in lineage_text)
        _add(report, "lineage_version_arc_v21", "v1" in lineage_text and "v21" in lineage_text)
        _add(report, "lineage_historical_vs_release", "authoritative" in lineage_text and "historical" in lineage_text)
        _add(report, "lineage_verifier_first", "verifier" in lineage_text)
        _add(report, "lineage_demo_work", "demo" in lineage_text)
        _add(report, "lineage_runtime_safety", "shadow" in lineage_text or "safety" in lineage_text or "l3" in lineage_text)
        _add(report, "lineage_no_go_claims", "no-go" in lineage_text or "forbidden" in lineage_text)

    artifact_index = root / "docs/ARTIFACT_INDEX.md"
    if artifact_index.is_file():
        idx = artifact_index.read_text(encoding="utf-8", errors="replace").lower()
        _add(report, "artifact_index_historical_section", "historical" in idx and "pre-release" in idx)

    checklist = root / "docs/RELEASE_CHECKLIST.md"
    if checklist.is_file():
        cl = checklist.read_text(encoding="utf-8", errors="replace").lower()
        _add(report, "checklist_lineage_review", "lineage" in cl and ("pre-a" in cl or "pre a" in cl))

    tech = root / "docs/EXACTKV_TECHNICAL_REPORT.md"
    outline = root / "docs/EXACTKV_TECHNICAL_REPORT_OUTLINE.md"
    if tech.is_file():
        t = tech.read_text(encoding="utf-8", errors="replace").lower()
        _add(report, "technical_report_lineage_section", "project lineage" in t)
        _add(report, "technical_report_version_arc_v21", "v1" in t and "v21" in t)
    elif outline.is_file():
        o = outline.read_text(encoding="utf-8", errors="replace").lower()
        _add(report, "technical_report_outline_lineage", "project lineage" in o)
    else:
        _add(report, "technical_report_or_outline", False, "missing technical report and outline")

    if required["inventory_json"].is_file():
        data = json.loads(required["inventory_json"].read_text(encoding="utf-8"))
        count = int(data.get("artifact_count") or 0)
        _add(report, "inventory_min_count", count >= 100, f"count={count}")
        pre = int(data.get("pre_formal_pipeline_count") or 0)
        _add(report, "inventory_pre_a_artifacts", pre >= 50, f"pre_a={pre}")
        arts = data.get("artifacts") or []
        buckets = {a.get("chronological_bucket") for a in arts}
        outside = buckets - {
            "formal_phase_A_to_C",
            "runtime_probe_phase_D",
            "kernel_phase_E_F",
            "truth_engine_phase_G",
            "platform_phase_H",
            "evidence_gate_R0_R1",
            "novelty_release_phase_I_J",
            "launch_phase_K_preparation",
        }
        _add(report, "inventory_outside_phase_aj", bool(outside), f"buckets={len(outside)}")

    if required["graph_json"].is_file():
        graph = json.loads(required["graph_json"].read_text(encoding="utf-8"))
        _add(report, "graph_has_nodes", len(graph.get("nodes") or []) >= 5)
        _add(report, "graph_has_edges", len(graph.get("edges") or []) >= 3)

    version_json = root / "reports/version_lineage.json"
    if version_json.is_file():
        vdata = json.loads(version_json.read_text(encoding="utf-8"))
        vids = {v.get("version_id") for v in vdata.get("versions") or []}
        for n in range(1, 22):
            _add(report, f"version_entry_V{n}", f"V{n}" in vids)
        pending = [
            v for v in vdata.get("versions") or []
            if v.get("evidence_status") == "context_known_source_pending"
        ]
        for v in pending:
            if v.get("evidence_status") != "context_known_source_pending":
                continue
            _add(
                report,
                f"source_pending_marked_{v.get('version_id')}",
                "source_pending" in str(v.get("caveats", "")).lower()
                or "scope_statement" in str(v.get("caveats", "")).lower()
                or "manual" in str(v.get("caveats", "")).lower()
                or "inferred" in str(v.get("caveats", "")).lower()
                or "phase" in str(v.get("caveats", "")).lower(),
                v.get("version_id", ""),
            )

    for rel in PUBLIC_LAUNCH_VERSION_SCAN:
        path = root / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        lower = text.lower()
        if V13_UNDERCOUNT_RE.search(lower) and "v21" not in lower.replace("v1–v13", "").replace("v1-v13", ""):
            # Allow "V1–V13 scope statements" subrange if V21 also mentioned
            if not re.search(r"v1[–\-]v21|v1\s*through\s*v21|v1–v21", lower):
                _add(report, f"no_v13_only_arc_{path.name}", False, "states V1-V13 as full arc")
        if rel.endswith("TECHNICAL_REPORT.md") or rel.endswith("PROJECT_LINEAGE.md"):
            if "v21" not in lower:
                _add(report, f"mentions_v21_{path.name}", False)

    for path in (required["lineage_md"], required["inventory_md"], required.get("version_lineage_md", root / "x")):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if re.search(r"\b(forbidden|not claim|does not claim|no-go)\b", line, re.I):
                continue
            for pat in FORBIDDEN_PHRASES:
                m = re.search(pat, line, re.I)
                if not m:
                    continue
                before = line[: m.start()]
                if re.search(r"\b(not|no|never|forbidden|without)\b", before[-80:], re.I):
                    continue
                _add(report, "no_forbidden_claims_introduced", False, f"{path.name}: {m.group()}")
                break

    report.status = "pass" if report.valid else "fail"
    return report
