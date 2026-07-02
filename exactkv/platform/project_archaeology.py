"""Project archaeology and lineage reconstruction (Release Gate R2).

Discovers historical artifacts from the repository file tree, git history,
and document/report metadata. Does not invent paths or results.
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Search seeds — not exhaustive; scanner discovers more from repo.
MARKER_RE = re.compile(
    r"\b(phase\s*[a-z0-9_+-]+|exp(?:eriment)?[_\s-]?\d{3}|v\d{1,2}|l[0-5]|"
    r"verifier|draft|shadow|observer|no-commit|dry-run|exactkv_failures|"
    r"first[_\s-]?divergence|token[_\s-]?drift|vllm|lmcache|vericache|"
    r"triton|cuda|no-go|claim|leaderboard)\b",
    re.I,
)

EXP_DOC_RE = re.compile(r"EXPERIMENT_(\d{3}[A-Z]?)_", re.I)
PHASE_DOC_RE = re.compile(r"PHASE_(\d+[A-Z]?)_", re.I)
EXP_SCRIPT_RE = re.compile(r"run_experiment_(\d{3})", re.I)
EXP_TEST_RE = re.compile(r"test_experiment_(\d{3})|test_exp(\d{3})", re.I)
V_SCOPE_RE = re.compile(r"V(\d{1,2})_SCOPE", re.I)
V_VERSION_RE = re.compile(r"\b[Vv](\d{1,2})\b|\bversion\s+(\d{1,2})\b", re.I)
RELEASE_TAG_RE = re.compile(r"^v0\.(\d+)\.0", re.I)

CHRONOLOGICAL_BUCKETS_ORDER = (
    "early_foundation",
    "verifier_core",
    "trace_correctness",
    "compression_simulation",
    "adversarial_demos",
    "structured_output_demos",
    "v_series_demos",
    "benchmark_prototypes",
    "safety_ladder",
    "shadow_observer_runtime",
    "l3_draft_shadow",
    "l4_verifier_mediated_dry_run",
    "runtime_coupling",
    "instability_analysis",
    "visualization_layer",
    "no_go_serving_probe",
    "memory_timing_claim_boundary",
    "external_compressor_investigation",
    "formal_phase_A_to_C",
    "runtime_probe_phase_D",
    "kernel_phase_E_F",
    "truth_engine_phase_G",
    "platform_phase_H",
    "evidence_gate_R0_R1",
    "novelty_release_phase_I_J",
    "launch_phase_K_preparation",
    "unknown",
)

SUBSYSTEMS = (
    "verifier",
    "generator",
    "compressor",
    "trace",
    "benchmark",
    "leaderboard",
    "runtime_probe",
    "shadow_observer",
    "safety",
    "external_adapter",
    "serving_probe",
    "memory_probe",
    "timing_probe",
    "visualization",
    "publication",
    "release",
    "claim_audit",
    "test_infra",
    "unknown",
)


@dataclass
class ArtifactRecord:
    artifact_id: str
    file_path: str
    file_exists: bool
    artifact_type: str
    phase_or_experiment_name: str = ""
    inferred_time_order: int = 9999
    chronological_bucket: str = "unknown"
    subsystem: str = "unknown"
    purpose: str = ""
    key_result_or_contribution: str = ""
    evidence_source: str = "file_name"
    claim_relevance: str = ""
    public_safe_summary: str = ""
    caveats: str = ""
    still_used_by_release: bool = False
    superseded_by: str = ""
    public_facing: bool = False
    confidence: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LineageGraph:
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": self.nodes, "edges": self.edges}


@dataclass
class VersionRecord:
    version_id: str
    evidence_status: str
    evidence_files: list[str] = field(default_factory=list)
    git_evidence: list[str] = field(default_factory=list)
    inferred_date_or_order: int = 0
    title_or_theme: str = ""
    purpose: str = ""
    key_contribution: str = ""
    relation_to_next_version: str = ""
    relation_to_current_release: str = ""
    public_safe_summary: str = ""
    caveats: str = ""
    superseded_by: str = ""
    still_supports_release: bool = False
    confidence: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


VERSION_SEEDS: dict[int, dict[str, Any]] = {
    1: {"title": "Correctness prototype", "theme": "verifier-first core", "tag": None},
    2: {"title": "Framework generalization", "theme": "registry, CLI", "tag": None},
    3: {"title": "Benchmark seriously", "theme": "sweeps, markdown reports", "tag": None},
    4: {"title": "Asymmetric K/V compression", "theme": "Experiment 003", "tag": "v0.4.0"},
    5: {"title": "Workspace-aware memory accounting", "theme": "Experiment 004", "tag": "v0.5.0"},
    6: {"title": "Backend adapter interface", "theme": "kvpress KnormPress", "tag": "v0.6.0"},
    7: {"title": "Layer-aware V policies", "theme": "Experiments 006/006C", "tag": "v0.7.0"},
    8: {"title": "Serving harness", "theme": "Experiment 007", "tag": "v0.8.0"},
    9: {"title": "Real backend gauntlet", "theme": "Exp 008–011", "tag": "v0.9.0"},
    10: {"title": "Suite hardening", "theme": "Exp 012–014", "tag": "v0.10.0"},
    11: {"title": "Launch hardening", "theme": "Exp 015–020", "tag": "v-release"},
    12: {"title": "Deferred work completion gauntlet", "theme": "Exp 021–027", "tag": None},
    13: {"title": "Practicality proof", "theme": "span verification, demos, external methods", "tag": None},
    14: {
        "title": "CUDA restored verifier & GPU memory diagnostics",
        "theme": "Phase 14A–14C; Exp 055–058",
        "phase_globs": ("EXPERIMENT_05[5-8]", "EXPERIMENT_056", "Phase 14"),
    },
    15: {
        "title": "vLLM feasibility probes",
        "theme": "Phase 15A–15E; Exp 059–065",
        "phase_globs": ("EXPERIMENT_059", "EXPERIMENT_06[0-5]", "Phase 15"),
    },
    16: {
        "title": "Shadow observers & streaming quant feasibility",
        "theme": "Phase 16A–16T; Exp 066–085",
        "phase_globs": ("EXPERIMENT_07[6-9]", "EXPERIMENT_08[0-5]", "PHASE_16", "Phase 16"),
    },
    17: {
        "title": "Claim-safe demo & broader validation",
        "theme": "Phase 17A–17D",
        "phase_globs": ("PHASE_17", "Phase 17"),
    },
    18: {
        "title": "Integration safety & L3 draft-shadow no-commit",
        "theme": "Phase 18A–18E; Exp 090–091",
        "phase_globs": ("PHASE_18", "EXPERIMENT_090", "EXPERIMENT_091", "Phase 18"),
    },
    19: {
        "title": "L3 round-log proposal source",
        "theme": "Phase 19A–19C",
        "phase_globs": ("PHASE_19", "EXPERIMENT_09[5-7]", "Phase 19"),
    },
    20: {
        "title": "L4 pre-gate & verifier-mediated design",
        "theme": "Phase 20A–20D",
        "phase_globs": ("PHASE_20", "EXPERIMENT_09[89]", "EXPERIMENT_10[01]", "Phase 20"),
    },
    21: {
        "title": "L4 scaffolds & trace-only dry-run",
        "theme": "Phase 21A–21L",
        "phase_globs": ("PHASE_21", "EXPERIMENT_10[2-9]", "EXPERIMENT_11[0-9]", "Phase 21"),
    },
}


def _version_from_path(path: str) -> int | None:
    m = V_SCOPE_RE.search(path)
    if m:
        return int(m.group(1))
    m2 = re.search(r"RELEASE_NOTES_V0\.(\d+)\.0", path, re.I)
    if m2:
        return int(m2.group(1))
    return None


def _file_matches_version_seed(path: str, version: int) -> bool:
    p = path.lower()
    if re.search(rf"\bphase\s+{version}[a-z]?\b", path, re.I):
        return True
    phase_file = re.search(rf"phase_{version}[a-z]", p)
    if phase_file:
        return True
    exp_ranges: dict[int, tuple[int, int]] = {
        14: (55, 58),
        15: (59, 65),
        16: (66, 85),
        17: (86, 89),
        18: (90, 94),
        19: (95, 97),
        20: (98, 101),
        21: (102, 115),
    }
    if version in exp_ranges:
        lo, hi = exp_ranges[version]
        m = re.search(r"experiment_(\d{3})", p)
        if m and lo <= int(m.group(1)) <= hi:
            return True
        m2 = re.search(r"run_exp(\d{3})", p)
        if m2 and lo <= int(m2.group(1)) <= hi:
            return True
        m3 = re.search(r"test_exp(\d{3})", p)
        if m3 and lo <= int(m3.group(1)) <= hi:
            return True
    seed = VERSION_SEEDS.get(version, {})
    for g in seed.get("phase_globs") or ():
        if g.lower() in p:
            return True
    return False


def discover_version_evidence(root: Path) -> dict[int, dict[str, Any]]:
    """Scan tracked files and git history for V1–V21 evidence."""
    tracked = git_tracked_files(root)
    by_version: dict[int, dict[str, Any]] = {
        v: {"files": set(), "git": set(), "content_hits": 0} for v in range(1, 22)
    }

    for rel in tracked:
        v = _version_from_path(rel)
        if v and 1 <= v <= 21:
            by_version[v]["files"].add(rel)
        for n in range(1, 22):
            if _file_matches_version_seed(rel, n):
                by_version[n]["files"].add(rel)

    for rel in tracked:
        if not rel.endswith((".md", ".py", ".json")):
            continue
        if not any(rel.startswith(p) for p in ("docs/", "exactkv/", "scripts/", "tests/")):
            continue
        full = root / rel
        if not full.is_file() or full.stat().st_size > 500_000:
            continue
        try:
            text = full.read_text(encoding="utf-8", errors="replace")[:100_000]
        except OSError:
            continue
        for m in V_VERSION_RE.finditer(text):
            num = m.group(1) or m.group(2)
            if num and 1 <= int(num) <= 21:
                by_version[int(num)]["content_hits"] += 1

    log_out = _run_git(["log", "--oneline", "--decorate", "-500"], root)
    for line in log_out.splitlines():
        for n in range(1, 22):
            if re.search(rf"\b[Vv]{n}\b", line) or re.search(rf"\bversion\s+{n}\b", line, re.I):
                by_version[n]["git"].add(line.strip()[:200])

    return by_version


def build_version_lineage(root: Path | str = ".") -> list[VersionRecord]:
    root = Path(root)
    discovered = discover_version_evidence(root)
    records: list[VersionRecord] = []

    for n in range(1, 22):
        seed = VERSION_SEEDS.get(n, {})
        files = sorted(discovered[n]["files"])
        git_hits = sorted(discovered[n]["git"])[:10]
        scope_path = f"docs/V{n}_SCOPE_STATEMENT.md"
        has_scope = (root / scope_path).is_file()
        if has_scope and scope_path not in files:
            files.insert(0, scope_path)

        if has_scope:
            status = "verified"
            confidence = "high"
            caveats_extra = ""
        elif n <= 13 and files:
            status = "partial"
            confidence = "medium"
            caveats_extra = ""
        elif n >= 14 and files:
            status = "partial"
            confidence = "medium"
            caveats_extra = (
                f"No V{n}_SCOPE_STATEMENT.md; version inferred from Phase {n} / experiment docs."
            )
        elif n >= 14:
            status = "context_known_source_pending"
            confidence = "low"
            caveats_extra = (
                f"Version V{n} is part of the project version arc but lacks dedicated "
                f"V{n}_SCOPE_STATEMENT.md in the repository."
            )
        else:
            status = "missing"
            confidence = "low"
            caveats_extra = "Insufficient repository evidence."

        title = seed.get("title") or f"Version {n}"
        theme = seed.get("theme") or ""
        tag = seed.get("tag")
        if tag:
            rn = f"docs/RELEASE_NOTES_{tag.upper()}.md"
            if (root / rn).is_file() and rn not in files:
                files.append(rn)

        purpose = f"Pre-formal-release milestone V{n}: {title}."
        if n >= 14:
            purpose += (
                f" Aligns with Phase {n} safety/runtime ladder work "
                "(distinct from formal A–K release phases)."
            )

        key_contrib = f"{theme}; git tag `{tag}`" if tag else theme
        relation_next = f"V{n + 1}" if n < 21 else "Formal release phases A–K"
        still_supports = n >= 10

        if n <= 13:
            caveats = "Historical prototype milestone; not the 1500-cell public headline."
        elif status == "partial":
            caveats = (
                f"{caveats_extra} Historical lineage only — not release benchmark evidence."
            )
        elif status == "context_known_source_pending":
            caveats = caveats_extra
        else:
            caveats = caveats_extra or "Historical lineage only."

        records.append(
            VersionRecord(
                version_id=f"V{n}",
                evidence_status=status,
                evidence_files=files[:40],
                git_evidence=git_hits,
                inferred_date_or_order=n,
                title_or_theme=title,
                purpose=purpose,
                key_contribution=key_contrib,
                relation_to_next_version=relation_next,
                relation_to_current_release="historical_support",
                public_safe_summary=(
                    f"V{n} ({title}): {theme}. Evidence: {status}. "
                    "Project lineage — not benchmark evidence."
                ),
                caveats=caveats,
                superseded_by="Formal A–K release pipeline",
                still_supports_release=still_supports,
                confidence=confidence,
            ),
        )
    return records


def render_version_lineage_md(versions: list[VersionRecord]) -> str:
    verified = sum(1 for v in versions if v.evidence_status == "verified")
    partial = sum(1 for v in versions if v.evidence_status == "partial")
    pending = sum(1 for v in versions if v.evidence_status == "context_known_source_pending")
    lines = [
        "# ExactKV Version Lineage (V1–V21)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "The **version arc** (V1–V21) spans pre-formal-release research milestones. "
        "It is **distinct** from formal release phases A–K and from the authoritative "
        "**1500-cell** benchmark (`reports/scale_7b/raw.json`).",
        "",
        f"- **Verified (scope statements):** {verified}",
        f"- **Partial (phase/experiment docs):** {partial}",
        f"- **Source pending:** {pending}",
        "",
        "## Version table",
        "",
        "| Version | Status | Theme | Sample evidence |",
        "|---------|--------|-------|-----------------|",
    ]
    for v in versions:
        ev = v.evidence_files[0] if v.evidence_files else "—"
        lines.append(f"| {v.version_id} | {v.evidence_status} | {v.title_or_theme} | `{ev}` |")
    lines.extend(["", "## Per-version detail", ""])
    for v in versions:
        lines.extend(
            [
                f"### {v.version_id} — {v.title_or_theme}",
                "",
                f"- **Evidence status:** `{v.evidence_status}` ({v.confidence})",
                f"- **Purpose:** {v.purpose}",
                f"- **Key contribution:** {v.key_contribution}",
                f"- **Caveats:** {v.caveats}",
            ],
        )
        if v.evidence_files:
            lines.append("- **Evidence files:**")
            for f in v.evidence_files[:10]:
                lines.append(f"  - `{f}`")
        lines.append("")
    lines.extend(
        [
            "## Version-lineage entries requiring manual source attachment",
            "",
            "V14–V21 lack dedicated `V{N}_SCOPE_STATEMENT.md` files. Evidence is drawn from "
            "Phase N / experiment documentation. Do not cite as verified benchmark evidence.",
            "",
        ],
    )
    for v in versions:
        if int(v.version_id[1:]) >= 14 and v.evidence_status != "verified":
            lines.append(f"- **{v.version_id}** ({v.evidence_status}): {v.caveats}")
    lines.append("")
    return "\n".join(lines)


def write_version_lineage_csv(versions: list[VersionRecord], path: Path) -> None:
    fields = list(VersionRecord.__dataclass_fields__.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for v in versions:
            row = v.to_dict()
            row["evidence_files"] = ";".join(v.evidence_files)
            row["git_evidence"] = ";".join(v.git_evidence)
            writer.writerow(row)


def _run_git(args: list[str], root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.stdout if proc.returncode == 0 else ""
    except OSError:
        return ""


def git_tracked_files(root: Path) -> list[str]:
    out = _run_git(["ls-files"], root)
    return [line.strip() for line in out.splitlines() if line.strip()]


def git_file_first_commit(root: Path, rel_path: str) -> str:
    out = _run_git(["log", "--diff-filter=A", "--format=%ci", "--", rel_path], root)
    line = out.strip().split("\n")[0] if out.strip() else ""
    return line


def git_tags(root: Path) -> list[str]:
    out = _run_git(["tag", "--list"], root)
    return sorted(out.splitlines())


def _artifact_type(path: str) -> str:
    p = path.lower()
    if p.endswith(".py") and "/tests/" in p or p.startswith("tests/"):
        return "test"
    if p.endswith(".py") and "/scripts/" in p or p.startswith("scripts/"):
        return "script"
    if p.endswith(".py"):
        return "code"
    if p.endswith(".json"):
        if "reports/" in p:
            return "report_json"
        return "config"
    if p.endswith((".md", ".rst")):
        return "doc"
    if p.endswith((".png", ".svg", ".gif", ".webp")):
        return "plot"
    if p.endswith((".yaml", ".yml", ".toml", ".ini", ".cfg")):
        return "config"
    if p.endswith((".csv",)):
        return "report_json"
    if p.endswith((".ipynb",)):
        return "doc"
    if p.endswith((".pt", ".pth", ".bin")):
        return "benchmark_output"
    if "public_release" in p:
        return "release_artifact"
    return "unknown"


def _exp_number_from_path(path: str) -> int | None:
    m = EXP_DOC_RE.search(path) or EXP_SCRIPT_RE.search(path) or EXP_TEST_RE.search(path)
    if not m:
        return None
    num = m.group(1) or m.group(2)
    return int(re.sub(r"\D", "", num))


def _phase_letter_order(path: str) -> int | None:
    m = re.search(r"PHASE_(\d+)([A-Z]?)", path, re.I)
    if m:
        return int(m.group(1)) * 100 + (ord(m.group(2).upper()) - ord("A") + 1 if m.group(2) else 0)
    m2 = re.search(r"phase([A-Z])", path, re.I)
    if m2:
        return 1000 + ord(m2.group(1).upper())
    return None


def infer_chronological_bucket(path: str, artifact_type: str) -> str:
    p = path.lower()
    exp = _exp_number_from_path(path)
    if "novelty_audit" in p or "claim_boundaries" in p or "phase i" in p:
        return "novelty_release_phase_I_J"
    if "release_checklist" in p or "repro_manifest" in p or "project_lineage" in p:
        return "launch_phase_K_preparation"
    if "release_evidence" in p or "evidence_integrity" in p or "gate_r0" in p:
        return "evidence_gate_R0_R1"
    if "public_release" in p or "release_packager" in p or "scale_7b" in p:
        if "phasef" in p or "phase_f" in p or "kernel" in p:
            return "kernel_phase_E_F"
        if "phaseg" in p or "phase_g" in p or "unified_truth" in p:
            return "truth_engine_phase_G"
        if "phaseh" in p or "public_leaderboard" in p:
            return "platform_phase_H"
    if "phasef" in p or "triton" in p and "kernel" in p:
        return "kernel_phase_E_F"
    if "phaseg" in p or "firstdivergence" in p or "unified_truth" in p:
        return "truth_engine_phase_G"
    if "phasea" in p or "phase_a" in p or "phaseb" in p or "phasec" in p:
        return "formal_phase_A_to_C"
    if "phased" in p or "runtime_probe" in p:
        return "runtime_probe_phase_D"
    if re.search(r"phase_[ef]", p):
        return "kernel_phase_E_F"
    if "l4" in p and ("dry" in p or "verifier_mediated" in p):
        return "l4_verifier_mediated_dry_run"
    if "l3" in p or "guarded_draft_shadow" in p or "shadow" in p and "no_commit" in p:
        return "l3_draft_shadow"
    if "shadow" in p or "observer" in p:
        return "shadow_observer_runtime"
    if re.search(r"\bl[0-5]\b", p) or "safety" in p or "integration_safety" in p:
        return "safety_ladder"
    if "vllm" in p or "lmcache" in p or "serving" in p:
        return "no_go_serving_probe"
    if "gpu_memory" in p or "timing" in p or "performance_memory" in p or "exp027" in p or "exp_027" in p:
        return "memory_timing_claim_boundary"
    if any(x in p for x in ("spectralquant", "shard", "kivi", "kvquant", "turboquant", "snapkv")):
        return "external_compressor_investigation"
    if "instability" in p or "exp116" in p or "exp117" in p or "phase_diagram" in p:
        return "instability_analysis"
    if "visual" in p or "plot" in p or "leaderboard.html" in p:
        return "visualization_layer"
    if "crash_test" in p or "terminal" in p or "demo" in p or "killer" in p or "correction" in p:
        return "structured_output_demos"
    if "verifier" in p or "exactkv" in p and artifact_type == "code":
        return "verifier_core"
    if "trace" in p or "divergence" in p or "acceptance" in p:
        return "trace_correctness"
    if "compress" in p or "int8" in p or "int4" in p:
        return "compression_simulation"
    if V_SCOPE_RE.search(path):
        v = int(V_SCOPE_RE.search(path).group(1))  # type: ignore[union-attr]
        if v <= 3:
            return "early_foundation"
        if v <= 9:
            return "benchmark_prototypes"
        return "v_series_demos"
    if exp is not None:
        if exp <= 5:
            return "early_foundation"
        if exp <= 20:
            return "benchmark_prototypes"
        if exp <= 45:
            return "external_compressor_investigation"
        if exp <= 85:
            return "shadow_observer_runtime"
        return "safety_ladder"
    if "v1_scope" in p or "v2_scope" in p or "vision" in p:
        return "early_foundation"
    if "release_notes" in p:
        return "v_series_demos"
    return "unknown"


def infer_subsystem(path: str, artifact_type: str) -> str:
    p = path.lower()
    if "test_" in p and artifact_type == "test":
        return "test_infra"
    if "claim" in p or "novelty" in p or "audit" in p:
        return "claim_audit"
    if "public_release" in p or "publish" in p or "paper_draft" in p:
        return "publication"
    if "leaderboard" in p:
        return "leaderboard"
    if "benchmark" in p or "sweep" in p or "experiment" in p:
        return "benchmark"
    if "shadow" in p or "observer" in p:
        return "shadow_observer"
    if "vllm" in p or "lmcache" in p or "serving" in p:
        return "serving_probe"
    if "gpu_memory" in p or "memory" in p:
        return "memory_probe"
    if "timing" in p or "throughput" in p:
        return "timing_probe"
    if "visual" in p or p.endswith(".png"):
        return "visualization"
    if any(x in p for x in ("spectralquant", "shard", "kivi", "kvquant", "turboquant", "adapter")):
        return "external_adapter"
    if "safety" in p or re.search(r"l[0-5]", p):
        return "safety"
    if "runtime" in p or "coupling" in p:
        return "runtime_probe"
    if "verifier" in p:
        return "verifier"
    if "compressor" in p or "compression" in p:
        return "compressor"
    if "trace" in p or "divergence" in p:
        return "trace"
    if "generator" in p:
        return "generator"
    if "release" in p:
        return "release"
    return "unknown"


def _infer_time_order(path: str) -> int:
    exp = _exp_number_from_path(path)
    if exp is not None:
        return 100 + exp
    ph = _phase_letter_order(path)
    if ph is not None:
        return 2000 + ph
    m = V_SCOPE_RE.search(path)
    if m:
        return 50 + int(m.group(1))
    if "scale_7b" in path:
        return 9000
    if "novelty_audit" in path:
        return 9100
    if "project_lineage" in path:
        return 9200
    if RELEASE_TAG_RE.search(path):
        return 40 + int(RELEASE_TAG_RE.search(path).group(1))  # type: ignore[union-attr]
    return 5000


def _read_title(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")[:4000]
    except OSError:
        return ""
    for line in text.splitlines()[:20]:
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _extract_json_metadata(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 2_000_000:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, Any] = {}
    for key in (
        "phase_id",
        "experiment_id",
        "status",
        "deterministic_mode",
        "total_cells",
        "exactkv_failures",
        "models",
        "models_evaluated",
        "compressors",
    ):
        if key in data:
            out[key] = data[key]
    return out


def _should_inventory(path: str) -> bool:
    """Include meaningful artifacts; skip noise binaries individually."""
    if path.endswith(".pt") or path.endswith(".pth"):
        return False
    if "/__pycache__/" in path:
        return False
    if path.endswith(".pyc"):
        return False
    prefixes = (
        "docs/",
        "scripts/",
        "tests/",
        "exactkv/",
        "reports/",
        "examples/",
        "notebooks/",
        "benchmarks/",
    )
    if any(path.startswith(p) for p in prefixes):
        return True
    if path in ("README.md", "pyproject.toml", "Makefile"):
        return True
    return False


def _group_key(path: str) -> str | None:
    """Group repetitive binary artifact dirs into one inventory row."""
    m = re.match(r"reports/exp(\d{3})_kv_files/", path)
    if m:
        return f"reports/exp{m.group(1)}_kv_files/"
    return None


def build_artifact_inventory(root: Path | str = ".") -> list[ArtifactRecord]:
    root = Path(root)
    tracked = git_tracked_files(root)
    seen_groups: set[str] = set()
    records: list[ArtifactRecord] = []
    counter = 0

    release_authoritative = {
        "reports/scale_7b/raw.json",
        "reports/scale_7b/leaderboard.json",
        "reports/scale_7b/scale_summary.json",
        "reports/public_release/leaderboard_final.json",
        "reports/phaseF_kernel_benchmark.json",
        "reports/phaseG_unified_truth.json",
        "reports/novelty_audit.json",
        "reports/release_evidence_status.json",
    }

    for rel in sorted(tracked):
        if not _should_inventory(rel):
            continue
        gk = _group_key(rel)
        if gk:
            if gk in seen_groups:
                continue
            seen_groups.add(gk)
            rel = gk

        counter += 1
        full = root / rel
        exists = full.is_file() or full.is_dir()
        atype = _artifact_type(rel)
        bucket = infer_chronological_bucket(rel, atype)
        subsystem = infer_subsystem(rel, atype)
        time_order = _infer_time_order(rel)

        phase_or_exp = ""
        if m := EXP_DOC_RE.search(rel):
            phase_or_exp = f"Experiment {m.group(1)}"
        elif m := PHASE_DOC_RE.search(rel):
            phase_or_exp = f"Phase {m.group(1)}"
        elif m := V_SCOPE_RE.search(rel):
            phase_or_exp = f"V{m.group(1)}"

        title = _read_title(full) if full.is_file() and atype == "doc" else ""
        meta = _extract_json_metadata(full) if full.is_file() and atype == "report_json" else {}
        if meta.get("phase_id"):
            phase_or_exp = str(meta["phase_id"])

        purpose = title or phase_or_exp or rel.split("/")[-1]
        key_result = ""
        if meta.get("total_cells"):
            key_result = f"{meta.get('total_cells')} cells; failures={meta.get('exactkv_failures', 'n/a')}"
        elif meta.get("status"):
            key_result = str(meta["status"])

        still_used = rel in release_authoritative or rel.startswith("reports/public_release/")
        public_facing = (
            still_used
            or rel.startswith("docs/NOVELTY")
            or rel.startswith("docs/CLAIM")
            or "public_release" in rel
            or rel in ("README.md", "docs/paper_draft.md", "docs/blog_post.md")
        )

        caveats = ""
        if "vllm" in rel.lower() or "lmcache" in rel.lower():
            caveats = "Serving probe / no-go; not production integration."
        elif "shadow" in rel.lower() or "l3" in rel.lower() or "l4" in rel.lower():
            caveats = "Diagnostic / dry-run / no-commit; not default runtime."
        elif "spectralquant" in rel.lower() or "shard" in rel.lower():
            caveats = "External adapter investigation; fallback/probe in current release."

        superseded = ""
        if "phaseA_benchmark" in rel and "scale_7b" not in rel:
            superseded = "reports/scale_7b/raw.json for public headline"
        if "DEMO_EXACTKV_LIVE_CORRECTION" in rel:
            superseded = "docs/EXACTKV_TERMINAL_CRASH_TEST.md"

        records.append(
            ArtifactRecord(
                artifact_id=f"art_{counter:05d}",
                file_path=rel,
                file_exists=exists,
                artifact_type=atype if not rel.endswith("/") else "benchmark_output",
                phase_or_experiment_name=phase_or_exp,
                inferred_time_order=time_order,
                chronological_bucket=bucket,
                subsystem=subsystem,
                purpose=purpose[:240],
                key_result_or_contribution=key_result[:240],
                evidence_source="generated_report_metadata" if meta else ("file_content" if title else "file_name"),
                claim_relevance="release_authoritative" if still_used else ("historical_support" if public_facing else "internal"),
                public_safe_summary=purpose[:180],
                caveats=caveats,
                still_used_by_release=still_used,
                superseded_by=superseded,
                public_facing=public_facing,
                confidence="high" if meta or title else "medium",
            ),
        )

    records.sort(key=lambda r: (r.inferred_time_order, r.file_path))
    return records


def build_lineage_graph(artifacts: Iterable[ArtifactRecord]) -> LineageGraph:
    graph = LineageGraph()
    by_bucket: dict[str, list[ArtifactRecord]] = defaultdict(list)
    for art in artifacts:
        by_bucket[art.chronological_bucket].append(art)

    bucket_ids = {b: f"bucket:{b}" for b in CHRONOLOGICAL_BUCKETS_ORDER if b in by_bucket}
    for bucket, bid in bucket_ids.items():
        paths = [a.file_path for a in by_bucket[bucket][:8]]
        graph.nodes.append(
            {
                "id": bid,
                "label": bucket.replace("_", " "),
                "type": "chronological_bucket",
                "file_paths": paths,
                "bucket": bucket,
                "subsystem": by_bucket[bucket][0].subsystem if by_bucket[bucket] else "unknown",
                "public_claim_status": "historical" if bucket not in (
                    "platform_phase_H",
                    "evidence_gate_R0_R1",
                    "novelty_release_phase_I_J",
                    "launch_phase_K_preparation",
                ) else "release_supporting",
            },
        )

    ordered_buckets = [b for b in CHRONOLOGICAL_BUCKETS_ORDER if b in bucket_ids]
    for i in range(len(ordered_buckets) - 1):
        graph.edges.append(
            {
                "source": bucket_ids[ordered_buckets[i]],
                "target": bucket_ids[ordered_buckets[i + 1]],
                "relationship": "evolved_into",
            },
        )

    # Key release nodes
    for rel, label in (
        ("reports/scale_7b/raw.json", "Scale 7B/8B benchmark (1500 cells)"),
        ("reports/public_release/leaderboard_final.json", "Public release leaderboard"),
        ("docs/NOVELTY_AUDIT.md", "Novelty audit"),
    ):
        graph.nodes.append(
            {
                "id": f"artifact:{rel}",
                "label": label,
                "type": "release_artifact",
                "file_paths": [rel],
                "bucket": "launch_phase_K_preparation",
                "subsystem": "release",
                "public_claim_status": "authoritative",
            },
        )

    if "bucket:verifier_core" in bucket_ids and "bucket:trace_correctness" in bucket_ids:
        graph.edges.append(
            {"source": "bucket:verifier_core", "target": "bucket:trace_correctness", "relationship": "supports"},
        )
    if "bucket:trace_correctness" in bucket_ids and "bucket:structured_output_demos" in bucket_ids:
        graph.edges.append(
            {
                "source": "bucket:trace_correctness",
                "target": "bucket:structured_output_demos",
                "relationship": "informs_claim_boundary",
            },
        )
    if "bucket:no_go_serving_probe" in bucket_ids and "bucket:claim_audit" not in bucket_ids:
        graph.edges.append(
            {
                "source": "bucket:no_go_serving_probe",
                "target": "bucket:memory_timing_claim_boundary",
                "relationship": "informs_claim_boundary",
            },
        )
    if "bucket:formal_phase_A_to_C" in bucket_ids:
        graph.edges.append(
            {
                "source": "bucket:formal_phase_A_to_C",
                "target": "artifact:reports/scale_7b/raw.json",
                "relationship": "evolved_into",
            },
        )
    graph.edges.append(
        {
            "source": "artifact:reports/scale_7b/raw.json",
            "target": "artifact:reports/public_release/leaderboard_final.json",
            "relationship": "generated",
        },
    )
    return graph


def render_historical_inventory_md(artifacts: list[ArtifactRecord], root: Path) -> str:
    lines = [
        "# ExactKV Historical Artifact Inventory",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Evidence-based inventory from git-tracked files, document titles, and report metadata.",
        "Version arc: **V1–V21** (see [`VERSION_LINEAGE.md`](VERSION_LINEAGE.md)).",
        "This inventory is **not** limited to Phase A–J.",
        "",
        f"**Total artifacts catalogued:** {len(artifacts)}",
        "",
        "## Summary by chronological bucket",
        "",
        "| Bucket | Count |",
        "|--------|------:|",
    ]
    bucket_counts: dict[str, int] = defaultdict(int)
    for a in artifacts:
        bucket_counts[a.chronological_bucket] += 1
    for bucket in CHRONOLOGICAL_BUCKETS_ORDER:
        if bucket_counts.get(bucket):
            lines.append(f"| `{bucket}` | {bucket_counts[bucket]} |")
    lines.extend(["", "## Inventory (sample by bucket)", ""])
    shown_per_bucket: dict[str, int] = defaultdict(int)
    for art in artifacts:
        if shown_per_bucket[art.chronological_bucket] >= 15:
            continue
        shown_per_bucket[art.chronological_bucket] += 1
        lines.append(f"### `{art.file_path}`")
        lines.append(f"- **Type:** {art.artifact_type}")
        lines.append(f"- **Bucket:** {art.chronological_bucket}")
        lines.append(f"- **Subsystem:** {art.subsystem}")
        if art.phase_or_experiment_name:
            lines.append(f"- **Phase/Experiment:** {art.phase_or_experiment_name}")
        lines.append(f"- **Purpose:** {art.purpose}")
        if art.key_result_or_contribution:
            lines.append(f"- **Key result:** {art.key_result_or_contribution}")
        if art.caveats:
            lines.append(f"- **Caveats:** {art.caveats}")
        if art.superseded_by:
            lines.append(f"- **Superseded by:** {art.superseded_by}")
        lines.append(f"- **Release use:** {'yes' if art.still_used_by_release else 'historical'}")
        lines.append("")
    lines.extend(
        [
            "## Version-lineage entries requiring manual source attachment",
            "",
            "V14–V21 lack dedicated `V{N}_SCOPE_STATEMENT.md` files in this repository. "
            "Related artifacts appear under Phase 14–21 experiment and phase docs. "
            "See [`VERSION_LINEAGE.md`](VERSION_LINEAGE.md) for evidence status per version.",
            "",
        ],
    )
    lines.append("Full machine-readable inventory: `reports/historical_artifact_inventory.json`")
    return "\n".join(lines)


def render_project_lineage_md(artifacts: list[ArtifactRecord], graph: LineageGraph, root: Path) -> str:
    bucket_counts: dict[str, int] = defaultdict(int)
    for a in artifacts:
        bucket_counts[a.chronological_bucket] += 1
    pre_a = sum(
        c for b, c in bucket_counts.items()
        if b not in (
            "formal_phase_A_to_C",
            "runtime_probe_phase_D",
            "kernel_phase_E_F",
            "truth_engine_phase_G",
            "platform_phase_H",
            "evidence_gate_R0_R1",
            "novelty_release_phase_I_J",
            "launch_phase_K_preparation",
        )
    )
    exp_docs = sum(1 for a in artifacts if "EXPERIMENT_" in a.file_path and a.artifact_type == "doc")
    phase_docs = sum(1 for a in artifacts if "PHASE_" in a.file_path and a.artifact_type == "doc")

    return f"""# ExactKV Project Lineage

> **ExactKV did not start at Phase A.** The formal A–J release pipeline formalized, scaled, packaged, and validated a much larger pre-release research arc spanning experiments, demos, safety ladders, runtime probes, and claim-boundary work.

Generated: {datetime.now(timezone.utc).isoformat()}

## 1. Executive summary

ExactKV evolved over a long research arc before the formal Phase A–J release pipeline. The repository contains **{len(artifacts)}** catalogued meaningful artifacts, including approximately **{pre_a}** pre–Phase A–J bucket entries, **{exp_docs}** experiment documents, and **{phase_docs}** phase documents discovered from the file tree — not from memory.

Phase A–J **formalized** cross-compressor benchmarking, kernel microbenchmarks, truth-engine divergence authority, public leaderboard packaging, novelty audit, and release gates. It did **not** originate verifier-mediated exactness, trace-level drift measurement, or demo-driven failure-case development.

The **version arc spans V1–V21** — pre-formal-release prototype milestones distinct from phases A–K. V1–V13 have dedicated scope statements; V14–V21 are documented primarily via Phase 14–21 experiment/phase artifacts (see [`VERSION_LINEAGE.md`](VERSION_LINEAGE.md)).

## 2. Methodology

This lineage was reconstructed by:

1. **File tree scan** — `git ls-files` across `exactkv/`, `scripts/`, `tests/`, `docs/`, `reports/`
2. **Git history** — tags (`v0.1.0`–`v0.11.0`), commit messages referencing exp/phase/L3/L4/shadow/runtime
3. **Report metadata** — `phase_id`, `total_cells`, `exactkv_failures` from JSON reports where present
4. **Document titles** — experiment index, V-series scope statements, phase closeouts, claim audits

Tooling: `exactkv/platform/project_archaeology.py` · `scripts/build_project_lineage.py`

## 3. Full chronological timeline (discovered buckets)

| Era (bucket) | Artifacts found |
|--------------|----------------:|
{chr(10).join(f'| `{b}` | {bucket_counts.get(b, 0)} |' for b in CHRONOLOGICAL_BUCKETS_ORDER if bucket_counts.get(b))}

## 4. Early problem framing

[`docs/VISION.md`](VISION.md) and V1–V3 scope statements frame the original problem: **lossy KV-cache compression can change greedy token generation**. ExactKV asked whether compressed caches remain compatible with full-precision greedy decoding through draft/verify/commit semantics.

## 4b. Version arc V1–V21

| Range | Role |
|-------|------|
| V1–V3 | Early foundation: correctness prototype, framework, sweeps |
| V4–V9 | Compression simulation, adapters, serving harness, backend gauntlet |
| V10–V11 | Evaluation suite hardening, launch hardening (tagged v0.10–v0.11) |
| V12–V13 | Deferred-work gauntlet, practicality proof, demos, external methods |
| V14–V15 | CUDA restored verifier, GPU memory diagnostics, vLLM no-go probes |
| V16 | Shadow observers, streaming quant feasibility, Phase 16 closeout |
| V17 | Claim-safe demo packaging, broader model / long-context validation |
| V18–V19 | Integration safety, L3 guarded draft-shadow no-commit, round-log sources |
| V20–V21 | L4 pre-gate, verifier-mediated dry-run scaffolds (no runtime commit) |

Full per-version evidence: [`VERSION_LINEAGE.md`](VERSION_LINEAGE.md) · `reports/version_lineage.json`

## 5. Verifier-first core

Early foundation and verifier-core artifacts include `exactkv/` generator/verifier modules, Experiments 001–002 smoke/core sweeps, and span verification work (Exp 028–029). The invariant **`exactkv_failures == 0`** on tested panels became the hard gate.

## 6. Trace-level correctness work

Trace correctness spans acceptance rate, first divergence index, verifier agreement, and ExactKV failure tracking — developed across V10 suites (Exp 012–016), sensitivity forensics (Exp 013), and Phase G `FirstDivergenceAuthority`.

## 7. Compression simulation and compressor studies

Discovered work includes int8/int4 simulators, asymmetric K/V (Exp 003), layer-aware V (Exp 006), TurboQuant/KIVI/KVQuant adapters (Exp 008–010, 023–024), SpectralQuant probes (Exp 042–045), and Shard bounded probes (Exp 038–041). Current release uses **fallback/proxy** for SpectralQuant and **probe-first** for Shard where real dependencies are absent.

## 8. Demo and failure-case development

Structured-output and adversarial demos include Exp 034/034b pharmacy correction, terminal crash-test demos (`EXACTKV_TERMINAL_CRASH_TEST.md`), LongBench-style drift demo (Exp 037), and V-series visual packages (Exp 035–036). These are **illustrative exactness evidence**, not throughput claims.

## 9. Safety ladder and runtime boundary work

Discovered L0–L5 / Phase 16–21 artifacts include guarded draft-shadow no-commit scaffolds (Phase 18B/Exp 091), L4 verifier-mediated **dry-run** design specs, integration safety specs (Phase 18A), and pre-L4 safety gate reviews. **No-commit / dry-run** constraints remain in force.

## 10. Runtime coupling and live-probe work

Shadow observer panels (Exp 076–085), live round observers (Exp 081–082), and decode-time shadow smoke tests (Exp 083–084) explored runtime instrumentation without authorizing production commit paths.

## 11. Instability analysis and visualization

Exp 116 instability regime analysis and Exp 117 phase diagrams connect to visualization layers (Exp 035, public visual package). These inform leaderboard insights but are not standalone public performance claims.

## 12. Serving, vLLM, LMCache, memory, and speed investigations

Experiments 017, 059–065 (vLLM feasibility), LMCache prototype path docs (Phase 11G), GPU memory pilots (Exp 018, 031, 057–058), and Exp 027/030 performance-memory truth boundaries established **no-go** and **forbidden claim** lists for serving throughput, unqualified end-to-end speedups, and unqualified GPU memory savings claims.

## 13. Transition into formal Phase A–J release pipeline

| Phase | Role |
|-------|------|
| A–C | Formal cross-model benchmark + leaderboard + visuals |
| D | Runtime probe layer |
| E–F | KV compression kernels (Phase F = kernel microbenchmark only) |
| G | Canonical divergence truth engine |
| H/H+ | Public leaderboard platform + 7B/8B scale |
| I | Novelty audit + claim lock |
| J | Public release freeze + reproducibility |
| R0/R1 | Evidence integrity + Mistral leaderboard repair |

## 14. Current release evidence (authoritative)

- **1500-cell** Phase H+ `reports/scale_7b/raw.json` (real GPU, float16, `deterministic_mode=false`)
- Models: `meta-llama/Llama-3.1-8B`, `mistralai/Mistral-7B-Instruct-v0.3`
- **`exactkv_failures = 0`**
- Phase F kernel microbenchmark (int8 ~1.63×, int4 ~1.54×) — **not end-to-end**
- Phase G truth engine; public bundle under `reports/public_release/`

## 15. What earlier work still supports

- Verifier/draft semantics and trace metrics methodology
- Demo narratives (terminal crash-test, structured-output drift)
- Claim boundaries (`CLAIMS_AUDIT.md`, Exp 027, VeriCache parity gate)
- Compressor adapter honesty disclosures
- Test infrastructure validating exactness gates

## 16. What earlier work was superseded

- Phase A 336-cell panel → **scale_7b 1500-cell** public headline
- Legacy live correction demo → terminal crash-test demo
- Stale per_model_tables without Mistral → R1 aggregate repair

## 17. What earlier work is intentionally not claimed

- vLLM/LMCache integration (probes only)
- Production serving / throughput / latency wins
- Active GPU memory savings (forbidden as public claim unless explicitly measured)
- Real SpectralQuant / full Shard product integration in current environment
- L4 runtime commit paths (dry-run / scaffold only)

## 18. How to read the release

| Tier | Artifacts |
|------|-----------|
| **Authoritative** | `reports/scale_7b/raw.json`, `reports/public_release/*`, `reports/release_evidence_status.json` |
| **Supporting historical** | Experiment docs, V-series suites, demos, safety ladder specs |
| **Exploratory / no-go** | vLLM probes, serving sidecars, timing harnesses — claim-boundary evidence only |

See also: [`ARTIFACT_INDEX.md`](ARTIFACT_INDEX.md) · [`HISTORICAL_ARTIFACT_INVENTORY.md`](HISTORICAL_ARTIFACT_INVENTORY.md)
"""


def write_inventory_csv(artifacts: list[ArtifactRecord], path: Path) -> None:
    fields = list(ArtifactRecord.__dataclass_fields__.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for art in artifacts:
            writer.writerow(art.to_dict())


def build_project_lineage(root: Path | str = ".") -> dict[str, Any]:
    root = Path(root)
    artifacts = build_artifact_inventory(root)
    graph = build_lineage_graph(artifacts)
    versions = build_version_lineage(root)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_count": len(artifacts),
        "version_arc": "V1-V21",
        "version_count": len(versions),
        "pre_formal_pipeline_count": sum(
            1 for a in artifacts
            if a.chronological_bucket not in (
                "formal_phase_A_to_C",
                "runtime_probe_phase_D",
                "kernel_phase_E_F",
                "truth_engine_phase_G",
                "platform_phase_H",
                "evidence_gate_R0_R1",
                "novelty_release_phase_I_J",
                "launch_phase_K_preparation",
            )
        ),
        "git_tags": git_tags(root),
        "versions": [v.to_dict() for v in versions],
        "artifacts": [a.to_dict() for a in artifacts],
        "graph": graph.to_dict(),
    }
