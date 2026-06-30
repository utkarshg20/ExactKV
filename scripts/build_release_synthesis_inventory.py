#!/usr/bin/env python3
"""Repository-wide forensic artifact inventory generator (Release Synthesis Part 1).

Walks `git ls-files`, classifies every tracked artifact with conservative
heuristics, and writes:

  release_synthesis/artifact_inventory.json
  release_synthesis/artifact_inventory.csv
  release_synthesis/artifact_inventory.md   (human summary; full rows in json/csv)

Classification is heuristic-assisted and self-labels a confidence level. No
results are invented: every row is derived from a real tracked path. Report JSON
files are opened (best effort) to surface a few headline metrics where present.
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "release_synthesis"

# Artifacts considered authoritative release-grade benchmark/claim evidence.
RELEASE_GRADE = {
    "reports/scale_7b/raw.json",
    "reports/scale_7b/leaderboard.json",
    "reports/scale_7b/leaderboard.md",
    "reports/scale_7b/leaderboard.csv",
    "reports/scale_7b/scale_summary.json",
    "reports/public_release/leaderboard_final.json",
    "reports/public_release/release_manifest.json",
    "reports/public_release/methodology.md",
    "reports/public_release/demo_cards.json",
    "reports/public_release/demo_cards.md",
    "reports/public_release/README_PUBLIC.md",
    "reports/public_release/benchmark_summary.md",
    "reports/phaseF_kernel_benchmark.json",
    "reports/phaseG_unified_truth.json",
    "reports/release_evidence_status.json",
    "reports/novelty_audit.json",
    "reports/novelty_audit_matrix.csv",
    "docs/CLAIM_BOUNDARIES.md",
    "docs/NOVELTY_AUDIT.md",
    "docs/METRIC_DEFINITIONS.md",
    "docs/EXACTKV_TECHNICAL_REPORT.md",
    "docs/PROJECT_LINEAGE.md",
    "docs/VERSION_LINEAGE.md",
    "docs/RELEASE_EVIDENCE_STATUS.md",
}

# Phase F kernel = qualified microbenchmark, not end-to-end.
QUALIFIED_CAVEATS = {
    "reports/phaseF_kernel_benchmark.json": "Kernel microbenchmark only; not end-to-end speedup.",
    "reports/phaseA_benchmark.json": "Historical 336-cell panel; superseded by scale_7b for public headline.",
}


def git_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return sorted(p for p in out.stdout.splitlines() if p.strip())


def classify_type(path: str) -> str:
    p = path.lower()
    ext = Path(p).suffix
    if p.startswith("tests/") and ext == ".py":
        return "test"
    if p.startswith("scripts/") and ext == ".py":
        return "script"
    if p.startswith("exactkv/") and ext == ".py":
        return "code"
    if ext == ".py":
        return "code"
    if ext == ".json":
        if "report" in p or p.startswith("reports/"):
            return "report_json"
        return "config" if ("config" in p or p.endswith("pyproject.toml")) else "report_json"
    if ext == ".csv":
        return "benchmark_output"
    if ext in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".mp4", ".webm"}:
        return "plot"
    if ext in {".html"}:
        return "launch_artifact" if ("leaderboard" in p or "site" in p) else "doc"
    if ext == ".md":
        if "launch" in p or "x_thread" in p or "linkedin" in p or "blog" in p:
            return "launch_artifact"
        return "report_md" if p.startswith("docs/") else "doc"
    if ext in {".toml", ".cfg", ".ini", ".yaml", ".yml", ".txt"}:
        return "config"
    if ext in {".sh"}:
        return "script"
    return "unknown"


def classify_timeline(path: str) -> tuple[str, str]:
    """Return (timeline, stage)."""
    name = Path(path).name
    p = path
    # Final launch / public release surface
    if (
        p.startswith("site/")
        or p.startswith("launch/")
        or p.startswith("release_synthesis/")
        or p.startswith("paper/")
        or "public_release" in p
        or "launch_" in name
        or name in {"RELEASE.md", "x_thread.md", "linkedin_post.md", "blog_post.md"}
    ):
        return "final launch", "Phase K / launch package"
    # Release gates
    if (
        "release_evidence" in p
        or "release_manifest" in p
        or name.startswith("RELEASE_CHECKLIST")
        or "RELEASE_EVIDENCE" in name
        or re.search(r"\bR[012]\b", name)
    ):
        return "release gate", "Gate R0/R1/R2"
    # Formal phase pipeline (A-K) via run_phase_* and phase{A..H} report stems
    if re.search(r"run_phase_[a-k]", p.lower()) or re.search(r"phase[A-H]\b", name) or re.search(r"phase[A-H]_", name):
        return "Phase-lineage", "Formal Phase A-K pipeline"
    # Numbered safety-ladder phases 11-21 (distinct from A-K, mapped to V14-V21)
    m = re.search(r"PHASE_(\d{1,2})", name)
    if m:
        return "Phase-lineage", f"Numbered phase {m.group(1)} (safety/runtime ladder)"
    # V-series scope/release notes
    if re.search(r"\bV\d{1,2}_SCOPE", name) or re.search(r"RELEASE_NOTES_V0", name) or re.search(r"PROJECT_STATUS_V0", name):
        return "V-lineage", "Versioned prototype milestone"
    # Experiment docs/scripts map to V-lineage research arc
    if re.search(r"EXPERIMENT_\d", name) or re.search(r"run_experiment_\d", p.lower()) or re.search(r"exp\d{2,3}", p.lower()):
        return "V-lineage", "Experiment (research arc)"
    return "unknown", ""


def is_historical(path: str, timeline: str) -> bool:
    p = path.lower()
    if path in RELEASE_GRADE:
        return False
    if "experiment_0" in p or "experiment_1" in p or "exp0" in p or "exp1" in p:
        return True
    if "vllm" in p or "sidecar" in p or "serving" in p or "shadow" in p:
        return True
    if timeline in {"V-lineage"}:
        return True
    if p.startswith("reports/") and re.search(r"experiment_\d", p):
        return True
    return False


def is_superseded(path: str) -> bool:
    p = path.lower()
    if "phasea_benchmark" in p:
        return True  # 336-cell -> superseded by scale_7b for public headline
    if re.search(r"reports/experiment_00[1-9]", p):
        return True
    if "leaderboard.json" == Path(p).name and "scale_7b" not in p and "public_release" not in p:
        return False
    return False


def headline_metric(path: str) -> str:
    if not path.endswith(".json"):
        return ""
    fp = ROOT / path
    try:
        if fp.stat().st_size > 6_000_000:
            return "large report (not parsed inline)"
        data = json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if not isinstance(data, dict):
        return ""
    bits = []
    for key in ("total_cells", "exactkv_failures", "deterministic_mode", "status", "phase_id", "phase_f_id"):
        if key in data:
            bits.append(f"{key}={data[key]}")
    return "; ".join(bits[:4])


def should_cite(path: str, release_grade: bool) -> bool:
    if release_grade:
        return True
    keep = {
        "reports/phaseA_benchmark.json",
        "docs/HISTORICAL_ARTIFACT_INVENTORY.md",
        "docs/REPRODUCIBILITY.md",
        "docs/ARTIFACT_INDEX.md",
        "reports/public_release/launch_manifest.json",
    }
    return path in keep


def build() -> dict:
    files = git_files()
    rows = []
    for path in files:
        typ = classify_type(path)
        timeline, stage = classify_timeline(path)
        release_grade = path in RELEASE_GRADE
        historical = is_historical(path, timeline)
        superseded = is_superseded(path)
        caveat = QUALIFIED_CAVEATS.get(path, "")
        metric = headline_metric(path)
        # Purpose heuristic
        purpose = {
            "code": "Library / framework implementation module.",
            "script": "Executable experiment, benchmark, or tooling script.",
            "test": "Automated contract/regression test.",
            "report_json": "Machine-readable benchmark/report artifact.",
            "report_md": "Markdown report / design / claim document.",
            "benchmark_output": "Tabular benchmark output (CSV).",
            "plot": "Visual asset (plot/figure/video frame).",
            "config": "Configuration / metadata file.",
            "launch_artifact": "Public launch / leaderboard surface.",
            "doc": "Documentation / supporting note.",
            "unknown": "Unclassified tracked artifact.",
        }[typ]
        confidence = "high" if (release_grade or typ in {"test", "code", "script"}) else (
            "medium" if timeline != "unknown" else "low"
        )
        rows.append(
            {
                "path": path,
                "type": typ,
                "likely_timeline": timeline,
                "likely_stage": stage,
                "purpose": purpose,
                "key_evidence": metric,
                "release_grade_evidence": release_grade,
                "historical_or_exploratory_only": historical and not release_grade,
                "superseded": superseded,
                "cite_in_final_paper": should_cite(path, release_grade),
                "caveats": caveat,
                "confidence": confidence,
            }
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "total_artifacts": len(rows),
        "note": "Heuristic-assisted classification. Every row is a real tracked path (git ls-files). "
        "release_grade_evidence flags the curated authoritative set; all other rows are supporting/historical.",
        "artifacts": rows,
    }


def write_outputs(inv: dict) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "artifact_inventory.json").write_text(
        json.dumps(inv, indent=2), encoding="utf-8"
    )

    rows = inv["artifacts"]
    fields = list(rows[0].keys())
    with (OUT / "artifact_inventory.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    by_type = Counter(r["type"] for r in rows)
    by_timeline = Counter(r["likely_timeline"] for r in rows)
    n_release = sum(1 for r in rows if r["release_grade_evidence"])
    n_hist = sum(1 for r in rows if r["historical_or_exploratory_only"])
    n_super = sum(1 for r in rows if r["superseded"])

    md = []
    md.append("# ExactKV Artifact Inventory (Release Synthesis — Part 1)\n")
    md.append(f"Generated: {inv['generated_at']}\n")
    md.append(
        "> Heuristic-assisted forensic inventory of **every tracked artifact** "
        "(`git ls-files`). Full per-artifact rows are in "
        "[`artifact_inventory.json`](artifact_inventory.json) and "
        "[`artifact_inventory.csv`](artifact_inventory.csv). This document summarizes counts and "
        "the curated release-grade evidence set.\n"
    )
    md.append(f"**Total tracked artifacts:** {inv['total_artifacts']}\n")
    md.append("## Counts by type\n")
    md.append("| Type | Count |\n|------|------:|")
    for t, c in by_type.most_common():
        md.append(f"| `{t}` | {c} |")
    md.append("\n## Counts by likely timeline\n")
    md.append("| Timeline | Count |\n|----------|------:|")
    for t, c in by_timeline.most_common():
        md.append(f"| {t} | {c} |")
    md.append("\n## Evidence tiers\n")
    md.append("| Tier | Count |\n|------|------:|")
    md.append(f"| Release-grade (curated authoritative) | {n_release} |")
    md.append(f"| Historical / exploratory only | {n_hist} |")
    md.append(f"| Superseded | {n_super} |")
    md.append("\n## Release-grade evidence set (authoritative)\n")
    md.append("| Path | Type | Key evidence | Caveat |\n|------|------|--------------|--------|")
    for r in rows:
        if r["release_grade_evidence"]:
            md.append(
                f"| `{r['path']}` | {r['type']} | {r['key_evidence'] or '—'} | {r['caveats'] or '—'} |"
            )
    md.append(
        "\n## Notes\n"
        "- The single benchmark **source of truth** for public claims is `reports/scale_7b/raw.json` "
        "(1500 cells, `exactkv_failures=0`, `deterministic_mode=false`).\n"
        "- `reports/phaseA_benchmark.json` (336 cells) is **superseded** as the public headline but retained as supporting cross-model evidence.\n"
        "- `reports/phaseF_kernel_benchmark.json` is a **kernel microbenchmark** — not an end-to-end speedup.\n"
        "- See [`source_of_truth_map.md`](source_of_truth_map.md) for the full hierarchy.\n"
    )
    (OUT / "artifact_inventory.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    inventory = build()
    write_outputs(inventory)
    print(f"Wrote inventory for {inventory['total_artifacts']} artifacts to {OUT}")
