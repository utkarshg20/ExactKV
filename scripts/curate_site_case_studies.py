#!/usr/bin/env python3
"""Curate readable divergence case studies for the public site.

Pulls from headline GPU panels (HF LongBench, BFCL validity, core scale),
not early synthetic pilot JSONL runs. Prefers snippets without deterministic
padding filler so the landing page shows real benchmark-shaped examples.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "reports" / "external_panels"
OUT = EXT / "case_studies_extracted.json"
SITE_OUT = ROOT / "site" / "data" / "case_studies.json"

FILLER_MARKERS = (
    "deterministic filler",
    "ExactKV evidence-plus panel",
    "segment_",
)

PANEL_SOURCES: tuple[tuple[str, Path, str], ...] = (
    ("core_scale", ROOT / "reports" / "scale_7b" / "raw.json", "core_scale"),
    ("hf_longbench_v26", EXT / "hf_longbench_v26_merged_raw.json", "hf_longbench_v26"),
    ("bfcl_validity_v27", EXT / "bfcl_validity_v27_merged_raw.json", "bfcl_validity_v27"),
    ("bfcl_export_50", EXT / "bfcl_export_50_raw.json", "bfcl_export_50"),
    ("faithful_wave1_longbench", EXT / "faithful" / "longbench_Llama_3_1_8B_raw.json", "faithful_wave1"),
)

TARGET_SLOTS: tuple[tuple[str, str | None], ...] = (
    ("core_scale", "capital"),
    ("hf_longbench_v26", "narrativeqa"),
    ("hf_longbench_v26", "gov_report"),
    ("bfcl_validity_v27", "parallel"),
    ("bfcl_validity_v27", "ast_eval"),
    ("bfcl_export_50", "multi_turn"),
    ("faithful_wave1_longbench", "kivi_catastrophic"),
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cell_ok(c: dict[str, Any]) -> bool:
    status = c.get("status")
    if status is not None and status != "ok":
        return False
    if c.get("exactkv_failure"):
        return False
    return True


def metrics(c: dict[str, Any]) -> dict[str, Any]:
    return c.get("metrics") or {}


def is_divergent(c: dict[str, Any]) -> bool:
    if metrics(c).get("token_level_divergence"):
        return True
    full = (c.get("full") or {}).get("output_text") or ""
    lossy = (c.get("lossy") or {}).get("output_text") or ""
    return bool(full or lossy) and full != lossy


def output_text(c: dict[str, Any], key: str) -> str:
    return str((c.get(key) or {}).get("output_text") or "")


def filler_penalty(text: str) -> int:
    penalty = 0
    for mark in FILLER_MARKERS:
        if mark in text:
            penalty += 5
    if re.search(r"\bsegment_\d+\b", text):
        penalty += 5
    return penalty


def readability_score(c: dict[str, Any]) -> float:
    if not is_divergent(c):
        return -1.0
    full = output_text(c, "full") or output_text(c, "exactkv")
    lossy = output_text(c, "lossy")
    if not full or not lossy or full == lossy:
        return -1.0

    score = 10.0
    score -= filler_penalty(full + lossy)

    m = metrics(c)
    fdi = m.get("first_divergence_index")
    if fdi is not None and fdi <= 8:
        score += 3.0
    if m.get("acceptance_rate", 0.0) >= 0.9:
        score += 1.0

    fam = str(c.get("dataset_family") or "")
    cat = str(c.get("task_category") or c.get("category") or "").lower()
    comp = str(c.get("compressor_name") or "")
    if fam == "longbench" or "narrative" in cat:
        score += 4.0
    if fam == "bfcl" or "tool" in cat or "ast" in cat:
        score += 3.0
    if comp == "int4_sim":
        score += 1.0
    if len(full) <= 80 and len(lossy) <= 80:
        score += 2.0
    return score


def make_snippet(text: str, *, max_len: int = 220) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[-max_len:].lstrip()


def snippet_fields(c: dict[str, Any]) -> dict[str, Any]:
    full = output_text(c, "full") or output_text(c, "exactkv")
    lossy = output_text(c, "lossy")
    exact = output_text(c, "exactkv") or full
    avail = bool(full) and bool(lossy)
    return {
        "full_snippet": make_snippet(full) if full else None,
        "lossy_snippet": make_snippet(lossy) if lossy else None,
        "exactkv_snippet": make_snippet(exact) if exact else None,
        "snippets_available": avail,
    }


def interpret_cell(c: dict[str, Any]) -> str:
    comp = str(c.get("compressor_name") or "")
    if comp.startswith("kivi"):
        return "catastrophic"
    fam = str(c.get("dataset_family") or "")
    cat = str(c.get("task_category") or c.get("category") or "").lower()
    if fam == "bfcl" or "ast" in cat or "tool" in cat:
        return "tool-risk"
    if "code" in cat:
        return "code-risk"
    if fam == "longbench" or "qa" in cat or "report" in cat:
        return "semantic"
    return "semantic"


def category_key(c: dict[str, Any]) -> str:
    return str(c.get("task_category") or c.get("category") or c.get("prompt_id") or "unknown").lower()


def slot_match(panel: str, slot_cat: str | None, c: dict[str, Any]) -> bool:
    if slot_cat is None:
        return True
    cat = category_key(c)
    if slot_cat == "capital":
        return "capital" in str(c.get("prompt_id") or "").lower()
    if slot_cat == "kivi_catastrophic":
        return c.get("compressor_name") == "kivi_offline_r32" and "!!!!" in output_text(c, "lossy")
    return slot_cat in cat


def build_case_entry(c: dict[str, Any], *, panel: str, source_file: Path) -> dict[str, Any]:
    m = metrics(c)
    return {
        "dataset_family": c.get("dataset_family") or ("longbench" if "longbench" in panel else panel),
        "task_category": c.get("task_category") or c.get("category"),
        "prompt_id": c.get("prompt_id"),
        "model_name": c.get("model_name"),
        "compressor_name": c.get("compressor_name"),
        "context_bucket": c.get("context_bucket"),
        "max_new_tokens": c.get("max_new_tokens"),
        "first_divergence_index": m.get("first_divergence_index"),
        "acceptance_rate": m.get("acceptance_rate"),
        "exactkv_failure": c.get("exactkv_failure"),
        "panel": panel,
        "source_file": str(source_file),
        **snippet_fields(c),
        "interpretation": interpret_cell(c),
    }


def iter_ok_cells(path: Path) -> list[dict[str, Any]]:
    report = load(path)
    return [c for c in report.get("cells", []) if cell_ok(c)]


def pick_best(candidates: list[dict[str, Any]], *, panel: str, source_file: Path) -> dict[str, Any] | None:
    ranked = sorted(
        ((readability_score(c), c) for c in candidates),
        key=lambda x: x[0],
        reverse=True,
    )
    for score, c in ranked:
        if score < 0:
            break
        entry = build_case_entry(c, panel=panel, source_file=source_file)
        if entry["snippets_available"]:
            return entry
    return None


def curate(*, max_cases: int = 8) -> dict[str, Any]:
    by_panel: dict[str, list[dict[str, Any]]] = {}
    for panel_name, path, _ in PANEL_SOURCES:
        if not path.is_file():
            continue
        by_panel[panel_name] = iter_ok_cells(path)

    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()

    for panel_name, slot_cat in TARGET_SLOTS:
        path = next(p for n, p, _ in PANEL_SOURCES if n == panel_name)
        cells = by_panel.get(panel_name, [])
        pool = [c for c in cells if is_divergent(c) and slot_match(panel_name, slot_cat, c)]
        entry = pick_best(pool, panel=panel_name, source_file=path)
        if entry is None:
            continue
        dedupe = f"{entry.get('prompt_id')}|{entry.get('compressor_name')}|{entry.get('context_bucket')}|{entry.get('max_new_tokens')}"
        if dedupe in used_ids:
            continue
        used_ids.add(dedupe)
        selected.append(entry)
        if len(selected) >= max_cases:
            break

    if len(selected) < max_cases:
        for panel_name, path, _ in PANEL_SOURCES:
            cells = by_panel.get(panel_name, [])
            pool = [c for c in cells if is_divergent(c)]
            entry = pick_best(pool, panel=panel_name, source_file=path)
            if entry is None:
                continue
            dedupe = f"{entry.get('prompt_id')}|{entry.get('compressor_name')}|{entry.get('context_bucket')}|{entry.get('max_new_tokens')}"
            if dedupe in used_ids:
                continue
            used_ids.add(dedupe)
            selected.append(entry)
            if len(selected) >= max_cases:
                break

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "divergent_count": len(selected),
        "total_case_entries": len(selected),
        "note": (
            "Curated from headline GPU panels (HF LongBench, BFCL validity/export, core scale). "
            "Drift diagnostics only. Not official benchmark leaderboard scores."
        ),
        "case_studies": selected,
    }


def write_site_payload(report: dict[str, Any]) -> None:
    cases = []
    for c in report.get("case_studies", []):
        if not c.get("snippets_available"):
            continue
        trimmed = {k: v for k, v in c.items() if k not in ("source_file", "timing_ms")}
        cases.append(trimmed)
    payload = {
        "generated_at": report.get("generated_at"),
        "note": report.get("note"),
        "case_studies": cases,
    }
    SITE_OUT.parent.mkdir(parents=True, exist_ok=True)
    SITE_OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    report = curate()
    if not report["case_studies"]:
        raise SystemExit("No readable divergent case studies found in headline panels.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_site_payload(report)

    panels = {c.get("panel") for c in report["case_studies"]}
    print(f"Wrote {len(report['case_studies'])} case studies -> {OUT.relative_to(ROOT)}")
    print(f"Synced site payload -> {SITE_OUT.relative_to(ROOT)}")
    print("Panels:", ", ".join(sorted(panels)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
