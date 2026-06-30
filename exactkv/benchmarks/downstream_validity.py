"""Downstream output validity metrics on panel cell artifacts (not official benchmark scores)."""
from __future__ import annotations

import ast
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


def bfcl_tool_call_valid(text: str) -> bool:
    """Balanced-brace scan: output contains a complete JSON object or array."""
    text = text.strip()
    depth = 0
    in_string = False
    escape = False
    started = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ("{", "["):
            depth += 1
            started = True
        elif ch in ("}", "]"):
            depth -= 1
            if started and depth == 0:
                return True
    return False


_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_candidate(text: str) -> str:
    """Best-effort extract Python from model output (fences or raw tail)."""
    text = text.strip()
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    if "def " in text:
        idx = text.find("def ")
        return text[idx:].strip()
    return text


def mbpp_syntax_valid(text: str) -> bool:
    """True if extracted Python parses under ast.parse."""
    candidate = extract_python_candidate(text)
    if not candidate.strip():
        return False
    try:
        ast.parse(candidate)
        return True
    except SyntaxError:
        return False


def annotate_cell_downstream(cell: Mapping[str, Any], *, family: str) -> dict[str, Any]:
    out = dict(cell)
    full_text = (cell.get("full") or {}).get("output_text", "") or ""
    exactkv_text = (cell.get("exactkv") or {}).get("output_text", "") or ""
    fam = (family or cell.get("dataset_family") or "").lower()

    if fam == "bfcl":
        out["full_kv_downstream_valid"] = bfcl_tool_call_valid(full_text)
        out["exactkv_downstream_valid"] = bfcl_tool_call_valid(exactkv_text)
        out["downstream_metric"] = "bfcl_json_tool_call"
    elif fam in ("mbpp", "humaneval"):
        out["full_kv_downstream_valid"] = mbpp_syntax_valid(full_text)
        out["exactkv_downstream_valid"] = mbpp_syntax_valid(exactkv_text)
        out["downstream_metric"] = "python_ast_syntax"
    else:
        out["full_kv_downstream_valid"] = None
        out["exactkv_downstream_valid"] = None
        out["downstream_metric"] = None
    return out


def summarise_downstream(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ok = [c for c in cells if c.get("status") == "ok"]
    metric = next((c.get("downstream_metric") for c in ok if c.get("downstream_metric")), None)
    n = len(ok)
    vf = sum(1 for c in ok if c.get("full_kv_downstream_valid"))
    ve = sum(1 for c in ok if c.get("exactkv_downstream_valid"))
    div = sum(1 for c in ok if (c.get("metrics") or {}).get("token_level_divergence"))
    preserved = sum(
        1 for c in ok
        if c.get("full_kv_downstream_valid") and c.get("exactkv_downstream_valid")
    )
    lost = sum(
        1 for c in ok
        if c.get("full_kv_downstream_valid") and not c.get("exactkv_downstream_valid")
    )
    accs = [(c.get("metrics") or {}).get("acceptance_rate", 1.0) for c in ok]
    return {
        "downstream_metric": metric,
        "cells_ok": n,
        "divergence_rate": div / n if n else 0.0,
        "full_kv_valid": vf,
        "exactkv_valid": ve,
        "full_kv_valid_rate": vf / n if n else 0.0,
        "exactkv_valid_rate": ve / n if n else 0.0,
        "valid_preserved_among_full_kv_valid": preserved,
        "valid_lost_among_full_kv_valid": lost,
        "preservation_rate_given_full_valid": preserved / vf if vf else None,
        "mean_acceptance_rate": statistics.mean(accs) if accs else 1.0,
    }


def summarise_by_compressor(cells: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in cells:
        if c.get("status") != "ok":
            continue
        groups[str(c.get("compressor_name") or "unknown")].append(dict(c))
    return {name: summarise_downstream(group) for name, group in sorted(groups.items())}


def process_panel_json(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    family = str(report.get("dataset_family") or path.stem)
    annotated = [
        annotate_cell_downstream(c, family=family)
        for c in report.get("cells") or []
    ]
    return {
        "path": str(path),
        "dataset_family": family,
        "overall": summarise_downstream(annotated),
        "by_compressor": summarise_by_compressor(annotated),
        "cells_annotated": len(annotated),
    }


def build_downstream_pack(paths: Sequence[Path]) -> dict[str, Any]:
    panels = [process_panel_json(p) for p in paths if p.is_file()]
    return {
        "panels": panels,
        "note": (
            "Downstream validity is diagnostic only — not official BFCL/MBPP/HumanEval scores. "
            "BFCL: balanced-brace JSON tool-call scan. MBPP/HumanEval: ast.parse on extracted Python."
        ),
    }
