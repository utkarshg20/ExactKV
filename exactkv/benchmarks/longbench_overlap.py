"""LongBench answer overlap diagnostics (not official LongBench scores)."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _tokenize(text: str) -> list[str]:
    return _normalize(text).split()


def token_f1(prediction: str, reference: str) -> float:
    pred_toks = _tokenize(prediction)
    ref_toks = _tokenize(reference)
    if not pred_toks and not ref_toks:
        return 1.0
    if not pred_toks or not ref_toks:
        return 0.0
    common = defaultdict(int)
    for t in pred_toks:
        common[t] += 1
    overlap = 0
    for t in ref_toks:
        if common[t] > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_toks)
    recall = overlap / len(ref_toks)
    return 2 * precision * recall / (precision + recall)


def normalize_references(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    if isinstance(raw, (list, tuple)):
        out: list[str] = []
        for item in raw:
            if isinstance(item, str) and item.strip():
                out.append(item)
            elif isinstance(item, (list, tuple)):
                out.extend(str(x) for x in item if str(x).strip())
        return out
    return [str(raw)] if str(raw).strip() else []


def max_token_f1(prediction: str, references: Sequence[str]) -> float | None:
    refs = [r for r in references if r.strip()]
    if not refs or not prediction.strip():
        return None
    return max(token_f1(prediction, ref) for ref in refs)


def base_longbench_prompt_id(prompt_id: str) -> str:
    """lb_narrativeqa_000_ctx2048 -> lb_narrativeqa_000"""
    if "_ctx" in prompt_id:
        return prompt_id.rsplit("_ctx", 1)[0]
    return prompt_id


def load_export_reference_index(path: Path | str) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pid = str(row.get("prompt_id") or "")
            if pid:
                index[pid] = normalize_references(row.get("reference_answer"))
    return index


def load_hf_reference_index(
    *,
    subsets: Sequence[str] | None = None,
    max_per_subset: int = 2,
) -> dict[str, list[str]]:
    from exactkv.benchmarks.external_dataset_loaders import load_longbench_hf  # noqa: PLC0415

    rows = load_longbench_hf(
        subsets=subsets,
        max_per_subset=max_per_subset,
    )
    index: dict[str, list[str]] = {}
    for row in rows:
        pid = str(row.get("prompt_id") or "")
        index[pid] = normalize_references(row.get("reference_answer"))
    return index


def score_cell_overlap(
    cell: Mapping[str, Any],
    references: Sequence[str],
) -> dict[str, Any]:
    refs = list(references)
    out: dict[str, Any] = {
        "prompt_id": cell.get("prompt_id"),
        "compressor_name": cell.get("compressor_name"),
        "model_name": cell.get("model_name"),
        "max_new_tokens": cell.get("max_new_tokens"),
        "reference_count": len(refs),
    }
    for path in ("full", "lossy", "exactkv"):
        text = (cell.get(path) or {}).get("output_text", "") or ""
        out[f"{path}_max_f1"] = max_token_f1(text, refs)
    out["exactkv_matches_full_f1"] = (
        out.get("full_max_f1") is not None
        and out.get("exactkv_max_f1") is not None
        and abs(out["full_max_f1"] - out["exactkv_max_f1"]) < 1e-9
    )
    return out


def analyse_longbench_panel(
    report: Mapping[str, Any],
    *,
    reference_index: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    missing_refs = 0
    for cell in report.get("cells") or []:
        if cell.get("status") != "ok":
            continue
        if (cell.get("dataset_family") or report.get("dataset_family")) != "longbench":
            continue
        refs = normalize_references(cell.get("reference_answer"))
        if not refs and reference_index is not None:
            base_id = base_longbench_prompt_id(str(cell.get("prompt_id") or ""))
            refs = list(reference_index.get(base_id) or [])
        if not refs:
            missing_refs += 1
            continue
        scored.append(score_cell_overlap(cell, refs))

    def _mean(key: str, rows: Sequence[Mapping[str, Any]]) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    by_comp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_comp[str(row.get("compressor_name") or "unknown")].append(row)

    summary: dict[str, Any] = {}
    for comp, rows in sorted(by_comp.items()):
        summary[comp] = {
            "cells_scored": len(rows),
            "mean_full_max_f1": _mean("full_max_f1", rows),
            "mean_lossy_max_f1": _mean("lossy_max_f1", rows),
            "mean_exactkv_max_f1": _mean("exactkv_max_f1", rows),
            "exactkv_matches_full_rate": sum(1 for r in rows if r.get("exactkv_matches_full_f1")) / len(rows),
        }

    return {
        "dataset_family": "longbench",
        "cells_scored": len(scored),
        "cells_missing_reference": missing_refs,
        "by_compressor": summary,
        "note": (
            "Diagnostic max token-F1 vs HF LongBench reference answers — NOT official LongBench scores. "
            "Short max_new_tokens limits answer overlap; use for relative compressor comparison only."
        ),
    }


def analyse_longbench_json(path: Path, *, reference_index: Mapping[str, Sequence[str]] | None = None) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    result = analyse_longbench_panel(report, reference_index=reference_index)
    result["path"] = str(path)
    return result
