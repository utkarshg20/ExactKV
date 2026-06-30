#!/usr/bin/env python3
"""Build LongBench answer-overlap diagnostic pack from panel JSON artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR in sys.path:
    sys.path.remove(_SCRIPT_DIR)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.benchmarks.longbench_overlap import (  # noqa: E402
    analyse_longbench_json,
    load_export_reference_index,
    load_hf_reference_index,
)

DEFAULT_EXPORT = "benchmarks/prompts/longbench_export.jsonl"

DEFAULT_PATHS = [
    "reports/external_panels/hf_longbench_v26_merged_raw.json",
    "reports/external_panels/hf_longbench_v26_Llama_3_1_8B_raw.json",
    "reports/external_panels/hf_longbench_v26_Mistral_7B_raw.json",
    "reports/external_panels/faithful/longbench_Mistral_7B_Instruct_v0_3_raw.json",
]


def _md(pack: dict) -> str:
    lines = [
        "# LongBench Answer-Overlap Pack",
        "",
        pack.get("note", ""),
        "",
    ]
    for panel in pack.get("panels") or []:
        lines.extend([
            f"## {Path(panel['path']).name}",
            "",
            f"- Cells scored: {panel.get('cells_scored', 0)}",
            f"- Missing reference: {panel.get('cells_missing_reference', 0)}",
            "",
            "| Compressor | n | Mean F1 (full-KV) | Mean F1 (lossy) | Mean F1 (ExactKV) | ExactKV=full |",
            "|------------|--:|------------------:|----------------:|------------------:|-------------:|",
        ])
        for comp, stats in sorted((panel.get("by_compressor") or {}).items()):
            def _f(v: float | None) -> str:
                return f"{v:.3f}" if v is not None else "—"

            lines.append(
                f"| `{comp}` | {stats['cells_scored']} | "
                f"{_f(stats.get('mean_full_max_f1'))} | {_f(stats.get('mean_lossy_max_f1'))} | "
                f"{_f(stats.get('mean_exactkv_max_f1'))} | "
                f"{100 * stats.get('exactkv_matches_full_rate', 0):.0f}% |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="*", default=DEFAULT_PATHS)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--export-jsonl", default=DEFAULT_EXPORT)
    parser.add_argument("--out-json", default="reports/external_panels/longbench_overlap_pack.json")
    parser.add_argument("--out-md", default="reports/external_panels/longbench_overlap_pack.md")
    args = parser.parse_args()

    paths = [Path(p) for p in args.paths if Path(p).is_file()]
    if not paths:
        print("No LongBench panel JSON files found.")
        return 1

    ref_index: dict[str, list[str]] = {}
    export_path = Path(args.export_jsonl)
    if export_path.is_file():
        ref_index = load_export_reference_index(export_path)
        print(f"Loaded {len(ref_index)} references from {export_path}")
    elif not args.skip_hf:
        try:
            ref_index = load_hf_reference_index()
            print(f"Loaded {len(ref_index)} references from HuggingFace")
        except Exception as exc:
            print(f"WARN: could not load HF references ({exc})")

    panels = [analyse_longbench_json(p, reference_index=ref_index) for p in paths]
    pack = {
        "panels": panels,
        "note": panels[0]["note"] if panels else "",
        "reference_index_size": len(ref_index),
    }
    md = _md(pack)
    print(md)

    if args.write:
        out_json = Path(args.out_json)
        out_md = Path(args.out_md)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(pack, indent=2) + "\n", encoding="utf-8")
        out_md.write_text(md + "\n", encoding="utf-8")
        print(f"Wrote {out_json}")
        print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
