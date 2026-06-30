#!/usr/bin/env python3
"""Build downstream validity summary from panel JSON artifacts."""
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

from exactkv.benchmarks.downstream_validity import build_downstream_pack  # noqa: E402

DEFAULT_PATHS = [
    "reports/external_panels/bfcl_validity_v27_merged_raw.json",
    "reports/external_panels/v30/bfcl_Llama_3_1_8B_raw.json",
    "reports/external_panels/v30/bfcl_Mistral_7B_Instruct_v0_3_raw.json",
    "reports/external_panels/v30/mbpp_Llama_3_1_8B_raw.json",
    "reports/external_panels/v30/mbpp_Mistral_7B_Instruct_v0_3_raw.json",
    "reports/external_panels/faithful/mbpp_mistral_smoke_raw.json",
]


def _md(pack: dict) -> str:
    lines = [
        "# Downstream Validity Pack",
        "",
        pack.get("note", ""),
        "",
    ]
    for panel in pack.get("panels") or []:
        o = panel["overall"]
        if o.get("downstream_metric") is None:
            continue
        lines.extend([
            f"## {panel['dataset_family']} — `{Path(panel['path']).name}`",
            "",
            f"- Metric: `{o['downstream_metric']}`",
            f"- Cells: {o['cells_ok']}",
            f"- Full-KV valid: {o['full_kv_valid']} ({100 * o['full_kv_valid_rate']:.1f}%)",
            f"- ExactKV valid: {o['exactkv_valid']} ({100 * o['exactkv_valid_rate']:.1f}%)",
            f"- Preserved among full-KV valid: {o['valid_preserved_among_full_kv_valid']}"
            f" / {o['full_kv_valid']}"
            + (f" ({100 * o['preservation_rate_given_full_valid']:.1f}%)" if o['preservation_rate_given_full_valid'] is not None else ""),
            f"- Lost among full-KV valid: {o['valid_lost_among_full_kv_valid']}",
            f"- Divergence rate: {100 * o['divergence_rate']:.1f}%",
            "",
            "| Compressor | full valid | exactkv valid | preserved | lost | div |",
            "|------------|----------:|-------------:|----------:|-----:|----:|",
        ])
        for comp, stats in sorted(panel.get("by_compressor", {}).items()):
            lines.append(
                f"| `{comp}` | {stats['full_kv_valid']} | {stats['exactkv_valid']} | "
                f"{stats['valid_preserved_among_full_kv_valid']} | {stats['valid_lost_among_full_kv_valid']} | "
                f"{100 * stats['divergence_rate']:.1f}% |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="*", default=DEFAULT_PATHS)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--out-json", default="reports/external_panels/downstream_validity_pack.json")
    parser.add_argument("--out-md", default="reports/external_panels/downstream_validity_pack.md")
    args = parser.parse_args()

    paths = [Path(p) for p in args.paths if Path(p).is_file()]
    if not paths:
        print("No panel JSON files found.")
        return 1

    pack = build_downstream_pack(paths)
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
