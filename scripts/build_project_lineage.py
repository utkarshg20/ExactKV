#!/usr/bin/env python3
"""Build project lineage and historical artifact inventory (Release Gate R2)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.platform.project_archaeology import (  # noqa: E402
    build_artifact_inventory,
    build_lineage_graph,
    build_project_lineage,
    build_version_lineage,
    render_historical_inventory_md,
    render_project_lineage_md,
    render_version_lineage_md,
    write_inventory_csv,
    write_version_lineage_csv,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ExactKV project lineage")
    parser.add_argument("--root", type=Path, default=_ROOT)
    args = parser.parse_args()

    report = build_project_lineage(args.root)
    artifacts = build_artifact_inventory(args.root)
    graph = build_lineage_graph(artifacts)

    json_path = args.root / "reports/historical_artifact_inventory.json"
    csv_path = args.root / "reports/historical_artifact_inventory.csv"
    graph_path = args.root / "reports/project_lineage_graph.json"
    inv_md = args.root / "docs/HISTORICAL_ARTIFACT_INVENTORY.md"
    lineage_md = args.root / "docs/PROJECT_LINEAGE.md"
    version_md = args.root / "docs/VERSION_LINEAGE.md"
    version_json = args.root / "reports/version_lineage.json"
    version_csv = args.root / "reports/version_lineage.csv"

    versions = build_version_lineage(args.root)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_inventory_csv(artifacts, csv_path)
    graph_path.write_text(json.dumps(graph.to_dict(), indent=2) + "\n", encoding="utf-8")
    inv_md.write_text(render_historical_inventory_md(artifacts, args.root), encoding="utf-8")
    lineage_md.write_text(render_project_lineage_md(artifacts, graph, args.root), encoding="utf-8")
    version_md.write_text(render_version_lineage_md(versions), encoding="utf-8")
    version_json.write_text(
        json.dumps(
            {
                "generated_at": report["generated_at"],
                "version_arc": "V1-V21",
                "versions": [v.to_dict() for v in versions],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_version_lineage_csv(versions, version_csv)

    print(f"artifacts={report['artifact_count']}")
    print(f"versions={len(versions)}")
    print(f"pre_formal_pipeline={report['pre_formal_pipeline_count']}")
    print(f"wrote={json_path}")
    print(f"wrote={csv_path}")
    print(f"wrote={graph_path}")
    print(f"wrote={inv_md}")
    print(f"wrote={lineage_md}")
    print(f"wrote={version_md}")
    print(f"wrote={version_json}")
    print(f"wrote={version_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
