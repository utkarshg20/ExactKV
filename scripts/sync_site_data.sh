#!/usr/bin/env bash
# Copy public JSON + figure assets into site/ for self-contained landing page.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/site/data" "$ROOT/site/assets"

cp "$ROOT/reports/public_release/leaderboard_final.json" "$ROOT/site/data/leaderboard.json"
python3 "$ROOT/scripts/curate_site_case_studies.py"

# Self-contained figure assets (dark-themed charts for the landing page).
python3 "$ROOT/scripts/prepare_site_figure_assets.py"
python3 "$ROOT/scripts/render_site_embedded_html.py"

python3 <<PY
import json
from pathlib import Path

root = Path("$ROOT")
manifest_path = root / "site/content_manifest.json"
lb_path = root / "site/data/leaderboard.json"

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
lb = json.loads(lb_path.read_text(encoding="utf-8"))

rows = []
for e in lb.get("entries") or []:
    rows.append({
        "rank": e.get("rank"),
        "compressor": e.get("compressor"),
        "model": e.get("model_short") or e.get("model"),
        "score": e.get("score"),
        "acceptance": e.get("acceptance_rate"),
        "divergence": e.get("divergence_score"),
        "availability": e.get("availability") or (
            "mock_fallback" if e.get("backend_tier") == "MOCK" else
            "probe_only" if e.get("probe_only") else "available"
        ),
    })
for e in lb.get("diagnostic_entries") or []:
    rows.append({
        "rank": e.get("rank"),
        "compressor": e.get("compressor"),
        "model": e.get("model_short") or e.get("model"),
        "score": e.get("score"),
        "acceptance": e.get("acceptance_rate"),
        "divergence": e.get("divergence_score"),
        "availability": e.get("availability") or "diagnostic",
    })

manifest["leaderboard_rows"] = rows
manifest["generated_at"] = lb.get("generated_at") or manifest.get("generated_at")
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print("synced content_manifest leaderboard_rows:", len(rows))
PY

echo "synced site/: leaderboard + case studies + assets"
