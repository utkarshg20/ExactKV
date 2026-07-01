#!/usr/bin/env bash
# Copy public JSON artifacts into site/data/ for self-contained landing page.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/site/data"
cp "$ROOT/reports/public_release/leaderboard_final.json" "$ROOT/site/data/leaderboard.json"
python3 <<PY
import json
from pathlib import Path
root = Path("$ROOT")
raw = json.loads((root / "reports/external_panels/case_studies_extracted.json").read_text())
cases = []
for c in raw.get("case_studies", []):
    if not c.get("snippets_available"):
        continue
    c = {k: v for k, v in c.items() if k not in ("source_file", "timing_ms")}
    cases.append(c)
out = {
    "generated_at": raw.get("generated_at"),
    "note": "Divergence snippets from external panels. Not official benchmark scores.",
    "case_studies": cases[:8],
}
(root / "site/data/case_studies.json").write_text(json.dumps(out, indent=2) + "\n")
print("synced site/data: leaderboard +", len(cases[:8]), "case studies")
PY
