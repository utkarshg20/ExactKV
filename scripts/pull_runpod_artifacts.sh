#!/usr/bin/env bash
# Aggressive local pull of RunPod ExactKV artifacts (safe if pod dies mid-run).
#
# Usage:
#   RUNPOD_HOST=... RUNPOD_PORT=... bash scripts/pull_runpod_artifacts.sh
# or:
#   bash scripts/pull_runpod_artifacts.sh   # uses Host runpod-marathon / launch/pod_ssh.env
set -uo pipefail
cd "$(dirname "$0")/.."
KEY="${RUNPOD_KEY:-$HOME/.ssh/runpod_exactkv}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_ROOT="reports/remote_backups/${STAMP}"

if [[ -z "${RUNPOD_HOST:-}" || -z "${RUNPOD_PORT:-}" ]]; then
  if [[ -f launch/pod_ssh.env ]]; then
    set -a
    # shellcheck disable=SC1091
    source launch/pod_ssh.env
    set +a
    RUNPOD_HOST="${RUNPOD_HOST:-${HOST:-}}"
    RUNPOD_PORT="${RUNPOD_PORT:-${PORT:-}}"
  fi
fi
if [[ -z "${RUNPOD_HOST:-}" || -z "${RUNPOD_PORT:-}" ]]; then
  # fall back to ssh config alias
  if ssh -G runpod-marathon 2>/dev/null | grep -q '^hostname '; then
    RUNPOD_HOST=$(ssh -G runpod-marathon | awk '/^hostname /{print $2; exit}')
    RUNPOD_PORT=$(ssh -G runpod-marathon | awk '/^port /{print $2; exit}')
  fi
fi
if [[ -z "${RUNPOD_HOST:-}" || -z "${RUNPOD_PORT:-}" ]]; then
  echo "No RUNPOD_HOST/PORT — skip pull"
  exit 0
fi

RSYNC_SSH="ssh -p ${RUNPOD_PORT} -i ${KEY} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"
HOST="$RUNPOD_HOST"

echo "==> pull $HOST:$RUNPOD_PORT @ $STAMP"
mkdir -p "$BACKUP_ROOT" \
  reports/external_panels/serving_microbench \
  reports/external_panels/systems_diagnostic \
  reports/systems

# Live working copies (overwrite local with remote)
rsync -az -e "$RSYNC_SSH" \
  "root@${HOST}:/workspace/ExactKV/reports/external_panels/serving_microbench/" \
  reports/external_panels/serving_microbench/ 2>/dev/null || echo "WARN serving pull"

rsync -az -e "$RSYNC_SSH" \
  "root@${HOST}:/workspace/ExactKV/reports/external_panels/systems_diagnostic/" \
  reports/external_panels/systems_diagnostic/ 2>/dev/null || echo "WARN systems pull"

rsync -az -e "$RSYNC_SSH" \
  "root@${HOST}:/workspace/ExactKV/reports/systems/" \
  reports/systems/ 2>/dev/null || echo "WARN systems pack pull"

# Also pull marathon logs
rsync -az -e "$RSYNC_SSH" \
  "root@${HOST}:/workspace/marathon_6h.log" \
  "reports/remote_backups/marathon_6h.log" 2>/dev/null || true
rsync -az -e "$RSYNC_SSH" \
  "root@${HOST}:/workspace/ExactKV/reports/external_panels/serving_microbench/" \
  "$BACKUP_ROOT/serving_microbench/" 2>/dev/null || true
rsync -az -e "$RSYNC_SSH" \
  "root@${HOST}:/workspace/ExactKV/reports/external_panels/systems_diagnostic/" \
  "$BACKUP_ROOT/systems_diagnostic/" 2>/dev/null || true
rsync -az -e "$RSYNC_SSH" \
  "root@${HOST}:/workspace/ExactKV/reports/systems/" \
  "$BACKUP_ROOT/systems/" 2>/dev/null || true

# Rebuild packs from whatever we have locally
python3 scripts/build_serving_microbench_pack.py 2>/dev/null || true
python3 scripts/build_systems_diagnostic_pack.py 2>/dev/null || true

# Manifest
python3 - <<PY
from pathlib import Path
import json
stamp = "$STAMP"
backup = Path("reports/remote_backups") / stamp
live = []
for rel in [
    "reports/external_panels/serving_microbench",
    "reports/external_panels/systems_diagnostic",
    "reports/systems",
]:
    p = Path(rel)
    if not p.exists():
        continue
    for f in p.rglob("*"):
        if f.is_file():
            live.append({"path": str(f), "bytes": f.stat().st_size})
manifest = {"pulled_at": stamp, "n_live_files": len(live), "files": live}
backup.mkdir(parents=True, exist_ok=True)
(backup / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"backup -> {backup} ({len(live)} live report files)")
PY

echo "==> pull done $STAMP"
