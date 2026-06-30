#!/usr/bin/env bash
# Poll RunPod faithful panel every 3 minutes. Logs to reports/faithful_monitor.log
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MONITOR="$ROOT/reports/faithful_monitor.log"
SSH_HOST="${RUNPOD_SSH:-runpod-a5000}"
INTERVAL="${FAITHFUL_MONITOR_INTERVAL:-180}"

mkdir -p "$(dirname "$MONITOR")"

poll_once() {
  ssh -o ConnectTimeout=15 -o BatchMode=yes "$SSH_HOST" '
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "n/a")
    RUN=$(pgrep -fc "run_external_panel.py" 2>/dev/null || echo 0)
    LAST=$(grep -hE "cell [0-9]+/|SUCCESS|FAILED|FAITHFUL_PANEL_DONE|Faithful panel complete|Starting Llama|FAITHFUL_LLAMA_DONE" \
      /workspace/ExactKV/reports/faithful_panel.log \
      /workspace/ExactKV/reports/faithful_llama_panel.log 2>/dev/null | tail -3)
    LB=$(python3 -c "
import json, os, glob
parts = []
for p in sorted(glob.glob(\"/workspace/ExactKV/reports/external_panels/faithful/*_raw.json\")):
    if not os.path.isfile(p): continue
    d = json.load(open(p))
    parts.append(f\"{os.path.basename(p)}:{d.get('status')}:{len(d.get('cells',[]))}ok:{d.get('cells_run',0)}\")
print(\" | \".join(parts) if parts else \"no artifacts\")
" 2>/dev/null)
    echo "gpu=$GPU procs=$RUN"
    echo "artifacts=$LB"
    echo "$LAST"
  ' 2>&1
}

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] monitor_start" >> "$MONITOR"

while true; do
  TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  OUT=$(poll_once) || OUT="SSH_FAIL: ${OUT:-connection error}"
  {
    echo "[$TS]"
    echo "$OUT"
    echo "---"
  } >> "$MONITOR"

  if echo "$OUT" | grep -q "SSH_FAIL"; then
    echo "[$TS] ssh_fail — will retry" >> "$MONITOR"
  elif echo "$OUT" | grep -q "FAITHFUL_LLAMA_DONE"; then
    echo "[$TS] all_done" >> "$MONITOR"
    exit 0
  fi

  sleep "$INTERVAL"
done
