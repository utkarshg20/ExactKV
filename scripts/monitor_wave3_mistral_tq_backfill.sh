#!/usr/bin/env bash
# Poll Mistral LongBench turboquant backfill; pull + rebuild when 576/576.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/reports/wave3_mistral_tq_monitor.log"
SSH_HOST="${RUNPOD_SSH:-runpod-a5000}"
INTERVAL="${WAVE3_TQ_MONITOR_INTERVAL:-120}"

mkdir -p "$(dirname "$LOG")"
echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] monitor_start" >>"$LOG"

poll_once() {
  ssh -o ConnectTimeout=25 -o BatchMode=yes "$SSH_HOST" '
    TMUX=$(tmux has-session -t wave3_mistral_tq 2>/dev/null && echo alive || echo dead)
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo n/a)
    PROG=$(python3 -c "
import json,glob,os
ok=0
for p in sorted(glob.glob(\"/workspace/ExactKV/reports/external_panels/faithful/wave3/*_wave3_raw.json\")):
    ok+=sum(1 for c in json.load(open(p)).get(\"cells\",[]) if c.get(\"status\")==\"ok\")
print(ok)
" 2>/dev/null)
    ML=$(python3 -c "
import json
from collections import Counter
d=json.load(open(\"/workspace/ExactKV/reports/external_panels/faithful/wave3/longbench_Mistral_7B_Instruct_v0_3_wave3_raw.json\"))
ok=[c for c in d.get(\"cells\",[]) if c.get(\"status\")==\"ok\"]
print(len(ok), dict(Counter(c.get(\"compressor_name\") for c in ok)))
" 2>/dev/null)
    LAST=$(grep -hE "\[longbench\] cell|WAVE3_MISTRAL_TQ_BACKFILL_DONE|Traceback|CUDA out of memory" \
      /workspace/ExactKV/reports/faithful_wave3_mistral_tq_backfill_*.log 2>/dev/null | tail -3)
    echo "tmux=$TMUX gpu=$GPU total_ok=$PROG/576 mistral_lb=$ML"
    echo "$LAST"
  ' 2>&1
}

while true; do
  TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  OUT=$(poll_once) || OUT="SSH_FAIL: ${OUT:-connection error}"
  { echo "[$TS]"; echo "$OUT"; echo "---"; } >>"$LOG"

  if echo "$OUT" | grep -q "WAVE3_MISTRAL_TQ_BACKFILL_DONE"; then
    echo "[$TS] backfill_complete" >>"$LOG"
    break
  fi
  if echo "$OUT" | grep -q "total_ok=576/576"; then
    echo "[$TS] grid_complete" >>"$LOG"
    break
  fi
  if echo "$OUT" | grep -q "tmux=dead" && echo "$OUT" | grep -q "mistral_lb=144"; then
    echo "[$TS] tmux_dead_grid_ok" >>"$LOG"
    break
  fi
  if echo "$OUT" | grep -qiE "SSH_FAIL|Traceback|CUDA out of memory"; then
    echo "[$TS] ALERT check log" >>"$LOG"
  fi
  sleep "$INTERVAL"
done

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] pulling..." >>"$LOG"
bash "$ROOT/scripts/pull_faithful_wave3_from_runpod.sh" >>"$LOG" 2>&1 || true
python3 "$ROOT/scripts/rebuild_wave3_panels.py" --dir "$ROOT/reports/external_panels/faithful/wave3" --write >>"$LOG" 2>&1
python3 "$ROOT/scripts/integrate_faithful_panel_results.py" --dir "$ROOT/reports/external_panels/faithful/wave3" --write >>"$LOG" 2>&1
echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] WAVE3_MONITOR_DONE" >>"$LOG"
