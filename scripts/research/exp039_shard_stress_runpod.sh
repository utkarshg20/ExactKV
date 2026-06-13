#!/usr/bin/env bash
# Experiment 039: Shard external-drafter stress panel on RunPod GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SHARD_DIR="${SHARD_REPO_PATH:-/root/shard}"
LOG="${EXP039_LOG:-/workspace/exp039_shard_stress.log}"
MAX_NEW="${EXP039_MAX_NEW_TOKENS:-64}"

exec > >(tee -a "$LOG") 2>&1
echo "=== Exp 039 Shard stress panel ==="
echo "date: $(date -Iseconds)"
echo "HF_TOKEN present: $(test -n "${HF_TOKEN:-}" && echo yes || echo no)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true

if [[ ! -d "$SHARD_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/krish1905/shard.git "$SHARD_DIR"
fi
export SHARD_REPO_PATH="$SHARD_DIR"

PYTHON="${EXP039_PYTHON:-python3}"
"$PYTHON" -m pip install -q -e "$SHARD_DIR" 2>/dev/null || "$PYTHON" -m pip install -q -e "$SHARD_DIR"

echo "--- dry plan ---"
"$PYTHON" scripts/research/run_exp039_shard_stress_panel.py --per-category 4 --max-prompts 48

echo "--- try-run max_new_tokens=$MAX_NEW ---"
"$PYTHON" scripts/research/run_exp039_shard_stress_panel.py --try-run \
  --device cuda \
  --dtype float16 \
  --max-new-tokens "$MAX_NEW" \
  --draft-len 4 \
  --per-category 4 \
  --max-prompts 48

echo "exp039_complete exit=0"
echo "Report: reports/experiment_039_shard_external_stress_panel.json"
