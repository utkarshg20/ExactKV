#!/usr/bin/env bash
# Experiment 040: Shard external-drafter lossiness and length ablation on RunPod.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SHARD_DIR="${SHARD_REPO_PATH:-/root/shard}"
LOG="${EXP040_LOG:-/workspace/exp040_shard_ablation.log}"

exec > >(tee -a "$LOG") 2>&1
echo "=== Exp 040 Shard ablation ==="
echo "date: $(date -Iseconds)"
echo "HF_TOKEN present: $(test -n "${HF_TOKEN:-}" && echo yes || echo no)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true

if [[ ! -d "$SHARD_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/krish1905/shard.git "$SHARD_DIR"
fi
export SHARD_REPO_PATH="$SHARD_DIR"

PYTHON="${EXP040_PYTHON:-python3}"
"$PYTHON" -m pip install -q -e "$SHARD_DIR" 2>/dev/null || "$PYTHON" -m pip install -q -e "$SHARD_DIR"

echo "--- dry plan ---"
"$PYTHON" scripts/research/run_exp040_shard_ablation.py --per-category 4 --max-prompts 48

echo "--- try-run ablation ---"
"$PYTHON" scripts/research/run_exp040_shard_ablation.py --try-run \
  --device cuda \
  --dtype float16 \
  --draft-len 4 \
  --per-category 4 \
  --max-prompts 48

echo "exp040_complete exit=0"
echo "Report: reports/experiment_040_shard_external_ablation.json"
