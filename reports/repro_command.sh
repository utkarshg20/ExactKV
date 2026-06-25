#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python scripts/exactkv.py run benchmark --deterministic
python scripts/exactkv.py run leaderboard
python scripts/exactkv.py run publish
python scripts/exactkv.py plot all
echo "ExactKV public release bundle complete."
