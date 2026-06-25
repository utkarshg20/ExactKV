#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 scripts/exactkv_repro.py --reports-only
echo "ExactKV public release bundle complete (reports-only, no inference)."
