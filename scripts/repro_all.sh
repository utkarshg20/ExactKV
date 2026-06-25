#!/usr/bin/env bash
# Thin wrapper around exactkv_repro.py (Phase J). Defaults to --release-check.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -eq 0 ]]; then
  exec python3 scripts/exactkv_repro.py --release-check
fi
exec python3 scripts/exactkv_repro.py "$@"
