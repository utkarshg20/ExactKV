#!/usr/bin/env bash
# Wait until no Python process is running scripts/run_external_panel.py.
#
# Do not use bare `pgrep -f run_external_panel.py`: shell wait loops often embed
# that substring in argv and match themselves forever.
set -euo pipefail

INTERVAL="${1:-30}"

external_panel_running() {
  pgrep -f '[p]ython.*scripts/run_external_panel\.py' >/dev/null 2>&1
}

while external_panel_running; do
  sleep "$INTERVAL"
done
