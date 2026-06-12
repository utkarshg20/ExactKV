#!/usr/bin/env bash
# Resume V12 Phase 1b after interrupted prep (GGUF + CLI smoke + gate only).
set -euo pipefail
WORKDIR="${TQ_PREP_WORKDIR:-/workspace/turboquant_prod_prep}"
export TQ_PREP_WORKDIR="$WORKDIR"
export SKIP_CLONE=1
export SKIP_BUILD=1
export GGUF_OUTTYPE="${GGUF_OUTTYPE:-auto}"
bash "$WORKDIR/run_prep.sh" 2>&1 | tee -a "$WORKDIR/prep_resume.log"
