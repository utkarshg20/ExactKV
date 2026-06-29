#!/usr/bin/env bash
# Build paper/ExactKV_Technical_Report.pdf from LaTeX.
# Requires: tectonic (brew install tectonic) OR a full TeX Live (pdflatex + bibtex).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/paper"

if command -v tectonic >/dev/null 2>&1; then
  echo "Building with tectonic..."
  tectonic ExactKV_Technical_Report.tex
elif command -v pdflatex >/dev/null 2>&1; then
  echo "Building with pdflatex..."
  pdflatex -interaction=nonstopmode ExactKV_Technical_Report.tex
  bibtex ExactKV_Technical_Report
  pdflatex -interaction=nonstopmode ExactKV_Technical_Report.tex
  pdflatex -interaction=nonstopmode ExactKV_Technical_Report.tex
else
  echo "ERROR: install tectonic: brew install tectonic" >&2
  exit 1
fi

echo "Wrote $ROOT/paper/ExactKV_Technical_Report.pdf"
