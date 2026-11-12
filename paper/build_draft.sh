#!/bin/bash
# Build a PDF reading copy of the current paper drafts: paper/build/draft.pdf
# Usage (from anywhere):  bash paper/build_draft.sh [--open]
set -euo pipefail
cd "$(dirname "$0")"
python3 build_draft.py
pandoc build/draft.md -o build/draft.pdf \
    --filter pandoc-citeproc \
    --pdf-engine=xelatex \
    --resource-path=. \
    -V mainfont="Times New Roman" -V mathfont="STIX Two Math" \
    -V geometry:margin=2.5cm -V fontsize=11pt -V linestretch=1.15 \
    -V colorlinks=true -V linkcolor=blue -V urlcolor=blue
echo "-> $(pwd)/build/draft.pdf"
if [[ "${1:-}" == "--open" ]]; then open build/draft.pdf; fi
