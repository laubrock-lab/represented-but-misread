#!/bin/bash
# Build the manuscript: paper/build/draft.pdf, and the LaTeX source for arXiv in
# paper/build/arxiv/ (draft.tex + figures/, compiles with xelatex; fonts from TeX Live).
# Usage (from anywhere):  bash paper/build_draft.sh [--open]
set -euo pipefail
cd "$(dirname "$0")"
python3 build_draft.py
OPTS=(--filter pandoc-citeproc --resource-path=.
      -V mainfont="STIXTwoText" -V mainfontoptions="Extension=.otf, UprightFont=*-Regular, BoldFont=*-Bold, ItalicFont=*-Italic, BoldItalicFont=*-BoldItalic"
      -V mathfont="STIX Two Math"
      -V geometry:margin=2.5cm -V fontsize=11pt -V linestretch=1.15
      -V colorlinks=true -V linkcolor=blue -V urlcolor=blue)
pandoc build/draft.md -o build/draft.pdf --pdf-engine=xelatex "${OPTS[@]}"
echo "-> $(pwd)/build/draft.pdf"
rm -rf build/arxiv && mkdir -p build/arxiv/figures
pandoc build/draft.md -s -o build/arxiv/draft.tex "${OPTS[@]}"
for f in $(grep -o "figures/[A-Za-z0-9_]*\.pdf" build/arxiv/draft.tex | sort -u); do cp "$f" build/arxiv/figures/; done
echo "-> $(pwd)/build/arxiv/ (LaTeX source and $(ls build/arxiv/figures | wc -l | tr -d ' ') figures)"
if [[ "${1:-}" == "--open" ]]; then open build/draft.pdf; fi
