"""Assemble the paper drafts into one Markdown file for pandoc (reading copy, not the
submission version).

Order: title (option 1 of abstract.md), abstract, introduction, methods, results,
discussion, appendix (appendix.md; terminology tables, lens analyses, full review
results, steering and few-shot tables), figures (images + captions from figures.md), references (ALL entries of
references/references.bib -- the drafts use plain-text citations, so the list is not
filtered to the cited works).

Usage:  bash paper/build_draft.sh   (calls this script, then pandoc)
"""
import re
from datetime import date
from pathlib import Path

P = Path(__file__).resolve().parent
D = P / "draft"


def section(name: str) -> str:
    return (D / name).read_text(encoding="utf-8")


abstract_md = section("abstract.md")
titles = re.findall(r"^1\. (.+?)(?:\n   (.+?))?\n", abstract_md, flags=re.M)
title = " ".join(t for t in titles[0] if t) if titles else "Draft"
abstract = abstract_md.split("## Abstract", 1)[1].strip()

figs = []
for para in section("figures.md").split("\n\n"):
    # "**Figure 2 (`stem`).**", "**Figure S2 (`stem`; draft ...).**", "**Figure (two-stage lens, `stem`; ...).**"
    m = re.match(r"\*\*(Figure(?: S?\d+)?) \(([^`]*)`(fig\w+)`[^)]*\)\.\*\*\s*(.*)", para.strip(), flags=re.S)
    if m:
        label, pre, stem, caption = m.groups()
        if label == "Figure":                       # unnumbered draft figure: name it by its description
            label = f"Figure ({pre.strip(' ,')})"
        caption = " ".join(caption.split())
        figs.append(f"![**{label}.** {caption}](figures/{stem}.pdf){{width=100%}}\n\n\\clearpage\n")

body = "\n\n".join(section(f) for f in ("introduction.md", "methods.md", "results.md", "discussion.md", "appendix.md"))
header = f"""---
title: "{title}"
subtitle: "Reading copy, assembled {date.today():%d %b %Y} -- draft, do not distribute"
author: "Jochen Laubrock et al."
abstract: |
{chr(10).join('  ' + l for l in abstract.splitlines())}
bibliography: references/references.bib
nocite: "@*"
header-includes: |
  \\usepackage{{caption}}
  \\captionsetup[figure]{{labelformat=empty}}
link-citations: true
---

"""
out = (header + body + "\n\n# Figures\n\n" + "\n".join(figs)
       + "\n\n# References (all entries of references.bib)\n\n::: {#refs}\n:::\n")
(P / "build").mkdir(exist_ok=True)
(P / "build" / "draft.md").write_text(out, encoding="utf-8")
print(f"title: {title}\nfigures: {len(figs)}\n-> {P / 'build' / 'draft.md'}")
