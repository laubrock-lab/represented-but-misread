"""Assemble the manuscript parts into one Markdown file for pandoc.

Order: title (first line of abstract.md), abstract, introduction, methods, results,
discussion (each numbered figure placed after the paragraph that first mentions it;
images and captions from figures.md), references (cited works only, via citeproc),
appendix, supplementary figures.

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
title = abstract_md.splitlines()[0].lstrip("# ").strip()
abstract = abstract_md.split("## Abstract", 1)[1].strip()

figs, supp = {}, []
for para in section("figures.md").split("\n\n"):
    # "**Figure 2 (`stem`).**", "**Figure S2 (`stem`; ...).**"
    m = re.match(r"\*\*(Figure(?: S?\d+)?) \(([^`]*)`(fig\w+)`[^)]*\)\.\*\*\s*(.*)", para.strip(), flags=re.S)
    if m:
        label, pre, stem, caption = m.groups()
        caption = " ".join(caption.split())
        block = f"![**{label}.** {caption}](figures/{stem}.pdf){{width=100% height=68%}}"
        if re.fullmatch(r"Figure \d+", label):
            figs[int(label.split()[1])] = block
        else:
            supp.append(block)


def place_figures(text: str) -> str:
    """Insert each numbered figure after the paragraph in which it is first mentioned."""
    paras = text.split("\n\n")
    placed = set()
    out = []
    for para in paras:
        out.append(para)
        if para.lstrip().startswith(("#", ">", "|", "![")):
            continue
        for n in sorted(figs):
            if n not in placed and re.search(rf"Figure {n}(?![0-9])", para):
                out.append(figs[n])
                placed.add(n)
    missing = sorted(set(figs) - placed)
    assert not missing, f"figures never mentioned in the text: {missing}"
    return "\n\n".join(out)


main_text = place_figures("\n\n".join(section(f) for f in ("introduction.md", "methods.md", "results.md", "discussion.md")))
appendix = section("appendix.md")
header = f"""---
title: "{title}"
author: "Jochen Laubrock"
date: "{date.today():%d %B %Y}"
abstract: |
{chr(10).join('  ' + l for l in abstract.splitlines())}
bibliography: references/references.bib
csl: references/apa.csl
header-includes: |
  \\usepackage{{caption}}
  \\captionsetup[figure]{{labelformat=empty}}
  \\usepackage{{float}}
  \\usepackage{{newunicodechar}}
  \\usepackage{{etoolbox}}
  \\AtBeginEnvironment{{longtable}}{{\\footnotesize\\setlength{{\\tabcolsep}}{{4pt}}}}
  \\emergencystretch=3em
  \\newunicodechar{{→}}{{\\ensuremath{{\\rightarrow}}}}
  \\newunicodechar{{←}}{{\\ensuremath{{\\leftarrow}}}}
  \\newunicodechar{{√}}{{\\ensuremath{{\\surd}}}}
  \\newunicodechar{{≈}}{{\\ensuremath{{\\approx}}}}
  \\newunicodechar{{≤}}{{\\ensuremath{{\\leq}}}}
  \\newunicodechar{{≥}}{{\\ensuremath{{\\geq}}}}
  \\newunicodechar{{ᵃ}}{{\\textsuperscript{{a}}}}
link-citations: true
---

"""
out = (header + main_text + "\n\n# References\n\n::: {{#refs}}\n:::\n\n" .replace("{{", "{").replace("}}", "}")
       + appendix + "\n\n## Supplementary figures\n\n" + "\n\n".join(supp) + "\n")
(P / "build").mkdir(exist_ok=True)
(P / "build" / "draft.md").write_text(out, encoding="utf-8")
print(f"title: {title}\nfigures: {len(figs)} in the text, {len(supp)} supplementary\n-> {P / 'build' / 'draft.md'}")
