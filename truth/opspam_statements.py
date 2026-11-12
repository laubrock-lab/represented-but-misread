"""Judgment set for factuality.py: truthful vs. deceptive hotel reviews, Deceptive Opinion
Spam Corpus v1.4 (Ott, Choi, Cardie & Hancock, 2011, ACL; Ott, Cardie & Hancock, 2013,
NAACL; CC BY-NC-SA 3.0). 20 Chicago hotels x {positive, negative} x 20 truthful (real
guests) + 20 deceptive (MTurk workers who never stayed there). Humans judge these close
to chance; n-gram classifiers reach ~90 % (Ott et al., 2011).

Third test case of representation vs. readout: a property of the text with known ground
truth that human judges cannot use, analogous to authorship.

Pairs: k-th truthful and k-th deceptive review of the same hotel and polarity (sorted by
file counter; the pairing itself is arbitrary). TOPIC = hotel, used as the grouping
factor for cross-validation and bootstrap (truth_stages.py). Signal class = truthful
("True" key -> answer word "Truthful").

Files kept out of the repository (data/external/op_spam, gitignored; public corpus).
"""
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1] / "data" / "external" / "op_spam" / "op_spam_v1.4"

SYSTEM = "You evaluate hotel reviews."
RAW = ("Is the following hotel review truthful or deceptive? A truthful review was written "
       "by a real guest about an actual stay; a deceptive review was written by someone who "
       "has never stayed at the hotel.\n\n"
       "Review: {s}\n\n"
       "Answer with one word ({opts}).\n"
       "Answer:")
USER = ("Is the following hotel review truthful or deceptive? A truthful review was written "
        "by a real guest about an actual stay; a deceptive review was written by someone who "
        "has never stayed at the hotel.\n\n"
        "Review: {s}\n\n"
        "Answer with exactly one word: {opts}.")
LABELS = ("Truthful", "Deceptive")      # (signal, noise)


def _load() -> dict:
    out = {}
    for f in _ROOT.glob("*_polarity/*/fold*/*.txt"):
        m = re.match(r"([td])_(\w+?)_(\d+)\.txt$", f.name)
        cls, hotel, i = m.group(1), m.group(2), int(m.group(3))
        pol = f.parts[-4].split("_")[0][:3]          # pos / neg
        out.setdefault((hotel, pol, cls), []).append((i, f.read_text(encoding="utf-8", errors="replace").strip()))
    return {k: [t for _, t in sorted(v)] for k, v in out.items()}


_D = _load()
TOPICS = sorted({h for h, _, _ in _D})
TARGET = MATCHED = TOPICS[0]
PAIRS = []
for (hotel, pol, cls), texts in sorted(_D.items()):
    if cls != "t":
        continue
    for k, (t, d) in enumerate(zip(texts, _D[(hotel, pol, "d")])):
        PAIRS.append((f"{hotel}_{pol}_{k:02d}", hotel, t, d))
