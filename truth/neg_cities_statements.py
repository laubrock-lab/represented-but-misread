"""Statement set for factuality.py: "neg_cities" from Marks & Tegmark (2024), The
Geometry of Truth (datasets/neg_cities.csv): negated statements "The city of X is not
in Y." — harder than cities. 748 matched pairs (one true, one false per city).
CSV kept out of the repository (data/external/geometry_of_truth, gitignored)."""
from pathlib import Path

import pandas as pd

_CSV = Path(__file__).resolve().parents[1] / "data" / "external" / "geometry_of_truth" / "neg_cities.csv"
_d = pd.read_csv(_CSV)
_true = _d[_d.label == 1].set_index("city").statement
_false = _d[_d.label == 0].set_index("city").statement
_cities = sorted(set(_true.index) & set(_false.index))

SYSTEM = "You evaluate statements."   # neutral, parallel to "Du bewertest Texte." (authorship)
TOPICS = ["neg_cities"]
TARGET = "neg_cities"
MATCHED = "neg_cities"
PAIRS = [(f"n{i:04d}", "neg_cities", _true[c], _false[c]) for i, c in enumerate(_cities)]
