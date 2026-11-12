"""Statement set for factuality.py: "cities" from Marks & Tegmark (2024), The Geometry
of Truth (https://github.com/saprmarks/geometry-of-truth, datasets/cities.csv).

748 matched pairs: for each city, one true and one false statement of the form
"The city of X is in Y." The CSV is public but kept out of the repository
(data/external/geometry_of_truth/cities.csv, gitignored).
"""
from pathlib import Path

import pandas as pd

_CSV = Path(__file__).resolve().parents[1] / "data" / "external" / "geometry_of_truth" / "cities.csv"
_d = pd.read_csv(_CSV)
_true = _d[_d.label == 1].set_index("city").statement
_false = _d[_d.label == 0].set_index("city").statement
_cities = sorted(set(_true.index) & set(_false.index))

SYSTEM = "You evaluate statements."   # neutral, parallel to "Du bewertest Texte." (authorship)
TOPICS = ["cities"]
TARGET = "cities"
MATCHED = "cities"
PAIRS = [(f"c{i:04d}", "cities", _true[c], _false[c]) for i, c in enumerate(_cities)]
