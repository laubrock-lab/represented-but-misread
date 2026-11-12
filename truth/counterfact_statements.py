"""Statement set for factuality.py: "counterfact_true_false" from Marks & Tegmark (2024),
The Geometry of Truth (datasets/counterfact_true_false.csv; derived from CounterFact,
Meng et al.). Diverse factual relations ("X was born in Y", ...). Pairs = one true and
one false statement with the same subject and relation; 748 pairs drawn with a fixed
seed (matching the size of the cities sets). CSV kept out of the repository."""
from pathlib import Path

import pandas as pd

_CSV = Path(__file__).resolve().parents[1] / "data" / "external" / "geometry_of_truth" / "counterfact_true_false.csv"
_d = pd.read_csv(_CSV)
_d["key"] = _d["subject"].astype(str) + " | " + _d["relation"].astype(str)
_both = _d.groupby("key").label.nunique()
_keys = pd.Series(sorted(_both[_both == 2].index)).sample(748, random_state=1).sort_values().tolist()
_g = _d[_d.key.isin(_keys)]
_true = _g[_g.label == 1].drop_duplicates("key").set_index("key").statement
_false = _g[_g.label == 0].drop_duplicates("key").set_index("key").statement

SYSTEM = "You evaluate statements."   # neutral, parallel to "Du bewertest Texte." (authorship)
TOPICS = ["counterfact"]
TARGET = "counterfact"
MATCHED = "counterfact"
PAIRS = [(f"f{i:04d}", "counterfact", _true[k], _false[k]) for i, k in enumerate(_keys)]
