"""Equivalence of the cached-prefix (batched) steering scores and the uncached ones.

Methods 2.7 states that the cached-prefix scoring "reproduced the scores of the uncached
computation (r = .999)". The number came from the paired run of slurm/steer_check.sh,
which scores the same texts and the same steering settings twice with steer_logits.py:

    steer_logits.py ... --suffix _chkA            uncached, one forward pass per setting
    steer_logits.py ... --batched --suffix _chkB  KV cache of the prompt prefix reused
                                                  for all settings, batched

Both runs were kept (outputs/<corpus>/steer_*_chk{A,B}.csv). This script only compares
the two saved tables; it needs no GPU and no model. For every such pair it writes one
row with the number of compared scores, the correlation of the decision variable dv and
the largest and mean absolute difference, plus the SD of dv as the scale against which
those differences should be read.

Usage:  python steer_cache_check.py
Output: outputs/steer_cache_check.csv
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["text_id", "order", "direction", "mode", "alpha"]


def compare(a_path: Path) -> dict:
    b_path = a_path.with_name(a_path.name.replace("_chkA.csv", "_chkB.csv"))
    a, b = pd.read_csv(a_path), pd.read_csv(b_path)
    m = a.merge(b, on=KEYS, suffixes=("_a", "_b"), validate="1:1")
    if len(m) != len(a) or len(m) != len(b):
        raise SystemExit(f"{a_path.name}: {len(a)} vs {len(b)} rows, {len(m)} matched")
    d = (m.dv_a - m.dv_b).abs()
    stem = re.match(r"steer_(.+?)_L(\d+)_", a_path.name)
    return dict(corpus=a_path.parent.name, cell=stem.group(1), layer=int(stem.group(2)),
                n_scores=len(m), n_items=m.text_id.nunique(),
                n_settings=m.groupby(["direction", "mode", "alpha"]).ngroups,
                n_orders=m.order.nunique(),
                r=float(np.corrcoef(m.dv_a, m.dv_b)[0, 1]),
                max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean()),
                dv_sd=float(m.dv_a.std()),
                max_abs_diff_p_mass=float((m.p_mass_a - m.p_mass_b).abs().max()))


def main() -> None:
    files = sorted(Path(f) for f in glob.glob("outputs/*/steer_*_chkA.csv")
                   if "summary" not in f and "decomp" not in f)
    if not files:
        raise SystemExit("No steer_*_chkA.csv found — run the paired check "
                         "(slurm/steer_check.sh) on a GPU node first.")
    res = pd.DataFrame([compare(f) for f in files])
    out = Path("outputs/steer_cache_check.csv")
    res.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(res.to_string(index=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
