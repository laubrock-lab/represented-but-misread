"""Training study: does the trained readout leave the representation intact?

For each adapter run (outputs/training_study/<corpus>/olmo2-7b_ts<C>s<seed>_raw_mk_acts.npz)
and the OLMo 2 base model (outputs/<corpus>/olmo2-7b_base_raw_mk_acts.npz), on the three
authorship corpora: layer-wise probes within each model and probe transfer between them,
at the last prompt position and the mean over the text positions (checkpoint_transfer.py
functions). H3 predicts: within-model probes unchanged, no transfer deficit at the text
positions; H4: any deficit at the answer position.

Usage (cluster, CPU): python training_study/transfer_ts.py [--layer-step 4]
Output: outputs/training_study/transfer_ts.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import config  # noqa: E402
from checkpoint_transfer import load, transfer  # noqa: E402

CORPORA = ["chat_tw25", "journalism_seg", "textbook"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer-step", type=int, default=4)
    a = ap.parse_args()
    out = config.OUTPUT_DIR / "training_study" / "transfer_ts.csv"
    rows = []
    for corpus in CORPORA:
        base_f = config.OUTPUT_DIR / corpus / "olmo2-7b_base_raw_mk_acts.npz"
        items_a, acts_a = load(base_f)
        for f in sorted((config.OUTPUT_DIR / "training_study" / corpus).glob("olmo2-7b_ts*_raw_mk_acts.npz")):
            m = re.search(r"_ts(C\d)s(\d+)_raw_mk_acts", f.name)
            items_b, acts_b = load(f)
            ib = items_b.reset_index().set_index("id").loc[items_a.id, "index"].to_numpy()
            y, g = items_a.y.to_numpy(), items_a.grp.to_numpy()
            for site in acts_a:
                A, B = acts_a[site], acts_b[site][ib]
                n_layers = A.shape[1] - 1
                for layer in range(a.layer_step, n_layers + 1, a.layer_step):
                    r = transfer(A[:, layer].astype(np.float64), B[:, layer].astype(np.float64), y, g)
                    rows.append(dict(corpus=corpus, condition=m.group(1), seed=int(m.group(2)), site=site,
                                     layer=layer, depth=layer / n_layers, within_base=r["aa"], within_post=r["bb"],
                                     base_to_post=r["ab"], post_to_base=r["ba"]))
            print(f"{corpus} {f.name}: done", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
