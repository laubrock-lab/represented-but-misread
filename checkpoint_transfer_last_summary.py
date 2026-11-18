"""Decodability at the last prompt position before and after post-training (Section 3.6).

From outputs/checkpoint_transfer_authorship*.csv (checkpoint_transfer.py): mean within-checkpoint
probe d' at the last prompt position in the upper third of the network (depth >= 2/3), base vs.
final model, per model family and authorship corpus.
Output: outputs/checkpoint_transfer_last_upper_third.csv
Usage:  python checkpoint_transfer_last_summary.py
"""
import glob

import pandas as pd

import config

PRESETS = ["gemma3-4b", "gemma3-12b", "qwen3-8b", "olmo2-7b", "apertus15-8b", "olmo3-7b-think"]
d = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(str(config.OUTPUT_DIR / "checkpoint_transfer_authorship*.csv"))))
x = d[(d.site == "last") & (d.depth >= 2 / 3) & (d.post == "instruct") & d.preset.isin(PRESETS)]
g = x.groupby(["preset", "set"]).agg(d_base=("within_base", "mean"), d_final=("within_post", "mean"),
                                     n_layers=("layer", "nunique")).reset_index()
g["change"] = g.d_final - g.d_base
g.to_csv(config.OUTPUT_DIR / "checkpoint_transfer_last_upper_third.csv", index=False)
print(g.round(2).to_string(index=False))
