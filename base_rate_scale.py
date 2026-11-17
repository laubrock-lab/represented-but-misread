"""Scale-free companion to base_rate_shift.py (17 Nov 2026).

The benchmark ln 16 applies to dv only if dv is a calibrated posterior log-odds. A model whose
dv is over- or under-dispersed (e.g., a post-trained model with more extreme logits) would shift
by more or fewer nats for that reason alone. This script therefore expresses the 20 -> 80% shift
of dv in units of the model's own evidence scale: the pooled within-class SD of dv across texts
(at the 50% prompt). It also reports that SD, and the SD a calibrated equal-variance observer
with the model's d' would show (= |d'| nats), as a calibration index.

Output: outputs/base_rate_scale.csv (per corpus x preset x variant x answer order x wording)
        outputs/base_rate_scale_models.csv (means per preset x variant x wording)
Usage:  python base_rate_scale.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from sdt_summary import sdt

PRESETS = ["gemma3-4b", "gemma3-12b", "qwen3-8b", "olmo2-7b", "apertus15-8b", "olmo3-7b-think"]
CORPORA = ["chat_tw25", "journalism_seg", "textbook"]


def main() -> None:
    rows = []
    for c in CORPORA:
        for p in PRESETS:
            for v in ("base", "instruct"):
                for o in ("mk", "km"):
                    for po in ("ai", "hu"):
                        f = {k: config.OUTPUT_DIR / c / f"{p}_{v}_raw_{o}_p{k}{po}.csv" for k in ("20", "50", "80")}
                        if not all(x.exists() for x in f.values()):
                            continue
                        d = {k: pd.read_csv(x, usecols=["text_id", "is_ai", "dv"]).set_index("text_id") for k, x in f.items()}
                        ids = d["20"].index.intersection(d["50"].index).intersection(d["80"].index)
                        y = d["50"].loc[ids, "is_ai"].to_numpy()
                        s50 = d["50"].loc[ids, "dv"].to_numpy()
                        shift = (d["80"].loc[ids, "dv"] - d["20"].loc[ids, "dv"]).to_numpy()
                        sd_w = np.sqrt(np.mean([s50[y == k].var(ddof=1) for k in (0, 1)]))
                        dp = sdt(y, s50)["d_auc"]
                        rows.append(dict(corpus=c, preset=p, variant=v, order=o, prior_order=po, n=len(ids),
                                         mean_shift=shift.mean(), sd_within=sd_w, shift_in_sd=shift.mean() / sd_w,
                                         d_auc=dp, sd_over_absd=sd_w / max(abs(dp), 1e-9),
                                         class_diff=s50[y == 1].mean() - s50[y == 0].mean()))
    r = pd.DataFrame(rows)
    r.to_csv(config.OUTPUT_DIR / "base_rate_scale.csv", index=False)
    m = r.groupby(["preset", "variant", "prior_order"]).agg(
        cells=("n", "size"), mean_shift=("mean_shift", "mean"), sd_within=("sd_within", "mean"),
        shift_in_sd=("shift_in_sd", "mean"), shift_in_sd_min=("shift_in_sd", "min"),
        shift_in_sd_max=("shift_in_sd", "max"), abs_class_diff=("class_diff", lambda x: x.abs().mean())).reset_index()
    m.to_csv(config.OUTPUT_DIR / "base_rate_scale_models.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(m.to_string(index=False))


if __name__ == "__main__":
    main()
