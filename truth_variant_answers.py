"""Answer-level SDT per checkpoint and raw prompt variant (factuality.py --prompt-variant).

truth_stages.py reads the default file names only and writes outputs/truth_<dataset>/stages.csv
and outputs/truth_stages_contrasts.csv; this script leaves those untouched and reads any
prompt variant via its file-name suffix, so that a new layout can be put next to the default.

Per dataset x preset x checkpoint x prompt variant (raw format, both answer orders averaged
per item, as in truth_stages.py / sdt_summary.sdt):

    n, n_pairs     items and bootstrap groups (matched true/false pairs)
    d_auc, lo, hi  AUC-based d' with a grouped bootstrap CI over pairs
    c0, c0_lo/hi   criterion at the dv > 0 split (c > 0 = conservative = towards "False")
    dz             equal-variance d' of dv (does not saturate)
    share_true     share of statements answered "True";  p_mass  mean answer probability

Usage
-----
    python truth_variant_answers.py --datasets neg_cities cities \
        --presets olmo2-7b apertus15-8b --variants base sft --prompt-variants default sfirst \
        --out outputs/truth_neg_cities/sfirst_answers_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config
from sdt_summary import grouped_bootstrap, sdt

STAGES = ["base", "sft", "rl", "dpo", "instruct"]


def dz(y: np.ndarray, s: np.ndarray) -> float:
    t, f = s[y == 1], s[y == 0]
    return float((t.mean() - f.mean()) / np.sqrt((t.var(ddof=1) + f.var(ddof=1)) / 2))


def answers(out: Path, preset: str, variant: str, sfx: str) -> pd.DataFrame | None:
    fs = [out / f"{preset}_{variant}_raw_{o}{sfx}.csv" for o in ("tf", "ft")]
    if not all(f.exists() for f in fs):
        return None
    df = pd.concat([pd.read_csv(f) for f in fs])
    n = df.groupby("item_id").size()
    df = df[df.item_id.isin(n[n == 2].index)]      # both answer orders present
    return df.groupby(["item_id", "pair_id", "topic", "is_true"], as_index=False).agg(
        dv=("dv", "mean"), p_mass=("p_mass", "mean"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["neg_cities", "cities"])
    ap.add_argument("--presets", nargs="+", default=["olmo2-7b", "apertus15-8b"])
    ap.add_argument("--variants", nargs="+", default=STAGES)
    ap.add_argument("--prompt-variants", nargs="+", default=["default", "sfirst"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for ds in args.datasets:
        out = config.OUTPUT_DIR / f"truth_{ds}"
        for preset in args.presets:
            for v in args.variants:
                for pv in args.prompt_variants:
                    sfx = "" if pv == "default" else f"_{pv}"
                    a = answers(out, preset, v, sfx)
                    if a is None:
                        continue
                    y, s = a.is_true.to_numpy(), a.dv.to_numpy()
                    st = sdt(y, s)
                    ci = grouped_bootstrap(
                        a.rename(columns={"pair_id": "source_id", "is_true": "is_ai"}),
                        args.n_boot, rng)
                    rows.append(dict(dataset=ds, preset=preset, checkpoint=v, prompt_variant=pv,
                                     n=len(a), n_pairs=a.pair_id.nunique(),
                                     d_auc=st["d_auc"], lo=ci["d_auc_lo"], hi=ci["d_auc_hi"],
                                     dz=dz(y, s), c0=st["c0"], c0_lo=ci["c0_lo"], c0_hi=ci["c0_hi"],
                                     auc=st["auc"], share_true=float((s > 0).mean()),
                                     p_mass=float(a.p_mass.mean())))
    if not rows:
        raise SystemExit("no judgment files found for the requested combinations")
    res = pd.DataFrame(rows).sort_values(["dataset", "preset", "prompt_variant", "checkpoint"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    with pd.option_context("display.width", 220, "display.max_columns", None,
                           "display.float_format", "{:.2f}".format):
        print(res.to_string(index=False))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
