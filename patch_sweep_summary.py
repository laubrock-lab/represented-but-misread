"""Summary of the position x layer patching sweep (patch_positions.py --layers).

For every corpus x direction (recipient-from-donor) x layer x position set:

    d, lo, hi        AUC-based d' of the patched recipient, order-averaged dv per item,
                     with a grouped bootstrap CI over the item groups (source_id),
                     exactly as the patch panel of plots/fig_data.py
    d_recipient      the unpatched recipient on the same items (the "none" set of the
                     same file)
    d_donor          the unpatched donor on the same items (the "none" set of a
                     reverse-direction file, i.e. of the run in which the donor is
                     the recipient; "none" does not depend on the layer, so any layer
                     of that direction serves. NaN if no such file exists)
    transfer         (d - d_recipient) / (d_donor - d_recipient): the fraction of the
                     donor-recipient difference that the patch transfers; transfer_lo/hi
                     from the same bootstrap resamples (NaN if |d_donor - d_recipient|
                     is below --min-gap, where the ratio is not interpretable)
    shift            mean |dv - dv(none)| per item, p_mass  mean answer mass
    n, n_groups      items and bootstrap groups

Usage
-----
    python patch_sweep_summary.py                       # outputs/patch_sweep/*/*.csv
    python patch_sweep_summary.py --in-dir outputs/patch --n-boot 2000
Output: <in-dir>/patch_sweep_summary.csv (numbers and IDs only, no text).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config
from sdt_summary import sdt

NAME = re.compile(r"^(?P<preset>.+?)_(?P<recipient>\w+)-from-(?P<donor>\w+)_L(?P<layer>\d+)"
                  r"(?:_n(?P<limit>\d+))?(?:_(?P<variant>[a-z]+))?\.csv$")


def dprime(y: np.ndarray, s: np.ndarray) -> float:
    return sdt(y, s)["d_auc"] if len(np.unique(y)) == 2 else np.nan


def order_avg(df: pd.DataFrame) -> pd.DataFrame:
    """One row per set x item: dv averaged over both answer orders (removes position bias)."""
    return (df.groupby(["set", "text_id", "source_id", "is_ai"], as_index=False)
            .agg(dv=("dv", "mean"), p_mass=("p_mass", "mean")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, default=config.OUTPUT_DIR / "patch_sweep")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--min-gap", type=float, default=0.25,
                    help="minimum |d_donor - d_recipient| for which the transfer ratio is reported")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    data, unpatched = {}, {}
    for f in sorted(args.in_dir.glob("*/*.csv")):
        m = NAME.match(f.name)
        if not m:
            continue
        cols = set(pd.read_csv(f, nrows=0).columns)
        if not {"text_id", "source_id", "is_ai", "order", "set", "dv"} <= cols:
            continue
        g = m.groupdict()
        key = (f.parent.name, g["preset"], g["recipient"], g["donor"], int(g["layer"]),
               g["limit"] or "", g["variant"] or "")
        avg = order_avg(pd.read_csv(f, usecols=["text_id", "source_id", "is_ai", "order", "set", "dv", "p_mass"]))
        data[key] = avg
        # "none" = the recipient without any patch; it does not depend on the layer
        unpatched.setdefault((key[0], key[1], key[2], key[5], key[6]),
                             avg[avg.set == "none"].set_index("text_id").dv)
    if not data:
        raise SystemExit(f"no patching CSVs in {args.in_dir}")
    print(f"{len(data)} per-item CSVs in {args.in_dir}")

    rng = np.random.default_rng(args.seed)
    rows = []
    for key, avg in data.items():
        corpus, preset, rec, don, layer, limit, variant = key
        none = avg[avg.set == "none"].set_index("text_id")
        # the donor's own unpatched sensitivity: the "none" set of the reverse direction
        donor_dv = unpatched.get((corpus, preset, don, limit, variant), pd.Series(dtype=float))

        for s, g in avg.groupby("set"):
            g = g.set_index("text_id")
            y = g.is_ai.to_numpy()
            s_patch = g.dv.to_numpy()
            s_none = none.dv.reindex(g.index).to_numpy()
            s_don = donor_dv.reindex(g.index).to_numpy() if len(donor_dv) else np.full(len(g), np.nan)
            groups = g.source_id.to_numpy()
            idx_by_group = pd.Series(np.arange(len(g))).groupby(groups).indices
            keys = np.array(list(idx_by_group), dtype=object)

            d, d_rec, d_don = dprime(y, s_patch), dprime(y, s_none), (
                dprime(y, s_don) if not np.isnan(s_don).any() else np.nan)
            boot_d, boot_t = [], []
            for _ in range(args.n_boot):
                i = np.concatenate([idx_by_group[k] for k in rng.choice(keys, len(keys))])
                if y[i].min() == y[i].max():
                    continue
                b = dprime(y[i], s_patch[i])
                boot_d.append(b)
                if not np.isnan(d_don):
                    br, bd = dprime(y[i], s_none[i]), dprime(y[i], s_don[i])
                    boot_t.append((b - br) / (bd - br) if abs(bd - br) > args.min_gap else np.nan)
            gap = d_don - d_rec
            ok = abs(gap) > args.min_gap
            rows.append(dict(
                corpus=corpus, preset=preset, recipient=rec, donor=don, layer=layer,
                limit=limit, variant=variant, set=s, n=len(g), n_groups=len(keys),
                d=d, lo=float(np.quantile(boot_d, .025)), hi=float(np.quantile(boot_d, .975)),
                d_recipient=d_rec, d_donor=d_don, gap=gap,
                transfer=(d - d_rec) / gap if ok else np.nan,
                transfer_lo=float(np.nanquantile(boot_t, .025)) if boot_t and ok else np.nan,
                transfer_hi=float(np.nanquantile(boot_t, .975)) if boot_t and ok else np.nan,
                shift=float(np.abs(s_patch - s_none).mean()), p_mass=float(g.p_mass.mean())))

    res = pd.DataFrame(rows)
    # the prompt variant comes from the file-name suffix ("" = default layout, e.g. "sfirst"
    # for the statement-first truth prompt); it is a key, not a detail: rows of different
    # layouts must never be pooled
    res["variant"] = res["variant"].replace("", "default")
    res = res.sort_values(["corpus", "preset", "variant", "recipient", "donor", "layer", "set"])
    out = args.out or (args.in_dir / "patch_sweep_summary.csv")
    res.to_csv(out, index=False)
    with pd.option_context("display.width", 220, "display.max_columns", None,
                           "display.max_rows", 400, "display.float_format", "{:.2f}".format):
        print(res[["corpus", "preset", "variant", "recipient", "donor", "layer", "set", "n",
                   "d", "lo", "hi", "d_recipient", "d_donor", "transfer"]].to_string(index=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
