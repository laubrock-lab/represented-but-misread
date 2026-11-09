"""Go/no-go readout for the Qwen3-8B third-family replication.

Prints, per corpus: the output-level d'/c of each condition (order-averaged,
from the probe CSV's out_* columns) and the best probe layer at the `last`
site (readout position), with grouped-bootstrap CI, shuffled-label AUC, and
the layer-0 sanity check (d' must be ~ 0 there). Also prints the steering
summary for journalism_seg (class vs decision vs random) at the steered
layers.

Usage:  python readout_qwen3.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CORPORA = ["journalism_seg", "chat_tw25", "journalism_win80", "chat"]
STEER_LAYERS = [24, 30]          # inversion band: onset (67%) and deep (83%)


def main() -> None:
    for corpus in CORPORA:
        path = Path("outputs") / corpus / "probe_layers_qwen3-8b.csv"
        if not path.exists():
            print(f"===== {corpus}: missing")
            continue
        df = pd.read_csv(path)
        print(f"\n===== {corpus}")

        out = df.groupby("condition")[["out_d", "out_d_lo", "out_d_hi", "out_c"]].first()
        print(out.round(2).to_string())

        print("  -- probes, site=last (best layer by d_auc) --")
        for cond, g in df[df.site == "last"].groupby("condition"):
            best = g.loc[g.d_auc.idxmax()]
            l0 = g[g.layer == 0].iloc[0]
            print(f"  {cond:42s} L{int(best.layer):2d}/{int(best.n_layers)}"
                  f" ({best.layer / best.n_layers:4.0%})  d'={best.d_auc:5.2f}"
                  f" [{best.d_auc_lo:5.2f},{best.d_auc_hi:5.2f}]"
                  f"  shuf={best.auc_shuffled:.3f}  L0 d'={l0.d_auc:+.2f}")


def steering_readout() -> None:
    """Steering summary (journalism_seg, instruct raw) at the steered layers."""
    out = Path("outputs/journalism_seg")
    for layer in STEER_LAYERS:
        sfx = f"qwen3-8b_instruct_raw_mk_L{layer}_sd"
        s = pd.read_csv(out / f"steer_summary_{sfx}.csv")
        d = pd.read_csv(out / f"steer_decomp_{sfx}.csv")
        r = d[d.direction.str.startswith("random")]
        print(f"\n===== steering L{layer} (random directional "
              f"{r.directional.mean():+.3f} ± {r.directional.std():.3f})")
        key = s[s.direction.isin(["class", "decision"])]
        with pd.option_context("display.width", 200, "display.float_format",
                               "{:+.2f}".format):
            print(key.to_string(index=False))


if __name__ == "__main__":
    main()
    steering_readout()
