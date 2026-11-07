"""Fluency hypothesis: does the model call surprising texts "KI" and fluent ones "Mensch"?

If AI-generated text is more predictable *for the judging model itself* and the
model uses "fluent = human", its answers invert. Per text, the model's own mean
surprisal (-mean log p per token of the text, BOS-only context) is computed for
base and instruct weights, then:

    (a) class difference:   AUC of surprisal for AI vs. human (< 0.5 = AI more fluent)
    (b) cue use:            within-class Spearman rho(surprisal, dv); > 0 = surprising
                            texts are called "KI" (the fluency heuristic)
    (c) mediation:          OLS dv ~ is_ai  vs.  dv ~ is_ai + surprisal (z-scored);
                            if the is_ai coefficient moves toward 0 / turns positive,
                            surprisal accounts for the inverted answers
    (d) representation:     per layer, alignment (correlation across dimensions) of the
                            class effect with the within-class surprisal effect

Output: <out_dir>/surprisal_<preset>.csv and a printed summary. No text is printed.

Usage
-----
    python fluency_check.py outputs/journalism_seg outputs/chat_tw25 --preset gemma3-4b
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import config
from alignment_profile import std_effect, within_class_decision_effect
from judge_logits import condition_stem, load_model, pick_device


@torch.no_grad()
def mean_surprisal(model, tok, text: str, device: str, max_tokens: int) -> tuple[float, int]:
    ids = tok(text, add_special_tokens=True, truncation=True, max_length=max_tokens,
              return_tensors="pt")["input_ids"].to(device)
    logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
    lp = torch.log_softmax(logits, -1).gather(1, ids[0, 1:, None])[:, 0]
    return float(-lp.mean()), int(ids.shape[1] - 1)


def ols_coef(y: np.ndarray, *xs: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones_like(y)] + list(xs))
    return np.linalg.lstsq(X, y, rcond=None)[0]


def order_avg_dv(out_dir: Path, preset: str, variant: str) -> pd.Series:
    stems = [condition_stem(preset, variant, "raw", o, "neutral") for o in ("mk", "km")]
    return pd.concat([pd.read_csv(out_dir / f"{s}.csv") for s in stems]).groupby("text_id")["dv"].mean()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dirs", type=Path, nargs="+")
    ap.add_argument("--preset", default="gemma3-4b", choices=list(config.MODEL_PRESETS))
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = pick_device(args.device)

    # 1) surprisal per text, base and instruct weights (cached in the output dir)
    for variant in ("base", "instruct"):
        todo = [d for d in args.out_dirs if not (d / f"surprisal_{args.preset}.csv").exists()
                or variant not in set(pd.read_csv(d / f"surprisal_{args.preset}.csv")["variant"])]
        if not todo:
            continue
        tok, model = load_model(config.MODEL_PRESETS[args.preset][variant], device, torch.bfloat16)
        for out_dir in todo:
            texts = pd.read_parquet(config.DATA_DIR / f"texts_{out_dir.name}.parquet",
                                    columns=["text_id", "text"])
            rows = []
            for r in texts.itertuples(index=False):
                s, n = mean_surprisal(model, tok, r.text, device, args.max_tokens)
                rows.append(dict(text_id=r.text_id, variant=variant, surprisal=s, n_tokens=n))
            f = out_dir / f"surprisal_{args.preset}.csv"
            new = pd.DataFrame(rows)
            if f.exists():
                old = pd.read_csv(f)
                new = pd.concat([old[old["variant"] != variant], new])
            new.to_csv(f, index=False)
            print(f"{out_dir.name}: surprisal under {variant} for {len(rows)} texts")
        del model
        if device == "mps":
            torch.mps.empty_cache()

    # 2) analysis
    for out_dir in args.out_dirs:
        s = pd.read_csv(out_dir / f"surprisal_{args.preset}.csv")
        meta = pd.read_parquet(config.DATA_DIR / f"texts_{out_dir.name}.parquet",
                               columns=["text_id", "source_id", "is_ai", "prompt_type"])
        print(f"\n=== {out_dir.name} ({args.preset})")
        for variant in ("base", "instruct"):
            d = meta.merge(s[s["variant"] == variant][["text_id", "surprisal"]], on="text_id")
            d["dv"] = d["text_id"].map(order_avg_dv(out_dir, args.preset, variant))
            d = d.dropna()
            y, sur, dv = d["is_ai"].to_numpy(), d["surprisal"].to_numpy(), d["dv"].to_numpy()
            by_type = d.groupby("prompt_type")["surprisal"].mean().round(2).to_dict()
            rho = {c: spearmanr(sur[y == c], dv[y == c])[0] for c in (0, 1)}
            z = (sur - sur.mean()) / sur.std()
            b1, b2 = ols_coef(dv, y)[1], ols_coef(dv, y, z)
            print(f"  {variant:8s} mean surprisal by type {by_type} | AUC(surprisal, AI) "
                  f"{roc_auc_score(y, sur):.2f}\n"
                  f"           cue use rho(surprisal, dv): human {rho[0]:+.2f}, AI {rho[1]:+.2f} | "
                  f"dv ~ is_ai: {b1:+.2f}  ->  + surprisal: is_ai {b2[1]:+.2f}, surprisal {b2[2]:+.2f}")
            # (d) representation: class effect vs. within-class surprisal effect, per layer
            f = out_dir / f"{condition_stem(args.preset, variant, 'raw', 'mk', 'neutral')}_acts.npz"
            if f.exists():
                zf = np.load(f)
                sur_map = d.set_index("text_id")["surprisal"]
                keep = np.isin(zf["text_id"], sur_map.index)
                A, yy = zf["acts_last"][keep], zf["is_ai"][keep].astype(int)
                ss = sur_map.reindex(zf["text_id"][keep]).to_numpy()
                al = [np.corrcoef(std_effect(A[:, l].astype(float), yy == 1),
                                  within_class_decision_effect(A[:, l].astype(float), yy, ss))[0, 1]
                      for l in range(1, A.shape[1])]
                n = len(al)
                print(f"           alignment(class, surprisal) early/middle/late: "
                      f"{np.mean(al[:n // 3]):+.2f} / {np.mean(al[n // 3:2 * n // 3]):+.2f} / "
                      f"{np.mean(al[2 * n // 3:]):+.2f}")


if __name__ == "__main__":
    main()
