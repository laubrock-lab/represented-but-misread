"""Step 3: steering — does the represented class direction causally drive the answer?

The probes show that AI vs. human is linearly decodable at the readout position,
while the model's answer is at or below chance. Steering tests whether the
answer pathway *uses* that direction. Two interventions on the residual stream
after decoder layer L (= hidden_states[L], the layer index of the probe tables),
applied at the last prompt token (default) or at all prompt tokens:

    add    h + alpha * Delta                        (bias: constant shift)
           Delta = mean(AI) - mean(human) along the class direction, so
           alpha = 1 moves a text by one class-mean distance.
           Prediction if the readout uses the direction: dv shifts (criterion c
           changes), d' stays the same.
    scale  h + (k - 1) * ((h . u) - m) * u          (gain on the class component)
           u = unit class direction, m = midpoint of the class means along u.
           k = 1 is no change, k = 0 removes the component, k > 1 amplifies it.
           Prediction if the readout uses the direction: d' increases with k.
           If d' does not move, the information is represented but not read out
           — the dissociation is causal, not just correlational.

Note: transformers records hidden_states before user hooks run, so the steered
representation does not show up in output_hidden_states; it is passed on to
the following layers (checked at startup by verify_alignment).

Directions are cross-fitted: for each text, Delta is estimated from the
activations (judge_logits.py *_acts.npz) of texts from *other* sources
(grouped K-fold), so no text steers itself. Controls with the same step size and procedure: a random direction (null
control, --controls random) and the decision direction, i.e. the direction
along which the model's own unsteered dv varies (positive control, --controls
decision). A class null effect only means "not used" if the decision direction
moves the answer at the same site.

Output: <out_dir>/steer_<stem>_L<layer>[_<positions>].csv (dv per text x order x
setting) and a summary with d' (AUC), c, mean dv per setting, grouped-bootstrap CIs.

Usage
-----
    python steer_logits.py outputs/journalism_seg --preset gemma3-4b --variant instruct
    python steer_logits.py outputs/chat_tw25 --preset gemma3-4b --variant instruct --layer 16
    python steer_logits.py outputs/journalism_seg --preset gemma3-4b --limit 6   # smoke test
True/false judgments (factuality.py outputs; prompt, answer words and scoring of the
statement set, e.g. "Truthful"/"Deceptive" for opspam): an out_dir named truth_<set>
switches to truth mode; orders tf/ft; --layer defaults to ~67 % depth.
    python steer_logits.py outputs/truth_neg_cities --preset olmo2-7b --variant base --controls decision random
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

import config
from judge_logits import (build_prompt, candidate_ids, condition_stem, load_model,
                          pick_device, score_one, truncate_text)
from sdt_summary import grouped_bootstrap, sdt

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **_):
        return x


# ---------------------------------------------------------------------------
# Hook
# ---------------------------------------------------------------------------

def find_decoder_layers(model) -> torch.nn.ModuleList:
    """The text decoder's layer stack (multimodal models also contain a vision tower)."""
    cfg = model.config
    n = getattr(cfg, "num_hidden_layers", None) or cfg.get_text_config().num_hidden_layers
    cands = [(name, m) for name, m in model.named_modules()
             if isinstance(m, torch.nn.ModuleList) and len(m) == n and name.endswith("layers")]
    if not cands:
        raise RuntimeError("Could not locate the decoder layer stack.")
    cands.sort(key=lambda x: ("language_model" not in x[0], len(x[0])))
    return cands[0][1]


class Steerer:
    """Forward hook on one decoder layer; parameters are set before each scoring."""

    def __init__(self, layer_module: torch.nn.Module):
        self.u: torch.Tensor | None = None     # unit direction [d]
        self.delta = 0.0                       # class-mean distance along u
        self.mid = 0.0                         # class midpoint along u
        self.mode, self.alpha = "add", 0.0
        self.pos: slice | None = None
        self.handle = layer_module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if self.u is None or self.pos is None:
            return output
        if self.mode == "add" and self.alpha == 0.0:
            return output
        if self.mode == "scale" and self.alpha == 1.0:
            return output
        h = output[0] if isinstance(output, tuple) else output
        hp = h[:, self.pos, :].float()
        u = self.u.to(hp.device)
        if self.mode == "add":
            hp = hp + self.alpha * self.delta * u
        else:
            proj = hp @ u - self.mid                               # [B, P]
            hp = hp + ((self.alpha - 1.0) * proj)[..., None] * u
        h = h.clone()
        h[:, self.pos, :] = hp.to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    def remove(self):
        self.handle.remove()


@torch.no_grad()
def verify_alignment(model, tok, layers, layer: int, device: str) -> None:
    """Check that hidden_states[layer] (what the directions were estimated on) is
    the input of decoder layer `layer`, i.e. the output we steer. transformers
    records hidden_states before user hooks run, so the check is done on the
    unsteered model with a pre-hook on the next layer."""
    if layer >= len(layers):
        return                                   # last layer: no next layer to check against
    ids = tok("Kurzer Test.", return_tensors="pt")["input_ids"].to(device)
    seen = {}
    def pre(mod, args, kwargs):
        seen["x"] = (args[0] if args else kwargs["hidden_states"]).detach()
    h = layers[layer].register_forward_pre_hook(pre, with_kwargs=True)
    out = model(input_ids=ids, output_hidden_states=True)
    h.remove()
    if not torch.allclose(out.hidden_states[layer].float(), seen["x"].float(), atol=1e-3):
        raise RuntimeError(f"hidden_states[{layer}] is not the input of decoder layer {layer}; "
                           "the steering hook would act on a different representation.")


# ---------------------------------------------------------------------------
# Cross-fitted directions
# ---------------------------------------------------------------------------

def crossfit_directions(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int,
                        control: str, seed: int, score: np.ndarray | None = None,
                        unit: str = "sd") -> list[dict]:
    """Per text: unit direction u, step size delta and midpoint along u,
    estimated without the text's own source.

    class     u = mean(AI) - mean(human)                     (what is represented)
    decision  u = mean(dv above median) - mean(dv below)     (what the answer follows;
              positive control: same site, same step size as the class direction)
    random    random unit vector                             (null control, isotropic)
    pcrandom  random unit vector within the top-20 principal components of the
              within-class-centred activations (matched null: lives in the
              high-variance subspace like the class direction does)
    split     mean difference between two random halves within each class
              (matched null: structured like a mean difference, no class info)
    classorth class direction with the decision direction projected out
              (does the class component act beyond the decision direction?)
    The step unit (delta) is common to all directions, so they take equal steps:
    unit "sd"    = pooled within-class SD of the projections onto the class
                   direction (natural spread; default)
    unit "delta" = class-mean distance (first runs; turned out to be only
                   0.2-0.5 SD, i.e. too small to be a strong intervention)
    """
    rng = np.random.default_rng(seed)
    out: list[dict | None] = [None] * len(y)
    for tr, te in GroupKFold(n_splits=min(n_splits, len(np.unique(groups)))).split(X, y, groups):
        mu1, mu0 = X[tr][y[tr] == 1].mean(0), X[tr][y[tr] == 0].mean(0)
        diff = mu1 - mu0
        dist = float(np.linalg.norm(diff))
        u_class = diff / dist
        p = X[tr] @ u_class
        sd = float(np.sqrt((p[y[tr] == 1].var() + p[y[tr] == 0].var()) / 2))
        delta = sd if unit == "sd" else dist               # common step size
        if control == "random":
            r = rng.standard_normal(X.shape[1])
            u, a, b = r / np.linalg.norm(r), mu1, mu0
        elif control == "pcrandom":
            Xc = X[tr] - np.where((y[tr] == 1)[:, None], mu1, mu0)    # within-class centred
            _, _, Vt = np.linalg.svd(Xc - Xc.mean(0), full_matrices=False)
            r = rng.standard_normal(min(20, Vt.shape[0])) @ Vt[:min(20, Vt.shape[0])]
            u, a, b = r / np.linalg.norm(r), mu1, mu0
        elif control == "split":
            half = np.zeros(len(tr), bool)
            for c in (0, 1):
                idx = np.where(y[tr] == c)[0]
                half[rng.permutation(idx)[: len(idx) // 2]] = True
            a, b = X[tr][half].mean(0), X[tr][~half].mean(0)
            u = (a - b) / np.linalg.norm(a - b)
        elif control in ("decision", "classorth"):
            hi = score[tr] > np.median(score[tr])
            a, b = X[tr][hi].mean(0), X[tr][~hi].mean(0)
            u_dec = (a - b) / np.linalg.norm(a - b)
            if control == "decision":
                u = u_dec
            else:
                v = u_class - (u_class @ u_dec) * u_dec
                u, a, b = v / np.linalg.norm(v), mu1, mu0
        else:
            u, a, b = u_class, mu1, mu0
        mid = float(((a + b) / 2) @ u)
        d = dict(u=torch.tensor(u, dtype=torch.float32), delta=delta, mid=mid,
                 cos_class=float(u @ u_class), dist=dist, sd=sd)
        for i in te:
            out[i] = d
    return out


def best_layer(out_dir: Path, stem: str) -> int:
    f = out_dir / "probe_layers.csv"
    if not f.exists():
        raise SystemExit(f"{f} not found — run probe_layers.py first or pass --layer.")
    p = pd.read_csv(f)
    p = p[(p["condition"] == stem) & (p["site"] == "last") & (p["layer"] > 0)]
    if p.empty:
        raise SystemExit(f"No probe results for {stem} — pass --layer.")
    return int(p.loc[p["d_auc"].idxmax(), "layer"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path, help="corpus output dir with *_acts.npz")
    ap.add_argument("--texts", type=Path, help="default: data/texts_<out_dir name>.parquet")
    ap.add_argument("--preset", default="gemma3-4b", choices=list(config.MODEL_PRESETS))
    ap.add_argument("--variant", default="instruct", choices=["base", "sft", "rl", "dpo", "instruct"])
    ap.add_argument("--fmt", default="raw", choices=["raw", "chat"])
    ap.add_argument("--system", default="neutral", choices=list(config.SYSTEM_PROMPTS))
    ap.add_argument("--orders", nargs="+", default=["mk", "km"], choices=list(config.ORDERS))
    ap.add_argument("--layer", type=int, help="hidden_states index (default: best probe layer)")
    ap.add_argument("--positions", default="last", choices=["last", "prompt"])
    ap.add_argument("--add", type=float, nargs="*", default=[-2, -1, -0.5, 0, 0.5, 1, 2],
                    help="alphas for additive steering (units of the class-mean distance)")
    ap.add_argument("--scale", type=float, nargs="*", default=[0, 0.5, 2, 4],
                    help="gains k for the class component (1 = unchanged)")
    ap.add_argument("--controls", nargs="*", default=["random"],
                    choices=["random", "decision", "pcrandom", "split", "classorth"],
                    help="random = null control; decision = positive control (direction the "
                         "model's own dv follows, same site and step size)")
    ap.add_argument("--n-random", type=int, default=1,
                    help="number of independent random directions (null distribution)")
    ap.add_argument("--suffix", default="", help="extra output suffix (e.g. _mnull), avoids overwriting")
    ap.add_argument("--only-controls", action="store_true",
                    help="run the control directions only (class run done separately)")
    ap.add_argument("--unit", default="sd", choices=["sd", "delta"],
                    help="step unit for --add: within-class SD along the class direction "
                         "(default) or the class-mean distance")
    ap.add_argument("--n-splits", type=int, default=7)
    ap.add_argument("--max-text-tokens", type=int, default=2048)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--limit", type=int, help="first N texts only (smoke test)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    truth = args.out_dir.name.startswith("truth_")
    if truth:      # true/false judgments: factuality.py layout and prompts
        import factuality as F
        items = F.truth_items("truth:" + args.out_dir.name.removeprefix("truth_"))   # sets prompts/labels
        if args.orders == ["mk", "km"]:
            args.orders = list(F.ORDERS)
        stem = f"{args.preset}_{args.variant}_raw_tf"
        acts = np.load(args.out_dir / f"{stem}_acts.npz")
        A = acts["acts"]
        layer = args.layer if args.layer is not None else round(.67 * (A.shape[1] - 1))
        meta = pd.DataFrame(dict(text_id=acts["item_id"].astype(str))).merge(
            items.assign(prompt_type=""), on="text_id", how="left", validate="1:1")
        X = A[:, layer, :].astype(np.float64)
    else:
        stem = condition_stem(args.preset, args.variant, args.fmt, "mk", args.system)
        acts = np.load(args.out_dir / f"{stem}_acts.npz")
        layer = args.layer if args.layer is not None else best_layer(args.out_dir, stem)
        meta = pd.DataFrame(dict(text_id=acts["text_id"], source_id=acts["source_id"],
                                 is_ai=acts["is_ai"]))
        texts_path = args.texts or config.DATA_DIR / f"texts_{args.out_dir.name}.parquet"
        texts = pd.read_parquet(texts_path, columns=["text_id", "prompt_type", "text"])
        meta = meta.merge(texts, on="text_id", how="left", validate="1:1")
        X = acts["acts_last"][:, layer, :].astype(np.float64)
    if layer < 1:
        raise SystemExit("--layer must be >= 1 (layer 0 = embeddings, not a decoder output).")
    if meta["text"].isna().any():
        raise SystemExit("Some activation texts are missing from the parquet file.")
    y, groups = meta["is_ai"].to_numpy(), meta["source_id"].to_numpy()
    score = None
    if {"decision", "classorth"} & set(args.controls):   # model's own unsteered dv, order-averaged
        if truth:
            js = [pd.read_csv(args.out_dir / f"{args.preset}_{args.variant}_raw_{o}.csv").rename(
                  columns={"item_id": "text_id"}) for o in ("tf", "ft")]
        else:
            js = [pd.read_csv(args.out_dir / f"{condition_stem(args.preset, args.variant, args.fmt, o, args.system)}.csv")
                  for o in ("mk", "km")]
        dv = pd.concat(js).groupby("text_id")["dv"].mean()
        score = meta["text_id"].map(dv).to_numpy(float)
        if np.isnan(score).any():
            raise SystemExit("Unsteered dv missing for some texts.")
    dirs = {"class": crossfit_directions(X, y, groups, args.n_splits, "class", args.seed,
                                         unit=args.unit)}
    for c in args.controls:
        names = ([f"{c}{r + 1}" for r in range(args.n_random)]
                 if c in ("random", "pcrandom", "split") else [c])
        for r, name in enumerate(names):
            dirs[name] = crossfit_directions(X, y, groups, args.n_splits, c, args.seed + 1000 * r,
                                             score, unit=args.unit)
            print(f"  {name} direction: mean cosine with class direction "
                  f"{np.mean([d['cos_class'] for d in dirs[name]]):+.2f}")
    if args.limit:
        meta = meta.head(args.limit)

    use = [d for d in dirs if not (args.only_controls and d == "class")]
    settings = [(d, "add", a) for d in use for a in args.add if d == "class" or a != 0] + \
               [(d, "scale", k) for d in use for k in args.scale if k != 1]
    if args.only_controls:           # unsteered reference, needed for the decomposition
        settings = [("class", "add", 0.0)] + settings
    device = pick_device(args.device)
    model_id = config.MODEL_PRESETS[args.preset][args.variant]
    tok, model = load_model(model_id, device, getattr(torch, args.dtype))
    layers = find_decoder_layers(model)
    verify_alignment(model, tok, layers, layer, device)
    steer = Steerer(layers[layer - 1])   # hidden_states[L] = output of layer L-1 = input of layer L
    cands, eow = (F.candidate_seqs(tok) if truth else candidate_ids(tok)), {}
    print(f"{stem}: layer {layer}, positions {args.positions}, {len(meta)} texts, "
          f"{len(settings)} settings x {len(args.orders)} orders; "
          f"step unit ({args.unit}) {np.mean([d['delta'] for d in dirs['class']]):.1f}; "
          f"class distance {np.mean([d['dist'] for d in dirs['class']]):.1f} = "
          f"{np.mean([d['dist'] / d['sd'] for d in dirs['class']]):.2f} SD; "
          f"mean residual norm {np.linalg.norm(X, axis=1).mean():.1f}")

    rows, t0 = [], time.time()
    for order in args.orders:
        for i, r in enumerate(tqdm(meta.itertuples(index=False), total=len(meta), desc=order)):
            if truth:
                prompt, add_special = F.build_prompt(tok, r.text, "raw", order)
            else:
                text, _, _ = truncate_text(tok, r.text, args.max_text_tokens)
                prompt, add_special = build_prompt(tok, text, args.fmt, order, args.system)
            n_prompt = len(tok(prompt, add_special_tokens=add_special)["input_ids"])
            steer.pos = slice(n_prompt - 1, n_prompt) if args.positions == "last" \
                else slice(0, n_prompt)
            for direction, mode, a in settings:
                d = dirs[direction][i]
                steer.u, steer.delta, steer.mid = d["u"], d["delta"], d["mid"]
                steer.mode, steer.alpha = mode, float(a)
                res = (F.score(model, tok, prompt, add_special, cands, eow, device, False) if truth else
                       score_one(model, tok, prompt, add_special, None, cands, eow, device, False))
                rows.append(dict(text_id=r.text_id, source_id=r.source_id, is_ai=r.is_ai,
                                 prompt_type=r.prompt_type, order=order, direction=direction,
                                 mode=mode, alpha=a, dv=res["dv"], p_mass=res["p_mass"]))
    steer.remove()
    print(f"{len(rows)} scorings in {time.time() - t0:.0f}s")

    df = pd.DataFrame(rows)
    sfx = f"_L{layer}" + ("_sd" if args.unit == "sd" else "") + \
          ("_prompt" if args.positions == "prompt" else "") + \
          ("_ctrl-" + "-".join(args.controls) if args.only_controls else "") + \
          ("_smoke" if args.limit else "") + args.suffix
    df.to_csv(args.out_dir / f"steer_{stem}{sfx}.csv", index=False)

    # summary: order-averaged dv per text and setting
    avg = (df.groupby(["direction", "mode", "alpha", "text_id", "source_id", "is_ai"],
                      as_index=False)["dv"].mean())
    rng = np.random.default_rng(args.seed)
    summ = []
    for (direction, mode, a), g in avg.groupby(["direction", "mode", "alpha"]):
        s = sdt(g["is_ai"].to_numpy(), g["dv"].to_numpy())
        ci = grouped_bootstrap(g.reset_index(drop=True), args.n_boot, rng) if not args.limit else {}
        summ.append(dict(direction=direction, mode=mode, alpha=a, d_auc=s["d_auc"],
                         d_lo=ci.get("d_auc_lo"), d_hi=ci.get("d_auc_hi"), c0=s["c0"],
                         mean_dv=g["dv"].mean(), n=len(g)))
    summ = pd.DataFrame(summ)
    # directional vs. non-specific part of each additive effect (vs. unsteered mean dv)
    ref = df[(df["mode"] == "add") & (df["alpha"] == 0)]
    if len(ref):
        base_dv = ref.groupby("text_id")["dv"].mean().mean()
        add = summ[summ["mode"] == "add"].set_index(["direction", "alpha"])["mean_dv"]
        decomp = []
        for (direction, a), v in add.items():
            if a > 0 and (direction, -a) in add.index:
                neg = add[(direction, -a)]
                decomp.append(dict(direction=direction, alpha=a,
                                   directional=(v - neg) / 2,
                                   nonspecific=(v + neg) / 2 - base_dv))
        decomp = pd.DataFrame(decomp)
        if len(decomp):
            decomp.to_csv(args.out_dir / f"steer_decomp_{stem}{sfx}.csv", index=False)
            rnd = decomp[decomp["direction"].str.startswith("random")]
            with pd.option_context("display.float_format", "{:+.2f}".format):
                print(decomp[~decomp["direction"].str.startswith("random")].to_string(index=False))
                if len(rnd):
                    print(rnd.groupby("alpha")[["directional", "nonspecific"]]
                          .agg(["mean", "std", "min", "max"]).to_string())
    summ.to_csv(args.out_dir / f"steer_summary_{stem}{sfx}.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:+.2f}".format):
        print(summ.to_string(index=False))
    print("Unsteered reference: direction=class, mode=add, alpha=0.")


if __name__ == "__main__":
    main()
