"""SAE latents per token: which text features drive the human heuristic part?

Runs Gemma 3 4B pt on each text alone (BOS + text, no judging prompt), takes
the residual stream after decoder layer 22 at every text token, encodes it with
Gemma Scope 2 SAEs (16k, JumpReLU; variants l0_big and l0_medium, the latter is
the one browsable on Neuronpedia as gemma-3-4b/22-gemmascope-2-res-16k) and averages the latent activations over
the tokens (the proper version of the text-mean code in human_cues.py, which
encoded the mean residual). Then, as in human_cues.py: within-class correlation
of every latent (active on >= 10 % of items) with the human heuristic part
h_res, ecological validity, BH-FDR. Texts are processed locally and never
printed.

Usage
-----
    python sae_tokens.py --root ../latent-detection --corpus chat
Output: outputs/human_encoding/sae_tokens_<corpus>_<variant>.csv (+ per-item codes .npz)
"""

from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd
import torch
from safetensors.numpy import load_file

import config
from human_cues import SAE_GLOB, bh, p_from_r, r_cols, within
from human_encoding import load_items
from judge_logits import load_model, pick_device
from steer_logits import find_decoder_layers

LAYER_MODULE = 22                     # model.layers.22.output, as in the SAE config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=__import__("pathlib").Path, default=config.REPO)
    ap.add_argument("--corpus", default="chat")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--variants", nargs="+", default=["big", "medium"])
    args = ap.parse_args()
    out = args.root / "outputs" / "human_encoding"
    d = load_items(args.corpus, args.root / "data", args.root / "outputs")
    text_col = "segment" if args.corpus == "journalism_seg" else "text"
    texts = pd.read_parquet(args.root / "data" / f"texts_{args.corpus}.parquet",
                            columns=["text_id", text_col]).set_index("text_id")[text_col]

    device = pick_device(args.device)
    saes = {}
    for v in args.variants:
        f = glob.glob(SAE_GLOB.replace("l0_big", f"l0_{v}") + "/params.safetensors")[0]
        p = load_file(f)
        saes[v] = [torch.tensor(p[k], dtype=torch.float32, device=device)
                   for k in ("w_enc", "b_enc", "threshold")]
    tok, model = load_model(config.MODEL_PRESETS["gemma3-4b"]["base"], device, torch.bfloat16)
    layer = find_decoder_layers(model)[LAYER_MODULE]
    grab = {}
    h = layer.register_forward_hook(
        lambda m, i, o: grab.__setitem__("x", (o[0] if isinstance(o, tuple) else o)[0]))

    codes = {v: [] for v in saes}
    l0 = {v: [] for v in saes}
    with torch.no_grad():
        for tid in d["text_id"]:
            ids = tok(texts[tid], return_tensors="pt", truncation=True,
                      max_length=args.max_tokens)["input_ids"].to(device)
            model(input_ids=ids, use_cache=False)
            x = grab["x"][1:].float()                       # text tokens, without BOS
            for v, (W_enc, b_enc, thr) in saes.items():
                pre = x @ W_enc + b_enc
                z = torch.where(pre > thr, pre, torch.zeros_like(pre))
                codes[v].append(z.mean(0).cpu().numpy())
                l0[v].append(float((z > 0).sum(1).float().mean()))
    h.remove()
    for v in saes:
        report(d, np.stack(codes[v]).astype(np.float64), np.mean(l0[v]), v, args, out)


def report(d: pd.DataFrame, Z: np.ndarray, l0: float, variant: str, args, out) -> None:
    np.savez(out / f"sae_tokens_{args.corpus}_{variant}.npz",
             text_id=d["text_id"].to_numpy(str), codes=Z)

    y, n = d["is_ai"].to_numpy(), len(d)
    freq = (Z > 0).mean(0)
    keep = np.where(freq >= 0.10)[0]
    Zk = Z[:, keep]
    r_h = r_cols(within(Zk, y), d["h_res"].to_numpy())
    p = p_from_r(r_h, n)
    res = pd.DataFrame(dict(latent=keep, freq=freq[keep], heuristic=r_h,
                            validity=r_cols(Zk, y.astype(float)),
                            utilization=r_cols(Zk, d["elogit"].to_numpy()), p=p, q=bh(p)))
    res = res.reindex(res["heuristic"].abs().sort_values(ascending=False).index)
    res.to_csv(out / f"sae_tokens_{args.corpus}_{variant}.csv", index=False)
    print(f"\n== {args.corpus}, SAE l0_{variant}: {n} texts, mean L0 per token {l0:.0f}; latents tested {len(keep)}, "
          f"q < .05: {(res.q < .05).sum()}, q < .10: {(res.q < .10).sum()}")
    print(res.head(args.top).round(3).to_string(index=False))
    if variant == "big":
        for lat in (1305, 14595, 12489):
            r = res[res.latent == lat]
            print(f"latent {lat}: " + (r.round(3).to_string(index=False, header=False) if len(r)
                                       else "active on < 10 % of texts"))


if __name__ == "__main__":
    main()
