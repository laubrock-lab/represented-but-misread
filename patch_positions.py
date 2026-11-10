"""Position-specific activation patching between two checkpoints of one family:
does post-training change the representation of the TEXT, or how the
answer-side positions take it up?

The donor model (e.g. base) is run over every prompt; its residual stream at layer
k (hidden_states[k] = output of decoder layer k) is stored for all prompt
positions. The recipient model (e.g. SFT) is then run with its layer-k output
replaced by the donor's at one set of prompt positions only:

    text  the tokens of the text being judged
    pre   the instruction before the text
    post  everything after the text (question + "Antwort:")
    last  the last prompt token only
    all   all prompt positions
    none  no patching (recipient reference)

Prediction if post-training changes the answer-side processing (not the text
representation): patching base activations into SFT at `post` (or `last`) restores
base-level discrimination, patching at `text` does not. Self-patching (donor =
recipient) must reproduce `none` exactly — a built-in check (--self-check).

Texts stay on university machines and are never printed.

Usage (cluster)
---------------
    python patch_positions.py --corpus textbook --preset apertus15-8b --recipient sft --donor base --layer 8
Output: outputs/patch/<corpus>/<preset>_<recipient>-from-<donor>_L<k>.csv (one row per text x order x set)
"""

from __future__ import annotations

import argparse
import gc

import numpy as np
import pandas as pd
import torch

import config
from judge_logits import (build_prompt, candidate_ids, load_model, pick_device, score_one,
                          truncate_text)
from steer_logits import find_decoder_layers

SETS = ["none", "text", "pre", "post", "last", "all"]


class Patcher:
    def __init__(self, layer_module):
        self.idx: torch.Tensor | None = None
        self.val: torch.Tensor | None = None
        self.handle = layer_module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if self.idx is None or len(self.idx) == 0:
            return output
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        h[0, self.idx] = self.val.to(h.device, h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h


def prompt_positions(tok, prompt: str, add_special: bool, text: str) -> tuple[int, dict]:
    enc = tok(prompt, add_special_tokens=add_special, return_offsets_mapping=True)
    offs = np.array(enc["offset_mapping"])
    n = len(enc["input_ids"])
    start = prompt.rfind(text)
    end = start + len(text)
    in_text = (offs[:, 0] >= start) & (offs[:, 1] <= end) & (offs[:, 1] > offs[:, 0])
    t_idx = np.where(in_text)[0]
    first, last_t = (t_idx.min(), t_idx.max()) if len(t_idx) else (n, n)
    pos = {"text": t_idx, "pre": np.arange(0, first), "post": np.arange(last_t + 1, n),
           "last": np.array([n - 1]), "all": np.arange(n), "none": np.array([], int)}
    return n, pos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--preset", required=True, choices=list(config.MODEL_PRESETS))
    ap.add_argument("--recipient", required=True)
    ap.add_argument("--donor", required=True)
    ap.add_argument("--layer", type=int, required=True, help="hidden_states index k (output of decoder layer k)")
    ap.add_argument("--sets", nargs="+", default=SETS, choices=SETS)
    ap.add_argument("--max-text-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    ids = config.MODEL_PRESETS[args.preset]
    texts = pd.read_parquet(config.REPO / "data" / f"texts_{args.corpus}.parquet",
                            columns=["text_id", "source_id", "is_ai", "text"])
    if args.limit:
        texts = texts.head(args.limit)

    # 1. donor activations at layer k for every prompt position
    tok, donor = load_model(ids[args.donor], device, torch.bfloat16)
    store = {}
    with torch.no_grad():
        for r in texts.itertuples(index=False):
            text = truncate_text(tok, r.text, args.max_text_tokens)[0]
            for order in ("mk", "km"):
                prompt, add = build_prompt(tok, text, "raw", order, "neutral")
                n, pos = prompt_positions(tok, prompt, add, text)
                ids_t = tok(prompt, add_special_tokens=add, return_tensors="pt")["input_ids"].to(device)
                hs = donor(input_ids=ids_t, output_hidden_states=True, use_cache=False).hidden_states
                store[(r.text_id, order)] = (prompt, add, pos, hs[args.layer][0].to("cpu", torch.bfloat16))
    del donor
    gc.collect()
    torch.cuda.empty_cache() if device == "cuda" else None

    # 2. recipient with patched positions
    tok, model = load_model(ids[args.recipient], device, torch.bfloat16)
    patcher = Patcher(find_decoder_layers(model)[args.layer - 1])
    cands, eow = candidate_ids(tok), {}
    rows = []
    with torch.no_grad():
        for r in texts.itertuples(index=False):
            for order in ("mk", "km"):
                prompt, add, pos, act = store[(r.text_id, order)]
                for s in args.sets:
                    idx = torch.tensor(pos[s], dtype=torch.long)
                    patcher.idx, patcher.val = (idx, act[idx]) if len(idx) else (None, None)
                    res = score_one(model, tok, prompt, add, None, cands, eow, device, False)
                    rows.append(dict(text_id=r.text_id, source_id=r.source_id, is_ai=r.is_ai, order=order,
                                     set=s, n_pos=len(idx), dv=res["dv"], p_mass=res["p_mass"]))
    patcher.handle.remove()
    out_dir = config.OUTPUT_DIR / "patch" / args.corpus
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.preset}_{args.recipient}-from-{args.donor}_L{args.layer}{'_smoke' if args.limit else ''}.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    from sdt_summary import sdt
    avg = df.groupby(["set", "text_id", "is_ai"], as_index=False).agg(dv=("dv", "mean"), pm=("p_mass", "mean"))
    ref = avg[avg.set == "none"].set_index("text_id").dv
    for s, g in avg.groupby("set"):
        d = sdt(g.is_ai.to_numpy(), g.dv.to_numpy())["d_auc"] if g.is_ai.nunique() == 2 else np.nan
        shift = np.abs(g.set_index("text_id").dv - ref.reindex(g.text_id).values).mean()
        print(f"  {s:5s} d' {d:+.2f}  p_mass {g.pm.mean():.2f}  mean |dv - none| {shift:.3f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
