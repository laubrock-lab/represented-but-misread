"""Position-specific activation patching between two checkpoints of one family:
does post-training change the representation of the TEXT, or how the
answer-side positions take it up?

The donor model (e.g. base) is run over every prompt; its residual stream at layer
k (hidden_states[k] = output of decoder layer k) is stored for all prompt
positions. The recipient model (e.g. SFT) is then run with its layer-k output
replaced by the donor's at one set of prompt positions only:

    text      the tokens of the text being judged
    pre       the instruction before the text
    post      everything after the text (question + "Antwort:")
    question  the question line, wherever it stands (empty where the prompt has none:
              authorship --prompt-variant qfirst puts it before the text, the truth
              prompts have one after the statement only with --prompt-variant sfirst)
    last      the last prompt token only
    all       all prompt positions
    none      no patching (recipient reference)

Prediction if post-training changes the answer-side processing (not the text
representation): patching base activations into SFT at `post` (or `last`) restores
base-level discrimination, patching at `text` does not. Self-patching (donor =
recipient) must reproduce `none` exactly — a built-in check (--self-check).

Texts stay on university machines and are never printed.

Usage (cluster)
---------------
    python patch_positions.py --corpus textbook --preset apertus15-8b --recipient sft --donor base --layer 8
Output: outputs/patch/<corpus>/<preset>_<recipient>-from-<donor>_L<k>.csv (one row per text x order x set)
True/false judgments (prompt and scoring of factuality.py): --corpus truth:<name>
(--limit = number of pairs); "text" = the statement, output in outputs/patch/truth_<name>/.
With --prompt-variant sfirst the truth prompt puts the statement first and the question
after it, so "question" is non-empty and "post" = question + answer cue; the output
file name gets the suffix _sfirst (the default layout keeps the existing names).

Layer sweep
-----------
--layers k1 k2 ... caches the donor states for all requested layers in one donor
pass (CPU, half precision) and loops over layers and position sets with the
recipient loaded once; one CSV per layer, named and formatted exactly as with
--layer. --out-root puts the files under outputs/<root>/<corpus>/ instead of
outputs/patch/<corpus>/. Memory of the cache is reported before the recipient is
loaded (n_layers x sum of prompt lengths x hidden size x 2 bytes).
"""

from __future__ import annotations

import argparse
import gc
import time

import numpy as np
import pandas as pd
import torch

import config
import factuality as F
from judge_logits import (build_prompt, candidate_ids, load_model, pick_device, score_one,
                          truncate_text)
from steer_logits import find_decoder_layers

SETS = ["none", "text", "pre", "post", "last", "all", "question"]


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


def prompt_positions(tok, prompt: str, add_special: bool, text: str,
                     qmarker: str | None = "Frage:") -> tuple[int, dict]:
    enc = tok(prompt, add_special_tokens=add_special, return_offsets_mapping=True)
    offs = np.array(enc["offset_mapping"])
    n = len(enc["input_ids"])
    start = prompt.rfind(text)
    end = start + len(text)
    in_text = (offs[:, 0] >= start) & (offs[:, 1] <= end) & (offs[:, 1] > offs[:, 0])
    t_idx = np.where(in_text)[0]
    first, last_t = (t_idx.min(), t_idx.max()) if len(t_idx) else (n, n)
    # the question line, found by its first words (authorship: "Frage: ..."; truth with
    # --prompt-variant sfirst: "Is this statement true or false?"), wherever it stands —
    # after the text by default, before it with --prompt-variant qfirst. Occurrences
    # inside the text itself are skipped; qmarker=None leaves the set empty (truth
    # prompts of the default layout, whose post-statement part is only the answer cue).
    q_idx = np.array([], int)
    k = prompt.find(qmarker) if qmarker else -1
    while k != -1:
        if not (start <= k < end):
            q_end = prompt.find("\n", k)
            q_end = len(prompt) if q_end == -1 else q_end
            q_idx = np.where((offs[:, 0] >= k) & (offs[:, 1] <= q_end) & (offs[:, 1] > offs[:, 0]))[0]
            break
        k = prompt.find(qmarker, k + 1)
    pos = {"text": t_idx, "pre": np.arange(0, first), "post": np.arange(last_t + 1, n),
           "last": np.array([n - 1]), "all": np.arange(n), "none": np.array([], int),
           "question": q_idx}
    return n, pos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--preset", required=True, choices=list(config.MODEL_PRESETS))
    ap.add_argument("--recipient", required=True)
    ap.add_argument("--donor", required=True)
    ap.add_argument("--layer", type=int, help="hidden_states index k (output of decoder layer k)")
    ap.add_argument("--layers", type=int, nargs="+",
                    help="several hidden_states indices in one job (one CSV per layer); "
                         "the donor is run once and its states are cached for all of them")
    ap.add_argument("--sets", nargs="+", default=SETS, choices=SETS)
    ap.add_argument("--max-text-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"],
                    help="precision of both models (float16 is faster on Apple silicon; the cluster runs of the paper used bfloat16)")
    ap.add_argument("--prompt-variant", default="default",
                    choices=sorted(set(config.RAW_VARIANTS) | set(F.RAW_VARIANTS)),
                    help="raw template variant: qfirst/para for the authorship prompts, "
                         "sfirst (statement first, question after it) for truth corpora; "
                         "adds a suffix to the output name")
    ap.add_argument("--out-root", default="patch",
                    help="write to outputs/<root>/<corpus>/ (default: patch)")
    ap.add_argument("--self-check", action="store_true",
                    help="sanity check: donor = recipient, so every set must reproduce the "
                         "unpatched dv; the maximum |dv - none| is reported per layer")
    ap.add_argument("--skip-existing", action="store_true",
                    help="leave layers whose output CSV already exists untouched (for restarts)")
    args = ap.parse_args()
    truth = args.corpus.startswith("truth:")
    # the two prompt families have their own variants; pick the matching one
    avail = F.RAW_VARIANTS if truth else config.RAW_VARIANTS
    if args.prompt_variant not in avail:
        ap.error(f"--prompt-variant {args.prompt_variant} is not defined for "
                 f"{'truth corpora' if truth else 'the authorship prompts'} "
                 f"(available: {', '.join(avail)})")
    (F.set_prompt_variant if truth else config.set_prompt_variant)(args.prompt_variant)

    layers = args.layers if args.layers else ([args.layer] if args.layer is not None else None)
    if layers and min(layers) < 1:
        ap.error("layers are hidden_states indices >= 1 (the output of decoder layer k); the embedding state (0) cannot be patched")
    if not layers:
        ap.error("give --layer or --layers")
    layers = sorted(dict.fromkeys(layers))
    donor_name = args.recipient if args.self_check else args.donor

    device = pick_device(args.device)
    ids = config.MODEL_PRESETS[args.preset]
    orders = tuple(F.ORDERS) if truth else ("mk", "km")
    if truth:     # true/false statements; --limit counts pairs
        texts = F.truth_items(args.corpus, args.limit)
    else:
        texts = pd.read_parquet(config.REPO / "data" / f"texts_{args.corpus}.parquet",
                                columns=["text_id", "source_id", "is_ai", "text"])
        if args.limit:
            texts = texts.head(args.limit)

    def make_prompt(tok, text, order):
        return F.build_prompt(tok, text, "raw", order) if truth else build_prompt(tok, text, "raw", order, "neutral")

    out_dir = config.OUTPUT_DIR / args.out_root / args.corpus.replace(":", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    pv = "" if args.prompt_variant == "default" else f"_{args.prompt_variant}"
    nlim = f"_n{args.limit}" if args.limit else ""

    def out_path(k: int):
        return out_dir / f"{args.preset}_{args.recipient}-from-{donor_name}_L{k}{nlim}{pv}.csv"

    todo = [k for k in layers if not (args.skip_existing and out_path(k).exists())]
    if not todo:
        print("nothing to do (all layers present)")
        return
    print(f"layers {todo} | sets {args.sets} | {len(texts)} texts x {len(orders)} orders", flush=True)

    # 1. donor activations at every requested layer k, for every prompt position
    t0 = time.time()
    tok, donor = load_model(ids[donor_name], device, getattr(torch, args.dtype))
    store, n_cached = {}, 0
    with torch.no_grad():
        for r in texts.itertuples(index=False):
            text = r.text if truth else truncate_text(tok, r.text, args.max_text_tokens)[0]
            for order in orders:
                prompt, add = make_prompt(tok, text, order)
                n, pos = prompt_positions(tok, prompt, add, text,
                                          F.question_marker() if truth else "Frage:")
                ids_t = tok(prompt, add_special_tokens=add, return_tensors="pt")["input_ids"].to(device)
                hs = donor(input_ids=ids_t, output_hidden_states=True, use_cache=False).hidden_states
                act = torch.stack([hs[k][0] for k in todo]).to("cpu", getattr(torch, args.dtype))
                store[(r.text_id, order)] = (prompt, add, pos, act)
                n_cached += act.numel()
    del donor
    gc.collect()
    torch.cuda.empty_cache() if device == "cuda" else None
    print(f"donor {donor_name} cached: {n_cached * 2 / 2**30:.2f} GiB "
          f"({len(store)} prompts x {len(todo)} layers) in {time.time() - t0:.0f}s", flush=True)

    # 2. recipient with patched positions, one pass per layer
    tok, model = load_model(ids[args.recipient], device, getattr(torch, args.dtype))
    dec = find_decoder_layers(model)
    cands, eow = (F.candidate_seqs(tok) if truth else candidate_ids(tok)), {}
    from sdt_summary import sdt

    def score(prompt, add):
        return (F.score(model, tok, prompt, add, cands, eow, device, False) if truth else
                score_one(model, tok, prompt, add, None, cands, eow, device, False))

    # the unpatched forward pass does not depend on the layer: compute it once and
    # reuse it for every empty position set ("none", and "question" where the prompt
    # has no question line). Identical values, 1 / len(sets) less compute per layer.
    plain: dict[tuple, dict] = {}
    for li, k in enumerate(todo):
        t0 = time.time()
        patcher = Patcher(dec[k - 1])
        rows = []
        with torch.no_grad():
            for r in texts.itertuples(index=False):
                for order in orders:
                    prompt, add, pos, act = store[(r.text_id, order)]
                    for s in args.sets:
                        idx = torch.tensor(pos[s], dtype=torch.long)
                        if len(idx):
                            patcher.idx, patcher.val = idx, act[li][idx]
                            res = score(prompt, add)
                        else:
                            patcher.idx = patcher.val = None
                            if (r.text_id, order) not in plain:
                                plain[(r.text_id, order)] = score(prompt, add)
                            res = plain[(r.text_id, order)]
                        rows.append(dict(text_id=r.text_id, source_id=r.source_id, is_ai=r.is_ai, order=order,
                                         set=s, n_pos=len(idx), dv=res["dv"], p_mass=res["p_mass"]))
        patcher.handle.remove()
        out = out_path(k)
        df = pd.DataFrame(rows)
        df.to_csv(out, index=False)

        print(f"L{k} ({time.time() - t0:.0f}s)", flush=True)
        avg = df.groupby(["set", "text_id", "is_ai"], as_index=False).agg(dv=("dv", "mean"), pm=("p_mass", "mean"))
        ref = avg[avg.set == "none"].set_index("text_id").dv
        worst = 0.0
        for s, g in avg.groupby("set"):
            d = sdt(g.is_ai.to_numpy(), g.dv.to_numpy())["d_auc"] if g.is_ai.nunique() == 2 else np.nan
            diff = np.abs(g.set_index("text_id").dv - ref.reindex(g.text_id).values)
            worst = max(worst, float(diff.max()))
            print(f"  {s:5s} d' {d:+.2f}  p_mass {g.pm.mean():.2f}  mean |dv - none| {diff.mean():.3f}")
        if args.self_check:
            print(f"  SELF-CHECK L{k}: max |dv - none| over all sets = {worst:.2e} "
                  f"({'OK' if worst < 1e-3 else 'FAILED'})")
        print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
