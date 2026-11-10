"""Answer gradients: how the model's own computation maps the residual stream onto
its answer (for the gain x angle readout decomposition).

For every text: dv = log P("KI") - log P("Mensch"), computed exactly as in
judge_logits.py (end-of-word corrected, log-sum-exp over the answer token
sequences), and its gradient with respect to the residual stream (all decoder
layer outputs, `hidden_states`) at the LAST PROMPT TOKEN. One forward + backward
pass per answer token sequence; the per-sequence gradients are combined with the
log-sum-exp weights, which gives the exact gradient of dv. No parameter gradients.

Answer order "Mensch oder KI" (mk) only, matching the activations saved by
judge_logits.py. See paper/notes_readout_decomposition.md for the analysis.
Texts are read locally / on the university cluster and never printed.

Usage
-----
    python gradient_readout.py outputs/textbook --preset apertus15-8b --variant sft --fmt raw
Output: outputs/<corpus>/grad_<condition stem>.npz
        (text_id, source_id, is_ai, dv, grad[n, n_layers+1, d] float32)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
from judge_logits import (build_prompt, candidate_ids, condition_stem, end_of_word_mask,
                          load_model, pick_device, truncate_text)


def answer_gradient(model, tok, prompt: str, add_special: bool, cands, eow: dict,
                    device: str) -> tuple[float, np.ndarray]:
    """Exact dv and d dv / d hidden_states[l] at the last prompt token, all l."""
    ids = tok(prompt, add_special_tokens=add_special, return_tensors="pt")["input_ids"].to(device)
    n_prompt = ids.shape[1]
    emb_layer = model.get_input_embeddings()
    lp, grads = {}, {}
    for label, seqs in cands.items():
        lp[label], grads[label] = [], []
        for seq in seqs:
            full = torch.cat([ids, torch.tensor([seq], device=device)], dim=1)
            emb = emb_layer(full).detach().requires_grad_(True)
            out = model(inputs_embeds=emb, output_hidden_states=True, use_cache=False)
            lps = torch.log_softmax(out.logits[0, -(len(seq) + 1):].float(), dim=-1)
            if "mask" not in eow:
                eow["mask"] = end_of_word_mask(tok, lps.shape[-1]).to(lps.device)
            lp_seq = sum(lps[i, t] for i, t in enumerate(seq))
            lp_word = lp_seq + torch.logsumexp(lps[-1][eow["mask"]], dim=0)
            hs = out.hidden_states
            g = torch.autograd.grad(lp_word, hs, allow_unused=True)
            gl = torch.stack([(gi[0, n_prompt - 1] if gi is not None else torch.zeros_like(h[0, 0]))
                              for gi, h in zip(g, hs)]).float().cpu()
            lp[label].append(float(lp_word.item()))
            grads[label].append(gl)
            del out, hs, g
    res = {}
    for label in cands:
        w = torch.softmax(torch.tensor(lp[label]), dim=0)          # d lse / d lp_seq
        res[label] = (float(torch.logsumexp(torch.tensor(lp[label]), dim=0)),
                      sum(wi * gi for wi, gi in zip(w, grads[label])))
    dv = res["KI"][0] - res["Mensch"][0]
    return dv, (res["KI"][1] - res["Mensch"][1]).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path, help="corpus output dir, e.g. outputs/textbook")
    ap.add_argument("--texts", type=Path, help="default: data/texts_<out_dir name>.parquet")
    ap.add_argument("--preset", required=True, choices=list(config.MODEL_PRESETS))
    ap.add_argument("--variant", default="instruct", choices=["base", "sft", "rl", "dpo", "instruct"])
    ap.add_argument("--fmt", default="raw", choices=["raw", "chat"])
    ap.add_argument("--system", default="neutral", choices=list(config.SYSTEM_PROMPTS))
    ap.add_argument("--max-text-tokens", type=int, default=2048)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    stem = condition_stem(args.preset, args.variant, args.fmt, "mk", args.system)
    texts = pd.read_parquet(args.texts or config.REPO / "data" / f"texts_{args.out_dir.name}.parquet",
                            columns=["text_id", "source_id", "is_ai", "text"])
    if args.limit:
        texts = texts.head(args.limit)
    device = pick_device(args.device)
    tok, model = load_model(config.MODEL_PRESETS[args.preset][args.variant], device,
                            getattr(torch, args.dtype))
    for p in model.parameters():
        p.requires_grad_(False)
    cands, eow = candidate_ids(tok), {}

    dvs, grads, t0 = [], [], time.time()
    for i, r in enumerate(texts.itertuples(index=False)):
        text = truncate_text(tok, r.text, args.max_text_tokens)[0]
        prompt, add_special = build_prompt(tok, text, args.fmt, "mk", args.system)
        dv, g = answer_gradient(model, tok, prompt, add_special, cands, eow, device)
        dvs.append(dv)
        grads.append(g.astype(np.float32))
        if i % 50 == 0:
            print(f"  {i + 1}/{len(texts)}  {time.time() - t0:.0f}s", flush=True)
    G = np.stack(grads)
    out = args.out_dir / f"grad_{stem}{'_smoke' if args.limit else ''}.npz"
    np.savez(out, text_id=texts.text_id.to_numpy(str), source_id=texts.source_id.to_numpy(str),
             is_ai=texts.is_ai.to_numpy(), dv=np.array(dvs), grad=G)

    # consistency check against judge_logits.py (dv must match up to numerics)
    ref = args.out_dir / f"{stem}.csv"
    if ref.exists():
        j = pd.read_csv(ref, usecols=["text_id", "dv"]).set_index("text_id")["dv"]
        r_ = np.corrcoef(np.array(dvs), j.reindex(texts.text_id).to_numpy())[0, 1]
        print(f"dv vs judge_logits: r = {r_:.4f}")
    print(f"{len(texts)} texts, grad shape {G.shape}, mean |grad| per layer "
          f"{np.round(np.linalg.norm(G, axis=2).mean(0)[[1, G.shape[1] // 2, -1]], 3)} -> {out}")


if __name__ == "__main__":
    main()
