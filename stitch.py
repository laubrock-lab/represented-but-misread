"""Model stitching between two checkpoints of the same family (e.g. base and SFT):
is the post-training change in the "representation" (lower layers) or in the
"readout" (upper layers)?

The stitched model = embedding + decoder layers 1..L_split from model A (lower),
decoder layers L_split+1..L + final norm + output layer from model B (upper). No
stitching weights: the residual stream at the split is passed on unchanged. This is
justified for checkpoints of one lineage (same coordinate system; text-token
representations of base and post-trained models have CKA 0.97–0.998 for OLMo 2 /
Apertus) and validated per stitch:
  * answer probability P(Mensch) + P(KI) (does the stitched model still answer?)
  * perplexity on the texts themselves, compared with the endpoint stitches
    (L_split = 0 ~ model B; L_split = L ~ model A)
Tied input/output embeddings are untied first, so the output layer stays B's.

Prediction if the post-training change sits in the readout: lower = post-trained,
upper = base reproduces base-level discrimination; lower = base, upper =
post-trained reproduces post-trained discrimination.

Texts stay on university machines and are never printed.

Usage (cluster)
---------------
    python stitch.py --corpus textbook --preset apertus15-8b --lower sft --upper base --split 16
Output: outputs/stitch/<corpus>/<preset>_<lower>-<upper>_L<split>_raw_{mk,km}.csv
        + ppl line in outputs/stitch/<corpus>/perplexity.csv
"""

from __future__ import annotations

import argparse
import gc

import numpy as np
import pandas as pd
import torch

import config
from judge_logits import load_model, pick_device, run_condition
from steer_logits import find_decoder_layers


@torch.no_grad()
def perplexity(model, tok, texts: list[str], device: str, max_tokens: int = 512) -> float:
    nll, n = 0.0, 0
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_tokens)["input_ids"].to(device)
        if ids.shape[1] < 2:
            continue
        out = model(input_ids=ids, labels=ids, use_cache=False)
        k = ids.shape[1] - 1
        nll += float(out.loss) * k
        n += k
    return float(np.exp(nll / n))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--preset", required=True, choices=list(config.MODEL_PRESETS))
    ap.add_argument("--lower", required=True, help="variant providing embedding + layers 1..split")
    ap.add_argument("--upper", required=True, help="variant providing layers split+1..L + head")
    ap.add_argument("--split", type=int, required=True, help="number of decoder layers from the lower model")
    ap.add_argument("--component", default="all", choices=["all", "attn", "mlp"],
                    help="all: embedding + whole layers 1..split from the lower model; attn / mlp: "
                         "only the attention (or MLP) block incl. its norm in layers 1..split, "
                         "everything else (incl. embedding) from the upper model")
    ap.add_argument("--n-ppl", type=int, default=40)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    ids = config.MODEL_PRESETS[args.preset]
    tok, model = load_model(ids[args.upper], device, torch.bfloat16)          # upper model B on GPU
    out_emb, in_emb = model.get_output_embeddings(), model.get_input_embeddings()
    if out_emb is not None and out_emb.weight.data_ptr() == in_emb.weight.data_ptr():
        out_emb.weight = torch.nn.Parameter(out_emb.weight.detach().clone())   # untie: head stays B's
        print("untied input/output embeddings")
    layers_b = find_decoder_layers(model)
    L = len(layers_b)
    if not 0 <= args.split <= L:
        raise SystemExit(f"--split must be in 0..{L}")

    _, lower = load_model(ids[args.lower], "cpu", torch.bfloat16)              # lower model A on CPU
    layers_a = find_decoder_layers(lower)
    def block(name: str) -> str:
        if "self_attn" in name or "attention_layernorm" in name:
            return "attn"
        if "mlp" in name or "feedforward_layernorm" in name:
            return "mlp"
        return "other"

    if args.component == "all":
        in_emb.weight.data.copy_(lower.get_input_embeddings().weight.data.to(device))
    copied = set()
    for i in range(args.split):
        sd_a = layers_a[i].state_dict()
        if args.component == "all":
            layers_b[i].load_state_dict(sd_a)
            continue
        sd_b = layers_b[i].state_dict()
        for k, v in sd_a.items():
            if block(k) == args.component:
                sd_b[k] = v
                copied.add(k)
        layers_b[i].load_state_dict(sd_b)
    if args.component != "all":
        print(f"copied {args.component} parameters per layer: {sorted(copied)}")
    del lower, layers_a
    gc.collect()
    model.eval()
    print(f"stitched {args.preset}: {args.lower} (embedding + layers 1..{args.split}) -> "
          f"{args.upper} (layers {args.split + 1}..{L} + head)")

    texts = pd.read_parquet(config.REPO / "data" / f"texts_{args.corpus}.parquet")
    if args.limit:
        texts = texts.head(args.limit)
    out_dir = config.OUTPUT_DIR / "stitch" / args.corpus
    out_dir.mkdir(parents=True, exist_ok=True)
    label = f"{args.lower}-{args.upper}_L{args.split}" + ("" if args.component == "all" else f"_{args.component}")
    ppl_texts = texts.sample(min(args.n_ppl, len(texts)), random_state=1)["text"].tolist()
    ppl = perplexity(model, tok, ppl_texts, device)
    print(f"perplexity on {len(ppl_texts)} texts: {ppl:.2f}")
    with open(out_dir / "perplexity.csv", "a") as f:
        f.write(f"{args.preset},{args.lower},{args.upper},{args.split},{L},{ppl:.4f},{len(ppl_texts)},{args.component}\n")
    for order in ("mk", "km"):
        run_condition(model, tok, texts, preset=args.preset, variant=label,
                      model_id=f"{ids[args.lower]}|{ids[args.upper]}@{args.split}", fmt="raw",
                      order=order, system="neutral", max_text_tokens=2048, device=device,
                      want_acts=False, out_dir=out_dir)


if __name__ == "__main__":
    main()
