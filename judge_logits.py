"""Logit-level human-vs-AI judgments plus residual-stream activations.

For every text, the model sees a prompt asking "Mensch oder KI?". Instead of
sampling an answer we read off the log-probabilities of the two answers,

    dv = log P("KI") - log P("Mensch")

which gives a continuous decision variable per text: the full ROC, d' from
the AUC, and the criterion (where dv = 0 falls) follow without any parsing.
In the same forward pass, the hidden states of all layers are stored for the
probe analysis (step 2):

    acts_last      residual stream at the last prompt token (where the
                   answer is read out)            [n_texts, n_layers+1, d]
    acts_textmean  mean over the tokens of the text itself  [same shape]

Conditions (one CSV per condition in outputs/<corpus>/):
    variant  base | instruct          (same family, same tokenizer)
    fmt      raw  | chat              (raw = completion prompt, identical for
                                       base and instruct; chat = instruct only)
    order    mk   | km                ("Mensch oder KI" vs "KI oder Mensch")

Everything runs locally; no text leaves the machine and no text is printed.

Usage
-----
    python judge_logits.py --texts data/texts_chat.parquet --preset gemma3-4b
    python judge_logits.py --texts data/texts_chat.parquet --preset gemma3-4b \\
        --variants instruct --formats chat --system expert
    python judge_logits.py --texts data/texts_chat.parquet --limit 10   # smoke test
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **_):
        return x


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_id: str, device: str, dtype: torch.dtype):
    tok = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    except ValueError:
        # Gemma 3 >= 4B is registered as an image-text-to-text model in some
        # transformers versions; the text path is identical.
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype)
    model.to(device).eval()
    return tok, model


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def truncate_text(tok, text: str, max_tokens: int) -> tuple[str, int, bool]:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text, len(ids), False
    return tok.decode(ids[:max_tokens]), max_tokens, True


def build_prompt(tok, text: str, fmt: str, order: str, system: str) -> tuple[str, bool]:
    """Return (prompt string, add_special_tokens flag for tokenization)."""
    opts = " oder ".join(config.ORDERS[order])
    if fmt == "raw":
        return config.RAW_TEMPLATE.format(text=text, opts=opts), True
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPTS[system]},
        {"role": "user", "content": config.CHAT_USER_TEMPLATE.format(text=text, opts=opts)},
    ]
    try:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)  # ignored unless Qwen3
    except Exception:
        # templates without a system role: fold the system prompt into the user turn
        merged = [{"role": "user",
                   "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}]
        prompt = tok.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)
    # The chat template already contains BOS; do not add it twice.
    return prompt, False


def candidate_ids(tok) -> dict[str, list[tuple[int, ...]]]:
    """Unique token sequences for each answer class."""
    out = {}
    for label, variants in config.CANDIDATES.items():
        seqs = {tuple(tok(v, add_special_tokens=False)["input_ids"]) for v in variants}
        out[label] = sorted(s for s in seqs if s)
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _forward(model, input_ids: torch.Tensor, keep: int, hidden: bool):
    kwargs = dict(input_ids=input_ids, use_cache=False, output_hidden_states=hidden)
    try:
        return model(**kwargs, logits_to_keep=keep)
    except TypeError:  # older architectures without logits_to_keep
        return model(**kwargs)


@torch.no_grad()
def score_one(model, tok, prompt: str, add_special: bool, text_span: tuple[int, int] | None,
              cands: dict[str, list[tuple[int, ...]]], device: str, want_acts: bool) -> dict:
    enc = tok(prompt, add_special_tokens=add_special, return_tensors="pt",
              return_offsets_mapping=text_span is not None and want_acts)
    input_ids = enc["input_ids"].to(device)
    n_prompt = input_ids.shape[1]

    out = _forward(model, input_ids, keep=1, hidden=want_acts)
    logp_next = torch.log_softmax(out.logits[0, -1].float(), dim=-1)

    # log P(answer class) = logsumexp over its token sequences
    class_lp = {}
    for label, seqs in cands.items():
        seq_lps = []
        for seq in seqs:
            lp = logp_next[seq[0]].item()
            if len(seq) > 1:  # rare: score remaining tokens with one extra pass
                ids = torch.cat([input_ids, torch.tensor([seq], device=device)], dim=1)
                o2 = _forward(model, ids, keep=len(seq) + 1, hidden=False)
                lps = torch.log_softmax(o2.logits[0, -(len(seq) + 1):-1].float(), dim=-1)
                lp = sum(lps[i, t].item() for i, t in enumerate(seq))
            seq_lps.append(lp)
        class_lp[label] = float(torch.logsumexp(torch.tensor(seq_lps), dim=0))

    res = dict(
        lp_mensch=class_lp["Mensch"],
        lp_ki=class_lp["KI"],
        dv=class_lp["KI"] - class_lp["Mensch"],
        p_mass=float(np.exp(class_lp["Mensch"]) + np.exp(class_lp["KI"])),
        n_prompt_tokens=n_prompt,
        top_tokens=[tok.decode([i]) for i in torch.topk(logp_next, 5).indices.tolist()],
    )

    if want_acts:
        hs = torch.stack([h[0] for h in out.hidden_states])  # [L+1, seq, d]
        res["act_last"] = hs[:, -1].float().cpu().numpy()
        mask = None
        if text_span is not None and "offset_mapping" in enc:
            offs = enc["offset_mapping"][0]
            start, end = text_span
            mask = (offs[:, 0] >= start) & (offs[:, 1] <= end) & (offs[:, 1] > offs[:, 0])
        if mask is None or not mask.any():
            res["act_textmean"] = hs.mean(dim=1).float().cpu().numpy()
            res["textmean_fallback"] = True
        else:
            res["act_textmean"] = hs[:, mask.to(hs.device)].mean(dim=1).float().cpu().numpy()
            res["textmean_fallback"] = False
    return res


def run_condition(model, tok, texts: pd.DataFrame, *, preset: str, variant: str,
                  model_id: str, fmt: str, order: str, system: str, max_text_tokens: int,
                  device: str, want_acts: bool, out_dir: Path) -> None:
    stem = f"{preset}_{variant}_{fmt}_{order}" + (f"_{system}" if fmt == "chat" else "")
    csv_path = out_dir / f"{stem}.csv"
    cands = candidate_ids(tok)
    print(f"\n=== {stem}  ({model_id})")
    print("    answer token sequences:",
          {k: [tok.decode(list(s)) for s in v] for k, v in cands.items()})

    rows, acts_last, acts_mean = [], [], []
    t0 = time.time()
    for i, r in enumerate(tqdm(texts.itertuples(index=False), total=len(texts), desc=stem)):
        text, n_tok, truncated = truncate_text(tok, r.text, max_text_tokens)
        prompt, add_special = build_prompt(tok, text, fmt, order, system)
        start = prompt.find(text)
        span = (start, start + len(text)) if start >= 0 else None
        res = score_one(model, tok, prompt, add_special, span, cands, device, want_acts)
        if i < 3:  # format sanity check — prints answer tokens only, never text
            print(f"    text {i}: top-5 next tokens {res['top_tokens']}  "
                  f"P(Mensch)+P(KI) = {res['p_mass']:.3f}")
        if want_acts:
            acts_last.append(res.pop("act_last"))
            acts_mean.append(res.pop("act_textmean"))
        res.pop("top_tokens")
        rows.append(dict(
            text_id=r.text_id, source_id=r.source_id, generator=r.generator,
            gen_model=r.gen_model, prompt_type=r.prompt_type, is_ai=r.is_ai,
            preset=preset, variant=variant, model_id=model_id, fmt=fmt, order=order,
            system=system if fmt == "chat" else "", n_text_tokens=n_tok,
            truncated=truncated, **res,
        ))

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    if want_acts:
        np.savez(out_dir / f"{stem}_acts.npz",
                 text_id=df["text_id"].to_numpy(dtype=str),
                 is_ai=df["is_ai"].to_numpy(),
                 source_id=df["source_id"].to_numpy(dtype=str),
                 acts_last=np.stack(acts_last).astype(np.float32),
                 acts_textmean=np.stack(acts_mean).astype(np.float32))
    print(f"    {len(df)} texts in {time.time() - t0:.0f}s; mean P(Mensch)+P(KI) = "
          f"{df['p_mass'].mean():.3f}; share 'KI' (dv>0) = {(df['dv'] > 0).mean():.2f}"
          f" -> {csv_path.relative_to(config.REPO) if csv_path.is_relative_to(config.REPO) else csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--texts", required=True, type=Path,
                    help="parquet or csv from prepare_texts.py")
    ap.add_argument("--preset", default="gemma3-4b", choices=list(config.MODEL_PRESETS))
    ap.add_argument("--base-id", help="override the base model id of the preset")
    ap.add_argument("--instruct-id", help="override the instruct model id of the preset")
    ap.add_argument("--variants", nargs="+", default=["base", "instruct"],
                    choices=["base", "instruct"])
    ap.add_argument("--formats", nargs="+", default=["raw", "chat"], choices=["raw", "chat"])
    ap.add_argument("--orders", nargs="+", default=["mk", "km"], choices=list(config.ORDERS))
    ap.add_argument("--system", default="neutral", choices=list(config.SYSTEM_PROMPTS))
    ap.add_argument("--acts-orders", nargs="*", default=["mk"],
                    help="orders for which activations are saved (none: pass no value)")
    ap.add_argument("--max-text-tokens", type=int, default=2048)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--limit", type=int, help="score only the first N texts (smoke test)")
    ap.add_argument("--out-dir", type=Path, help="default: outputs/<texts file stem>")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    texts = (pd.read_parquet(args.texts) if args.texts.suffix == ".parquet"
             else pd.read_csv(args.texts))
    if args.limit:
        texts = texts.head(args.limit)
    out_dir = args.out_dir or config.OUTPUT_DIR / args.texts.stem.removeprefix("texts_")
    if args.limit:
        out_dir = out_dir / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    dtype = getattr(torch, args.dtype)
    ids = dict(config.MODEL_PRESETS[args.preset])
    if args.base_id:
        ids["base"] = args.base_id
    if args.instruct_id:
        ids["instruct"] = args.instruct_id
    print(f"{len(texts)} texts ({texts['is_ai'].sum()} AI) | device {device} | {args.dtype}")

    for variant in args.variants:
        conditions = [(f, o) for f in args.formats for o in args.orders
                      if not (f == "chat" and variant == "base")]
        todo = []
        for fmt, order in conditions:
            stem = f"{args.preset}_{variant}_{fmt}_{order}" + (f"_{args.system}" if fmt == "chat" else "")
            if (out_dir / f"{stem}.csv").exists() and not args.overwrite:
                print(f"skip {stem} (exists; --overwrite to redo)")
            else:
                todo.append((fmt, order))
        if not todo:
            continue
        tok, model = load_model(ids[variant], device, dtype)
        for fmt, order in todo:
            run_condition(model, tok, texts, preset=args.preset, variant=variant,
                          model_id=ids[variant], fmt=fmt, order=order, system=args.system,
                          max_text_tokens=args.max_text_tokens, device=device,
                          want_acts=order in (args.acts_orders or []), out_dir=out_dir)
        del model
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
