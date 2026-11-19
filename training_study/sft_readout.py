"""Training study: LoRA SFT of a base model on background chat data plus (condition-
dependent) authorship-judgment examples. Tests whether the training signal sets the
direction of the readout (manuscript, Sections 2.9 and 3.7).

Data per run: --bg-n Tülu 3 conversations (loss on assistant tokens, rendered with the
chat template of the official SFT checkpoint) + the judgment examples of --condition
(training_study/judgment_examples.py; raw authorship prompt, loss on the answer word),
shuffled together, one epoch. LoRA on attention and MLP projections. Adapters saved at
fractions of training (ckpt-XXXX); a quick evaluation (d' on 100 held-out GNAD items,
raw prompt) is logged at every checkpoint.

Usage (cluster):
    python training_study/sft_readout.py --condition C1 --seed 1 --out-dir adapters/ts/olmo2_C1_s1
    python training_study/sft_readout.py --condition C1 --max-steps 4 --bg-n 50 --out-dir /tmp/x  # smoke
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "training_study"))
import config  # noqa: E402
from judge_logits import build_prompt, candidate_ids, load_model, pick_device, score_one  # noqa: E402
from judgment_examples import examples  # noqa: E402

TARGET_MODULES = r"^(?!.*vision).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def encode_chat(tok, messages, max_len):
    """input ids and labels (-100 except assistant tokens) for one conversation."""
    ids, labels = [], []
    prev = ""
    for i, m in enumerate(messages):
        upto = tok.apply_chat_template(messages[: i + 1], tokenize=False)
        seg = upto[len(prev):] if upto.startswith(prev) else upto
        seg_ids = tok(seg, add_special_tokens=False)["input_ids"]
        ids += seg_ids
        labels += seg_ids if m["role"] == "assistant" else [-100] * len(seg_ids)
        prev = upto
    return ids[:max_len], labels[:max_len]


def encode_judgment(tok, text, label, order, max_len, task="authorship"):
    if task == "topic":      # C4: unrelated judgment in the same raw layout
        from judgment_examples import TOPIC_ORDERS, TOPIC_TEMPLATE
        prompt, add_special = TOPIC_TEMPLATE.format(text=text, opts=" oder ".join(TOPIC_ORDERS[order])), True
    else:
        prompt, add_special = build_prompt(tok, text, "raw", order, "neutral")
    p = tok(prompt, add_special_tokens=add_special)["input_ids"]
    a = tok(" " + label, add_special_tokens=False)["input_ids"]
    p = p[-(max_len - len(a)):]
    return p + a, [-100] * len(p) + a


def quick_eval(model, tok, items, device):
    cands, eow = candidate_ids(tok), {}
    model.eval()
    dv = []
    with torch.no_grad():
        for r in items.itertuples(index=False):
            prompt, add = build_prompt(tok, r.text, "raw", "mk", "neutral")
            dv.append(score_one(model, tok, prompt, add, None, cands, eow, device, False)["dv"])
    model.train()
    from sklearn.metrics import roc_auc_score
    from scipy.stats import norm
    auc = roc_auc_score(items.is_ai, dv)
    return dict(auc=round(auc, 3), d=round(float(np.sqrt(2) * norm.ppf(np.clip(auc, .005, .995))), 3),
                share_ki=round(float(np.mean(np.array(dv) > 0)), 3))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="olmo2-7b")
    ap.add_argument("--template-variant", default="sft", help="checkpoint whose chat template is used")
    ap.add_argument("--condition", required=True, choices=["C0", "C1", "C2", "C3", "C4"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--bg-n", type=int, default=20_000)
    ap.add_argument("--n-examples", type=int, help="subsample the judgment examples (dose)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--save-fracs", type=float, nargs="+", default=[0, .25, .5, 1])
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = pick_device(a.device)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    ids = config.MODEL_PRESETS[a.preset]
    tok, model = load_model(ids["base"], device, torch.bfloat16)
    tmpl = AutoTokenizer.from_pretrained(ids[a.template_variant]).chat_template
    if tok.chat_template is None:
        tok.chat_template = tmpl
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.lora_alpha, lora_dropout=0.05,
                                             target_modules=TARGET_MODULES, task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    with gzip.open(REPO / "data" / "training_study" / "tulu_bg.jsonl.gz", "rt") as fh:
        bg = [json.loads(l)["messages"] for _, l in zip(range(a.bg_n), fh)]
    data = [("bg",) + encode_chat(tok, m, a.max_len) for m in bg]
    ex = examples(a.condition, seed=a.seed)
    if a.n_examples and len(ex):
        ex = ex.sample(a.n_examples, random_state=a.seed)
    data += [("judg",) + encode_judgment(tok, r.text, r.label, random.choice(["mk", "km"]), a.max_len,
                                         getattr(r, "task", "authorship"))
             for r in ex.itertuples(index=False)]
    data = [d for d in data if any(x != -100 for x in d[2])]
    random.shuffle(data)
    total = math.ceil(len(data) / a.grad_accum) if a.max_steps is None else a.max_steps
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)))
    save_at = {int(round(f * total)) for f in a.save_fracs}
    held = pd.read_parquet(REPO / "data" / "texts_gnad_heldout.parquet").groupby("is_ai").head(50)
    meta = dict(vars(a), out_dir=str(a.out_dir), n_bg=len(bg), n_judgment=len(ex), total_steps=total, device=device)
    (a.out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log = open(a.out_dir / "train_log.jsonl", "a")
    print(f"{len(data)} sequences ({len(bg)} background, {len(ex)} judgment, condition {a.condition}); "
          f"{total} optimizer steps; checkpoints at {sorted(save_at)}", flush=True)

    def checkpoint(step, losses):
        model.save_pretrained(a.out_dir / f"ckpt-{step:04d}")
        rec = dict(step=step, time=round(time.time() - t0), **{f"loss_{k}": (round(float(np.mean(v[-200:])), 4) if v else None)
                                                                for k, v in losses.items()},
                   **quick_eval(model, tok, held, device))
        log.write(json.dumps(rec) + "\n"); log.flush()
        print("checkpoint", rec, flush=True)

    t0, step, micro = time.time(), 0, 0
    losses = {"bg": [], "judg": []}
    model.train()
    if 0 in save_at:
        checkpoint(0, losses)
    for kind, x, y in data:
        if step >= total:
            break
        inp = torch.tensor([x], device=device)
        lab = torch.tensor([y], device=device)
        loss = model(input_ids=inp, labels=lab, use_cache=False).loss
        (loss / a.grad_accum).backward()
        losses[kind].append(loss.item())
        micro += 1
        if micro % a.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0:
                print(f"step {step}/{total}  bg {np.mean(losses['bg'][-200:] or [np.nan]):.3f}  "
                      f"judg {np.mean(losses['judg'][-200:] or [np.nan]):.3f}  {time.time() - t0:.0f}s", flush=True)
            if step in save_at:
                checkpoint(step, losses)
    if step not in save_at:
        checkpoint(step, losses)
    log.close()
    print(f"done in {time.time() - t0:.0f}s -> {a.out_dir}")


if __name__ == "__main__":
    main()
