"""EditLens (Pangram; Thai et al., ICLR 2026) as an external, trained detector.

EditLens is Llama-3.2-3B with a LoRA adapter and an ordinal classification
head over "AI-edit buckets" (0 = human ... n-1 = fully AI), trained on English
data. Score = expected bucket / (n - 1) in [0, 1]. This re-implements the
released inference script (pangramlabs/EditLens, scripts/inference.py) without
running external code:

  * text cleaning as in scripts/preprocess.py: demojize (if the `emoji`
    package is installed), drop a <think> block, drop an English boilerplate
    first paragraph ("Sure", "Here", ...), lower-case, collapse whitespace
  * head = LayerNorm + bias-free Linear (NormedLinear), loaded with the adapter
  * base model in 4-bit NF4 if `bitsandbytes` is installed (as in the original),
    otherwise bf16 (recorded in the output as `quant`)
  * max_length 1024 tokens, left padding with EOS as pad token

Model license: CC BY-NC-SA 4.0 (non-commercial research use only); base model
under the Llama 3.2 community license. Texts never leave university machines
(local model only, never the Pangram API) and are never printed.

Usage (cluster)
---------------
    python editlens_score.py --corpus textbook
Output: outputs/<corpus>/editlens_llama32-3b.csv (text_id, source_id, is_ai, score, bucket, p_*)
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config

ADAPTER = "pangram/editlens_Llama-3.2-3B"
BASE = "meta-llama/Llama-3.2-3B"
BOILERPLATE_STARTS = ["Sure", "Here", "Abstract", "Title", "I'm happy to help", "Certainly"]
TEXT_COL = {"journalism_seg": "text"}       # model input as the models/participants saw it

try:
    import emoji
except ImportError:                          # pragma: no cover
    emoji = None


class NormedLinear(torch.nn.Module):
    """LayerNorm + bias-free Linear (EditLens scripts/train.py)."""

    def __init__(self, hidden_size, num_labels, device=None, dtype=None):
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        self.linear = torch.nn.Linear(hidden_size, num_labels, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.linear(self.norm(x))


def clean_text(text: str) -> str:
    if emoji is not None:
        text = emoji.demojize(text)
    if "</think>" in text:
        text = text.split("</think>")[1].strip()
    paragraphs = [p for p in text.split("\n") if p.strip()]
    if paragraphs:
        first = re.sub(r"^[^a-zA-Z0-9]*", "", paragraphs[0])
        if emoji is not None:
            first = emoji.replace_emoji(first, "")
        if any(first.startswith(s) for s in BOILERPLATE_STARTS) and len(paragraphs) > 1:
            text = "\n".join(paragraphs[1:])
    return re.sub(r"\s+", " ", text.lower()).strip()


def n_buckets_from_adapter(path: str) -> int:
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            if "score" in key and "linear.weight" in key:
                return f.get_tensor(key).shape[0]
    raise ValueError("no score head in adapter")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from peft import PeftModel

    n_b = n_buckets_from_adapter(hf_hub_download(ADAPTER, "adapter_model.safetensors"))
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.padding_side = "left"
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes  # noqa: F401
        qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16)
        base = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=n_b,
                                                                  quantization_config=qc)
        quant = "nf4"
    except ImportError:
        base = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=n_b,
                                                                  dtype=torch.bfloat16).to("cuda")
        quant = "bf16"
    base.config.pad_token_id = tok.pad_token_id
    dev = next(base.parameters()).device
    base.score = NormedLinear(base.config.hidden_size, n_b, device=dev, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, ADAPTER).eval()
    print(f"EditLens: {n_b} buckets | base {quant} | emoji normalisation {'on' if emoji else 'off'}")

    col = TEXT_COL.get(args.corpus, "text")
    d = pd.read_parquet(config.REPO / "data" / f"texts_{args.corpus}.parquet",
                        columns=["text_id", "source_id", "is_ai", col])
    if args.limit:
        d = d.head(args.limit)
    probs = []
    with torch.no_grad():
        for i in range(0, len(d), args.batch_size):
            batch = [clean_text(t) for t in d[col].iloc[i:i + args.batch_size]]
            enc = tok(batch, truncation=True, max_length=args.max_length, padding=True,
                      return_tensors="pt").to(dev)
            logits = model(**enc).logits.float()
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    p = np.concatenate(probs)
    out = d[["text_id", "source_id", "is_ai"]].copy()
    out["score"] = p @ np.arange(n_b) / (n_b - 1)
    out["bucket"] = p.argmax(1)
    for j in range(n_b):
        out[f"p_{j}"] = p[:, j]
    out["quant"] = quant
    out["emoji_norm"] = emoji is not None
    path = config.OUTPUT_DIR / args.corpus / "editlens_llama32-3b.csv"
    if args.limit:
        path = path.with_name("editlens_llama32-3b_smoke.csv")
    out.to_csv(path, index=False)
    from sklearn.metrics import roc_auc_score
    print(f"{len(out)} texts | mean score AI {out.score[out.is_ai == 1].mean():.2f} "
          f"human {out.score[out.is_ai == 0].mean():.2f} | AUC {roc_auc_score(out.is_ai, out.score):.3f} -> {path}")


if __name__ == "__main__":
    main()
