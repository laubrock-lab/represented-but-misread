"""True/false judgments on matched statement pairs: output level vs. representation.

The same SDT logic as for the AI-text task, with "true" as the signal class.
A statement set is a Python file defining

    PAIRS    [(pair_id, topic, true_statement, false_statement), ...]
    TOPICS   all topics; TARGET  the topic of interest
    MATCHED  the control topic matched to TARGET in kind (e.g. same domain)

Each pair differs in one fact, so true and false statements of a topic are
matched in content and length.

Output level, per model condition x topic (both answer orders averaged):
    dv                log P("True") - log P("False"), end-of-word corrected
    out_d, out_c      d' from the AUC, c from the dv > 0 split
    d_dv              equal-variance d' of dv; does not saturate (the AUC-based
                      d' is capped at 2.45 for 12 pairs)
    bias_nats         mean dv; true_called_false  share of true statements with dv < 0
    p_mass            P("True") + P("False"); low = the model wants to say something else
Representation level: a truth direction fitted on the control topics only and
applied to the held-out topic (last prompt token, standardized with the training
statistics); controls are evaluated leave-one-topic-out among the controls. The
layer is chosen on the controls. Methods: mass-mean and L2 logistic regression;
probe_d (AUC-based) and probe_dz (equal-variance).

Fine-tuned models: --adapter DIR --tag NAME applies a LoRA adapter (peft) to the
instruct weights and stores the condition as variant NAME.

Usage
-----
    python factuality.py --statements S.py --out-dir D \
        --presets gemma3-4b
    python factuality.py --statements S.py --out-dir D --presets gemma3-4b \
        --adapter D/adapters/<run>/ckpt-<step> --tag <name>
    python factuality.py --statements S.py --out-dir D --analyze-only
    python factuality.py --statements truth/cities_statements.py --out-dir outputs/truth_cities \
        --presets olmo2-7b --prompt-variant sfirst     # -> *_raw_tf_sfirst.csv, summary_sfirst.csv

Prompt variants (raw format): --prompt-variant default|sfirst, see RAW_VARIANTS. The
variant is part of every output file name (suffix _<variant>), so runs of different
layouts live side by side in one output directory; the default keeps the old names.
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

import config
from judge_logits import _forward, end_of_word_mask, load_model, pick_device
from sdt_summary import grouped_bootstrap, sdt

ORDERS = {"tf": ("True", "False"), "ft": ("False", "True")}
CANDIDATES = {"True": ["True", " True"], "False": ["False", " False"]}
RAW = ("Is the following statement true or false?\n\n"
       "Statement: {s}\n\n"
       "Answer with one word ({opts}).\n"
       "Answer:")
SYSTEM = "You answer questions about historical facts."   # default; truth/ sets override
USER = ("Is the following statement true or false?\n\n"
        "Statement: {s}\n\n"
        "Answer with exactly one word: {opts}.")
CONDITIONS = [("base", "raw")] + [(v, f) for v in ("sft", "rl", "dpo", "instruct") for f in ("raw", "chat")]

# ---------------------------------------------------------------------------
# Raw prompt variants (completion format only; the chat format is unaffected).
# "default": the question precedes the statement, so the statement is read with the
# task already known and the positions after it hold only the answer cue.
# "sfirst": the statement comes first and the question follows it, so the positions
# after the statement carry the question. Answer words, candidate sets and scoring
# are identical in both. Selected with --prompt-variant; output file names get the
# suffix "_<variant>" (the default variant keeps the existing names).
# ---------------------------------------------------------------------------
RAW_SFIRST = ("Statement: {s}\n\n"
              "Is this statement true or false?\n\n"
              "Answer with one word ({opts}).\n"
              "Answer:")
RAW_VARIANTS = {"default": None, "sfirst": RAW_SFIRST}   # None = the set's own RAW
# first characters of the question line, for the "question" position set of
# patch_positions.py; None = this variant has no question line after the statement
QUESTION_MARKERS = {"default": None, "sfirst": "Is this statement true or false?"}
PROMPT_VARIANT = "default"


def set_prompt_variant(name: str) -> None:
    """Switch the raw template globally (build_prompt reads raw_template())."""
    global PROMPT_VARIANT
    if name not in RAW_VARIANTS:
        raise SystemExit(f"unknown prompt variant {name!r} (have: {', '.join(RAW_VARIANTS)})")
    PROMPT_VARIANT = name


def raw_template() -> str:
    """Default variant: RAW, i.e. whatever the statement set put there (apply_set_overrides);
    any other variant: its own template."""
    return RAW if PROMPT_VARIANT == "default" else RAW_VARIANTS[PROMPT_VARIANT]


def variant_suffix() -> str:
    """File-name suffix of the active prompt variant ("" for the default)."""
    return "" if PROMPT_VARIANT == "default" else f"_{PROMPT_VARIANT}"


def question_marker() -> str | None:
    return QUESTION_MARKERS[PROMPT_VARIANT]


def load_statements(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"statements_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("PAIRS", "TOPICS", "TARGET", "MATCHED"):
        if not hasattr(mod, name):
            raise SystemExit(f"{path} does not define {name}")
    return mod


def items(st: ModuleType) -> pd.DataFrame:
    rows = []
    for pair_id, topic, true_s, false_s in st.PAIRS:
        rows.append(dict(item_id=f"{pair_id}T", pair_id=pair_id, topic=topic, is_true=1, statement=true_s))
        rows.append(dict(item_id=f"{pair_id}F", pair_id=pair_id, topic=topic, is_true=0, statement=false_s))
    return pd.DataFrame(rows)


def apply_set_overrides(st: ModuleType) -> None:
    """A statement set may define its own SYSTEM, RAW, USER prompts and answer LABELS
    (signal word, noise word); defaults are the true/false prompts above."""
    global SYSTEM, RAW, USER, ORDERS, CANDIDATES
    SYSTEM = getattr(st, "SYSTEM", SYSTEM)
    RAW = getattr(st, "RAW", RAW)
    USER = getattr(st, "USER", USER)
    if hasattr(st, "LABELS"):
        a, b = st.LABELS
        ORDERS = {"tf": (a, b), "ft": (b, a)}
        CANDIDATES = {"True": [a, " " + a], "False": [b, " " + b]}


def truth_items(corpus: str, limit_pairs: int | None = None) -> pd.DataFrame:
    """'truth:<name>' -> items of truth/<name>_statements.py in the column layout of the
    text corpora (text_id, source_id, is_ai, text), so that stitch.py and
    patch_positions.py can run on true/false judgments (is_ai = is_true = signal)."""
    name = corpus.split(":", 1)[1]
    st = load_statements(config.REPO / "truth" / f"{name}_statements.py")
    apply_set_overrides(st)
    if limit_pairs:
        st.PAIRS = st.PAIRS[:limit_pairs]
    return items(st).rename(columns={"item_id": "text_id", "pair_id": "source_id",
                                     "is_true": "is_ai", "statement": "text"})


def build_prompt(tok, s: str, fmt: str, order: str) -> tuple[str, bool]:
    opts = " or ".join(ORDERS[order])
    if fmt == "raw":
        return raw_template().format(s=s, opts=opts), True
    return chat_prompt(tok, USER.format(s=s, opts=opts)), False


def chat_prompt(tok, user: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except Exception:  # no system role: fold it into the user turn
        merged = [{"role": "user", "content": SYSTEM + "\n\n" + user}]
        return tok.apply_chat_template(merged, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)


def candidate_seqs(tok) -> dict:
    return {k: sorted({tuple(tok(v, add_special_tokens=False)["input_ids"]) for v in vs} - {()})
            for k, vs in CANDIDATES.items()}


@torch.no_grad()
def score(model, tok, prompt: str, add_special: bool, cands: dict, eow: dict,
          device: str, want_acts: bool) -> dict:
    ids = tok(prompt, add_special_tokens=add_special, return_tensors="pt")["input_ids"].to(device)
    out = _forward(model, ids, keep=1, hidden=want_acts)
    logp_next = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
    lp_word, lp_prefix = {}, {}
    for label, seqs in cands.items():
        word, pre = [], []
        for seq in seqs:
            full = torch.cat([ids, torch.tensor([seq], device=device)], dim=1)
            o = _forward(model, full, keep=len(seq) + 1, hidden=False)
            lps = torch.log_softmax(o.logits[0, -(len(seq) + 1):].float(), dim=-1)
            if "mask" not in eow:
                eow["mask"] = end_of_word_mask(tok, lps.shape[-1]).to(lps.device)
            lp_seq = sum(lps[i, t].item() for i, t in enumerate(seq))
            pre.append(lp_seq)
            word.append(lp_seq + torch.logsumexp(lps[-1][eow["mask"]], dim=0).item())
        lp_word[label] = float(torch.logsumexp(torch.tensor(word), dim=0))
        lp_prefix[label] = float(torch.logsumexp(torch.tensor(pre), dim=0))
    res = dict(lp_true=lp_word["True"], lp_false=lp_word["False"],
               dv=lp_word["True"] - lp_word["False"],
               p_mass=float(np.exp(lp_prefix["True"]) + np.exp(lp_prefix["False"])),
               top1=tok.decode([int(logp_next.argmax())]))
    if want_acts:
        res["act"] = torch.stack([h[0, -1] for h in out.hidden_states]).float().cpu().numpy()
    return res


@torch.no_grad()
def generate(model, tok, prompt: str, add_special: bool, device: str, n: int) -> str:
    ids = tok(prompt, add_special_tokens=add_special, return_tensors="pt")["input_ids"].to(device)
    g = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, do_sample=False)
    return tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip()


def load_condition(preset: str, variant: str, device: str, adapter: Path | None = None):
    """Base or instruct weights of a preset; with an adapter, the instruct model
    with the LoRA adapter merged in."""
    tok, model = load_model(config.MODEL_PRESETS[preset][variant], device, torch.bfloat16)
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload().eval()
    return tok, model


def run_model(preset: str, variant: str, fmts: list[str], df: pd.DataFrame, out: Path,
              device: str, gen_n: int, adapter: Path | None = None, tag: str | None = None) -> None:
    label = tag or variant
    sfx = variant_suffix()
    todo = [(f, o) for f in fmts for o in ORDERS
            if not (out / f"{preset}_{label}_{f}_{o}{sfx}.csv").exists()]
    if not todo:
        return
    tok, model = load_condition(preset, variant, device, adapter)
    model_id = config.MODEL_PRESETS[preset][variant] + (f"+{adapter}" if adapter else "")
    cands, eow = candidate_seqs(tok), {}
    for fmt, order in todo:
        stem = f"{preset}_{label}_{fmt}_{order}{sfx}"
        want_acts = order == "tf"
        rows, acts, t0 = [], [], time.time()
        for r in df.itertuples(index=False):
            prompt, add = build_prompt(tok, r.statement, fmt, order)
            res = score(model, tok, prompt, add, cands, eow, device, want_acts)
            if want_acts:
                acts.append(res.pop("act"))
            if gen_n and variant == "instruct" and fmt == "chat" and order == "tf":
                res["generated"] = generate(model, tok, prompt, add, device, gen_n)
            rows.append(dict(item_id=r.item_id, pair_id=r.pair_id, topic=r.topic,
                             is_true=r.is_true, preset=preset, variant=label, fmt=fmt,
                             order=order, model_id=model_id, **res))
        res_df = pd.DataFrame(rows)
        res_df.to_csv(out / f"{stem}.csv", index=False)
        if want_acts:
            np.savez(out / f"{stem}_acts.npz", item_id=res_df["item_id"].to_numpy(dtype=str),
                     acts=np.stack(acts).astype(np.float32))
        print(f"{stem}: {len(res_df)} statements in {time.time() - t0:.0f}s, "
              f"mean p_mass {res_df['p_mass'].mean():.2f}, share 'True' {(res_df['dv'] > 0).mean():.2f}")
    del model
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def eq_var_d(score: np.ndarray, y: np.ndarray) -> float:
    """Equal-variance d' = (mean true - mean false) / pooled SD."""
    t, f = score[y == 1], score[y == 0]
    return float((t.mean() - f.mean()) / np.sqrt((t.var(ddof=1) + f.var(ddof=1)) / 2))


def judgment_files(out: Path, sfx: str = "") -> list[Path]:
    return [f for f in out.glob("*.csv") if f.stem.endswith((f"_tf{sfx}", f"_ft{sfx}"))
            and not f.name.startswith("steer")]


def output_level(out: Path, n_boot: int, sfx: str = "") -> pd.DataFrame:
    df = pd.concat([pd.read_csv(f) for f in judgment_files(out, sfx)])
    keys = ["preset", "variant", "fmt"]
    avg = (df.groupby(keys + ["item_id", "pair_id", "topic", "is_true"], as_index=False)
             .agg(dv=("dv", "mean"), p_mass=("p_mass", "mean")))
    rng = np.random.default_rng(1)
    rows = []
    for key, g in avg.groupby(keys + ["topic"]):
        g = g.reset_index(drop=True)
        s = sdt(g["is_true"].to_numpy(), g["dv"].to_numpy())
        ci = grouped_bootstrap(g.rename(columns={"pair_id": "source_id", "is_true": "is_ai"}),
                               n_boot, rng)
        t, f = g.loc[g["is_true"] == 1, "dv"], g.loc[g["is_true"] == 0, "dv"]
        rows.append(dict(zip(keys + ["topic"], key), out_d=s["d_auc"],
                         out_d_lo=ci["d_auc_lo"], out_d_hi=ci["d_auc_hi"], out_c=s["c0"],
                         d_dv=(t.mean() - f.mean()) / np.sqrt((t.var() + f.var()) / 2),
                         bias_nats=g["dv"].mean(), true_called_false=(t < 0).mean(),
                         share_true=(g["dv"] > 0).mean(), p_mass=g["p_mass"].mean(), n=len(g)))
    return pd.DataFrame(rows)


def probe_curves(st: ModuleType, out: Path, sfx: str = "") -> pd.DataFrame:
    meta = items(st).set_index("item_id")
    controls = [t for t in st.TOPICS if t != st.TARGET]
    rows = []
    for f in sorted(out.glob(f"*_tf{sfx}_acts.npz")):
        cond = f.name[: -len(f"_tf{sfx}_acts.npz")]
        preset, variant, fmt = cond.rsplit("_", 2)
        z = np.load(f)
        m = meta.loc[z["item_id"]]
        topic, y, A = m["topic"].to_numpy(), m["is_true"].to_numpy(), z["acts"]
        n_layers = A.shape[1] - 1
        for layer in range(1, n_layers + 1):
            X = A[:, layer].astype(np.float64)
            for target in st.TOPICS:
                train = np.isin(topic, controls) & (topic != target)
                test = topic == target
                mu, sd = X[train].mean(0), X[train].std(0)
                sd[sd < 1e-6 * max(sd.max(), 1e-12)] = np.inf          # dead dimensions -> 0
                Xtr, Xte = (X[train] - mu) / sd, (X[test] - mu) / sd
                ytr = y[train]
                w_mm = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
                lr = LogisticRegression(C=0.01, max_iter=5000).fit(Xtr, ytr)
                for method, s in (("massmean", Xte @ w_mm), ("logreg", lr.decision_function(Xte))):
                    rows.append(dict(preset=preset, variant=variant, fmt=fmt, layer=layer,
                                     depth=layer / n_layers, target=target, method=method,
                                     probe_d=sdt(y[test], s)["d_auc"],
                                     probe_dz=eq_var_d(s, y[test])))
    return pd.DataFrame(rows)


def analyze(st: ModuleType, out: Path, n_boot: int) -> pd.DataFrame:
    sfx = variant_suffix()
    outl = output_level(out, n_boot, sfx)
    curves = probe_curves(st, out, sfx)
    curves.to_csv(out / f"probe_curves{sfx}.csv", index=False)
    keys = ["preset", "variant", "fmt"]
    sel = []
    for key, g in curves.groupby(keys + ["method"]):
        ctrl = g[g["target"] != st.TARGET].groupby("layer")["probe_dz"].mean()
        best = int(ctrl.idxmax())
        for target, r in g[g["layer"] == best].set_index("target").iterrows():
            sel.append(dict(zip(keys + ["method"], key), topic=target, layer=best,
                            probe_d=r["probe_d"], probe_dz=r["probe_dz"]))
    sel = (pd.DataFrame(sel)
           .pivot_table(index=keys + ["topic"], columns="method", values=["probe_d", "probe_dz"]))
    sel.columns = [f"{v}_{m}" for v, m in sel.columns]
    table = outl.merge(sel.reset_index(), on=keys + ["topic"], how="left")
    table.to_csv(out / f"summary{sfx}.csv", index=False)
    pd.set_option("display.width", 200)
    cols = keys + ["topic", "out_d", "d_dv", "out_c", "bias_nats", "true_called_false",
                   "p_mass", "probe_dz_massmean", "probe_dz_logreg"]
    print(table[[c for c in cols if c in table]].round(2).to_string(index=False))
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--statements", type=Path, required=True, help="statement-set file (.py)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--presets", nargs="+", default=["gemma3-4b"], choices=list(config.MODEL_PRESETS))
    ap.add_argument("--variants", nargs="+", default=["base", "instruct"],
                    choices=["base", "sft", "rl", "dpo", "instruct"])
    ap.add_argument("--adapter", type=Path, help="LoRA adapter applied to the instruct weights")
    ap.add_argument("--tag", help="variant label for the adapter condition (no underscores)")
    ap.add_argument("--generate", type=int, default=0,
                    help="also store a greedy free answer of N tokens (instruct, chat format)")
    ap.add_argument("--limit", type=int, help="smoke test: first N pairs per topic")
    ap.add_argument("--prompt-variant", default="default", choices=list(RAW_VARIANTS),
                    help="raw prompt layout: default = question before the statement; "
                         "sfirst = statement first, question after it. Output names get "
                         "the suffix _<variant> (default: unchanged names)")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--no-analyze", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    if args.adapter and (not args.tag or "_" in args.tag):
        ap.error("--adapter needs a --tag without underscores")

    st = load_statements(args.statements)
    apply_set_overrides(st)
    set_prompt_variant(args.prompt_variant)       # after the set overrides: a variant wins over RAW
    print(f"system prompt (chat format): {SYSTEM!r}; answer words: {ORDERS['tf']}"
          + (f"; prompt variant: {args.prompt_variant}" if variant_suffix() else ""))
    out = args.out_dir
    df = items(st)
    if args.limit:
        keep = df.groupby("topic")["pair_id"].unique().apply(lambda p: list(p[: args.limit]))
        df = df[df["pair_id"].isin(sum(keep.tolist(), []))]
        out = out / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        device = pick_device(args.device)
        for preset in args.presets:
            if args.adapter:
                run_model(preset, "instruct", ["raw", "chat"], df, out, device, args.generate,
                          args.adapter, args.tag)
                continue
            for variant in args.variants:
                # the prompt variants are raw-format layouts; the chat format is identical
                # to the default run, so it is not repeated (and not stored under the suffix)
                fmts = [f for v, f in CONDITIONS if v == variant
                        and (f == "raw" or not variant_suffix())]
                run_model(preset, variant, fmts, df, out, device, args.generate)
    if not args.no_analyze:
        analyze(st, out, args.n_boot)


if __name__ == "__main__":
    main()
