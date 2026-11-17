"""Check regenerated outputs of the negated-statements analysis against the paper's values.

reproduce/reference_negation.csv holds one row per number that Section 3.1 (and Table 2)
reports for the negated statements: where it lives (file, filter, column), the value of the
run that the manuscript was written from (`reference`), the value as printed in the
manuscript (`manuscript`), and a tolerance. This script re-reads those cells from an
outputs directory, compares them, and prints a pass/fail table.

    python reproduce/check_negation.py                        # against outputs/
    python reproduce/check_negation.py --outputs /scratch/neg --steps b
    python reproduce/check_negation.py --tol 0.1 --only olmo2-7b

Status per row
    PASS       |regenerated - reference| <= tol
    FAIL       the cell exists but differs by more than tol
    MISSING    the file or the filtered row does not exist (step not run, model not scored)
    AMBIGUOUS  the filter matches more than one row (should not happen)

Besides the comparison, each row is checked against the manuscript: `ms` is OK when the
reference value, rounded to the manuscript's number of decimals, equals the printed
number. A mismatch there means the manuscript and the stored outputs disagree, which is a
problem of the paper, not of the rerun.

Exit code 0 only if every selected row is PASS (MISSING is tolerated with --allow-missing).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def lookup(df: pd.DataFrame, filt: str) -> pd.DataFrame:
    for kv in filt.split(";"):
        key, val = kv.split("=", 1)
        if key not in df.columns:
            return df.iloc[0:0]
        df = df[df[key].astype(str) == val]
    return df


def ms_ok(reference: float, manuscript: str) -> bool:
    nd = len(manuscript.split(".")[1]) if "." in manuscript else 0
    # the printed number must be the reference rounded to nd decimals; exact ties
    # (x.xx5) may be rounded either way, hence the epsilon
    return abs(reference - float(manuscript)) <= 10 ** (-nd) / 2 + 1e-9


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs",
                    help="output root to check (default: the repository's outputs/)")
    ap.add_argument("--reference", type=Path, default=HERE / "reference_negation.csv")
    ap.add_argument("--steps", default="", help="comma-separated steps of negation.sh (default: all)")
    ap.add_argument("--only", default="", help="substring filter on the filter column, e.g. a preset")
    ap.add_argument("--tol", type=float, help="override the per-row tolerance")
    ap.add_argument("--allow-missing", action="store_true",
                    help="do not fail on cells that are absent (e.g. a step that was not run)")
    args = ap.parse_args()

    ref = pd.read_csv(args.reference)
    if args.steps:
        ref = ref[ref["step"].isin(args.steps.split(","))]
    if args.only:
        ref = ref[ref["filter"].str.contains(args.only, regex=False)]
    if ref.empty:
        print("no reference rows selected", file=sys.stderr)
        return 2

    cache: dict[Path, pd.DataFrame | None] = {}
    rows = []
    for r in ref.itertuples(index=False):
        path = args.outputs / r.file
        if path not in cache:
            cache[path] = pd.read_csv(path) if path.exists() else None
        df = cache[path]
        tol = args.tol if args.tol is not None else float(r.tol)
        got, diff, status = float("nan"), float("nan"), "MISSING"
        if df is None:
            note = "file not found"
        else:
            sub = lookup(df, r.filter)
            if len(sub) > 1:
                status, note = "AMBIGUOUS", f"{len(sub)} rows match"
            elif len(sub) == 0 or r.column not in sub.columns:
                note = "row/column not found"
            elif pd.isna(sub.iloc[0][r.column]):
                note = "value is NaN"
            else:
                got = float(sub.iloc[0][r.column])
                diff = got - float(r.reference)
                status = "PASS" if abs(diff) <= tol else "FAIL"
                note = ""
        rows.append(dict(id=r.id, step=r.step, file=r.file, column=r.column,
                         manuscript=r.manuscript, reference=float(r.reference), regenerated=got,
                         diff=diff, tol=tol, status=status,
                         ms="OK" if ms_ok(float(r.reference), str(r.manuscript)) else "MISMATCH",
                         note=note))

    res = pd.DataFrame(rows)
    with pd.option_context("display.width", 220, "display.max_rows", 500,
                           "display.max_colwidth", 40):
        print(res[["id", "step", "file", "column", "manuscript", "reference", "regenerated",
                   "diff", "tol", "status", "ms"]].to_string(
            index=False, float_format=lambda v: f"{v:+.3f}" if pd.notna(v) else "-"))
    counts = res["status"].value_counts().to_dict()
    print(f"\noutputs: {args.outputs}")
    print("summary: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    bad_ms = res[res["ms"] != "OK"]
    if len(bad_ms):
        print("\nreference values that do not match the printed manuscript number:")
        print(bad_ms[["id", "reference", "manuscript", "note"]].to_string(index=False))
    for r in res[res["status"].isin(["MISSING", "AMBIGUOUS"])].itertuples(index=False):
        print(f"  {r.status}: {r.id} ({r.file}: {r.note})")

    failed = int(counts.get("FAIL", 0)) + int(counts.get("AMBIGUOUS", 0)) + len(bad_ms)
    if not args.allow_missing:
        failed += int(counts.get("MISSING", 0))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
