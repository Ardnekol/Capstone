#!/usr/bin/env python3
"""Aggregate multi-seed cross-domain results for the unified Florence-2 model.

Reads the per-seed evaluation summaries written by run_multiseed.sh and reports
mean +/- sample std for the Florence row on each of the 7 cross-domain
benchmarks, plus a LaTeX-ready snippet.

Usage:
    python aggregate_seeds.py --seeds 1,2,3
    python aggregate_seeds.py --seeds 1,2,3 --results-root eval_results
"""
import argparse
import glob
import os
import re
import statistics as st

# benchmark label -> (subdir under eval_results/seed<seed>/, kind)
# kind: cls -> (acc%, macroF1); det -> F1 (last col); seg -> mIoU (first col)
BENCHMARKS = [
    ("Classification RealWaste", "realwaste", "cls"),
    ("Classification WaRP-C",    "warpc",     "cls"),
    ("Detection ICRA19",         "det_icra19",    "det"),
    ("Detection ZeroWaste-f",    "det_zerowaste", "det"),
    ("Detection WaRP-D",         "det_warpd",     "det"),
    ("Segmentation DWSD",        "seg_dwsd",      "seg"),
    ("Segmentation ZeroWaste-f", "seg_zerowaste", "seg"),
]

NUMCELL = re.compile(r"-?\d+\.?\d*")


def florence_row(summary_dir):
    """Return the numeric cells on the Florence row of the first *summary*.md.

    Parses the markdown table cell-by-cell and keeps only cells that are
    *entirely* a number (after stripping '%'). This avoids grabbing the '-2'
    inside the model name 'Florence-2'.
    """
    files = sorted(glob.glob(os.path.join(summary_dir, "*summary*.md")))
    for f in files:
        with open(f) as fh:
            for line in fh:
                if "florence" in line.lower() and "|" in line:
                    vals = []
                    for cell in line.split("|"):
                        c = cell.replace("%", "").strip()
                        if NUMCELL.fullmatch(c):
                            vals.append(float(c))
                    if vals:
                        return vals
    return None


def extract(kind, vals):
    """Pull the headline metric(s) for a benchmark kind from the row floats."""
    if vals is None:
        return None
    if kind == "cls":
        # | acc% | macroF1 |  -> last two floats
        return {"acc": vals[-2], "f1": vals[-1]}
    if kind == "det":
        # | P | R | F1 | -> last float is F1
        return {"f1": vals[-1]}
    if kind == "seg":
        # | mIoU | pixacc | -> first float is mIoU
        return {"miou": vals[0]}
    return None


def fmt(mean, sd, pct=False):
    if pct:
        return f"{mean:.2f} $\\pm$ {sd:.2f}" if sd is not None else f"{mean:.2f}"
    return f"{mean:.3f} $\\pm$ {sd:.3f}" if sd is not None else f"{mean:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True, help="comma list, e.g. 1,2,3")
    ap.add_argument("--results-root", default="eval_results")
    args = ap.parse_args()
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]

    print(f"Seeds: {seeds}\n")
    print(f"{'Benchmark':28s} {'Metric':6s} {'per-seed values':28s} {'mean +/- std'}")
    print("-" * 90)

    latex_rows = []
    missing = []
    for label, sub, kind in BENCHMARKS:
        per_seed = {}
        for sd in seeds:
            d = os.path.join(args.results_root, f"seed{sd}", sub)
            m = extract(kind, florence_row(d))
            if m is None:
                missing.append(f"seed{sd}/{sub}")
                continue
            for k, v in m.items():
                per_seed.setdefault(k, []).append(v)

        for metric, vals in per_seed.items():
            n = len(vals)
            mean = st.mean(vals)
            sd_ = st.stdev(vals) if n > 1 else None
            vs = ", ".join(f"{v:.3f}" for v in vals)
            sd_str = f"{sd_:.4f}" if sd_ is not None else "  n/a"
            print(f"{label:28s} {metric:6s} [{vs:26s}] {mean:.4f} +/- {sd_str}  (n={n})")
            pct = (metric == "acc")
            latex_rows.append((label, metric, fmt(mean, sd_, pct)))

    if missing:
        print("\n[!] missing summaries (run/finish those evals):")
        for m in missing:
            print("    -", m)

    print("\n% ---- LaTeX (mean $\\pm$ std over seeds) ----")
    for label, metric, cell in latex_rows:
        print(f"% {label} ({metric}): {cell}")


if __name__ == "__main__":
    main()
