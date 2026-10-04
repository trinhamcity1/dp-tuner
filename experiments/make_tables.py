"""Emit paper-ready Markdown tables (mean +- std over seeds) from the results file.

Usage: RESULTS_FILE=results_snapshot.jsonl python experiments/make_tables.py > paper/tables.md
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["adult", "diabetes130", "brfss"]
ROWS = [("dpvae", "DP-VAE (ours)"), ("mst", "MST"), ("aim", "AIM"), ("dpctgan", "DP-CTGAN"),
        ("patectgan", "PATE-CTGAN (smartnoise)"), ("dpvae_argmax", "DP-VAE, argmax decoding")]
REFS = [("identity", "Real training data"), ("ctgan", "CTGAN (non-private)"), ("tvae", "TVAE (non-private)")]
EPS = [1.0, 2.0, 4.0]


def cell(sub, metric):
    v = sub[metric].dropna()
    if not len(v):
        return "–"
    return f"{v.mean():.3f}" if len(v) == 1 else f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"


def table(df, metric, title):
    head = "| Method | " + " | ".join(f"{d} ε={e:g}" for d in DATASETS for e in EPS) + " |"
    lines = [f"### {title}", "", head, "|" + "---|" * (1 + len(DATASETS) * len(EPS))]
    for key, name in ROWS:
        cells = [cell(df[(df.dataset == d) & (df.method == key) & (df.eps == e)], metric)
                 for d in DATASETS for e in EPS]
        if any(c != "–" for c in cells):
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    for key, name in REFS:
        cells = []
        for d in DATASETS:
            c = cell(df[(df.dataset == d) & (df.method == key)], metric)
            cells += [c] + ["″"] * (len(EPS) - 1)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    path = os.path.join(HERE, os.environ.get("RESULTS_FILE", "results.jsonl"))
    df = pd.DataFrame([json.loads(l) for l in open(path) if l.strip()])
    print(table(df, "auroc_lr", "TSTR AUROC, logistic regression (higher is better)"))
    print(table(df, "auroc_hgb", "TSTR AUROC, gradient boosting (higher is better)"))
    print(table(df, "tvd_1way", "Mean 1-way marginal TVD (lower is better)"))
    print(table(df, "tvd_2way", "Mean 2-way marginal TVD (lower is better)"))
    print(table(df, "mia_auc", "DCR membership-inference AUC (0.5 = no detectable leakage)"))
    t = df[df.method.isin(["dpvae", "mst", "aim", "dpctgan", "patectgan"])].groupby(["dataset", "method"])["train_time_s"]
    print("### Mean fit+sample time (s), averaged over ε\n")
    print(t.mean().unstack(0).round(0).to_markdown())


if __name__ == "__main__":
    main()
