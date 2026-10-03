"""Summarize results.jsonl: per-(dataset, method, eps) mean +- std, and paired seed-matched tests.

Tests compare DP-VAE against every other DP method on each metric, pairing runs by seed, with a
Holm-Bonferroni correction across all (dataset, eps, baseline) comparisons of one metric.
Usage: python experiments/analyze.py [--metric auroc_lr] [--markdown out.md]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = ["auroc_lr", "auroc_hgb", "tvd_1way", "tvd_2way", "mia_auc"]
HIGHER_IS_BETTER = {"auroc_lr": True, "auroc_hgb": True, "tvd_1way": False, "tvd_2way": False, "mia_auc": None}
DP_METHODS = ["dpvae", "mst", "patectgan", "dpctgan", "dpvae_argmax"]
REFERENCES = ["identity", "ctgan", "tvae"]


def load():
    rows = [json.loads(l) for l in open(os.path.join(HERE, "results.jsonl")) if l.strip()]
    return pd.DataFrame(rows)


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def summary(df):
    g = df.groupby(["dataset", "method", "eps"])
    out = g[METRICS].agg(["mean", "std"])
    out["n"] = g.size()
    return out


def paired_tests(df, metric, target="dpvae"):
    rows = []
    for (ds, eps), sub in df[df.method.isin(DP_METHODS)].groupby(["dataset", "eps"]):
        a = sub[sub.method == target].set_index("seed")[metric]
        for base in DP_METHODS:
            if base == target:
                continue
            b = sub[sub.method == base].set_index("seed")[metric]
            common = a.index.intersection(b.index)
            if len(common) < 3:
                continue
            x, y = a.loc[common].values, b.loc[common].values
            diff = x - y
            p = stats.ttest_rel(x, y).pvalue if np.std(diff) > 0 else 0.0
            rows.append({"dataset": ds, "eps": eps, "baseline": base, "n": len(common),
                         "target_mean": x.mean(), "baseline_mean": y.mean(), "diff": diff.mean(),
                         "diff_ci95": stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff)), "p": p})
    res = pd.DataFrame(rows)
    if len(res):
        res["p_holm"] = holm(res.p.values)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="auroc_lr")
    ap.add_argument("--markdown", default="")
    a = ap.parse_args()
    df = load()
    pd.set_option("display.width", 200, "display.max_rows", 500)
    s = summary(df)
    print(s.round(4).to_string())
    print()
    blocks = []
    for m in METRICS:
        t = paired_tests(df, m)
        if not len(t):
            continue
        print(f"== DP-VAE vs baselines on {m} (paired by seed, Holm-corrected within metric)")
        print(t.round(4).to_string(index=False))
        print()
        blocks.append((m, t))
    if a.markdown:
        with open(a.markdown, "w") as f:
            f.write("# Benchmark summary\n\n")
            f.write(s.round(4).to_markdown() + "\n\n")
            for m, t in blocks:
                f.write(f"## DP-VAE vs baselines: {m}\n\n" + t.round(4).to_markdown(index=False) + "\n\n")


if __name__ == "__main__":
    main()
