"""Diagnostic (non-private, analysis only): why does DP-VAE marginal fidelity worsen as epsilon grows?

For one trained DP-VAE, compares marginal TVD of rows decoded from
  (a) the prior z ~ N(0, I)                      -- what the released synthesizer does
  (b) the aggregate posterior q(z|x) of real rows -- non-private; isolates the prior/posterior gap
and reports how far the aggregate posterior is from N(0, I) (mean |mu|, mean posterior variance,
covariance of posterior means).
Usage: python experiments/diag_prior.py --dataset adult --eps 1 4 --seed 0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import bench  # noqa: E402
from dp_synth_data_gen.dptvae import DPTVAE  # noqa: E402


def decode(gen, z, c_idx):
    C = torch.nn.functional.one_hot(torch.from_numpy(c_idx), num_classes=gen._d_cond).float()
    m = gen._model.decode if hasattr(gen._model, "decode") else gen._model._module.decode
    with torch.no_grad():
        return gen._inverse_transform(m(z, C).numpy().astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="adult")
    ap.add_argument("--eps", type=float, nargs="+", default=[1.0, 4.0])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    schema, train, test = bench.load(a.dataset)
    label = schema["label"]
    sizes = [len(s["domain"]) for s in schema["columns"].values()]
    tr_codes = bench.codes(train, schema)
    for eps in a.eps:
        torch.manual_seed(a.seed); np.random.seed(a.seed)
        n = len(train)
        epochs = max(1, int(round(bench.STEP_BUDGET / (n // bench.BATCH))))
        sigma = bench.dp_sgd_sigma(eps * (1 - bench.LABEL_EPS_FRACTION), n, epochs)
        gen = DPTVAE(epochs=epochs, batch_size=bench.BATCH, max_grad_norm=bench.CLIP, noise_multiplier=sigma,
                     delta=bench.DELTA, random_state=a.seed)
        X, y = train.drop(columns=[label]), train[label]
        gen.fit(X, y)
        mod = gen._model._module if hasattr(gen._model, "_module") else gen._model
        mod.eval()
        M, _ = gen._fit_transform_X(X)
        mapping = {v: i for i, v in enumerate(gen._y_classes.tolist())}
        c_idx = np.array([mapping[v] for v in y.values])
        C = torch.nn.functional.one_hot(torch.from_numpy(c_idx), num_classes=gen._d_cond).float()
        with torch.no_grad():
            mu, logvar = mod.encode(torch.from_numpy(M).float(), C)
        z_post = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        z_prior = torch.randn_like(mu)
        out = {"dataset": a.dataset, "eps": eps, "seed": a.seed, "sigma": sigma,
               "mean_abs_mu": float(mu.abs().mean()), "mean_post_var": float(torch.exp(logvar).mean()),
               "agg_post_var_per_dim": float(z_post.var(0).mean()),
               "active_dims_var_mu_gt_0.01": int((mu.var(0) > 0.01).sum())}
        for name, z in [("prior", z_prior), ("posterior", z_post)]:
            syn = decode(gen, z, c_idx)
            syn[label] = y.values
            syn = syn[train.columns].astype(str)
            fid = bench.fidelity(tr_codes, bench.codes(syn, schema), sizes)
            out[f"{name}_tvd1"] = round(fid["tvd_1way"], 4)
            out[f"{name}_tvd2"] = round(fid["tvd_2way"], 4)
        print("DIAG " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
