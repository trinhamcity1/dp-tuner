"""Run one (dataset, method, epsilon, seed) benchmark job and print its result as one JSON line.

Every method receives the same public-domain, all-categorical training table (see prepare_data.py),
so no generator gets non-private access to data statistics through preprocessing. Methods that
condition on the label (our DP-VAE and DP-CTGAN) spend LABEL_EPS_FRACTION of the budget on a
Laplace-noised label histogram and the rest on DP-SGD; budgets compose additively at the same delta.
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

DELTA = 1e-6
SPLIT_SEED = 7
TEST_FRACTION = 0.2
STEP_BUDGET = int(os.environ.get("STEP_BUDGET", 2000))  # DP-SGD optimizer steps, held fixed across datasets
NONPRIVATE_STEPS = 5000
BATCH = 512
CLIP = 1.6
LABEL_EPS_FRACTION = 0.05
MIA_QUERIES = 2000
MIA_SYNTH_ROWS = 20000


def load(name):
    with open(os.path.join(HERE, "data", f"{name}.schema.json")) as f:
        schema = json.load(f)
    df = pd.read_csv(os.path.join(HERE, "data", f"{name}.csv"), dtype=str, keep_default_na=False)
    label = schema["label"]
    rng = np.random.RandomState(SPLIT_SEED)
    test_idx = []
    for _, grp in df.groupby(label):
        n_test = int(round(len(grp) * TEST_FRACTION))
        test_idx.extend(rng.choice(grp.index.values, n_test, replace=False))
    test_mask = df.index.isin(test_idx)
    return schema, df[~test_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)


def codes(df, schema):
    """Map every column to integer codes over its public domain."""
    out = np.empty((len(df), len(schema["columns"])), dtype=np.int32)
    for j, (c, spec) in enumerate(schema["columns"].items()):
        lookup = {v: i for i, v in enumerate(spec["domain"])}
        out[:, j] = df[c].map(lookup).fillna(0).astype(np.int32).values
    return out


def onehot(df, schema, exclude):
    blocks = []
    for c, spec in schema["columns"].items():
        if c == exclude:
            continue
        lookup = {v: i for i, v in enumerate(spec["domain"])}
        idx = df[c].map(lookup).fillna(0).astype(int).values
        m = np.zeros((len(df), len(spec["domain"])), dtype=np.float32)
        m[np.arange(len(df)), idx] = 1.0
        blocks.append(m)
    return np.concatenate(blocks, axis=1)


def utility(train_syn, test, schema):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    label = schema["label"]
    y_tr = train_syn[label].astype(int).values
    y_te = test[label].astype(int).values
    if len(np.unique(y_tr)) < 2:
        return {"auroc_lr": float("nan"), "auroc_hgb": float("nan")}
    X_tr = onehot(train_syn, schema, label)
    X_te = onehot(test, schema, label)
    classes = np.unique(y_te)
    out = {}
    for name, clf in [("lr", LogisticRegression(max_iter=2000)),
                      ("hgb", HistGradientBoostingClassifier(max_iter=200, random_state=0))]:
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)
        if len(classes) > 2:
            # Columns of predict_proba follow clf.classes_; a class absent from synthetic data gets probability 0.
            full = np.zeros((len(X_te), len(classes)))
            for k, cl in enumerate(clf.classes_):
                full[:, np.searchsorted(classes, cl)] = proba[:, k]
            full = full / np.clip(full.sum(1, keepdims=True), 1e-12, None)
            out[f"auroc_{name}"] = float(roc_auc_score(y_te, full, multi_class="ovr", average="macro", labels=classes))
        else:
            pos = list(clf.classes_).index(1) if 1 in clf.classes_ else None
            score = proba[:, pos] if pos is not None else np.zeros(len(X_te))
            out[f"auroc_{name}"] = float(roc_auc_score(y_te, score))
    return out


def fidelity(real_codes, syn_codes, domain_sizes):
    """Mean total variation distance over all 1-way and 2-way marginals."""
    d = real_codes.shape[1]
    tvd1 = []
    for j in range(d):
        p = np.bincount(real_codes[:, j], minlength=domain_sizes[j]) / len(real_codes)
        q = np.bincount(syn_codes[:, j], minlength=domain_sizes[j]) / len(syn_codes)
        tvd1.append(0.5 * np.abs(p - q).sum())
    tvd2 = []
    for i in range(d):
        for j in range(i + 1, d):
            size = domain_sizes[i] * domain_sizes[j]
            p = np.bincount(real_codes[:, i] * domain_sizes[j] + real_codes[:, j], minlength=size) / len(real_codes)
            q = np.bincount(syn_codes[:, i] * domain_sizes[j] + syn_codes[:, j], minlength=size) / len(syn_codes)
            tvd2.append(0.5 * np.abs(p - q).sum())
    return {"tvd_1way": float(np.mean(tvd1)), "tvd_2way": float(np.mean(tvd2))}


def mia(train_codes, test_codes, syn_codes, seed):
    """Distance-to-closest-synthetic-record membership attack (Hamming distance over coded columns).

    AUC near 0.5 means the attacker cannot tell training members from held-out non-members.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(1000 + seed)
    mem = train_codes[rng.choice(len(train_codes), MIA_QUERIES, replace=False)]
    non = test_codes[rng.choice(len(test_codes), MIA_QUERIES, replace=False)]
    syn = syn_codes[rng.choice(len(syn_codes), min(MIA_SYNTH_ROWS, len(syn_codes)), replace=False)]
    queries = np.concatenate([mem, non])
    dmin = np.empty(len(queries))
    for s in range(0, len(queries), 50):
        block = queries[s:s + 50]
        dmin[s:s + 50] = (block[:, None, :] != syn[None, :, :]).sum(-1).min(1)
    y = np.r_[np.ones(len(mem)), np.zeros(len(non))]
    return {"mia_auc": float(roc_auc_score(y, -dmin)),
            "mia_exact_match_member": float((dmin[:len(mem)] == 0).mean()),
            "mia_exact_match_nonmember": float((dmin[len(mem):] == 0).mean())}


def noisy_label_sampler(train, label, schema, eps_label, rng):
    """Laplace-noised class histogram (pure eps_label-DP, sensitivity 1 under add/remove)."""
    classes = schema["columns"][label]["domain"]
    counts = train[label].value_counts().reindex(classes, fill_value=0).values.astype(float)
    noisy = np.clip(counts + rng.laplace(0, 1.0 / eps_label, size=len(counts)), 0, None)
    probs = noisy / noisy.sum() if noisy.sum() > 0 else np.ones(len(classes)) / len(classes)
    return lambda n: rng.choice(classes, size=n, p=probs)


def dp_sgd_sigma(eps, n_train, epochs):
    from opacus.accountants.utils import get_noise_multiplier
    sample_rate = 1.0 / (n_train // BATCH)
    return get_noise_multiplier(target_epsilon=eps, target_delta=DELTA, sample_rate=sample_rate,
                                epochs=epochs, accountant="prv")


def run_method(method, eps, seed, train, schema):
    import torch
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 4)))
    torch.manual_seed(seed)
    np.random.seed(seed)
    label = schema["label"]
    n = len(train)
    info = {}

    if method == "identity":
        # Releases the training data itself: the worst case, calibrating what the attack can detect.
        return train.sample(n, replace=True, random_state=seed).reset_index(drop=True), info

    if method in ("ctgan", "tvae"):
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
        md = SingleTableMetadata()
        md.detect_from_dataframe(train)
        for c in train.columns:
            md.update_column(c, sdtype="categorical")
        epochs = max(1, int(round(NONPRIVATE_STEPS * 500 / n)))  # non-private references get a larger budget
        cls = CTGANSynthesizer if method == "ctgan" else TVAESynthesizer
        synth = cls(md, epochs=epochs, batch_size=500, enforce_rounding=False) if method == "tvae" else \
            cls(md, epochs=epochs, batch_size=500, pac=10)
        synth.fit(train)
        info["epochs"] = epochs
        return synth.sample(n).astype(str), info

    if method == "mst":
        from snsynth import Synthesizer
        synth = Synthesizer.create("mst", epsilon=eps, delta=DELTA)
        synth.fit(train, categorical_columns=list(train.columns), preprocessor_eps=0.0)
        return synth.sample(n).astype(str), info

    if method == "patectgan":
        from snsynth import Synthesizer
        epochs = max(1, int(round(STEP_BUDGET * 500 / n)))
        synth = Synthesizer.create("patectgan", epsilon=eps, delta=DELTA, epochs=epochs, cuda=False, verbose=False)
        synth.fit(train, categorical_columns=list(train.columns), preprocessor_eps=0.0)
        info["epochs"] = epochs
        return synth.sample(n).astype(str), info

    if method in ("dpvae", "dpvae_argmax", "dpctgan"):
        rng = np.random.RandomState(seed)
        eps_label = LABEL_EPS_FRACTION * eps
        eps_sgd = eps - eps_label
        epochs = max(1, int(round(STEP_BUDGET / (n // BATCH))))
        sigma = dp_sgd_sigma(eps_sgd, n, epochs)
        X, y = train.drop(columns=[label]), train[label]
        if method.startswith("dpvae"):
            from dp_synth_data_gen.dptvae import DPTVAE
            gen = DPTVAE(epochs=epochs, batch_size=BATCH, max_grad_norm=CLIP, noise_multiplier=sigma, delta=DELTA,
                         decode="argmax" if method == "dpvae_argmax" else "sample", random_state=seed)
        else:
            from dp_synth_data_gen.dpctgan_v2 import DPCTGAN
            gen = DPCTGAN(epochs=epochs, batch_size=BATCH, max_grad_norm=CLIP, noise_multiplier=sigma, delta=DELTA)
        gen.fit(X, y)
        sample_labels = noisy_label_sampler(train, label, schema, eps_label, rng)
        y_req = sample_labels(n)
        y_req = np.array([type(gen._y_classes[0])(v) for v in y_req]) if gen._y_classes is not None else y_req
        syn_X, y_out = gen.sample(n, return_y=True, y_cond=y_req)
        syn = syn_X.astype(str)
        syn[label] = pd.Series(y_out).astype(str).values
        info.update({"epochs": epochs, "sigma": float(sigma), "eps_sgd_target": float(eps_sgd),
                     "eps_sgd_spent": gen.get_epsilon(DELTA), "eps_label": float(eps_label)})
        return syn[train.columns], info

    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--eps", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--save-synth", default="")
    a = ap.parse_args()

    schema, train, test = load(a.dataset)
    t0 = time.time()
    syn, info = run_method(a.method, a.eps, a.seed, train, schema)
    train_time = time.time() - t0

    for c, spec in schema["columns"].items():
        bad = ~syn[c].isin(spec["domain"])
        if bad.any():
            syn.loc[bad, c] = spec["domain"][0]
            info.setdefault("out_of_domain_fixed", {})[c] = int(bad.sum())
    if a.save_synth:
        syn.head(MIA_SYNTH_ROWS).to_csv(a.save_synth, index=False, compression="gzip")

    domain_sizes = [len(s["domain"]) for s in schema["columns"].values()]
    tr_c, te_c, sy_c = codes(train, schema), codes(test, schema), codes(syn, schema)
    result = {"dataset": a.dataset, "method": a.method, "eps": a.eps, "seed": a.seed, "delta": DELTA,
              "n_train": len(train), "train_time_s": round(train_time, 1), **info}
    result.update(utility(syn, test, schema))
    result.update(fidelity(tr_c, sy_c, domain_sizes))
    result.update(mia(tr_c, te_c, sy_c, a.seed))
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
