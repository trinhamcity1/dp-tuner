#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DP ε Auto-Tuner (Demo Skeleton)
-------------------------------

Purpose:
    - Wrap a DP generator (e.g., DP-TVAE or DP-CTGAN) and *automatically* search
      for the smallest ε that still achieves a target utility (e.g., AUROC ≥ τ)
      on a downstream task (e.g., 30-day readmission), measured on a REAL holdout set.
    - This file is a *teaching/demo skeleton* with clear step-by-step comments.
      It includes fallbacks so you can run it even if some libraries (ctgan, opacus)
      aren't installed. Replace stubs with your concrete models when ready.

High-level steps (matching our research plan):
    0) Imports & helper utilities
    1) Load & preprocess dataset  -> collect N (dataset size)
    2) Fix policy & training knobs -> B (batch), C (clip), epochs (→ T), δ (delta)
    3) Define σ search space       -> noise multiplier candidates (tuner dials)
    4) Tuning loop per σ:
         4.1) Train DP generator (TVAE/CTGAN with DP-SGD)
         4.2) Ask privacy accountant for ε(N, B, T, C, σ, δ)
         4.3) Sample synthetic data
         4.4) Train downstream classifier (logistic regression)
         4.5) Repeat across seeds; collect mean ± 95% CI
    5) Selection rule: pick smallest ε whose AUROC ≥ τ
    6) Output plots/tables & save results

NOTE: In practice, ε is computed by a *privacy accountant* (e.g., RDP in Opacus/TF-Privacy)
      from (N, B, T, C, σ, δ). Here we provide a stubbed accountant if Opacus is missing.
"""

# ========== 0) IMPORTS & HELPERS ===============================================================
import os
import math
import json
import random
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

# Try to import optional libs; if not present, we use stubs.
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
except Exception as e:
    raise RuntimeError("Please install scikit-learn to run this demo: pip install scikit-learn") from e

# Use your SDV wrappers from generators/
try:
    from generators.sdv_ctgan import CtganGenerator
    from generators.sdv_tvae import TvaeGenerator
    HAS_SDV = True
except Exception as e:
    print(f"[WARN] SDV wrappers not available, using stub: {e}")
    HAS_SDV = False

# Optional: Opacus for DP-SGD accounting
try:
    from opacus.accountants import RDPAccountant
    HAS_OPACUS = True
    print("[HAS_OPACUS] has successfully imported")
except Exception:
    HAS_OPACUS = False

# Utility: set seeds for reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

# Utility: 95% CI for a list of numbers
def mean_and_ci(xs: List[float], alpha: float = 0.05) -> Tuple[float, float]:
    arr = np.array(xs, dtype=float)
    mean = float(np.mean(arr))
    # Normal approx CI
    se = float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
    z = 1.96  # ~95%
    return mean, z * se

# ========== 1) LOAD & PREPROCESS DATA  ==========================================================
"""
STEP 1 collects: N (dataset size) after preprocessing/splitting.

For the demo, we'll synthesize a small tabular dataset that *behaves like* a readmission dataset:
    - mixed numeric/categorical columns
    - binary label y
Replace this with your Kaggle Diabetes 30-Day Readmission dataset loader.
"""

def make_fake_healthcare(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    set_seed(seed)
    # Numeric features
    age = np.random.normal(60, 12, size=n).clip(18, 95)
    bmi = np.random.normal(28, 6, size=n).clip(14, 60)
    a1c = np.random.normal(6.8, 1.1, size=n).clip(4.5, 14.0)

    # Categorical features (low/high risk buckets)
    sex = np.random.choice(["F", "M"], size=n)
    smoker = np.random.choice(["yes", "no"], size=n, p=[0.2, 0.8])
    comorb = np.random.choice(["none", "htn", "dm", "cardio"], size=n, p=[0.25, 0.30, 0.30, 0.15])

    # Label: "readmit_30d" with some nonlinear truth
    base = (age - 50) / 30 + (bmi - 25) / 20 + (a1c - 6.5) / 2
    risk = base + (smoker == "yes") * 0.4 + (comorb == "cardio") * 0.6 + (comorb == "dm") * 0.3
    prob = 1 / (1 + np.exp(-risk))
    y = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        "age": age, "bmi": bmi, "a1c": a1c,
        "sex": sex, "smoker": smoker, "comorb": comorb,
        "readmit_30d": y
    })
    return df

# Preprocess into X, y and train/val/test splits; returns N (size of train)
def preprocess_and_split(df: pd.DataFrame, label: str = "readmit_30d", test_size=0.15, val_size=0.15, seed=0):
    """Return both raw splits and preprocessed matrices.
    New keys in meta:
        - preprocessor
        - cat_cols, num_cols
        - classes_
        - label
        - X_train_raw, X_val_raw, X_test_raw (pandas DataFrames with label column)
    """
    # First split on RAW for generator training
    y_all = df[label].values
    X_all = df.drop(columns=[label])
    X_train_raw, X_temp_raw, y_train_raw, y_temp_raw = train_test_split(X_all, y_all, test_size=(test_size + val_size), random_state=seed, stratify=y_all)
    rel_test = test_size / (test_size + val_size)
    X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(X_temp_raw, y_temp_raw, test_size=rel_test, random_state=seed, stratify=y_temp_raw)

    # Identify column types on RAW X
    cat_cols = [c for c in X_all.columns if X_all[c].dtype == "object"]
    num_cols = [c for c in X_all.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    # Fit on train RAW features
    pre.fit(X_train_raw)

    # Transform to matrices for downstream classifier
    X_train_t = pre.transform(X_train_raw)
    X_val_t   = pre.transform(X_val_raw)
    X_test_t  = pre.transform(X_test_raw)

    N = X_train_t.shape[0]
    meta = {
        "preprocessor": pre,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "classes_": np.unique(y_all).tolist(),
        "label": label,
        "X_train_raw": X_train_raw.assign(**{label: y_train_raw}),
        "X_val_raw": X_val_raw.assign(**{label: y_val_raw}),
        "X_test_raw": X_test_raw.assign(**{label: y_test_raw}),
    }
    return (X_train_t, y_train_raw, X_val_t, y_val_raw, X_test_t, y_test_raw, N, meta)

# ========== 2) POLICY DEFAULTS (B, C, epochs, δ) ===============================================
"""
STEP 2 sets human-chosen knobs and derives T (total steps).
These stay fixed during the σ search (unless you run sensitivity tests).
"""

class Policy:
    def __init__(self, B=128, C=1.0, epochs=50, delta=None):
        self.B = B
        self.C = C
        self.epochs = epochs
        self.delta = delta  # if None, we'll set to 1/N later

# ========== 3) PRIVACY ACCOUNTANT (ε from N, B, T, C, σ, δ) ====================================
"""
If Opacus is available, we use RDPAccountant. Otherwise, we provide a *very rough* placeholder
so the script still runs end-to-end. Replace the placeholder with a proper accountant in production.
"""

def compute_steps(N: int, B: int, epochs: int) -> int:
    steps_per_epoch = math.ceil(N / B)
    return steps_per_epoch * epochs

def epsilon_from_accountant(N: int, B: int, epochs: int, C: float, sigma: float, delta: float) -> float:
    T = compute_steps(N, B, epochs)
    q = B / N  # sampling rate

    if HAS_OPACUS:
        # Opacus version compatibility: some versions return float, some (eps, order)
        acc = RDPAccountant()
        for _ in range(T):
            acc.step(noise_multiplier=sigma, sample_rate=q)

        res = acc.get_epsilon(delta=delta)
        # If it's a tuple/list, take the first item; else it's already a float
        eps = float(res[0]) if isinstance(res, (tuple, list)) else float(res)
        print("[HAS_OPACUS] has been used")
        return eps
    else:
        # ---- Placeholder heuristic (for demo only!) -----------------------------------------
        T = compute_steps(N, B, epochs)
        q = B / N
        approx_eps = (q * math.sqrt(T)) / max(1e-6, sigma) * 2.0
        approx_eps *= (1.0 + max(0.0, math.log(1.0 / max(delta, 1e-12))) / 100.0)
        print("[heuristic] has been used")
        return float(approx_eps)

# ========== 4) GENERATORS (STUBS or REAL) ======================================================
"""
We provide a stub "generator" so the script runs without ctgan/tvae.
If SDV (CTGAN/TVAE) is available, we switch to those for a strong NON-DP baseline.
For real DP training, you'll need to implement DP-SGD inside the generator trainer.
"""

class StubGenerator:
    """
    A stand-in for a DP generator.
    - "Trains" by storing config.
    - "Samples" by drawing from a fitted Gaussian over the preprocessed feature space.
    This is only to demonstrate the tuner flow and interfaces.
    """
    def __init__(self, input_dim: int, sigma: float, clip: float, epochs: int, seed: int = 0):
        self.input_dim = input_dim
        self.sigma = sigma
        self.clip = clip
        self.epochs = epochs
        self.seed = seed
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        set_seed(self.seed)
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0) + 1e-6
        return self

    def sample(self, n: int) -> np.ndarray:
        set_seed(self.seed + 123)
        X_syn = np.random.normal(loc=self.mean_, scale=self.std_, size=(n, self.input_dim))
        return X_syn

# ========== 5) DOWNSTREAM TASK (UTILITY EVALUATION) ============================================
def downstream_auroc(X_syn: np.ndarray, y_syn: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, seed=0) -> float:
    set_seed(seed)
    clf = LogisticRegression(max_iter=1000, n_jobs=None)
    clf.fit(X_syn, y_syn)
    y_prob = clf.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, y_prob))

# ========== 6) TUNER LOOP ======================================================================
"""
This loop varies σ, computes ε, generates synthetic data, trains a downstream classifier,
evaluates AUROC on real test, and aggregates mean ± 95% CI over seeds.
"""

def run_tuner(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    policy: Policy,
    delta: float,
    sigma_grid: List[float],
    seeds: List[int],
    target_auroc: float = 0.78,
    synth_size: int = None,
    *,
    gen_kind: str = "ctgan",
    preprocessor: ColumnTransformer = None,
    raw_train_df: pd.DataFrame = None,
    label_col: str = "readmit_30d",
) -> pd.DataFrame:
    """Run tuner using SDV CTGAN/TVAE as non-DP baselines.
    We still compute ε from (N,B,epochs,C,σ,δ) for reporting, but the generator itself
    is non-DP (baseline). Later you can plug in DP-enabled trainers with the same interface.
    """
    if preprocessor is None or raw_train_df is None:
        raise ValueError("run_tuner now expects preprocessor and raw_train_df so it can train SDV models on RAW.")
    N = X_train.shape[0]
    if synth_size is None:
        synth_size = N

    results = []
    for sigma in sigma_grid:
        eps = epsilon_from_accountant(N=N, B=policy.B, epochs=policy.epochs, C=policy.C, sigma=sigma, delta=delta)

        # Collect AUROCs across seeds
        aurocs = []
        for s in seeds:
            # Choose generator
            if HAS_SDV:
                if gen_kind.lower() == "ctgan":
                    gen = CtganGenerator(epochs=policy.epochs, batch_size=policy.B, verbose=False, label_col=label_col)
                elif gen_kind.lower() == "tvae":
                    gen = TvaeGenerator(epochs=policy.epochs, batch_size=policy.B, verbose=False, label_col=label_col)
                else:
                    raise ValueError(f"Unknown gen_kind: {gen_kind}")
            else:
                # Fallback to stub if SDV not available
                gen = StubGenerator(input_dim=X_train.shape[1], sigma=sigma, clip=policy.C, epochs=policy.epochs, seed=s)

            # Fit on RAW (DataFrame) including label column if supervised
            if isinstance(raw_train_df, pd.DataFrame) and (label_col in raw_train_df.columns):
                X_df = raw_train_df.drop(columns=[label_col])
                y_series = raw_train_df[label_col]
                gen.fit(X_df, y_series)
            else:
                # last resort: use matrix
                gen.fit(pd.DataFrame(X_train), pd.Series(y_train))

            # Sample synthetic as RAW DataFrame
            if hasattr(gen, "sample"):
                syn_df = gen.sample(synth_size)
            else:
                # fallback stub
                X_syn = gen.sample(synth_size)
                aux_clf = LogisticRegression(max_iter=1000)
                aux_clf.fit(X_train, y_train)
                y_prob_syn = aux_clf.predict_proba(X_syn)[:, 1]
                y_syn = (y_prob_syn > 0.5).astype(int)
                auroc = downstream_auroc(X_syn, y_syn, X_test, y_test, seed=s)
                aurocs.append(auroc)
                continue

            # Split X/y from synthetic RAW df
            if label_col in syn_df.columns:
                y_syn = syn_df[label_col].astype(int).to_numpy()
                X_syn_raw = syn_df.drop(columns=[label_col])
            else:
                # if model didn't include label, create pseudo-labels using aux model
                aux_clf = LogisticRegression(max_iter=1000)
                aux_clf.fit(X_train, y_train)
                y_prob_syn = aux_clf.predict_proba(preprocessor.transform(syn_df))[:, 1]
                y_syn = (y_prob_syn > 0.5).astype(int)
                X_syn_raw = syn_df

            # Transform synthetic RAW X to feature matrix using the same preprocessor
            X_syn = preprocessor.transform(X_syn_raw)

            # Downstream evaluation
            auroc = downstream_auroc(X_syn, y_syn, X_test, y_test, seed=s)
            aurocs.append(auroc)

        mean_auc, ci_auc = mean_and_ci(aurocs)
        results.append({
            "sigma": sigma,
            "epsilon": eps,
            "auroc_mean": mean_auc,
            "auroc_ci": ci_auc,
            "B": policy.B, "C": policy.C, "epochs": policy.epochs, "delta": delta,
        })

    df = pd.DataFrame(results).sort_values(by="epsilon")
    df["meets_target"] = df["auroc_mean"] >= target_auroc
    return df

# ========== 7) MAIN ============================================================================
def main():
    # ---- Step 1: Load data & preprocess (collect N) ----
    raw = make_fake_healthcare(n=6000, seed=7)
    X_train, y_train, X_val, y_val, X_test, y_test, N, meta = preprocess_and_split(raw, label="readmit_30d", seed=7)
    print(f"[INFO] N (train size) = {N}")

    # ---- Step 2: Fix policy knobs (B, C, epochs, δ) ----
    policy = Policy(B=256, C=1.0, epochs=50, delta=None)
    delta = policy.delta or (1.0 / N)  # choose δ = 1/N if not provided
    print(f"[INFO] Policy: B={policy.B}, C={policy.C}, epochs={policy.epochs}, delta={delta:.2e}")
    
    # ---- Step 2.5: Compute baseline AUROC (upper bound) ----
    real_clf = LogisticRegression(max_iter=1000)
    real_clf.fit(X_train, y_train)
    real_auc = roc_auc_score(y_test, real_clf.predict_proba(X_test)[:, 1])
    target_auroc = 0.9 * real_auc
    print(f"[BASELINE] Real-data AUROC = {real_auc:.3f}, target τ = {target_auroc:.3f}")

    # ---- Step 3: Define σ search space ----
    sigma_grid = [0.6, 0.9, 1.2, 1.6, 2.0]
    seeds = [0, 1]  # keep short for demo; bump for more stable CI
    # target_auroc = 0.78   #comment as step 2.5 determine the target_auroc

    # ---- Step 4–6: Run tuner with SDV CTGAN baseline ----
    df = run_tuner(
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        policy=policy, delta=delta,
        sigma_grid=sigma_grid, seeds=seeds,
        target_auroc=target_auroc, synth_size=N,
        gen_kind="tvae",
        preprocessor=meta["preprocessor"],
        raw_train_df=meta["X_train_raw"],
        label_col=meta["label"],
    )
    print("\n[TUNER RESULTS] (sorted by epsilon)")
    print(df.to_string(index=False))

    # ---- Step 7: Save results to JSON/CSV for your report ----
    out_dir = "./outputs"
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "tuner_results.csv"), index=False)

    summary = {
        "target_auroc": target_auroc,
        "best_row": df[df["meets_target"]].sort_values("epsilon").head(1).to_dict(orient="records")
    }
    with open(os.path.join(out_dir, "tuner_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[SUMMARY]")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
