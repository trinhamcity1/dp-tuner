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
         4.4) Train downstream classifier on synthetic, eval AUROC on REAL test
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

# Optional: CTGAN/TVAE (tabular generators)
try:
    from ctgan import CTGAN
    from ctgan.sdv import TVAE
    HAS_CTGAN = True
except Exception:
    HAS_CTGAN = False

# Optional: Opacus for DP-SGD accounting
try:
    from opacus.accountants import RDPAccountant
    HAS_OPACUS = True
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
    y = df[label].values
    X = df.drop(columns=[label])

    # Identify column types
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Column transformer: one-hot for categoricals; scale numerics
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    # First split train+temp
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + val_size), random_state=seed, stratify=y)
    # Split temp into val/test
    rel_test = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=rel_test, random_state=seed, stratify=y_temp)

    # Fit transformer on train, transform all
    pre.fit(X_train)
    X_train_t = pre.transform(X_train)
    X_val_t = pre.transform(X_val)
    X_test_t = pre.transform(X_test)

    # N is size of training set after preprocessing
    N = X_train_t.shape[0]

    meta = {
        "preprocessor": pre,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "classes_": np.unique(y).tolist(),
    }
    return (X_train_t, y_train, X_val_t, y_val, X_test_t, y_test, N, meta)

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
        # Using Opacus' RDP accountant in a simplified way:
        acc = RDPAccountant()
        # We "simulate" T steps with Poisson subsampling. Opacus expects to be hooked into a trainer,
        # but we'll approximate by manually stepping the accountant.
        # NOTE: This is still a simplification; in a real trainer you'd let Opacus track per-step.
        # Here we use the analytical RDP step for Gaussian mechanism with sampling rate q.
        for _ in range(T):
            acc.step(noise_multiplier=sigma, sample_rate=q)

        # Opacus >= 1.3 returns a float; older versions may return (eps, best_alpha)
        val = acc.get_epsilon(delta)
        eps = float(val if isinstance(val, (int, float)) else val[0])
        return eps
    else:
        # ---- Placeholder heuristic (for demo only!) -----------------------------------------
        # ε roughly decreases with larger σ; increases with T and q. This is NOT a real bound.
        # Replace with a true accountant in your real code.
        T = compute_steps(N, B, epochs)
        q = B / N
        approx_eps = (q * math.sqrt(T)) / max(1e-6, sigma) * 2.0
        # Adjust by delta magnitude (smaller delta -> slightly larger effective epsilon)
        approx_eps *= (1.0 + max(0.0, math.log(1.0 / max(delta, 1e-12))) / 100.0)
        return float(approx_eps)

# ========== 4) GENERATORS (STUBS or REAL) ======================================================
"""
We provide a stub "generator" so the script runs without ctgan/tvae.
If ctgan is installed, you can switch to real CTGAN/TVAE training (without DP).
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
) -> pd.DataFrame:
    N = X_train.shape[0]
    if synth_size is None:
        synth_size = N  # generate same number of synthetic samples

    results = []
    for sigma in sigma_grid:
        eps = epsilon_from_accountant(N=N, B=policy.B, epochs=policy.epochs, C=policy.C, sigma=sigma, delta=delta)

        # Collect AUROCs across seeds
        aurocs = []
        for s in seeds:
            # STEP 4.1: "Train DP generator" (stubbed here)
            gen = StubGenerator(input_dim=X_train.shape[1], sigma=sigma, clip=policy.C, epochs=policy.epochs, seed=s)
            gen.fit(X_train)

            # STEP 4.3: Sample synthetic
            X_syn = gen.sample(synth_size)
            # For the stub, synth labels are built by a noisy projection of X_train->y_train correlation
            # In real code, you'd sample (X,y) jointly from the learned tabular model.
            # We'll hack labels by training a quick LR on real train, then label synthetic via that LR.
            aux_clf = LogisticRegression(max_iter=1000)
            aux_clf.fit(X_train, y_train)
            y_prob_syn = aux_clf.predict_proba(X_syn)[:, 1]
            y_syn = (y_prob_syn > 0.5).astype(int)

            # STEP 4.4: Train downstream model on synthetic, eval on REAL test
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
    # Selection rule: smallest ε with AUROC ≥ target
    df["meets_target"] = df["auroc_mean"] >= target_auroc
    return df

# ========== 7) MAIN ============================================================================
def main():
    # ---- Step 1: Load data & preprocess (collect N) ----
    raw = make_fake_healthcare(n=6000, seed=7)
    X_train, y_train, X_val, y_val, X_test, y_test, N, meta = preprocess_and_split(raw, label="readmit_30d", seed=7)
    print(f"[INFO] N (train size) = {N}")

    # ---- Step 2: Fix policy knobs (B, C, epochs, δ) ----
    policy = Policy(B=128, C=1.0, epochs=50, delta=None)
    delta = policy.delta or (1.0 / N)  # choose δ = 1/N if not provided
    print(f"[INFO] Policy: B={policy.B}, C={policy.C}, epochs={policy.epochs}, delta={delta:.2e}")

    # ---- Step 3: Define σ search space ----
    sigma_grid = [0.6, 0.9, 1.2, 1.6, 2.0]
    seeds = [0, 1, 2]
    target_auroc = 0.78

    # ---- Step 4–6: Run tuner ----
    df = run_tuner(
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        policy=policy, delta=delta,
        sigma_grid=sigma_grid, seeds=seeds,
        target_auroc=target_auroc, synth_size=N,
    )
    print("\n[TUNER RESULTS] (sorted by epsilon)")
    print(df.to_string(index=False))

    # ---- Step 7: Save results to JSON/CSV for your report ----
    out_dir = "./outputs"
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "tuner_results.csv"), index=False)

    # Persist a lightweight JSON summary
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
