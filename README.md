# DP Epsilon Tuner

A research framework for exploring the privacy–utility tradeoff in differential privacy: automatically tunes the DP noise level in synthetic tabular data generators and finds the smallest privacy budget (ε) that still preserves downstream utility.

## Research problem

Hospitals and AI centers struggle to select a differential privacy budget (ε) when releasing or training on synthetic versions of private patient data.

- Too small ε → strong privacy but poor utility (synthetic data is useless downstream).
- Too large ε → weak privacy but good accuracy.

This project automates finding

$$
\epsilon^* = \min \epsilon \quad \text{s.t. utility (AUROC)} \geq \tau
$$

where τ is defined relative to a baseline (90% of real-data AUROC on a held-out test set), by sweeping the DP-SGD noise multiplier σ, computing the corresponding ε via an Opacus RDP accountant, training a DP generator at each σ, and measuring downstream classifier AUROC on real held-out data.

## Architecture

- `dp_tuner.py` — the tuner loop: preprocessing/split, the RDP accountant (`epsilon_from_accountant`), the σ sweep (`run_tuner`), and downstream AUROC evaluation.
- `generators/` — non-DP SDV baselines (CTGAN, TVAE) used as an upper-bound reference.
- `dp_synth_data_gen/` — DP generators trained with Opacus DP-SGD:
  - `dpctgan.py` / `dpctgan_v2.py` — DP-CTGAN (adversarial; v2 adds spectral norm / quantile transforms).
  - `dptvae.py` — a conditional VAE trained end-to-end with DP-SGD (no adversarial component). **Currently the best-performing generator in this project's own experiments — see Findings below.**
- `demo.py` — entry point; auto-detects a CSV in `input_data/`, guesses its label column, and runs the full σ sweep.

## Installation

```bash
git clone https://github.com/trinhamcity1/dp-tuner.git
cd dp-tuner
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Drop exactly one CSV into `input_data/` (label column is auto-detected by name; see `demo.py`), then:

```bash
python3 demo.py
```

Results are written to `outputs/tuner_results.csv` and `outputs/tuner_summary.json`.

## Findings (pilot experiment, one dataset)

Experiments below were run on the [BRFSS 2021 diabetes health-indicators dataset](https://www.cdc.gov/brfss/) (`input_data/diabetes_012_health_indicators_BRFSS2021.csv`, ~236k rows, 21 features, 3-class label), with a logistic-regression downstream classifier and macro one-vs-rest AUROC as the utility metric. **This is a single-dataset pilot study, not a validated general result** — see Limitations below.

Real-data baseline AUROC: **0.776**. Target τ (90% of baseline): **0.698**.

| Generator | Best AUROC | ε at best point | Notes |
|---|---|---|---|
| Non-DP CTGAN (no privacy, reference ceiling) | 0.723 | — | |
| DP-CTGAN v2 (GAN + DP-SGD, untuned) | 0.604 | 2.10 | Non-monotonic AUROC-vs-ε curve; wide run-to-run variance (±0.045–0.163) |
| DP-TVAE, default KL weight (β=0.02) | 0.697 (n=2) | 1.80 | Looked promising at n=2; **a 5-seed re-run showed the true mean was ~0.62–0.65** — the n=2 result was a favorable statistical fluke, not a stable finding |
| **DP-TVAE, tuned KL weight (β=1.0)** | **0.722** | **1.58** | Matches the non-DP ceiling; tight variance (±0.017–0.048 across σ); 4 of 5 tested σ clear the target |

### The core finding

Replacing the DP generator's architecture from a GAN (DP-CTGAN) to a conditional VAE (DP-TVAE) substantially improved both **utility** (0.60 → ~0.70 AUROC at comparable ε) and **training stability** (much tighter run-to-run variance) under identical DP-SGD noise. The likely mechanism: DP-CTGAN trains an adversarial minimax game, and DP-SGD noise on the discriminator's gradients destabilizes that game far more than it destabilizes a VAE's single smooth reconstruction+KL objective — there's no second network to fall out of sync with.

A follow-up sweep of the VAE's KL-divergence weight (β) found a clean, monotonic relationship: increasing β from the default 0.02 up to ~0.8–1.0 both raised mean AUROC (0.68 → 0.72) *and* collapsed the variance across seeds (±0.046 → ±0.006 at a fixed σ), before over-regularizing and degrading again past β≈2.5–4.0. The interpretation: DP-SGD's injected noise gives the encoder/decoder something spurious to overfit to, and a stronger KL penalty (regularizing the latent space back toward its prior) fights that overfitting — i.e., **under DP-SGD noise, more regularization helps more than it normally would**, a pattern worth checking against other DP-training regimes beyond tabular VAEs.

With β tuned, the full σ sweep at full scale (150 epochs, all 236k rows) cleared the 90%-of-baseline utility target at 4 of 5 tested σ values, with the strongest-privacy point tested (σ=2.0, ε=1.58) achieving AUROC 0.722 — matching the non-DP ceiling almost exactly while providing a meaningfully small ε.

### Limitations (read before citing this)

- **Single dataset.** Everything above is one BRFSS table. No claim of generalization to other tabular domains, feature distributions, or label types has been tested.
- **Thin seed counts at the final operating points.** The confirmed full-scale sweep uses 2 seeds per σ. An earlier result at n=2 (the β=0.02 DP-TVAE row above) turned out to be a fluke once re-run with 5 seeds — the same risk applies to the β=1.0 numbers until they get the same treatment.
- **No comparison to the published DP-tabular-synthesis literature** (DP-CTGAN as originally published, PATE-GAN, DP-WGAN, other private VAE variants). This project's DP-CTGAN implementation is a from-scratch one, not a reproduction of any specific paper's reported numbers.
- **Single downstream model, single metric.** Only logistic regression / macro AUROC. No distributional fidelity checks (marginal or correlation preservation) and no privacy-attack evaluation (e.g., membership inference) to empirically corroborate that the DP guarantee is doing real protective work.
- **The regularization-vs-DP-noise explanation is a hypothesis** supported by a clean correlational sweep, not by a mechanistic diagnostic (e.g., gradient variance/noise decomposition).

## Status

Functional end-to-end: bug-fixed DP-CTGAN v2, a rebuilt DP-TVAE (the previous version was a non-functional stub), label auto-detection, and a working σ→ε→AUROC pipeline validated at full scale on real data. The VAE-vs-GAN and KL-weight findings above are a promising pilot result worth extending (more datasets, more seeds, literature baselines) before treating them as established.
