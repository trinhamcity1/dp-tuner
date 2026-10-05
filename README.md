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

## Findings (leakage-free benchmark, three datasets)

Full write-up: [`paper/DRAFT.md`](paper/DRAFT.md). Tables: [`paper/tables.md`](paper/tables.md).
Paired tests: [`experiments/summary.md`](experiments/summary.md). Raw runs:
`experiments/results_snapshot.jsonl` (474 runs).

**Setup.** Adult, Diabetes130-US and BRFSS 2021 (39k–189k training rows), each discretized over
public codebook domains so that no preprocessing statistic leaks outside the privacy budget.
ε ∈ {1, 2, 4}, δ = 10⁻⁶, 10 paired seeds for the main methods. Utility is train-on-synthetic /
test-on-real AUROC; fidelity is mean 1-way and 2-way marginal TVD; empirical privacy is a
distance-to-closest-record membership attack.

**Main result.** DP-VAE + DP latent prior samples categorical columns from the decoder's
likelihood. It also replaces the N(0, I) sampling prior with a Gaussian fit to the aggregate
posterior, released with the analytic Gaussian mechanism for 5% of ε. In all nine
dataset × ε settings it beats MST on both AUROC (+0.045 to +0.144) and 2-way marginal error
(1.4–3.9× lower), Holm-corrected p ≤ 0.007. MST and AIM keep the edge on 1-way marginals, and AIM
on 2-way marginals where it ran (Adult, ε = 1). No DP method shows detectable membership leakage
(attack AUC 0.49–0.51). The smartnoise-synth PATE-CTGAN implementation shares one discriminator
object between its teachers and student; see the draft, Section 6.

**Superseded pilot.** An earlier single-dataset pilot on BRFSS reported DP-TVAE results under a
pipeline that fitted preprocessors (quantile transforms, class proportions) outside the privacy
budget. Those numbers are not comparable to, and are replaced by, the benchmark above.

## Reproducing

```bash
python experiments/prepare_data.py        # build the three datasets (Adult/Diabetes130 raw files from UCI)
experiments/launch.sh                     # restart-safe job queue (experiments/runner.py)
RESULTS_FILE=results_snapshot.jsonl python experiments/make_tables.py > paper/tables.md
RESULTS_FILE=results_snapshot.jsonl python experiments/analyze.py --markdown experiments/summary.md
```
