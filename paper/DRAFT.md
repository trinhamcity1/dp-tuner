# Sampling, Not Argmax: A Leakage-Free Benchmark Showing DP-VAEs Beat Marginal and GAN Synthesizers on Downstream Utility

*Working draft — target venues: DPLICIT 2027 workshop, PETS 2027. All numbers come from
`experiments/results_snapshot.jsonl` via `experiments/make_tables.py` and `experiments/analyze.py`;
do not edit numbers by hand.*

## Abstract

Differentially private (DP) synthetic tabular data is usually evaluated with pipelines that fit
data-dependent preprocessors (quantile transforms, observed category sets, class priors) outside the
privacy budget, which both leaks information and makes methods incomparable. We build a leakage-free
benchmark in which every method receives the same table discretized over public codebook domains and
spends its entire budget inside the mechanism. On three public health and census datasets (Adult,
Diabetes130-US, BRFSS 2021; 39k–189k training rows) at ε ∈ {1, 2, 4}, δ = 10⁻⁶, a conditional
variational autoencoder trained with DP-SGD (DP-VAE) beats MST on train-on-synthetic/test-on-real
AUROC in all nine settings (+0.05 to +0.15 AUROC; Holm-corrected p ≤ 0.008, 10 paired seeds), and
beats DP-SGD and PATE-based CTGAN variants. At ε = 1 its utility matches that of non-private CTGAN and
TVAE. A single design choice drives much of this: decoding categorical columns by sampling from the decoder's
likelihood instead of taking the argmax, which costs 0.03–0.14 AUROC and multiplies 2-way marginal
error 2–7× when omitted. MST keeps an advantage on one-way marginals, so the result is a
utility/fidelity trade-off rather than uniform dominance. A distance-to-closest-record membership
attack finds no detectable leakage for any DP method (mean AUC 0.494–0.512, versus 0.56–0.66 when the
training data itself is released). Finally, we find that the widely used smartnoise-synth PATE-CTGAN
implementation shares one discriminator object between all teachers and the student, so its PATE
privacy analysis does not apply as implemented.

## 1. Introduction

Health agencies, survey programmes and companies increasingly want to share tabular microdata for
model development without exposing the individuals in it. Differentially private (DP) synthetic data
promises a release that can be reused indefinitely under a single formal privacy budget. For many
users the question is practical: will a classifier trained on the synthetic table still work on real
people? This is the train-on-synthetic, test-on-real (TSTR) setting we study.

Recent benchmarks have largely settled on marginal-based mechanisms such as MST and AIM as the
state of the art, with deep generative models (GANs and VAEs trained with DP-SGD) reported as weaker
under realistic budgets. We argue that two methodological issues blur this comparison.

First, **preprocessing leakage.** Deep tabular generators are usually wrapped in data-dependent
transforms: quantile or mode-specific normalisers fitted on the raw data, category vocabularies read
off the training set, and class proportions copied from it. None of these is covered by the
mechanism's privacy accounting. Marginal-based libraries, by contrast, ask for a public domain or
spend budget to learn one. Comparisons that mix the two give the deep models non-private information.
Our own earlier pipeline made exactly this mistake, and its reported advantage shrank when we removed
it; we therefore treat a leakage-free protocol as a first-class contribution.

Second, **decoding.** VAE-style tabular synthesizers model each categorical column with a softmax, but
implementations often emit the argmax at sampling time. Under DP noise the decoder is uncertain, so
the argmax collapses columns toward their modes. That destroys the marginals and the label–feature
dependencies a downstream classifier needs.

With both issues fixed, a simple conditional VAE trained with DP-SGD becomes a strong baseline. Our
contributions are:

1. A leakage-free evaluation protocol and an open, restart-safe benchmark harness, in which every
   method receives the same public-domain table and spends its whole budget inside the mechanism
   (Section 3).
2. A controlled comparison on three public datasets at three budgets, with 10 paired seeds and
   multiplicity-corrected tests. DP-VAE beats MST on TSTR AUROC in every setting and beats DP GAN
   baselines in most, at modest CPU cost (Section 5.1).
3. A characterisation of the trade-off: marginal methods remain better at reproducing low-order
   marginals, while DP-VAE preserves more predictive signal (Section 5.2).
4. An ablation showing that sampling, rather than argmax decoding, accounts for much of DP-VAE's
   advantage (Section 5.3).
5. An empirical membership-inference check (Section 5.4), and an implementation flaw in a widely used
   PATE-CTGAN library that invalidates its privacy analysis (Section 6).

## 2. Background and related work

**Differential privacy and DP-SGD.** A mechanism M is (ε, δ)-DP if, for neighbouring datasets D and D′,
Pr[M(D) ∈ S] ≤ e^ε Pr[M(D′) ∈ S] + δ. DP-SGD (Abadi et al., 2016) clips per-example gradients and
adds Gaussian noise; its cumulative privacy loss is tracked with Rényi DP (Mironov, 2017) or
numerical privacy-loss-distribution accountants such as PRV (Gopi et al., 2021). Post-processing
preserves DP, so sampling from a DP-trained generator costs no additional budget.

**Marginal-based synthesis.** MST (McKenna et al., 2021), the winner of the 2018 NIST DP synthetic data
challenge, privately selects a maximum spanning tree of 2-way marginals, measures them with Gaussian
noise, and fits a graphical model with Private-PGM. AIM (McKenna et al., 2022) adaptively selects
marginals against a workload and is generally the strongest method on marginal-preservation
benchmarks.

**Deep DP synthesis.** DP-GAN variants clip and noise discriminator gradients. DP-CTGAN (Fang et al.,
2022) applies this to CTGAN's conditional architecture (Xu et al., 2019). PATE-GAN (Jordon et al.,
2019) trains teacher discriminators on disjoint partitions and a student on their noisy votes, and
PATE-CTGAN (Rosenblatt et al., 2020) combines this with CTGAN. VAE-based synthesizers such as TVAE
(Xu et al., 2019) can be trained with DP-SGD directly, since the whole model sees the data.

**Benchmarks and auditing.** Tao et al. (2021) and Ganev and De Cristofaro compare graphical and deep
DP synthesizers and generally favour marginal methods. Annamalai et al. (2024) show that tight
auditing can expose gaps between claimed and actual privacy in synthetic data generators, and Ganev,
Annamalai and De Cristofaro (TMLR 2025) find 19 privacy violations across six open-source PATE-GAN
implementations. Similarity-based privacy metrics such as distance to closest record are known to be
weak evidence of privacy; we use one only as a sanity check alongside formal DP.

## 3. Leakage-free evaluation protocol

- **Public domains.** Every column is categorical over a domain fixed from the dataset's public
  codebook. Numeric columns with large public ranges are cut into 16 equal-width bins over the
  codebook range (e.g. BRFSS BMI 12–99, Adult age 17–90), so no quantiles or observed extrema are
  computed from data (`experiments/prepare_data.py`).
- **Identical inputs.** All methods receive the same discretized table; marginal methods run with
  `preprocessor_eps = 0` because no preprocessing statistics are needed.
- **One individual per row.** Diabetes130 keeps each patient's first encounter, so record-level DP is
  individual-level DP.
- **Budget accounting.** δ = 10⁻⁶ (< 1/n for every dataset). Conditional deep models spend 5% of ε on a
  Laplace-noised class histogram (pure DP, sensitivity 1) used to draw labels at sampling time, and
  95% on DP-SGD; budgets compose additively. DP-SGD noise is calibrated with the PRV accountant, the
  same accountant used to report spent ε (spent ε within 1% of target in every run).
- **Fixed optimisation budget.** DP-SGD methods take 2,000 steps with batch 512 and clip norm 1.6 on
  every dataset (epochs = steps / ⌊n/512⌋). Non-private references take 5,000 steps.
- **Splits and seeds.** One stratified 80/20 train/test split per dataset (seed 7). DP-VAE and MST:
  10 seeds; DP-CTGAN and PATE-CTGAN: 5 seeds; non-private references and ablation: 3 seeds.

### Metrics

- **Utility:** TSTR AUROC on the real held-out test set, logistic regression and histogram gradient
  boosting on one-hot features (macro one-vs-rest for 3-class BRFSS).
- **Fidelity:** mean total variation distance over all 1-way and 2-way marginals.
- **Empirical privacy:** distance-to-closest-record attack. Hamming distance from 2,000 training
  members and 2,000 held-out non-members to 20,000 synthetic rows, scored by AUC. Releasing the
  training data itself calibrates what the attack can detect.

## 4. Methods compared

| Method | Type | Implementation |
|---|---|---|
| DP-VAE (ours) | Conditional VAE, DP-SGD | `dp_synth_data_gen/dptvae.py`, Opacus 1.6 |
| DP-CTGAN | Conditional GAN, DP-SGD discriminator, Gumbel-softmax generator | `dp_synth_data_gen/dpctgan_v2.py` |
| PATE-CTGAN | GAN with PATE teacher ensemble | smartnoise-synth 1.0.8 |
| MST | Marginal-based (maximum spanning tree + Private-PGM) | smartnoise-synth 1.0.8 |
| AIM | Workload-adaptive marginal-based | smartnoise-synth 1.0.8, model size capped at 20 MB (*runs pending*) |
| CTGAN, TVAE | Non-private references | SDV |

**DP-VAE.** Encoder and decoder MLPs (256–128), 16-d latent space, conditioned on the one-hot label,
KL weight β = 1. Each categorical column is decoded as a softmax over its domain and trained with
cross-entropy. At sampling time we draw z ~ N(0, I) and **sample each column from its softmax**
rather than taking the argmax.

## 5. Results

Full tables: `paper/tables.md`. Paired tests: `experiments/summary.md`.

### 5.1 Downstream utility

TSTR AUROC, logistic regression (mean ± std over seeds):

| Method | Adult ε=1 | Adult ε=4 | Diabetes130 ε=1 | Diabetes130 ε=4 | BRFSS ε=1 | BRFSS ε=4 |
|---|---|---|---|---|---|---|
| DP-VAE (ours) | **0.877 ± 0.004** | **0.889 ± 0.006** | **0.570 ± 0.016** | **0.582 ± 0.012** | **0.733 ± 0.012** | **0.731 ± 0.010** |
| MST | 0.761 ± 0.034 | 0.770 ± 0.019 | 0.522 ± 0.016 | 0.520 ± 0.009 | 0.585 ± 0.004 | 0.586 ± 0.004 |
| DP-CTGAN | 0.758 ± 0.045 | 0.785 ± 0.013 | 0.510 ± 0.037 | 0.515 ± 0.018 | 0.630 ± 0.035 | 0.642 ± 0.091 |
| PATE-CTGAN | 0.460 ± 0.053 | 0.516 ± 0.060 | 0.529 ± 0.030 | 0.497 ± 0.011 | 0.509 ± 0.037 | 0.600 ± 0.025 |
| *Non-private CTGAN* | *0.883* | | *0.521* | | *0.732* | |
| *Non-private TVAE* | *0.887* | | *0.551†* | | *0.677* | |
| *Real data* | *0.912* | | *0.635* | | *0.768* | |

† Two of three TVAE seeds on Diabetes130 generated only the majority class (≈9% positive rate), so
AUROC is undefined for them.

- DP-VAE beats MST in all nine (dataset, ε) cells: +0.116 to +0.131 on Adult, +0.048 to +0.062 on
  Diabetes130, +0.145 to +0.147 on BRFSS (10 paired seeds; Holm-corrected p ≤ 0.008 everywhere).
- DP-VAE beats PATE-CTGAN in 7 of 9 cells and DP-CTGAN in 5 of 9 at Holm-corrected α = 0.05; in the
  remaining cells the mean difference still favours DP-VAE (+0.05 to +0.12) but intervals are wide
  (5 seeds).
- At ε = 1, DP-VAE is within 0.01 of non-private CTGAN/TVAE on Adult and matches or exceeds them on
  BRFSS and Diabetes130. The non-private references use default hyperparameters, so this says the DP
  cost is small, not that DP-VAE is better than a tuned non-private model.
- Gradient-boosting AUROC shows the same ordering (`paper/tables.md`).

### 5.2 Fidelity: a trade-off with MST

- **1-way marginals:** MST is near-exact (TVD ≤ 0.003), as it measures every 1-way marginal directly;
  DP-VAE's error is 0.03–0.08.
- **2-way marginals:** mixed. MST is better on Adult (0.091–0.093 vs 0.119–0.141). DP-VAE is better on
  Diabetes130 at ε = 1, 2 (0.062 vs 0.111; 0.084 vs 0.110) and on BRFSS at ε = 1 (0.087 vs 0.101).
  The other cells show no significant difference.
- **DP-VAE fidelity worsens as ε grows** on every dataset (e.g. Adult 1-way TVD 0.055 → 0.082 from
  ε = 1 to 4), while utility rises or stays flat. Hypothesis to test: with less gradient noise the
  aggregate posterior drifts from the N(0, I) prior used at sampling time, and the decoder's
  per-column softmaxes become sharper and less calibrated.
- Both GAN baselines have poor marginals (2-way TVD 0.2–0.68), which is consistent with mode collapse
  under DP.

### 5.3 Ablation: categorical decoding

Same model and training; only the decoding rule changes (argmax: 3 seeds; sampling: 10 seeds):

| Decoding | Adult ε=1 AUROC | Adult ε=1 2-way TVD | Diabetes130 ε=1 AUROC | BRFSS ε=1 AUROC |
|---|---|---|---|---|
| Sample from softmax (default) | 0.877 | 0.119 | 0.570 | 0.733 |
| Argmax | 0.791 | 0.418 | 0.523 | 0.668 |

Argmax collapses each column toward its conditional mode. That inflates 2-way error 2–7× and costs
0.03–0.14 AUROC; with argmax decoding, DP-VAE falls to roughly MST's utility. The direction is the
same in all 9 cells and every unadjusted paired p is below 0.05, but with only 3 seeds none survives
Holm correction across all 36 utility comparisons. *To do: run 10 seeds for the ablation.* Sampling is the decoding
rule that matches the model's own likelihood, so we recommend it as the default for any VAE-style
tabular synthesizer.

### 5.4 Empirical privacy check

The DCR membership-inference AUC is 0.494–0.512 for every DP method, dataset and ε, with standard
deviation about 0.01 across seeds. Releasing the training data itself scores 0.56–0.66, so the attack
can detect leakage when it exists. On Diabetes130 every method, the non-private ones included, sits
about 0.005 above 0.5. Because this offset does not depend on the method, we attribute it to the
dataset (ties in Hamming distance on a low-cardinality domain) rather than to the mechanisms. This
attack is a sanity check, not an audit; it cannot lower-bound ε. A tight audit with worst-case
canaries (Annamalai et al. 2024) is future work.

### 5.5 Cost

Mean fit-plus-sample time on 4 CPU cores: MST 49–92 s, DP-VAE 382–652 s, DP-CTGAN 395–550 s,
PATE-CTGAN 233–955 s. DP-VAE is roughly 7–13× slower than MST but runs on CPU in minutes.

## 6. An implementation flaw in smartnoise-synth PATE-CTGAN

In smartnoise-synth 1.0.8 (`snsynth/pytorch/nn/patectgan.py`), the student discriminator and every
teacher are the same object:

```python
student_disc = discriminator
teacher_disc = [discriminator for i in range(self.num_teachers)]
```

Every teacher update therefore trains the one shared network on its partition of raw data, and the
generator is trained against that same network. PATE's guarantee assumes teachers trained on disjoint
data whose only output is a noisy vote. With a shared network, the student and generator receive
updates computed directly from raw data, so the noisy-vote accounting no longer bounds the privacy loss.
The implementation also ignores its `epochs` argument and trains until the moments accountant reaches
ε. In our runs PATE-CTGAN's utility stays near chance (AUROC 0.46–0.60). Ganev et al. (TMLR 2025) audit
six PATE-GAN implementations, including smartnoise's PATE-GAN; we did not find this PATE-CTGAN issue
reported. *To do before submission:* check the smartnoise-sdk issue tracker, confirm the issue in
the latest release, run a canary-based audit, and disclose to the maintainers.

## 7. Limitations

- **Hyperparameter selection.** DP-VAE's architecture and KL weight were chosen during earlier
  development on BRFSS using non-private validation, and that tuning is not charged to the budget.
  Adult and Diabetes130 were never used for tuning, so they act as held-out datasets for the
  hyperparameter choice; the result holds on both. Baselines use library defaults. A fully private
  hyperparameter search (e.g. Papernot & Steinke 2022) is future work.
- **Categorical-only protocol.** Discretising numeric columns favours methods built for discrete data
  (MST, AIM) and limits fidelity on continuous attributes.
- **AIM model size.** AIM's model size is capped at 20 MB for CPU tractability, which may understate
  its performance (*AIM results pending*).
- **Single split and moderate seed counts.** Seeds vary the mechanism's randomness, not the data
  split.
- **The privacy check is heuristic** (Section 5.4).

## 8. Reproducibility

`experiments/prepare_data.py` builds the datasets. `experiments/bench.py` runs one
(dataset, method, ε, seed) job. `experiments/runner.py` and `experiments/launch.sh` run the
restart-safe queue. `experiments/analyze.py` and `experiments/make_tables.py` regenerate every number
in this draft.
