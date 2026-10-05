# Fixing the Sampler: Likelihood Decoding and a DP Latent Prior Make DP-VAEs Competitive Tabular Synthesizers

*Working draft — target venues: DPLICIT 2027 workshop, PETS 2027. All numbers come from
`experiments/results_snapshot.jsonl` via `experiments/make_tables.py` and `experiments/analyze.py`;
do not edit numbers by hand.*

## Abstract

Differentially private (DP) synthetic tabular data is usually evaluated with pipelines that fit
data-dependent preprocessors (quantile transforms, observed category sets, class priors) outside the
privacy budget, which both leaks information and makes methods incomparable. We build a leakage-free
benchmark in which every method receives the same table discretized over public codebook domains and
spends its entire budget inside the mechanism. Within it, we show that a conditional variational
autoencoder trained with DP-SGD (DP-VAE) is held back not by training but by how it *samples*. Two
fixes recover most of what is lost. (i) Categorical columns are drawn from the decoder's likelihood
rather than decoded by argmax. (ii) A DP latent prior: a Gaussian fit to the aggregate posterior,
released once with the analytic Gaussian mechanism for 5% of the budget, replaces N(0, I) at sampling
time. We evaluate on Adult, Diabetes130-US and BRFSS 2021 (39k–189k training rows) at ε ∈ {1, 2, 4},
δ = 10⁻⁶, with 10 paired seeds. The resulting DP-VAE + prior beats MST on both train-on-synthetic /
test-on-real AUROC (+0.045 to +0.144) and 2-way marginal error (1.4–3.9× lower) in all nine settings
(Holm-corrected p ≤ 0.007). Compared with plain DP-VAE, it cuts 2-way error 1.7–3.5× at a utility cost
of at most 0.021 AUROC, and its fidelity no longer degrades as ε grows. MST keeps the edge on one-way
marginals, and AIM on low-order marginals where it is tractable. A distance-to-closest-record
membership attack finds no detectable leakage for any DP method (AUC 0.49–0.51, versus 0.56–0.66 when
the training data itself is released). We also find that the smartnoise-synth PATE-CTGAN
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

Third, **the sampling prior.** Even with likelihood decoding, a DP-VAE's marginal fidelity gets
*worse* as the budget grows. We trace this to a mismatch between the N(0, I) prior used at sampling
time and the aggregate posterior the encoder actually learned. The mismatch can be fixed privately
and cheaply.

With these issues fixed, a conditional VAE trained with DP-SGD becomes a strong DP synthesizer. Our
contributions are:

1. A leakage-free evaluation protocol and an open, restart-safe benchmark harness, in which every
   method receives the same public-domain table and spends its whole budget inside the mechanism
   (Section 3).
2. A DP latent prior: a one-shot Gaussian-mechanism release of the aggregate posterior's mean and
   covariance (Section 4). It halves or better DP-VAE's 2-way marginal error at a small utility cost.
3. A controlled comparison on three public datasets at three budgets, with 10 paired seeds and
   multiplicity-corrected tests. DP-VAE + prior beats MST on both TSTR AUROC and 2-way marginal error
   in every setting, and beats DP GAN baselines (Sections 5.1–5.2).
4. Ablations (10 seeds) isolating both sampling fixes: likelihood vs argmax decoding, and DP prior vs
   N(0, I) (Section 5.3), with a diagnostic explaining why fidelity degraded with ε.
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
- **Splits and seeds.** One stratified 80/20 train/test split per dataset (seed 7). DP-VAE (both
  priors), MST and the decoding ablation: 10 seeds; DP-CTGAN and PATE-CTGAN: 5 seeds; non-private
  references: 3 seeds; AIM: as many as finished (see Limitations).

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
| DP-VAE + DP prior (ours) | Conditional VAE, DP-SGD, DP latent prior | `dp_synth_data_gen/dptvae.py` (`fit_latent_prior`), Opacus 1.6 |
| DP-VAE (ours, ablation) | Conditional VAE, DP-SGD, N(0, I) prior | `dp_synth_data_gen/dptvae.py`, Opacus 1.6 |
| DP-CTGAN | Conditional GAN, DP-SGD discriminator, Gumbel-softmax generator | `dp_synth_data_gen/dpctgan_v2.py` |
| PATE-CTGAN | GAN with PATE teacher ensemble | smartnoise-synth 1.0.8 |
| MST | Marginal-based (maximum spanning tree + Private-PGM) | smartnoise-synth 1.0.8 |
| AIM | Workload-adaptive marginal-based | smartnoise-synth 1.0.8, model size capped at 20 MB |
| CTGAN, TVAE | Non-private references | SDV |

**DP-VAE.** Encoder and decoder MLPs (256–128), 16-d latent space, conditioned on the one-hot label,
KL weight β = 1. Each categorical column is decoded as a softmax over its domain and trained with
cross-entropy. At sampling time we draw z ~ N(0, I) and **sample each column from its softmax**
rather than taking the argmax.

**DP latent prior (DP-VAE + prior).** A VAE generates by decoding z ~ N(0, I), but the training
objective only pulls the *aggregate* posterior q(z) = E_x q(z|x) toward that prior. Under DP-SGD the
gap between them grows as ε increases (less noise lets the encoder use more of the latent space),
and decoding prior samples then lands in regions the decoder was rarely trained on. A
non-private diagnostic on Adult (seed 0) isolates the effect: decoding posterior samples of the
training rows gives 2-way TVD 0.055 (ε = 1) and 0.041 (ε = 4), while decoding prior samples gives
0.115 and 0.162. Fitting a single Gaussian to the aggregate posterior already recovers most of the
gap (0.065 and 0.066). We make this fit private:

1. After DP-SGD, draw one posterior sample z_i ~ q(z | x_i, y_i) per training row and clip it to
   L2 norm R (R = 6; under 2% of rows are clipped).
2. Release the vector [Σ z_i, vec(Σ z_i z_iᵀ) / R] once with the Gaussian mechanism. Under add/remove
   adjacency its L2 sensitivity is R√2. The noise scale comes from the exact analytic Gaussian
   mechanism (Balle & Wang, 2018) at (ε_prior, δ/2), with ε_prior = 0.05 ε.
3. Normalise by the noisy row count n̂ from the label histogram, which is already DP and so costs no
   extra budget. Then form the mean and covariance, project the covariance onto the PSD cone
   (eigenvalues ≥ 10⁻³), and sample z ~ N(μ̂, Σ̂) at generation time.

Total budget: ε_label + ε_prior + ε_SGD = ε (5% + 5% + 90%), and DP-SGD and the prior release each
use δ/2, so the whole pipeline is (ε, δ)-DP by basic composition. The extra cost is one forward pass
over the training data.

## 5. Results

Full tables: `paper/tables.md`. Paired tests: `experiments/summary.md`.

### 5.1 Downstream utility

TSTR AUROC, logistic regression (mean ± std over seeds):

| Method | Adult ε=1 | Adult ε=4 | Diabetes130 ε=1 | Diabetes130 ε=4 | BRFSS ε=1 | BRFSS ε=4 |
|---|---|---|---|---|---|---|
| DP-VAE + DP prior (ours) | 0.856 ± 0.006 | 0.884 ± 0.007 | 0.567 ± 0.014 | 0.574 ± 0.012 | 0.729 ± 0.011 | 0.727 ± 0.010 |
| DP-VAE, N(0, I) prior | **0.877 ± 0.004** | **0.889 ± 0.006** | **0.570 ± 0.016** | **0.582 ± 0.012** | **0.733 ± 0.012** | **0.731 ± 0.010** |
| MST | 0.761 ± 0.034 | 0.770 ± 0.019 | 0.522 ± 0.016 | 0.520 ± 0.009 | 0.585 ± 0.004 | 0.586 ± 0.004 |
| AIM‡ | 0.824 (2 seeds) | – | – | – | – | – |
| DP-CTGAN | 0.758 ± 0.045 | 0.785 ± 0.013 | 0.510 ± 0.037 | 0.515 ± 0.018 | 0.630 ± 0.035 | 0.642 ± 0.091 |
| PATE-CTGAN | 0.460 ± 0.053 | 0.516 ± 0.060 | 0.529 ± 0.030 | 0.497 ± 0.011 | 0.509 ± 0.037 | 0.600 ± 0.025 |
| *Non-private CTGAN* | *0.883* | | *0.521* | | *0.732* | |
| *Non-private TVAE* | *0.887* | | *0.551†* | | *0.677* | |
| *Real data* | *0.912* | | *0.635* | | *0.768* | |

† Two of three TVAE seeds on Diabetes130 generated only the majority class (≈9% positive rate), so
AUROC is undefined for them.
‡ AIM exceeded the 3-hour per-job limit (4 CPUs, 16 GB) on Diabetes130 at ε = 1; Adult ε ≥ 2 and
BRFSS runs are pending. See Limitations.

- DP-VAE + DP prior beats MST in all nine (dataset, ε) cells, by +0.045 to +0.144 AUROC (10 paired
  seeds; Holm-corrected p ≤ 0.007 everywhere). Plain DP-VAE's margin is slightly larger: +0.048 to
  +0.147, Holm-corrected p ≤ 0.008.
- The DP prior costs at most 0.021 AUROC relative to plain DP-VAE (largest on Adult at ε = 1; the
  difference is significant in 4 of 9 cells and below 0.01 in 7 of 9).
- Plain DP-VAE beats PATE-CTGAN in 8 of 9 cells and DP-CTGAN in 7 of 9 at Holm-corrected α = 0.05.
  DP-VAE + prior does so in 7 and 5 of 9. In the remaining cells the mean difference still favours
  the VAE, but intervals are wide (5 GAN seeds).
- On Adult at ε = 1, AIM's AUROC (0.824, 2 seeds) sits between MST and DP-VAE + prior (0.856).
- At ε = 1, DP-VAE is within 0.01 of non-private CTGAN/TVAE on Adult and matches or exceeds them on
  BRFSS and Diabetes130. The non-private references use default hyperparameters, so this says the DP
  cost is small, not that DP-VAE is better than a tuned non-private model.
- Gradient-boosting AUROC shows the same ordering (`paper/tables.md`).

### 5.2 Fidelity

Mean marginal TVD (lower is better), ε = 1 / ε = 4:

| Method | Adult 1-way | Adult 2-way | Diabetes130 1-way | Diabetes130 2-way | BRFSS 1-way | BRFSS 2-way |
|---|---|---|---|---|---|---|
| DP-VAE + DP prior (ours) | 0.022 / 0.029 | 0.066 / 0.067 | 0.012 / 0.017 | **0.036 / 0.039** | 0.012 / 0.015 | **0.026 / 0.031** |
| DP-VAE, N(0, I) prior | 0.055 / 0.082 | 0.119 / 0.141 | 0.028 / 0.061 | 0.062 / 0.104 | 0.053 / 0.065 | 0.087 / 0.107 |
| MST | **0.003 / 0.001** | 0.093 / 0.091 | **0.002 / 0.001** | 0.111 / 0.110 | **0.000 / 0.000** | 0.101 / 0.101 |
| AIM (ε = 1, 2 seeds) | 0.004 | **0.040** | – | – | – | – |

- **2-way marginals:** DP-VAE + DP prior beats MST in all nine cells, with 1.4–3.9× lower error
  (Holm-corrected p < 10⁻⁷). It beats plain DP-VAE in all nine, with 1.7–3.5× lower error. Where AIM
  finished (Adult, ε = 1), AIM remains best (0.040 vs 0.066).
- **1-way marginals:** MST and AIM, which measure every 1-way marginal directly, stay near-exact.
  The DP prior cuts DP-VAE's 1-way error 2.4–4.6× but does not close this gap.
- **No degradation with ε.** With the N(0, I) prior, DP-VAE's fidelity gets worse as ε grows (Adult
  2-way 0.119 → 0.141). With the DP prior it is flat (0.066 → 0.067). This matches the diagnostic in
  Section 4: the aggregate posterior drifts further from N(0, I) as DP-SGD noise falls (mean |μ| 0.33
  at ε = 1 vs 0.65 at ε = 4 on Adult), and the DP prior tracks the drift.
- Both GAN baselines have poor marginals (2-way TVD 0.2–0.68), which is consistent with mode collapse
  under DP.

### 5.3 Ablations

**Categorical decoding.**

Same model and training; only the decoding rule changes (10 paired seeds per cell, ε = 1 shown):

| Decoding | Adult AUROC | Adult 2-way TVD | Diabetes130 AUROC | Diabetes130 2-way TVD | BRFSS AUROC | BRFSS 2-way TVD |
|---|---|---|---|---|---|---|
| Sample from softmax (default) | **0.877** | **0.119** | **0.570** | **0.062** | **0.733** | **0.087** |
| Argmax | 0.785 | 0.420 | 0.519 | 0.449 | 0.680 | 0.174 |

Argmax collapses each column toward its conditional mode. Across all nine (dataset, ε) cells,
sampling improves AUROC by 0.02–0.12 (significant after Holm correction in 8 of 9 cells; the
exception is Diabetes130 at ε = 2, with p_Holm = 0.055). It also lowers both 1-way and 2-way marginal
error in all 9 cells (2-way TVD 2.0–7.3× lower at ε = 1). With argmax decoding, DP-VAE loses most of
its advantage over MST on Adult and Diabetes130. Sampling is the decoding rule that matches the
model's own likelihood, so we recommend it as the default for any VAE-style tabular synthesizer.

The argmax gap narrows as ε grows (e.g. Adult 2-way TVD gap 0.30 → 0.15 from ε = 1 to 4): with less
noise the decoder is more confident and the argmax discards less.

**Sampling prior.** The comparison of DP-VAE + DP prior with plain DP-VAE in Sections 5.1–5.2 is
itself an ablation of the prior, since the training is identical apart from moving 5% of ε from
DP-SGD to the prior release. Fidelity improves in all nine cells, and utility changes by
−0.021 to +0.001 AUROC. In a non-private diagnostic (Adult, seed 0), a 10-component Gaussian mixture
fit to the aggregate posterior reaches 2-way TVD 0.059 / 0.049 at ε = 1 / 4, against 0.065 / 0.066
for a single Gaussian. A private mixture is therefore a natural next step for higher budgets.

### 5.4 Empirical privacy check

The DCR membership-inference AUC is 0.494–0.512 for every DP method, dataset and ε, with standard
deviation about 0.01 across seeds. Releasing the training data itself scores 0.56–0.66, so the attack
can detect leakage when it exists. On Diabetes130 every method, the non-private ones included, sits
about 0.005 above 0.5. Because this offset does not depend on the method, we attribute it to the
dataset (ties in Hamming distance on a low-cardinality domain) rather than to the mechanisms. This
attack is a sanity check, not an audit; it cannot lower-bound ε. A tight audit with worst-case
canaries (Annamalai et al. 2024) is future work.

### 5.5 Cost

Mean fit-plus-sample time on 4 CPU cores: MST 49–92 s, DP-VAE 382–652 s, DP-VAE + DP prior 488–814 s
(some runs shared the CPU with AIM), DP-CTGAN 395–550 s, PATE-CTGAN 233–955 s. AIM took 22–27
minutes on Adult at ε = 1 and peaked at about 6 GB of memory. On Diabetes130 at ε = 1 it hit the
3-hour limit twice. On Adult at ε ≥ 2 its first attempts ran out of memory or time while sharing the
machine with a second AIM job; solo retries are queued. The VAE methods run on CPU in minutes.

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
- **AIM coverage.** AIM's model size is capped at 20 MB for CPU tractability, which may understate
  its performance. Even so, it exceeded our 3-hour per-job limit on 4 CPUs / 16 GB in most settings,
  so we report it only where it finished (Adult, ε = 1, 2 seeds; BRFSS pending). On those cells it
  has the best 2-way fidelity of any method, and a fair reading of our results is that AIM leads on
  low-order marginals while DP-VAE + prior leads on downstream utility.
- **Single split and moderate seed counts.** Seeds vary the mechanism's randomness, not the data
  split.
- **The privacy check is heuristic** (Section 5.4).

## 8. Reproducibility

`experiments/prepare_data.py` builds the datasets. `experiments/bench.py` runs one
(dataset, method, ε, seed) job. `experiments/runner.py` and `experiments/launch.sh` run the
restart-safe queue. `experiments/analyze.py` and `experiments/make_tables.py` regenerate every number
in this draft.
