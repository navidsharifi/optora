# Paper reproductions

The other pages of this tab check Optora against closed forms. The
[optora-reproductions](https://github.com/navidsharifi/optora-reproductions)
repository checks it against published experiments. Each reproduction
installs the released `optora` package from PyPI, calls the public API
only, and reruns an experiment from a paper on distributionally robust
optimization.

[:simple-googlecolab: Open the notebook in Colab](https://colab.research.google.com/github/navidsharifi/optora-reproductions/blob/main/notebooks/reproductions.ipynb){ .md-button .md-button--primary }
[:fontawesome-brands-github: Notebook on GitHub](https://github.com/navidsharifi/optora-reproductions/blob/main/notebooks/reproductions.ipynb){ .md-button }

One notebook holds every reproduction. It opens with the recorded results
of all papers, which are visible without running anything, and a setup
cell that clones and installs the repository on Colab. Each paper then has
its report, its recorded results, and cells that rerun its smoke and full
configurations.

## How a claim is checked

Each paper is reduced to a few numbered claims. For every claim, the
location in the paper, the metric, the value the paper reports and a
tolerance are committed before the full run. A claim counts as reproduced
when the mean over seeds lies within that tolerance. When the problem
has a convex reformulation, the reproduction also solves it with a
reference solver that shares no code with Optora, either HiGHS or
Clarabel. When a result disagrees with the paper,
the report traces the cause, and a cause that lies in the library becomes
an issue on the Optora tracker.

## Papers

| Paper | Experiment | Optora component | Claims within tolerance |
| --- | --- | --- | --- |
| [Duchi and Namkoong (2019)](https://github.com/navidsharifi/optora-reproductions/blob/main/papers/duchi2019_variance_regularization/README.md)[^duchi] | Simulation of Section 5.2 | [`ChiSquareAmbiguitySet`](../api/dro/phi_dro.md) | 2 of 3 |
| [Ho, Petrik and Wiesemann (2018)](https://github.com/navidsharifi/optora-reproductions/blob/main/papers/ho2018_fast_bellman_updates/README.md)[^ho] | Robust Bellman update of Section 5 under an $L_1$ ambiguity set | [`TotalVariationAmbiguitySet`](../api/dro/phi_dro.md) | 1 of 2 |
| [Levy, Carmon, Duchi and Sidford (2020)](https://github.com/navidsharifi/optora-reproductions/blob/main/papers/levy2020_large_scale_dro/README.md)[^levy] | Digits experiment of Section 7 | [`CVaRAmbiguitySet`](../api/dro/phi_dro.md), [`ChiSquareAmbiguitySet`](../api/dro/phi_dro.md), [`PenalizedChiSquareDRO`](../api/dro/phi_dro.md) | 3 of 4 |
| [Mohajerin Esfahani and Kuhn (2018)](https://github.com/navidsharifi/optora-reproductions/blob/main/papers/mohajerin2018_wasserstein_dro/README.md)[^mohajerin] | Mean-risk portfolio of Section 7.2 | [`WassersteinAmbiguitySet`](../api/dro/wasserstein_dro.md) | 3 of 3 |
| [Sagawa, Koh, Hashimoto and Liang (2020)](https://github.com/navidsharifi/optora-reproductions/blob/main/papers/sagawa2020_group_dro/README.md)[^sagawa] | Waterbirds experiment of Section 3 | [`OnlineGroupDRO`](../api/dro/group_dro.md), [`group_losses`](../api/dro/group_dro.md) | 5 of 5 |
| [Sinha, Namkoong and Duchi (2018)](https://github.com/navidsharifi/optora-reproductions/blob/main/papers/sinha2018_certifiable_robustness/README.md)[^sinha] | Synthetic data and MNIST of Section 4 | [`transport_map`](../api/dro/continuous_wasserstein.md), prototyped in the reproductions repository first ([#122](https://github.com/navidsharifi/optora/issues/122)) | 6 of 6 |

Each link opens the full report, with the setup, every deviation from the
paper and the findings. The numbers below are taken from those reports as
of 21 November 2026. The
[results table](https://github.com/navidsharifi/optora-reproductions#results)
of the repository is generated from the result records and stays the
reference for every value. The figures below are those of the reports.
The repository draws them from the same records and this page loads them
from its `main` branch.

### Variance-based regularization

Duchi and Namkoong minimize the robustly regularized risk

$$
R_n(\theta) = \sup_{P} \left\{ \mathbb{E}_P[\ell(\theta; X)] \,:\,
D_\varphi(P \,\|\, \hat{P}_n) \le \rho / n \right\},
\qquad \varphi(t) = \tfrac{1}{2}(t - 1)^2,
$$

and compare its minimizer with the empirical risk minimizer. Optora
defines the chi-square divergence as
$D_{\chi^2}(q \,\|\, p) = \sum_i (q_i - p_i)^2 / p_i = 2 D_\varphi(q \,\|\, p)$,
so the same ball has radius $2\rho / n$.

```python
ambiguity_set = ChiSquareAmbiguitySet(nominal, radius=2.0 * rho / n)
robust_risk = ambiguity_set.worst_case_expectation(losses)
```

The robust solution has a lower population risk than the empirical risk
minimizer in all 24,000 replications, and for $n \ge 500$ it recovers the
true parameter, as the paper reports. The third claim does not reproduce.
At $n = 100$ the paper reports a robust risk of 0.18 times the empirical
one, and the exact minimizer of $R_n$ gives 0.0055. At this sample size
the ball is large enough to contain every distribution on the tied highest
losses, so $R_n$ is not differentiable at its minimizer. The gradient
method with a line search described in the paper stalls there at a ratio
of 0.52. A projected subgradient method with averaged iterates reaches the
minimizer that an independent conic solver confirms. The value in the
paper lies between the two, and without the simulation code of the authors
the report cannot say which solver setting produces it. The
[training guide](../training.md) describes where the worst case of each
set has such kinks.

<figure class="figure" markdown="span">
  ![Ratio of the mean robust risk to the mean ERM risk at n = 100 for four noise levels and two solvers, against the value of the paper](https://raw.githubusercontent.com/navidsharifi/optora-reproductions/main/papers/duchi2019_variance_regularization/figures/risk_ratio-light.svg#only-light){ loading=lazy }
  ![Ratio of the mean robust risk to the mean ERM risk at n = 100 for four noise levels and two solvers, against the value of the paper](https://raw.githubusercontent.com/navidsharifi/optora-reproductions/main/papers/duchi2019_variance_regularization/figures/risk_ratio-dark.svg#only-dark){ loading=lazy }
  <figcaption>Mean robust risk over mean ERM risk at $n = 100$. The band is the
  tolerance around the value of the paper, which lies between the
  line-search method and the exact minimizer at every noise level.</figcaption>
</figure>

### Fast Bellman updates for robust MDPs

Ho, Petrik and Wiesemann compute the robust value of one state-action
pair under an $L_1$ ambiguity set. Total variation is half the $L_1$
distance, so this value is one worst-case expectation,

$$
q(\xi) = \min_{p \in \Delta^S} \left\{ p^\top z \,:\, \lVert p - \bar{p} \rVert_1 \le \xi \right\}
= -\sup_{p} \left\{ \mathbb{E}_p[-z] \,:\, \mathrm{TV}(p, \bar{p}) \le \xi / 2 \right\}.
$$

```python
ambiguity_set = TotalVariationAmbiguitySet(nominal, radius=budget / 2.0)
q = -ambiguity_set.worst_case_expectation(-z)
```

The values agree with the linear program solved by HiGHS to
$1.9 \cdot 10^{-13}$ over every problem size, budget and run. The speed
claim does not reproduce. The paper reports a speedup of about three
orders of magnitude over a general linear programming solver. A single
Optora call at $S = 400$ takes 0.109 ms on a CPU against 7.4 ms for HiGHS,
a factor of about 68. The time of that call barely changes with $S$,
because it is spent in the fixed overhead of about fifteen small eager
PyTorch operations rather than in the sort. The paper times a C++
implementation against Gurobi, so its ratio compares a different pair of
programs. A measurement made after the run, and not part of any claim,
evaluates 400 instances at nine budgets in one batched call in 22.7 ms,
about 6 $\mu$s per value.

<figure class="figure" markdown="span">
  ![Largest error of Optora against the linear program and the time of one call for HiGHS and Optora, against the number of states](https://raw.githubusercontent.com/navidsharifi/optora-reproductions/main/papers/ho2018_fast_bellman_updates/figures/bellman_update-light.svg#only-light){ loading=lazy }
  ![Largest error of Optora against the linear program and the time of one call for HiGHS and Optora, against the number of states](https://raw.githubusercontent.com/navidsharifi/optora-reproductions/main/papers/ho2018_fast_bellman_updates/figures/bellman_update-dark.svg#only-dark){ loading=lazy }
  <figcaption>Left, the largest error against HiGHS for each number of states,
  with the tolerance of the first claim. Right, the time of a single
  call, where the HiGHS time grows with $S$ and the Optora time does
  not.</figcaption>
</figure>

### Large-scale methods for DRO

Levy, Carmon, Duchi and Sidford train a linear classifier on fixed
features of the Digits data, the 60,000 MNIST training images and 600
typed digits, under three robust objectives over the training
distribution $P_0$,

$$
\mathrm{CVaR}_{\alpha}(\ell), \qquad
\sup_{Q} \left\{ \mathbb{E}_Q[\ell] \,:\, D_{\chi^2}(Q \,\|\, P_0) \le \rho \right\}, \qquad
\sup_{Q} \left\{ \mathbb{E}_Q[\ell] - \lambda D_{\chi^2}(Q \,\|\, P_0) \right\},
$$

with $\alpha = 0.02$, $\rho = 1$ and $\lambda = 0.05$. Each minibatch
method takes stochastic gradient steps on the plug-in robust loss of a
batch of $n$ samples. The paper defines
$D_{\chi^2}(Q \,\|\, P) = \tfrac{1}{2} \mathbb{E}_P[(dQ/dP - 1)^2]$, half
of the chi-square divergence of Optora, so the ball has radius $2\rho$ and
the penalty is $\lambda / 2$.

```python
nominal = torch.full((n,), 1.0 / n)
cvar = CVaRAmbiguitySet(nominal, alpha=0.02).worst_case_expectation(losses)
ball = ChiSquareAmbiguitySet(nominal, radius=2.0).worst_case_expectation(losses)
penalized = PenalizedChiSquareDRO(nominal, penalty=0.025)(losses)
```

Three of the four claims reproduce. The best minibatch size reaches 2% of
the optimal training objective in 8.7, 9.5 and 11.6 times fewer epochs
than the full-batch method, against 9.0, 9.5 and 16.2 in the paper. Every
minibatch epoch count lies within about one standard deviation of the
paper. The count that differs is the full batch on the penalized
objective, 1215 epochs against 1680, and the report could not trace its
cause. The fourth claim, a worst-group test loss 17.5% to 27% lower than
ERM, does not reproduce under the comparison fixed before the run. After
the last epoch, DRO lowers the loss of the worst group, the typed digit 9,
by 14.9% against the better of the two ERM baselines and by about 25%
against the other one. The paper reads its range from training curves
that show both baselines and does not name the epoch. With the group
losses recorded after every epoch, the same comparison lies inside the
range from epoch 17 to epoch 47, while ERM is still fitting the typed
digits, and settles near 15% once both methods converge.

<figure class="figure" markdown="span">
  ![Reduction of the worst-group test loss against ERM read at every epoch, against the range of the paper](https://raw.githubusercontent.com/navidsharifi/optora-reproductions/main/papers/levy2020_large_scale_dro/figures/group_loss_reduction-light.svg#only-light){ loading=lazy }
  ![Reduction of the worst-group test loss against ERM read at every epoch, against the range of the paper](https://raw.githubusercontent.com/navidsharifi/optora-reproductions/main/papers/levy2020_large_scale_dro/figures/group_loss_reduction-dark.svg#only-dark){ loading=lazy }
  <figcaption>The fourth claim read at every epoch, mean over five seeds with one
  sample standard deviation. The band is the range of the paper, and
  the claim fixed before the run is the last point.</figcaption>
</figure>

### Wasserstein DRO with performance guarantees

Mohajerin Esfahani and Kuhn solve a mean-risk portfolio problem over a
type-1 Wasserstein ball around the empirical distribution of $N$ return
vectors in $\mathbb{R}^m$. `WassersteinAmbiguitySet` works on a finite
support, so the reproduction uses the $N$ samples together with $2m$
distant points that carry no nominal mass. By Corollary 5.1 of the paper
this support gives the worst case over $\mathbb{R}^m$ up to an error that
vanishes with the distance of those points. One batched ground cost covers
the 200 simulated datasets, and the radii form a second batch dimension.

```python
ambiguity_set = WassersteinAmbiguitySet(nominal, cost.unsqueeze(1), radii)
value = ambiguity_set.worst_case_expectation(portfolio_loss(points, x, tau))
```

All three claims reproduce. At a large radius the robust portfolio is the
equally weighted one in every simulation, with a largest weight error of
$3.7 \cdot 10^{-8}$. The certificate bounds the out-of-sample performance
in all 200 simulations at $\varepsilon = 1$ and $N = 300$. With $N = 30$
samples, the best radius improves the out-of-sample performance over the
nearly nonrobust one by 0.154 on average, against about 0.2 read from the
figure in the paper. Every certificate is within $1.6 \cdot 10^{-6}$ of
the optimal value of the linear program in the paper, solved by HiGHS.
The panels for $N = 3000$ are not reproduced, because a dense ground cost
for 200 such datasets needs about 14.6 GB.

<figure class="figure" markdown="span">
  ![Out-of-sample performance and reliability of the certificate against the Wasserstein radius for N = 30](https://raw.githubusercontent.com/navidsharifi/optora-reproductions/main/papers/mohajerin2018_wasserstein_dro/figures/radius_sweep-light.svg#only-light){ loading=lazy }
  ![Out-of-sample performance and reliability of the certificate against the Wasserstein radius for N = 30](https://raw.githubusercontent.com/navidsharifi/optora-reproductions/main/papers/mohajerin2018_wasserstein_dro/figures/radius_sweep-dark.svg#only-dark){ loading=lazy }
  <figcaption>Out-of-sample performance and reliability over the radius for
  $N = 30$, as in Figure 5(a) of the paper. The performance is best
  at an intermediate radius, where the certificate already holds in
  90% of the simulations.</figcaption>
</figure>

The example [Batched ground costs](ambiguity_sets/batched_ground_costs.md)
shows the same batching on a small problem.

### Group DRO under subgroup shift

Sagawa, Koh, Hashimoto and Liang train a ResNet-50 on Waterbirds, where
the background matches the label in 95% of the training images. Group DRO
minimizes the worst empirical risk over the $m = 4$ groups of label and
background,

$$
\min_{\theta} \max_{g \le m} \; \mathbb{E}_{\hat{P}_g}[\ell(\theta; (x, y))],
$$

with the exponentiated-gradient update of the group weights $q$ from
their Algorithm 1. `OnlineGroupDRO` holds $q$, and `group_losses` averages
the per-sample losses within each group.

```python
adversary = OnlineGroupDRO(torch.full((4,), 0.25), step_size=0.01)
adversary.step(group_losses(losses.detach(), groups, 4))
loss = adversary(group_losses(losses, groups, 4))
```

All five claims reproduce. With the default $\ell_2$ penalty and 300
epochs both methods fit every training group, and the worst-group test
accuracy of ERM ends at 69.3% (60.0% in the paper). With the penalty
$\lambda = 1$, group DRO reaches a worst-group test accuracy of 84.6%
against 27.6% for ERM (84.6% and 21.3%). With early stopping after one
epoch it reaches 85.0% against 17.8%, means over five seeds (86.0% and
6.7%). In both regularized regimes the average test accuracy of group DRO
stays within about one point of ERM. The runs used micro-batches of 64
under bfloat16 autocast to fit a 6 GB GPU, which raised the question of
how the online update combines with gradient accumulation. The
[training guide](../training.md) now answers it.

<figure class="figure" markdown="span">
  ![Average and worst-group test accuracy of ERM and group DRO in the three training regimes](https://raw.githubusercontent.com/navidsharifi/optora-reproductions/main/papers/sagawa2020_group_dro/figures/test_accuracy-light.svg#only-light){ loading=lazy }
  ![Average and worst-group test accuracy of ERM and group DRO in the three training regimes](https://raw.githubusercontent.com/navidsharifi/optora-reproductions/main/papers/sagawa2020_group_dro/figures/test_accuracy-dark.svg#only-dark){ loading=lazy }
  <figcaption>Test accuracy on Waterbirds in the three regimes of Table 1. The
  average accuracies of the two methods agree within two points,
  and the worst-group accuracies differ by up to 67 points.</figcaption>
</figure>

### Certifiable distributional robustness

Sinha, Namkoong and Duchi train with the Lagrangian relaxation of the
worst case over a Wasserstein ball with transport cost
$c(z, z_0) = \lVert x - x_0 \rVert_2^2$ on the inputs,

$$
\min_{\theta} \; \mathbb{E}_{\hat{P}_n}\big[\phi_\gamma(\theta; Z)\big], \qquad
\phi_\gamma(\theta; z_0) = \sup_{z} \big\{ \ell(\theta; z) - \gamma\, c(z, z_0) \big\},
$$

and certify the trained model with the bound
$\gamma \rho + \mathbb{E}_{\hat{P}_n}[\phi_\gamma(\theta; Z)]$ on the
worst-case loss over the ball of radius $\rho$ (their Theorem 3). For a
smooth loss and a large enough $\gamma$ the inner problem is strongly
concave, so gradient ascent solves it. Optora `v0.4.0` had no continuous
adversary of this kind, so the reproduction ran the inner ascent from a
prototype in its own repository.

All six claims reproduce. On MNIST the robustness level reached on the
training set is $\hat\rho_n = 0.192$ (0.209 in the paper), where the
certificate of 0.695 bounds the test worst-case loss of 0.321 (0.707 and
0.329). Under an $\ell_2$ PGM attack at the training radius WRM errs on
2.2% of the test set and ERM on 11.3% (2.3% and 10.8%). Under an
$\ell_\infty$ attack with $\epsilon = 0.1$ WRM errs on 17.2% against 26.6%
for the best of the three adversarially trained baselines (14.7% and
25.2%). The interface the prototype needed, a batched ascent for any
differentiable loss with a step size of order $1/\sqrt{t}$ and a penalty
that may differ per example, is now part of Optora as
[`transport_map`](../api/dro/continuous_wasserstein.md)
([#122](https://github.com/navidsharifi/optora/issues/122)). It will ship with the next release, and the reproduction
will switch to it from the prototype then.

<figure class="figure" markdown="span">
  ![Certificate of the WRM model and its test worst-case loss against the robustness level, on synthetic data and on MNIST](https://raw.githubusercontent.com/navidsharifi/optora-reproductions/main/papers/sinha2018_certifiable_robustness/figures/certificate-light.svg#only-light){ loading=lazy }
  ![Certificate of the WRM model and its test worst-case loss against the robustness level, on synthetic data and on MNIST](https://raw.githubusercontent.com/navidsharifi/optora-reproductions/main/papers/sinha2018_certifiable_robustness/figures/certificate-dark.svg#only-dark){ loading=lazy }
  <figcaption>Certificate and test worst-case loss of the WRM model for each of
  the five seeds, as in Figure 2 of the paper. The marker is the
  level $\hat\rho_n$ reached in training.</figcaption>
</figure>

## What the reproductions changed in Optora

Eight issues came out of these runs, and all eight are closed, by the
`v0.3.0` and `v0.4.0` releases, by changes to the documentation, or on
`main` for the next release.

| Issue | Found in | Change |
| --- | --- | --- |
| [#108](https://github.com/navidsharifi/optora/issues/108) | Duchi and Namkoong | `ChiSquareAmbiguitySet` finds its tilt exactly from one sort instead of a bisection, which removes most of the kernel launches of a solve on a GPU (`v0.3.0`). |
| [#109](https://github.com/navidsharifi/optora/issues/109) | Duchi and Namkoong | The scalar search behind the KL, Sinkhorn and general $\varphi$ sets tests more tilts per step on a GPU, where small problems are bound by kernel launches (`v0.3.0`). |
| [#112](https://github.com/navidsharifi/optora/issues/112) | Mohajerin Esfahani and Kuhn | `WassersteinAmbiguitySet` accepts a batch of ground costs, so datasets with different supports are solved in one call (`v0.3.0`). |
| [#114](https://github.com/navidsharifi/optora/issues/114) | Duchi and Namkoong | The [training guide](../training.md) notes that a descent method with a line search stalls on a saturated chi-square ball and that a nonsmooth outer solver is needed. |
| [#116](https://github.com/navidsharifi/optora/issues/116) | Levy et al. | `PenalizedChiSquareDRO` adds the penalized chi-square objective (`v0.4.0`). |
| [#117](https://github.com/navidsharifi/optora/issues/117) | Levy et al. | Capturing a minibatch step in a CUDA graph is tested, and the chi-square set is built without a copy to the host (`v0.4.0`). |
| [#120](https://github.com/navidsharifi/optora/issues/120) | Sagawa et al. | The [training guide](../training.md) shows how to combine `OnlineGroupDRO` with gradient accumulation over micro-batches. |
| [#122](https://github.com/navidsharifi/optora/issues/122) | Sinha et al. | `transport_map` and `squared_distance` add the continuous Wasserstein adversary that the reproduction prototyped (on `main`, not yet released). |

The searches of #108 and #109 are described under
[Mathematical formulations](../formulations.md#tilt-solved-sets-and-the-bracketed-scalar-search).
A component that a reproduction needs and Optora does not ship is
prototyped in the reproductions repository first, and it is proposed here
once a reproduction has shown which interface it needs.

## Running a reproduction

The repository needs Python 3.11 or newer.

```bash
git clone https://github.com/navidsharifi/optora-reproductions
cd optora-reproductions
pip install -e ".[dev]"
python -m papers.ho2018_fast_bellman_updates.run --config papers/ho2018_fast_bellman_updates/configs/smoke.toml
```

Every paper has a `smoke.toml` that runs in seconds on a CPU and a
`full.toml` with the settings of the paper. The full runs recorded above
took about three minutes on a CPU for Ho, Petrik and Wiesemann. The other
five ran on a laptop GPU with 6 GB of memory and took between 2.2 hours
for Mohajerin Esfahani and Kuhn and 12 hours for Sagawa et al.

[^duchi]: John Duchi and Hongseok Namkoong. Variance-based regularization
    with convex objectives. *Journal of Machine Learning Research*,
    20(68):1-55, 2019.
[^ho]: Chin Pang Ho, Marek Petrik and Wolfram Wiesemann. Fast Bellman
    updates for robust MDPs. *Proceedings of the 35th International
    Conference on Machine Learning*, PMLR 80:1979-1988, 2018.
[^levy]: Daniel Levy, Yair Carmon, John C. Duchi and Aaron Sidford.
    Large-scale methods for distributionally robust optimization.
    *Advances in Neural Information Processing Systems 33*, pages
    8847-8860, 2020.
[^mohajerin]: Peyman Mohajerin Esfahani and Daniel Kuhn. Data-driven
    distributionally robust optimization using the Wasserstein metric:
    performance guarantees and tractable reformulations. *Mathematical
    Programming*, 171(1-2):115-166, 2018.
[^sagawa]: Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto and Percy
    Liang. Distributionally robust neural networks. *International
    Conference on Learning Representations*, 2020.
[^sinha]: Aman Sinha, Hongseok Namkoong and John Duchi. Certifying some
    distributional robustness with principled adversarial training.
    *International Conference on Learning Representations*, 2018.
