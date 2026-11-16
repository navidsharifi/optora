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
when the mean over seeds lies within that tolerance. Each reproduction
also solves its problem with a reference solver that shares no code with
Optora, either HiGHS or Clarabel. When a result disagrees with the paper,
the report traces the cause, and a cause that lies in the library becomes
an issue on the Optora tracker.

## Papers

| Paper | Experiment | Optora component | Claims within tolerance |
| --- | --- | --- | --- |
| [Duchi and Namkoong (2019)](https://github.com/navidsharifi/optora-reproductions/blob/main/papers/duchi2019_variance_regularization/README.md)[^duchi] | Simulation of Section 5.2 | [`ChiSquareAmbiguitySet`](../api/dro/phi_dro.md) | 2 of 3 |
| [Ho, Petrik and Wiesemann (2018)](https://github.com/navidsharifi/optora-reproductions/blob/main/papers/ho2018_fast_bellman_updates/README.md)[^ho] | Robust Bellman update of Section 5 under an $L_1$ ambiguity set | [`TotalVariationAmbiguitySet`](../api/dro/phi_dro.md) | 1 of 2 |
| [Mohajerin Esfahani and Kuhn (2018)](https://github.com/navidsharifi/optora-reproductions/blob/main/papers/mohajerin2018_wasserstein_dro/README.md)[^mohajerin] | Mean-risk portfolio of Section 7.2 | [`WassersteinAmbiguitySet`](../api/dro/wasserstein_dro.md) | 3 of 3 |

Each link opens the full report, with the setup, every deviation from the
paper and the findings. The numbers below are taken from those reports as
of 16 November 2026. The
[results table](https://github.com/navidsharifi/optora-reproductions#results)
of the repository is generated from the result records and stays the
reference for every value.

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

The example [Batched ground costs](ambiguity_sets/batched_ground_costs.md)
shows the same batching on a small problem.

## What the reproductions changed in Optora

Three issues came out of these runs, and all three are fixed in `v0.3.0`.

| Issue | Found in | Change |
| --- | --- | --- |
| [#108](https://github.com/navidsharifi/optora/issues/108) | Duchi and Namkoong | `ChiSquareAmbiguitySet` finds its tilt exactly from one sort instead of a bisection, which removes most of the kernel launches of a solve on a GPU. |
| [#109](https://github.com/navidsharifi/optora/issues/109) | Duchi and Namkoong | The scalar search behind the KL, Sinkhorn and general $\varphi$ sets tests more tilts per step on a GPU, where small problems are bound by kernel launches. |
| [#112](https://github.com/navidsharifi/optora/issues/112) | Mohajerin Esfahani and Kuhn | `WassersteinAmbiguitySet` accepts a batch of ground costs, so datasets with different supports are solved in one call. |

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
took about three minutes on a CPU for Ho, Petrik and Wiesemann, and 3.8
hours and 2.2 hours on a laptop GPU for the other two.

[^duchi]: John Duchi and Hongseok Namkoong. Variance-based regularization
    with convex objectives. *Journal of Machine Learning Research*,
    20(68):1-55, 2019.
[^ho]: Chin Pang Ho, Marek Petrik and Wolfram Wiesemann. Fast Bellman
    updates for robust MDPs. *Proceedings of the 35th International
    Conference on Machine Learning*, PMLR 80:1979-1988, 2018.
[^mohajerin]: Peyman Mohajerin Esfahani and Daniel Kuhn. Data-driven
    distributionally robust optimization using the Wasserstein metric:
    performance guarantees and tractable reformulations. *Mathematical
    Programming*, 171(1-2):115-166, 2018.
