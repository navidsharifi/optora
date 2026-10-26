# Minibatch DRO against the full batch

Every other page hands the ambiguity set the whole dataset. A training loop
at scale cannot: each step sees a minibatch, and the obvious move, running
the ambiguity set on that minibatch alone, quietly changes the objective.
This page measures by how much, and checks the two remedies in
[`optora.dro.minibatch`](../api/dro/minibatch.md) against the full-batch
objective on the same problem.

## Why the plug-in is biased

Write $\hat P_m$ for the uniform distribution over a minibatch of $m$
samples. The *plug-in* estimate is

$$
\hat V_m = \sup_{q:\, D_{\mathrm{KL}}(q \,\|\, \hat P_m) \,\le\, \rho}
    \mathbb{E}_q[\ell].
$$

The supremum is a nonlinear functional of the sample, so
$\mathbb{E}[\hat V_m]$ is not the full-batch worst case, and
$\nabla \hat V_m$ is not an unbiased stochastic gradient of it[^levy]. For
KL the plug-in is a log-sum-exp of a subsample, and it is biased **low**:
the adversary can only upweight the high-loss points the minibatch happens
to contain, and a small minibatch rarely contains the rare ones.

## The problem

The two-group regression of
[example 07](subpopulation_shift_training.md): $N = 4096$ points, 90% with
slope $(1, 0)$ and 10% with slope $(1, 1.5)$, noise $0.3$, a linear model
with a bias, squared loss, and a KL ball.

Minibatches are drawn **with replacement** from the dataset, so they are
i.i.d. draws from its empirical distribution $P_N$, and the full-batch
objective

$$
F(\theta) = \sup_{q:\, D_{\mathrm{KL}}(q \,\|\, P_N) \,\le\, \rho}
    \mathbb{E}_q[\ell(\theta)]
$$

is *exactly* the population objective every estimator targets. There is no
asymptotic argument anywhere in the comparison: `KLAmbiguitySet` computes
$F$ exactly on all $N$ points, and every number below is measured against
it.

## Part 1: bias of one estimate

The model is fixed at its least-squares fit. For each radius, the script
draws 4000 independent minibatches per estimator and reports the bias as a
fraction of the robust excess $F - \bar\ell$, the part of the objective the
ambiguity set adds:

| Radius | Plug-in, $m = 16$ | $m = 64$ | $m = 256$ | Multilevel | Stochastic dual |
| --- | --- | --- | --- | --- | --- |
| $4.7 \times 10^{-4}$ | $-0.381$ | $-0.188$ | $-0.075$ | $-0.074$ | $+0.010$ |
| $0.01$ | $-0.441$ | $-0.240$ | $-0.106$ | $-0.103$ | $+0.009$ |
| $0.05$ | $-0.509$ | $-0.302$ | $-0.147$ | $-0.140$ | $+0.008$ |
| $0.2$ | $-0.584$ | $-0.372$ | $-0.197$ | $-0.186$ | $+0.007$ |
| $1$ | $-0.679$ | $-0.464$ | $-0.265$ | $-0.250$ | $+0.006$ |

Standard errors are $0.005$ to $0.015$; the script prints them and asserts
every claim below within four of them.

- **The plug-in misses a large share of the robustness.** At $m = 64$ it
  reports 54% to 81% of the robust excess, and the shortfall grows with the
  radius, because a larger ball leans harder on the tail a minibatch
  undersamples.
- **The calibrated radius is not exempt.** The first row is
  `kl_radius(N, 0.95)`, the radius under which the ambiguity set gives a
  95% confidence interval for the risk (see [Calibrating the ambiguity
  radius](calibrated_ambiguity_radius.md)). A 16-sample plug-in reports
  only 62% of that correction.
- **`MultilevelMonteCarloEstimator` has the bias of the 256-sample plug-in
  at an expected 68 samples per estimate**, with base batch 16 and 4
  levels[^bg][^levy]. The price is variance: its standard error is about
  three times the 256-sample plug-in's.
- **`StochasticPhiDual` at the full batch's optimal dual variables is
  unbiased** on 64 samples: for fixed $(\eta, \lambda)$ the dual is an
  expectation, and its minibatch mean is an unbiased estimate of it.

!!! note "A control variate, not a correction"

    Each estimate is reported minus the mean loss of its own minibatch
    (for the multilevel estimator, of its base level). That difference has
    the same bias as the estimate, since the subtracted term's expectation
    is the full-batch mean loss exactly, but most of the sampling noise of
    the mean cancels. It is what makes standard errors near $0.01$ possible
    with 4000 draws.

## Part 2: training on minibatches

Each estimator now drives `torch.optim.Adam` on the model, over 8 seeds,
from a warm start fitted by least squares on a 64-sample pilot batch, at
radius $\rho = 0.2$. Progress is measured on the full-batch objective: the
suboptimality $F(\bar\theta) - \min F$ of the average $\bar\theta$ of the
second half of the iterates, against a full-batch L-BFGS reference.

| Estimator | Steps | Samples drawn | Mean suboptimality | Worst seed |
| --- | --- | --- | --- | --- |
| Plug-in, $m = 64$ | 800 | 51,200 | $0.0616$ | $0.0650$ |
| Multilevel | 500 | 36,448 | $0.0213$ | $0.0375$ |
| Stochastic dual, $m = 64$ | 3000 | 192,000 | $0.0028$ | $0.0191$ |

For scale, the warm start is $0.474$ above the optimum and the full-batch
least-squares fit $0.415$.

=== "Plug-in"

    It descends fastest and then stops at a **floor**. The worst of the
    eight seeds is only 6% above their mean of $0.06$: independent
    minibatch noise would spread them out, so what holds them there is
    bias. The plug-in
    converges, just to the minimizer of $\mathbb{E}[\hat V_{64}]$, which
    is a less robust model than the minimizer of $F$.

=== "Multilevel"

    It goes below the plug-in floor on fewer samples. It is not free: each
    step runs three exact solves (the base level, the full batch, and its
    two halves as one batch), so in this script a multilevel step costs
    about four plug-in steps of wall-clock time.

=== "Stochastic dual"

    It has no inner solve and no bias, so each step is the cheapest of the
    three, and given enough of them it goes furthest. Its weakness is
    variance: KL's conjugate is exponential, so a minibatch containing one
    high-loss point produces a gradient orders of magnitude above the
    typical one. The worst seed is seven times the mean.

!!! warning "Start the dual variables on the scale of the loss"

    With $\log \eta = 0$ and $\lambda = 0$ the first minibatches are far
    from the dual optimum, the exponential conjugate produces enormous
    gradients, and Adam's second-moment estimate never recovers: training
    stalls. The script instead starts $(\log \eta, \lambda)$ at the exact
    dual optimum of the pilot batch, which uses only those 64 samples. Any
    start of the right scale works; the default does not.

## Which one to use

| Situation | Use |
| --- | --- |
| Fixed batch size, many cheap steps, chi-square or KL ball | `StochasticPhiDual`, with its dual variables started on the loss scale |
| A set without a finite conjugate (total variation), or the exact solve wanted on every step | `MultilevelMonteCarloEstimator` |
| Small radius and large batches, where the measured bias is acceptable | The plug-in, knowing it is less robust than it reports |

## Reading the figure

- **Left**: bias as a fraction of the robust excess against the radius,
  with two-standard-error bars. The plug-in curves fan out below zero and
  sink with the radius; the multilevel curve sits on the 256-sample one;
  the stochastic dual sits on zero. The dotted line marks the calibrated
  radius.
- **Right**: suboptimality of the averaged iterate on the full-batch
  objective against samples drawn, on log axes, with the range over seeds
  shaded. The plug-in is noisy early and then settles on its floor; the
  other two keep going down past it.

!!! note "Runtime"

    This is the slowest example, around 20 to 30 seconds on a CPU. Almost
    all of it is the multilevel and plug-in training loops: every KL
    bisection is a fixed sequence of small kernels, and on 64-point
    minibatches the cost is launch overhead rather than arithmetic.

## Source

```py title="examples/08_minibatch_dro_training.py"
--8<-- "examples/08_minibatch_dro_training.py"
```

[^levy]:
    Daniel Levy, Yair Carmon, John C. Duchi and Aaron Sidford, "Large-Scale
    Methods for Distributionally Robust Optimization", *NeurIPS* (2020).
    [arXiv:2010.05893](https://arxiv.org/abs/2010.05893).

[^bg]:
    Jose H. Blanchet and Peter W. Glynn, "Unbiased Monte Carlo for
    Optimization and Functions of Expectations via Multi-Level
    Randomization", *Winter Simulation Conference* (2015).
