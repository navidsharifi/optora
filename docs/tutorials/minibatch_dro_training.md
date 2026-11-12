# Minibatch DRO against the full batch

Every other tutorial gives the ambiguity set the whole dataset. A training
loop at scale cannot do that, because each step sees only a minibatch. The
obvious approach is to run the ambiguity set on that minibatch alone, and
this changes the objective that is being minimized. This page measures how
large the change is and checks the two remedies in
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
$\nabla \hat V_m$ is not an unbiased stochastic gradient of it[^levy]. The
direction of the bias follows from the dual form of the worst case, the
one `StochasticPhiDual` uses below, which is an infimum of functionals
that are linear in the nominal distribution.
The worst case is therefore concave in the nominal, and since
$\mathbb{E}[\hat P_m]$ is the full-batch distribution, Jensen's inequality
makes the plug-in biased **low**. For KL the plug-in is a log-sum-exp of a
subsample. The adversary can only upweight the high-loss points that the
minibatch happens to contain, and a small minibatch rarely contains the
rare ones.

## The problem

The data come from the two-group regression of
[tutorial 07](subpopulation_shift_training.md). There are $N = 4096$
points, 90% of them with slope $(1, 0)$ and 10% with slope $(1, 1.5)$, and
the noise level is $0.3$. The model is linear with a bias, the loss is the
squared loss, and the ambiguity set is a KL ball.

Minibatches are drawn **with replacement** from the dataset, so they are
i.i.d. draws from its empirical distribution $P_N$, and the full-batch
objective

$$
F(\theta) = \sup_{q:\, D_{\mathrm{KL}}(q \,\|\, P_N) \,\le\, \rho}
    \mathbb{E}_q[\ell(\theta)]
$$

is *exactly* the population objective that every estimator targets. No
asymptotic argument enters the comparison. `KLAmbiguitySet` computes $F$
exactly on all $N$ points, and every number below is measured against it.

## Part 1, the bias of one estimate

The model is fixed at its least-squares fit. For each estimator the script
draws 4000 independent minibatches, evaluates the estimate at every radius,
and reports the bias as a fraction of the robust excess $F - \bar\ell$,
which is the amount the ambiguity set adds to the mean loss $\bar\ell$.

| Radius | Plug-in, $m = 16$ | $m = 64$ | $m = 256$ | Multilevel | Stochastic dual |
| --- | --- | --- | --- | --- | --- |
| $4.7 \times 10^{-4}$ | $-0.381$ | $-0.188$ | $-0.075$ | $-0.074$ | $+0.010$ |
| $0.01$ | $-0.441$ | $-0.240$ | $-0.106$ | $-0.103$ | $+0.009$ |
| $0.05$ | $-0.509$ | $-0.302$ | $-0.147$ | $-0.140$ | $+0.008$ |
| $0.2$ | $-0.584$ | $-0.372$ | $-0.197$ | $-0.186$ | $+0.007$ |
| $1$ | $-0.679$ | $-0.464$ | $-0.265$ | $-0.250$ | $+0.006$ |

The standard errors lie between $0.005$ and $0.015$, and the script prints
them next to each bias. It asserts that the plug-in bias is negative at
$m = 16$ and increases with $m$, that the multilevel bias matches the
256-sample plug-in within four combined standard errors, and that the bias
of the stochastic dual is within four standard errors of zero.

- **The plug-in recovers only part of the robust excess.** At $m = 64$ it
  reports 54% to 81% of the robust excess, and the shortfall grows with the
  radius because a larger ball puts more weight on the tail that a
  minibatch undersamples.
- **The calibrated radius has the same bias.** The first row is
  `kl_radius(N, 0.95)`, the radius under which the ambiguity set gives a
  95% confidence interval for the risk (see [Calibrating the ambiguity
  radius](calibrated_ambiguity_radius.md)). A 16-sample plug-in reports
  only 62% of that correction.
- **`MultilevelMonteCarloEstimator` has the bias of the 256-sample plug-in
  at an expected 68 samples per estimate**, with base batch 16 and 4
  levels[^bg][^levy]. Its variance is higher, and its standard error is
  about three times that of the 256-sample plug-in.
- **`StochasticPhiDual` is unbiased on 64 samples at the optimal dual
  variables of the full batch.** For fixed $(\eta, \lambda)$ the dual
  objective is an expectation, so its minibatch mean is an unbiased
  estimate of it, and at the optimal dual variables that expectation is
  $F$.

!!! note "A control variate, not a correction"

    Each estimate is reported minus the mean loss of its own minibatch,
    and for the multilevel estimator minus the mean loss of its base
    level. The subtracted term has the full-batch mean loss as its
    expectation, so the difference has the same bias as the estimate,
    while most of the sampling noise of the mean cancels. This keeps the
    standard errors near $0.01$ with 4000 draws.

## Part 2, training on minibatches

Each estimator now drives `torch.optim.Adam` on the model at radius
$\rho = 0.2$, over 8 seeds, from a warm start fitted by least squares on a
64-sample pilot batch. Progress is measured on the full-batch objective as
the suboptimality $F(\bar\theta) - \min F$, where $\bar\theta$ is the
average of the second half of the iterates and $\min F$ comes from a
full-batch L-BFGS reference.

| Estimator | Steps | Samples drawn | Mean suboptimality | Worst seed |
| --- | --- | --- | --- | --- |
| Plug-in, $m = 64$ | 800 | 51,200 | $0.0616$ | $0.0650$ |
| Multilevel | 500 | 36,448 | $0.0213$ | $0.0375$ |
| Stochastic dual, $m = 64$ | 3000 | 192,000 | $0.0028$ | $0.0191$ |

For scale, the warm start is $0.474$ above the optimum and the full-batch
least-squares fit is $0.415$ above it.

=== "Plug-in"

    The plug-in descends quickly and then settles at a **floor**. The
    worst of the eight seeds is only 6% above their mean of $0.06$.
    Independent minibatch noise would spread the seeds apart, so the floor
    comes from the bias. The plug-in gradient is an unbiased stochastic
    gradient of $\mathbb{E}[\hat V_{64}]$, so training approaches the
    minimizer of that function, and that minimizer has a higher full-batch
    objective than the minimizer of $F$.

=== "Multilevel"

    The multilevel estimator goes below the plug-in floor while drawing
    fewer samples. Each of its steps runs three exact solves, namely the
    base level, the full batch, and its two halves as one batched solve,
    and the batches are larger on average. On a CPU a multilevel step
    takes more than twice as long as a plug-in step.

=== "Stochastic dual"

    The stochastic dual has no inner solve and no bias, so each of its
    steps is the cheapest of the three, and with enough steps it reaches
    the lowest suboptimality. Its weakness is variance. KL's conjugate is
    exponential, so a minibatch that contains one high-loss point can
    produce a gradient far larger than the typical one. The worst seed is
    seven times the mean.

!!! warning "Start the dual variables on the scale of the loss"

    With the defaults $\log \eta = 0$ and $\lambda = 0$, the first
    minibatches are far from the dual optimum, and the exponential
    conjugate produces very large gradients. The `StochasticPhiDual`
    docstring warns that this start can stall training under KL, and in
    this script it makes training considerably slower. The script instead
    starts $(\log \eta, \lambda)$ at the exact dual optimum of the pilot
    batch, which uses only those 64 samples.

## Which one to use

| Situation | Use |
| --- | --- |
| A fixed batch size, many cheap steps, and a chi-square or KL ball | `StochasticPhiDual`, with its dual variables started on the loss scale |
| A set without a finite conjugate (total variation), or an exact solve on every step | `MultilevelMonteCarloEstimator` |
| A small radius and large batches, where the measured bias is acceptable | The plug-in, which underestimates the full-batch worst case |

## Reading the figure

- The **left** panel shows the bias as a fraction of the robust excess
  against the radius, with two-standard-error bars. The plug-in curves
  lie below zero and move further down as the radius grows. The
  multilevel curve lies on the 256-sample plug-in curve, and the
  stochastic dual lies on zero. The dotted vertical line marks the
  calibrated radius.
- The **right** panel shows the suboptimality of the averaged iterate on
  the full-batch objective against the number of samples drawn, on log
  axes, with the range over seeds shaded. The dotted horizontal line marks
  the plug-in floor. The plug-in curve settles on that floor after an
  early transient, and the other two continue below it.

!!! note "Runtime"

    This tutorial takes around 20 to 30 seconds on a CPU. Every KL worst
    case is computed by a bisection with a fixed number of steps, so on
    64-point minibatches the cost of a plug-in or multilevel step is the
    launch overhead of many small kernels rather than arithmetic.

## The bias as the batch grows

The table of Part 1 measures the bias at three batch sizes. The animation
sweeps the batch size from 8 to 2048 at the training radius $0.2$ and
draws two thousand minibatches at every size. The left panel shows the
distribution of the plug-in estimate of the robust excess $F - \bar\ell$
over those minibatches, together with the full-batch value. The middle
panel shows the relative bias with its two-standard-error band, and it
marks the multilevel estimator and the stochastic dual at their own batch
sizes. The right panel shows the share of minibatches that contain at
least one of the top 1% of losses. The bias shrinks as this share grows,
because the adversary cannot upweight a point that the minibatch does not
contain.

<figure class="figure" markdown="span">
  ![The spread of the minibatch plug-in estimate, its relative bias and the share of minibatches holding a rare high loss, as the batch size grows](../assets/figures/plug-in-bias-light.gif#only-light){ loading=lazy }
  ![The spread of the minibatch plug-in estimate, its relative bias and the share of minibatches holding a rare high loss, as the batch size grows](../assets/figures/plug-in-bias-dark.gif#only-dark){ loading=lazy }
  <figcaption>The left panel shows the plug-in estimates of the robust
  excess over two thousand minibatches of the current size, with the
  full-batch value and the mean estimate marked. The middle panel shows
  the relative bias against the batch size, with the multilevel estimator
  and the stochastic dual at their own sizes. The right panel shows the
  share of minibatches that contain a top-1% loss, both measured and in
  closed form.</figcaption>
</figure>

## The figure

The figure below is rendered from this script when the site is built, so
it always shows the output of the current code.

<figure class="figure" markdown="span">
  ![Bias of the minibatch plug-in estimate and training curves of the three estimators](../assets/figures/08_minibatch_dro_training-light.png#only-light){ loading=lazy }
  ![Bias of the minibatch plug-in estimate and training curves of the three estimators](../assets/figures/08_minibatch_dro_training-dark.png#only-dark){ loading=lazy }
  <figcaption>The left panel shows the bias of each minibatch estimate as a fraction of the robust excess, against the radius. The right panel shows the suboptimality of the full-batch objective at the averaged iterate against the number of samples drawn, for the plug-in, the multilevel estimator and the stochastic dual.</figcaption>
</figure>

## Source

```py title="tutorials/08_minibatch_dro_training.py"
--8<-- "tutorials/08_minibatch_dro_training.py"
```

[^levy]:
    Daniel Levy, Yair Carmon, John C. Duchi and Aaron Sidford, "Large-Scale
    Methods for Distributionally Robust Optimization", *NeurIPS* (2020).
    [arXiv:2010.05893](https://arxiv.org/abs/2010.05893).

[^bg]:
    Jose H. Blanchet and Peter W. Glynn, "Unbiased Monte Carlo for
    Optimization and Functions of Expectations via Multi-Level
    Randomization", *Winter Simulation Conference* (2015).
