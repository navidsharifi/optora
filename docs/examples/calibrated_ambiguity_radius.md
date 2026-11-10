# Calibrating the ambiguity radius

The other pages in this section sweep the radius, compare several radii, or
pick one that makes the plot easy to read. None of them answers the
question a reader has the first time they use DRO, namely how to choose
the radius.

A radius chosen by intuition can fail in both directions. If it is too
small, the robust value collapses onto the empirical mean, and the nested
worst-case solve changes nothing. If it is too large, the robust value
saturates at $\max_i \ell_i$, which describes the worst observation rather
than the problem. Neither failure is visible from the robust value alone.

Finite-sample theory gives a principled answer. If the nominal
distribution is an empirical measure over $n$ observations, the radius can
be set as a *quantile* of a limit distribution, and the ambiguity set it
defines is then a confidence region.

## The calibration

Fix a functional of interest, here the mean $\mu = \mathbb{E}[Z]$, and
define the profile divergence of a candidate value $\mu$ against the
empirical distribution $\hat{P}_n$ as

$$
T_n(\mu) \;=\; \min_{q}
\Big\{ D_{\chi^2}\big(q \,\|\, \hat{P}_n\big)
\;:\; \mathbb{E}_q[Z] = \mu \Big\} .
$$

This is the empirical-likelihood statistic with the $\chi^2$-divergence in
place of the log-likelihood, and it obeys the same limit theorem. Evaluated
at the *true* mean $\mu_0$, it satisfies

$$
n \, T_n(\mu_0) \;\xrightarrow{\ d\ }\; \chi^2_1 .
$$

Duchi and Namkoong[^dn] base their statistical theory of $\phi$-divergence
DRO on this expansion. The robust objective is an asymptotically exact
upper confidence bound on the true risk, and its width is the variance term
that a $\chi^2$ ball generates. Read in reverse, the theorem tells us how
to set the radius. Choosing

$$
\rho_n(\alpha) \;=\; \frac{\chi^2_{1,\,1-\alpha}}{n}
$$

makes $\{q : D_{\chi^2}(q \| \hat{P}_n) \le \rho_n(\alpha)\}$ an
asymptotically valid $(1-\alpha)$ confidence region, so the interval it
induces on the mean,

$$
\Big[\; -\!\!\sup_{q \in \mathcal{U}} \mathbb{E}_q[-Z]
\;,\;\; \sup_{q \in \mathcal{U}} \mathbb{E}_q[Z] \;\Big],
$$

is an asymptotically valid $(1-\alpha)$ confidence interval. The radius is
then determined by the confidence level instead of being a free parameter.

The $\chi^2$ geometry is the simplest case, because the divergence itself
is the test statistic. The same expansion calibrates every
$\varphi$-divergence whose generator is smooth at one, with an extra factor
$\varphi''(1)/2$ in the radius, which is $1$ for $\chi^2$ and $\tfrac{1}{2}$
for KL. The Wasserstein geometry admits the same approach. With the
quadratic cost $\lVert x - y \rVert_2^2$ on the input space, Blanchet, Kang
and Murthy[^bkm] show that the robust Wasserstein profile of the true risk
again converges to $\chi^2_1$, up to the scale
$\mathrm{Var}(\ell) / \mathbb{E}\lVert \nabla_x \ell \rVert_2^2$, so the
radius again shrinks like $1/n$. Classical finite-sample guarantees for
Wasserstein balls instead tie the radius to the concentration rate of the
empirical measure, which degrades with the dimension[^gao].

## The helpers

`optora.dro` provides the calibration as plain functions, one per
geometry.

| Function | Radius | Calibrates |
| --- | --- | --- |
| `chi_square_radius(n, confidence)` | $\chi^2_{1,1-\alpha} / n$ | `ChiSquareAmbiguitySet` |
| `kl_radius(n, confidence)` | $\chi^2_{1,1-\alpha} / (2n)$ | `KLAmbiguitySet` |
| `phi_divergence_radius(n, confidence, curvature)` | $\varphi''(1)\, \chi^2_{1,1-\alpha} / (2n)$ | A `PhiAmbiguitySet` whose generator is smooth at one |
| `wasserstein_radius(loss, input_gradient, confidence)` | $\chi^2_{1,1-\alpha}\, \widehat{\mathrm{Var}}(\ell) / (n\, \overline{\lVert \nabla_x \ell \rVert_2^2})$ | Quadratic-cost transport over the input space |

`confidence` is the coverage of the two-sided interval. Pass
`one_sided=True` when only the upper bound
$\sup_{q} \mathbb{E}_q[\ell] \ge \mathbb{E}_{P_0}[\ell]$ matters. The
helpers then use $z_{1-\alpha}^2$ in place of $z_{1-\alpha/2}^2$.

!!! tip "Covering a $d$-dimensional parameter"

    The three $\varphi$-divergence helpers accept `degrees_of_freedom=d`,
    which replaces $\chi^2_{1,1-\alpha}$ by $\chi^2_{d,1-\alpha}$. The ball
    is then a simultaneous confidence region for a $d$-dimensional
    parameter, such as every component of a vector mean, or the minimizer
    of a risk over $d$ parameters[^dn]. A $d = 1$ radius used for a vector
    parameter under-covers. For $d = 3$ at a nominal 95%, the region covers
    the true mean only about 72% of the time, because
    $\Pr(\chi^2_3 \le \chi^2_{1,0.95}) \approx 0.72$. `one_sided` is
    defined for $d = 1$ only.

!!! warning "Wasserstein on a finite support"

    `wasserstein_radius` calibrates transport anywhere in the input space.
    A `WassersteinAmbiguitySet` whose support is the observations alone can
    only move mass among them, so under the same radius its worst case is
    no larger, and its coverage no higher, than the calibrated one.

!!! warning "Mind the factor of two"

    Optora's [`ChiSquareDivergence`](../api/divergences/f_divergence.md) is
    $D_{\chi^2}(q \| p) = \sum_i (q_i - p_i)^2 / p_i$, generated by
    $\phi(t) = (t-1)^2$. Duchi and Namkoong use
    $\phi(t) = \tfrac{1}{2}(t-1)^2$, so their radius is *half* of the one
    written above. A radius copied from a paper with the other
    normalization changes the half-width of every reported interval by a
    factor of $\sqrt{2}$.

## A sanity check against the normal interval

The calibration should reproduce a result that is already well
established, and it does. In the interior regime, where no candidate $q_i$
is pushed to zero, the $\chi^2$ worst-case expectation has the closed form
$\bar{Z} + \sqrt{\rho \cdot \mathrm{Var}(Z)}$. Substituting
$\rho_n(\alpha)$ and using $\chi^2_{1,1-\alpha} = z_{1-\alpha/2}^2$ gives

$$
\sup_{q \in \mathcal{U}} \mathbb{E}_q[Z]
\;=\; \bar{Z} + \sqrt{\frac{z_{1-\alpha/2}^2}{n} \cdot s_n^2}
\;=\; \bar{Z} + z_{1-\alpha/2} \, \frac{s_n}{\sqrt{n}} ,
$$

which is exactly the textbook normal interval, with $s_n^2$ the plug-in
variance. The script checks this identity against the solver rather than
against the algebra, and so does
`tests/dro/test_phi_dro.py::test_chi_square_calibrated_radius_gives_the_normal_interval`.

!!! tip "No `scipy` required"

    A $\chi^2$ variable with one degree of freedom is the square of a
    standard normal, so
    $\chi^2_{1,\,1-\alpha} = \big(\Phi^{-1}(1 - \alpha/2)\big)^2$, and the
    helpers read $\Phi^{-1}$ from the standard library's
    `statistics.NormalDist`. Optora's only runtime dependency stays
    `torch`, and computing a radius never touches a device.

## Measuring the coverage

An asymptotic guarantee holds as $n \to \infty$, so the relevant question
is how far it is from exact at the sample sizes we have. The script runs
the experiment the theorem describes. It draws 200 independent samples
from a standard exponential, builds the calibrated interval for each, and
counts how often the interval contains the true mean of $1$.

The exponential is a deliberately difficult case. Its skewness is $2$, and
the $\chi^2$ limit is a first-order approximation that does not account for
skewness, so under-coverage should be large enough to see above the Monte
Carlo noise.

| $1-\alpha$ | $n = 10$ | $n = 40$ | $n = 160$ |
| --- | --- | --- | --- |
| 0.50 | 0.415 | 0.510 | 0.485 |
| 0.80 | 0.725 | 0.725 | 0.785 |
| 0.90 | 0.790 | 0.830 | 0.865 |
| 0.95 | 0.855 | 0.880 | 0.935 |
| 0.99 | 0.920 | 0.945 | 0.975 |

Coverage is below nominal in every cell except $n = 40$ at
$1-\alpha = 0.50$, and the largest deviation shrinks from $0.110$ at
$n = 10$ to $0.035$ at $n = 160$. A calibrated radius is therefore not
exact, but its error can be measured and decreases as $n$ grows.

!!! note "Common random numbers"

    Each sample size draws its replications once and reuses them for every
    confidence level. The calibrated sets are nested in $\alpha$, so the
    measured coverage is *exactly* monotone in the nominal level rather
    than monotone up to noise. This is why the script can assert
    monotonicity instead of only plotting it.

## Where the closed form fails

The script evaluates the ambiguity set instead of computing
$\bar{Z} + \sqrt{\rho\, s_n^2}$ directly, because that formula is only the
*interior* optimum. It corresponds to the worst-case distribution

$$
q_i \;=\; \hat{p}_i \left( 1 + \frac{Z_i - \bar{Z}}{s_n} \sqrt{\rho} \right),
$$

which stops being a distribution once $\rho$ is large enough to drive some
$q_i$ negative. Beyond that point the true worst case is *smaller* than the
closed form, because the maximizer is truncated to
$q \propto \hat{p}\,(Z - c)_+$ and the closed form ignores the constraint
$q \ge 0$.

The script prints the most negative discrepancy over all replications, and
the discrepancy is clearly nonzero in only two cells.

| $1-\alpha$ | $n = 10$ | $n = 40$ | $n = 160$ |
| --- | --- | --- | --- |
| 0.90 | $\approx 0$ | $\approx 0$ | $\approx 0$ |
| 0.95 | $-1.8 \times 10^{-3}$ | $\approx 0$ | $\approx 0$ |
| 0.99 | $-2.1 \times 10^{-2}$ | $\approx 0$ | $\approx 0$ |

Entries marked $\approx 0$ are floating-point rounding, a few times
$10^{-15}$, so in those cells the set reproduces the closed form to working
precision. The two negative entries are not rounding. At $n = 10$ and high
confidence the calibrated radius ($\rho = 0.38$ and $\rho = 0.66$) is large
enough to reach the boundary of the simplex. This is the same small-$n$,
high-confidence region where the coverage table is worst. Where the
asymptotic approximation is least accurate, the closed form is also wrong,
and [`ChiSquareAmbiguitySet`](../api/dro/phi_dro.md) handles both regimes
without being told which one applies.

## Reading the figure

- The **left** panel plots empirical coverage against nominal confidence,
  one curve per sample size, with the diagonal marking exact calibration.
  The curves lie below the diagonal, apart from the single point at
  $n = 40$ and $1-\alpha = 0.50$, and approach it as $n$ grows.
- The **right** panel plots the signed error, coverage minus nominal,
  against sample size on a log axis. The error tends to zero, but not
  monotonically at every level. With 200 replications the standard error
  of a coverage estimate $c$ is $\sqrt{c(1-c)/200}$, which is about $0.02$
  near $0.9$ and $0.035$ near $0.5$. The non-monotone step at
  $1-\alpha = 0.50$ is within this Monte Carlo noise.

!!! note "One batched solve per side"

    Coverage is a Monte Carlo quantity, so the experiment needs
    $3 \times 5 \times 200 \times 2 = 6000$ worst-case solves. They are not
    run one at a time. `worst_case_expectation` accepts a loss of shape
    $(\dots, n)$ and a tensor radius that broadcasts against the batch, so
    the five confidence levels form a $(5, 1)$ column of radii and the 200
    replications a $(200, n)$ loss, and one call returns all
    $5 \times 200$ values. Each sample size therefore needs two calls, one
    per side of the interval, and the whole experiment takes well under a
    second on a CPU.

    Every element of that batch is bisected independently through
    elementwise reductions, to the same fixed depth set by the dtype, so
    the extreme radii are resolved as accurately as the mild ones. There is
    no shared starting point and no iteration budget that could leave part
    of the grid short of its optimum. This matters here because an error of
    a few $10^{-4}$ is enough to change a coverage estimate.

## Coverage as the sample grows

The animation runs the experiment on a nested sample. It makes one draw of
200 replications, and each frame uses the first $n$ observations of every
replication. The left panel shows the 90% intervals of the first fifty
replications, which shrink like $1/\sqrt{n}$ and recentre as the sample
grows, with the intervals that miss the true mean in a second colour. The
middle panel shows the coverage curve at the current $n$ against the
diagonal, and the right panel accumulates the signed error at five levels.
Consecutive frames share their observations, so the curves are smoother
than independent samples would make them, although the standard error of
about $0.02$ is still visible as small fluctuations.

<figure class="figure" markdown="span">
  ![Calibrated chi-square intervals of a nested sample, their coverage against the nominal level, and the coverage error, as the sample size grows](../assets/figures/calibration-coverage-light.gif#only-light){ loading=lazy }
  ![Calibrated chi-square intervals of a nested sample, their coverage against the nominal level, and the coverage error, as the sample size grows](../assets/figures/calibration-coverage-dark.gif#only-dark){ loading=lazy }
  <figcaption>The left panel shows the 90% intervals of fifty replications
  at the current sample size, with the true mean as a vertical line. The
  middle panel shows empirical coverage against nominal confidence. The
  right panel shows coverage minus nominal at five levels, accumulated as
  $n$ grows.</figcaption>
</figure>

## The figure

The figure is rendered from this script when the site is built, so it
always shows the output of the current code.

<figure class="figure" markdown="span">
  ![Empirical coverage of the calibrated chi-square interval against the nominal confidence](../assets/figures/06_calibrated_ambiguity_radius-light.png#only-light){ loading=lazy }
  ![Empirical coverage of the calibrated chi-square interval against the nominal confidence](../assets/figures/06_calibrated_ambiguity_radius-dark.png#only-dark){ loading=lazy }
  <figcaption>The left panel shows the empirical coverage of the true mean by the interval at the calibrated radius against the nominal confidence level. The right panel shows that the under-coverage shrinks as the sample size grows.</figcaption>
</figure>

## Source

```py title="examples/06_calibrated_ambiguity_radius.py"
--8<-- "examples/06_calibrated_ambiguity_radius.py"
```

[^dn]:
    John C. Duchi and Hongseok Namkoong, "Learning Models with Uniform
    Performance via Distributionally Robust Optimization", *Annals of
    Statistics* 49(3), 1378-1406 (2021).
    [arXiv:1810.08750](https://arxiv.org/abs/1810.08750).
    The companion asymptotic theory is Duchi, Glynn and Namkoong,
    "Statistics of Robust Optimization: A Generalized Empirical Likelihood
    Approach", *Mathematics of Operations Research* 46(3), 946-969 (2021).

[^bkm]:
    Jose Blanchet, Yang Kang and Karthyek Murthy, "Robust Wasserstein
    Profile Inference and Applications to Machine Learning", *Journal of
    Applied Probability* 56(3), 830-857 (2019).
    [arXiv:1610.05627](https://arxiv.org/abs/1610.05627).

[^gao]:
    Rui Gao, "Finite-Sample Guarantees for Wasserstein Distributionally
    Robust Optimization: Breaking the Curse of Dimensionality", *Operations
    Research* (2025). [arXiv:2009.04382](https://arxiv.org/abs/2009.04382).
    The paper discusses the dimension-dependent radii that follow from
    measure concentration and gives finite-sample guarantees for
    Wasserstein DRO that avoid them.
