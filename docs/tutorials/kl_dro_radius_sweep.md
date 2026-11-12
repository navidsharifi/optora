# KL-DRO radius sweep

This tutorial isolates the ambiguity set from everything else. We fix a
nominal distribution and a per-scenario loss, and we follow the worst-case
expectation

$$
\sup_{q \,:\, D_{\mathrm{KL}}(q \,\|\, \mathrm{nominal}) \le \rho}
\mathbb{E}_q[\mathrm{loss}]
$$

as the radius $\rho$ grows. There is no decision variable to optimize, so
every change in the curve comes from the ambiguity set alone.

## Why it is worth plotting

Both ends of the curve are known in closed form, which makes the sweep a
useful correctness check.

| Radius | What the ball is | Worst case |
| --- | --- | --- |
| $\rho = 0$ | the single distribution $\{\mathrm{nominal}\}$ | $\sum_i \mathrm{nominal}_i \,\mathrm{loss}_i$ |
| $\rho \to \infty$ | eventually contains the point mass on the worst scenario | $\max_i \mathrm{loss}_i$ |

The script asserts the first identity to within $10^{-12}$. When the
radius is the float `0.0`, Optora skips the solve and returns the nominal
expectation, so it never has to handle a dual whose minimizer diverges as
$\rho \to 0$. The second limit is reached at a finite radius. Let
$P^\star$ be the nominal mass on the highest-loss scenarios. Once $\rho$
reaches $-\log P^\star$, every point of the tilt path is feasible and the
maximizer is the point mass itself, so the curve meets the line at the
maximum loss instead of approaching it asymptotically.

Between the two ends the curve is nondecreasing, because a larger ball
contains a smaller one and its supremum cannot be smaller. The script
checks this on the sampled radii as well.

## Reading the output

There are three things to look at.

1. The printed comparison between $\mathbb{E}_{\mathrm{nominal}}[\mathrm{loss}]$
   and `worst_case_expectation(radius=0.0)`, which agree to within
   $10^{-12}$.
2. The `monotonically nondecreasing in radius: True` line.
3. The figure, on a log-scaled radius axis. It starts flat near the
   nominal expectation, rises steeply over the middle decades of the
   radius, and levels off at $\max_i \mathrm{loss}_i$.

!!! tip "Everything is `float64` here"

    The tolerances in the script assume double precision. The tilt is
    bisected to the precision of the dtype, which is $2^{-52}$ in
    `float64` but only $2^{-23}$ in `float32`. In single precision the
    rounding error on the flat part of the curve can be large enough to
    fail a tight check even though the value is correct to single
    precision.

## The adversary as the radius grows

The static figure further down summarizes the sweep, and the animation
shows where it comes from. At every radius it draws the worst-case
distribution $q^\star$ next to the nominal. The distribution is read off
the gradient of the worst-case value with respect to the loss (Danskin's
theorem). The exponential tilt moves mass gradually from the cheap
scenarios to the expensive one. The right panel checks that the
constraint is active, since $D_{\mathrm{KL}}(q^\star \,\|\, p)$ equals
$\rho$ until the radius reaches $-\log P^\star$. Beyond that radius the
point mass is feasible and the divergence stays constant.

<figure class="figure" markdown="span">
  ![The KL worst-case distribution, the worst-case value and the divergence of the adversary as the radius grows](../assets/figures/kl-radius-light.gif#only-light){ loading=lazy }
  ![The KL worst-case distribution, the worst-case value and the divergence of the adversary as the radius grows](../assets/figures/kl-radius-dark.gif#only-dark){ loading=lazy }
  <figcaption>The left panel shows the nominal and the worst-case
  distribution at the current radius. The middle panel shows the
  worst-case value on a log-scaled radius axis, with the nominal mean,
  the maximum loss and the saturation radius marked. The right panel
  shows the divergence of the adversary, which follows the diagonal while
  the constraint is active and stays flat once the ball contains the
  point mass.</figcaption>
</figure>

## The figure

The figure is rendered from this script when the site is built, so it always shows the output of the current code.

<figure class="figure" markdown="span">
  ![Worst-case expectation of the KL ambiguity set against the radius](../assets/figures/01_kl_dro_radius_sweep-light.png#only-light){ loading=lazy }
  ![Worst-case expectation of the KL ambiguity set against the radius](../assets/figures/01_kl_dro_radius_sweep-dark.png#only-dark){ loading=lazy }
  <figcaption>The worst case starts at the nominal expectation, rises along the log-scaled radius axis, and equals the maximum loss once the radius reaches $-\log P^\star$.</figcaption>
</figure>

## Source

```py title="tutorials/01_kl_dro_radius_sweep.py"
--8<-- "tutorials/01_kl_dro_radius_sweep.py"
```
