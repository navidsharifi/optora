# KL-DRO radius sweep

This is the smallest honest DRO experiment I could write: fix a nominal
distribution, fix a per-scenario loss, and watch what happens to

$$
\sup_{q \,:\, D_{\mathrm{KL}}(q \,\|\, \mathrm{nominal}) \le \rho}
\mathbb{E}_q[\mathrm{loss}]
$$

as the radius $\rho$ grows. Nothing is being optimized over a decision
variable here — the decision is out of the picture entirely, so whatever
the curve does is the ambiguity set's doing and nothing else's.

## Why it is worth plotting

The formulation has two endpoints you can write down without solving
anything, which makes it a genuinely useful sanity check rather than a
pretty picture:

| Radius | What the ball is | Worst case |
| --- | --- | --- |
| $\rho = 0$ | just $\{\mathrm{nominal}\}$ | $\sum_i \mathrm{nominal}_i \,\mathrm{loss}_i$ |
| $\rho \to \infty$ | eventually contains the point mass on the worst scenario | $\max_i \mathrm{loss}_i$ |

The script asserts the first one to machine precision (Optora special-cases
$\rho = 0$ and returns the nominal expectation exactly, rather than sending
the dual variable $\eta \to \infty$ and hoping). The second is only
*approached*: the dual optimum runs off to the boundary, so a
finite-iteration solve gets close and stops. That asymmetry is expected, and
seeing the curve flatten out just below the red line is the point.

In between, the curve should be nondecreasing — a bigger ball cannot contain
a less adversarial distribution — and the script checks that too, on the
sampled radii, rather than taking it on faith.

## Reading the output

Three things to look at:

1. The printed comparison between $\mathbb{E}_{\mathrm{nominal}}[\mathrm{loss}]$
   and `worst_case_expectation(radius=0.0)`; they should agree to about
   $10^{-12}$.
2. The `monotonically nondecreasing in radius: True` line.
3. The figure, on a log-scaled radius axis: a flat start near the nominal
   expectation, a steep middle where robustness actually costs something,
   and a plateau at $\max_i \mathrm{loss}_i$.

!!! tip "Everything is `float64` here"

    The tolerances above are only defensible in double precision. In
    `float32` the gradient norms in the dual solve plateau around
    $10^{-6}$, which is enough to make a tight convergence check fail even
    though the iterate is sitting on the answer.

## Source

```py title="examples/01_kl_dro_radius_sweep.py"
--8<-- "examples/01_kl_dro_radius_sweep.py"
```
