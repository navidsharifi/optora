# Sinkhorn relaxation of Wasserstein DRO

The worst-case expectation over a Wasserstein ball on a finite support is a
linear program. Its value is exact, cheap to compute, and piecewise linear
in the loss vector, which means that along a smooth path of decisions the
robust objective has kinks wherever the optimal transport plan changes.
The kinks are not a rounding artifact, and they do not sit in harmless
places: the minimizer of a maximum of smooth functions is typically a
point where two of them cross. An outer method with a gradient-norm
stopping rule never stops there, and a fixed-step gradient method
oscillates around the kink forever.

Sinkhorn DRO (Wang, Gao and Xie 2025) replaces the Wasserstein distance by
its entropic regularization. The ball becomes smooth, its worst case
becomes an exponential tilt with exact derivatives of every order, and a
single parameter $\epsilon$ controls how far it is from the exact ball.
This example puts `SinkhornAmbiguitySet` next to `WassersteinAmbiguitySet`
on one problem and measures both sides of the trade: how fast the
relaxation converges to the exact solution as $\epsilon \to 0$, and what
its smoothness buys an outer solver.

## The entropic ball as a subclass

Optora's ambiguity sets are subclasses of
[`AmbiguitySet`](../api/core/dro_base.md). The Wasserstein ball bisects its
own piecewise-linear dual; the entropic ball is a
`TiltedAmbiguitySet`, the same base the KL ball uses, because on a finite
support the Sinkhorn ball is a mixture of KL balls with one shared budget
(Wang, Gao and Xie, Lemma 3.11). With $Q_i \propto \nu\, e^{-c_{i\cdot}/
\epsilon}$ the row of the Gibbs kernel at support point $i$, nominal
weights $p$ and budget $\rho$,

$$
\sup_{P:\, \mathcal{W}_\epsilon(\hat P, P) \le \rho} \mathbb{E}_P[\ell]
= \sup_{\gamma_1, \ldots, \gamma_n}
  \Big\{ \sum_i p_i\, \mathbb{E}_{\gamma_i}[\ell] \;:\;
  \epsilon \sum_i p_i\, \mathrm{KL}(\gamma_i \,\|\, Q_i) \le \bar\rho \Big\},
$$

and every row's maximizer is the tilt $\gamma_i \propto Q_i\,
e^{\ell / (\lambda\epsilon)}$ at one shared temperature. The set therefore
needs no solver of its own: the bracketed bisection on the tilt that
`KLAmbiguitySet` uses pins the worst case exactly, with no host
synchronization, and autograd differentiates through it. The formulation
page has the [full statement](../formulations.md#sinkhornambiguityset).

!!! note "What `radius` means for the entropic ball"

    Entropic smoothing moves the centre of the ball. The smallest Sinkhorn
    distance any distribution can have from $\hat P$ is not zero but
    $-\epsilon \sum_i p_i \log \sum_k e^{-c_{ik}/\epsilon}$, attained at
    the kernel-smoothed nominal $P^0 = p^\top Q$, and
    `SinkhornAmbiguitySet.radius` is the budget $\bar\rho$ *in excess* of
    that bias. The script prints the bias next to every gap. It vanishes
    with $\epsilon$ (it is $2.9 \times 10^{-5}$ at $\epsilon = 10^{-3}$
    here), so the same number serves as the radius of both sets in the
    limit, and it is two orders of magnitude larger than the value gap at
    every $\epsilon$ below $0.1$. A ball parameterized by the raw Sinkhorn
    distance would converge to the Wasserstein ball at the rate of the
    bias, $O(\epsilon \log n)$; the excess-budget convention is what makes
    the comparison below tight.

## The problem

The forty log-normal demands of the regret example, with equal weight, the
ground cost $|\xi - \xi'|$, the quadratic loss $(\theta - \xi)^2$ of a
scalar location decision, and the radius $0.1$. The loss is smooth in
$\theta$, so every kink in the Wasserstein objective comes from the
ambiguity set, not from the loss.

## Convergence as $\epsilon \to 0$

At the fixed decision $\theta = 1$, the exact worst case is $0.41675$. The
Sinkhorn value at thirteen strengths from $1$ down to $10^{-3}$:

| $\epsilon$ | Sinkhorn value | $\lvert V_\epsilon - V_W \rvert$ | entropic bias |
| --- | --- | --- | --- |
| $1$ | $0.34832$ | $6.8 \times 10^{-2}$ | $3.25$ |
| $10^{-1}$ | $0.42618$ | $9.4 \times 10^{-3}$ | $0.17$ |
| $10^{-2}$ | $0.41670$ | $5.6 \times 10^{-5}$ | $4.3 \times 10^{-3}$ |
| $10^{-3}$ | $0.41681$ | $5.3 \times 10^{-5}$ | $2.9 \times 10^{-5}$ |

The gap is not monotone: it changes sign twice on the way down, once near
$\epsilon = 0.03$ and once near $0.007$, so a log plot of its absolute
value dips at the crossings. What it does obey is a linear bound, and the
script asserts it: for every $\epsilon \le 0.1$ the gap is below
$0.094\,\epsilon$. That is the rate one expects of entropic regularization
on a finite support (the entropy of the optimal plan is bounded), and it
holds here with a small constant.

Convergence is also uniform in the decision. Over a grid of $801$
decisions spanning the demand range, $\sup_\theta |V_\epsilon - V_W|$ falls
from $0.13$ at $\epsilon = 0.1$ to $0.053$ at $0.03$ and $0.013$ at
$0.01$, and the mean derivative gap falls with it, from $0.10$ to $0.037$
to $0.0097$.

<figure class="figure" markdown="span">
  ![The robust objective, its derivative and its minimizer under the Sinkhorn ball converging to the Wasserstein ones as the entropic strength vanishes](../assets/figures/sinkhorn-relaxation-light.gif#only-light){ loading=lazy }
  ![The robust objective, its derivative and its minimizer under the Sinkhorn ball converging to the Wasserstein ones as the entropic strength vanishes](../assets/figures/sinkhorn-relaxation-dark.gif#only-dark){ loading=lazy }
  <figcaption>On the site's twenty-four-scenario instance. Left: the
  robust objective in $\theta$ under both balls, with their minimizers.
  Middle: the derivative, a staircase for the Wasserstein ball and a
  smooth curve for the Sinkhorn ball that bends toward it. Right: the
  uniform value gap and the distance between the minimizers as
  $\epsilon$ falls.</figcaption>
</figure>

## Differentiable everywhere, measured

A twice differentiable function moves its derivative by at most the step
times its largest second derivative between neighbouring grid points. The
script evaluates both derivatives with autograd on the $801$-point grid
(step $2.25 \times 10^{-3}$) and compares the largest increment with that
bound:

| objective | largest increment of $\mathrm{d}V/\mathrm{d}\theta$ | step $\times \max V''$ | ratio |
| --- | --- | --- | --- |
| Wasserstein | $6.5 \times 10^{-2}$ | $4.5 \times 10^{-3}$ | $14.4$ |
| Sinkhorn, $\epsilon = 0.1$ | $8.05 \times 10^{-3}$ | $8.05 \times 10^{-3}$ | $1.00$ |
| Sinkhorn, $\epsilon = 0.03$ | $9.36 \times 10^{-3}$ | $9.36 \times 10^{-3}$ | $1.00$ |
| Sinkhorn, $\epsilon = 0.01$ | $1.26 \times 10^{-2}$ | $1.26 \times 10^{-2}$ | $1.00$ |

The Sinkhorn derivative attains the bound to three digits, which is what
a smooth function does when its curvature peaks are wide against the grid
step. The Wasserstein derivative exceeds it fourteenfold: it jumps. Its
autograd second derivative is $2$ everywhere, the curvature of the loss
under the active transport plan; the jumps have no finite second
derivative and autograd cannot report them. The Sinkhorn second derivative
is finite, with peaks at the Wasserstein kinks that grow from $3.6$ to
$5.6$ as $\epsilon$ falls from $0.1$ to $0.01$: the smooth curve is
sharpening toward the staircase.

The animation below sweeps the decision at a fixed $\epsilon = 0.02$ and
shows where the jumps come from. The Wasserstein adversary is a vertex of
the transport polytope: between kinks it does not move at all, and when
$\theta$ crosses one it switches vertices and relocates a block of mass
within a single frame. The Sinkhorn adversary is a tilt of the kernel
rows, and it moves a little in every frame, fastest where the Wasserstein
adversary jumps.

<figure class="figure" markdown="span">
  ![The Wasserstein and Sinkhorn worst-case distributions and the derivatives they imply as the decision sweeps across the support](../assets/figures/sinkhorn-adversary-light.gif#only-light){ loading=lazy }
  ![The Wasserstein and Sinkhorn worst-case distributions and the derivatives they imply as the decision sweeps across the support](../assets/figures/sinkhorn-adversary-dark.gif#only-dark){ loading=lazy }
  <figcaption>Left: the nominal, the Sinkhorn adversary (filled) and the
  Wasserstein adversary (outlined) at the current decision, with the
  total-variation mass each one moved since the previous frame. Right:
  the derivative of the robust objective under each ball, with the current
  decision marked.</figcaption>
</figure>

## The minimizers

The exact robust decision is $\theta_W = 1.2142$, and it sits on a kink:
the derivative of the Wasserstein objective jumps from $-2.1 \times
10^{-2}$ to $+8.9 \times 10^{-4}$ across it. The objective rises
twenty-four times faster to the left of its minimizer than to the right.

| $\epsilon$ | Sinkhorn minimizer $\theta_\epsilon$ | $\lvert \theta_\epsilon - \theta_W \rvert$ | $V_W(\theta_\epsilon) - V_W(\theta_W)$ |
| --- | --- | --- | --- |
| $10^{-1}$ | $1.1930$ | $2.1 \times 10^{-2}$ | $1.5 \times 10^{-3}$ |
| $10^{-2}$ | $1.2177$ | $3.5 \times 10^{-3}$ | $1.5 \times 10^{-5}$ |
| $10^{-3}$ | $1.2165$ | $2.3 \times 10^{-3}$ | $7.2 \times 10^{-6}$ |

The last column is the number that matters when the relaxed decision is
deployed against the exact objective, and the script asserts it is below
$0.022\,\epsilon$ at every $\epsilon \le 0.1$. The middle column converges
more slowly, and it is the flat right flank that explains why: a
minimizer can wander some distance to the right of the kink at almost no
cost in the exact objective, and the smooth relaxation does exactly that.
The location is ill-conditioned; the value is not.

## What the smoothness buys the outer solve

`MinimaxSolver` wraps a `GradientDescent` with step $0.2$, gradient
tolerance $10^{-8}$ and a budget of $400$ steps around each ball, starting
from $\theta = 2$:

| objective | converged | iterations | final decision |
| --- | --- | --- | --- |
| Wasserstein | no | $400$ (budget) | $1.21432$ |
| Sinkhorn, $\epsilon = 0.1$ | yes | $19$ | $1.19303$ |
| Sinkhorn, $\epsilon = 0.03$ | yes | $20$ | $1.20461$ |
| Sinkhorn, $\epsilon = 0.01$ | yes | $16$ | $1.21767$ |

Unrolling the same iteration by hand shows what the Wasserstein solve is
doing: after the first few steps its gradient magnitude never drops below
$1.2 \times 10^{-3}$ and keeps bouncing up to $2.1 \times 10^{-2}$, the two
one-sided slopes at the kink, while the Sinkhorn gradient at $\epsilon =
0.01$ decays geometrically to $10^{-14}$ in thirty steps. The Wasserstein
iterate is not wrong, it is within $1.5 \times 10^{-4}$ of the exact
minimizer, but no gradient test can tell the solver so. This is the
practical case for the relaxation in an outer loop: a smooth inner
problem whose gradient means what a line search or a stopping rule
assumes it means, at an accuracy cost that is linear in $\epsilon$ and
measurable.

## The figure

Rendered from this script when the site is built, so it always shows the
output of the current code:

<figure class="figure" markdown="span">
  ![Gaps between the Sinkhorn and Wasserstein solutions over the entropic strength, the two derivatives near the minimizer, and the gradient magnitude of a fixed-step outer solve on each objective](../assets/figures/12_sinkhorn_relaxation_of_wasserstein_dro-light.png#only-light){ loading=lazy }
  ![Gaps between the Sinkhorn and Wasserstein solutions over the entropic strength, the two derivatives near the minimizer, and the gradient magnitude of a fixed-step outer solve on each objective](../assets/figures/12_sinkhorn_relaxation_of_wasserstein_dro-dark.png#only-dark){ loading=lazy }
  <figcaption>Left: the value gap, the minimizer distance and the exact
  suboptimality of the relaxed decision against $\epsilon$, with the
  entropic bias dotted. Middle: the derivative of each objective around
  the Wasserstein minimizer (dashed). Right: the gradient magnitude along
  sixty fixed steps on each objective, against the solver's
  tolerance.</figcaption>
</figure>

## What to take away

- `SinkhornAmbiguitySet` is the smooth counterpart of
  `WassersteinAmbiguitySet`: on a finite support it is a budgeted mixture
  of KL balls, so it reuses the tilt machinery rather than a transport
  solver, and it drops into `MinimaxSolver` unchanged.
- With the excess-budget convention for its radius, the gap to the exact
  worst case is linear in $\epsilon$ with a small constant, here under
  $0.1\,\epsilon$ in value and $0.03\,\epsilon$ in the exact suboptimality
  of the relaxed decision.
- The Wasserstein objective has derivative jumps, one of them at its own
  minimizer. The Sinkhorn objective is twice differentiable, and its
  curvature peaks sharpen toward those jumps as $\epsilon$ falls.
- A fixed-step gradient method with a gradient-norm stopping rule
  converges on the relaxation in about twenty steps and never converges on
  the exact objective; the exact ball is better served by a method that
  tolerates kinks, such as the derivative-sign bisection the script uses
  as its reference.

## References

- Jie Wang, Rui Gao and Yao Xie, "Sinkhorn Distributionally Robust
  Optimization", *Operations Research*, 2025 (arXiv:2109.11926).
- Marco Cuturi, "Sinkhorn Distances: Lightspeed Computation of Optimal
  Transport", *Advances in Neural Information Processing Systems* 26, 2013.
- Peyman Mohajerin Esfahani and Daniel Kuhn, "Data-Driven Distributionally
  Robust Optimization Using the Wasserstein Metric", *Mathematical
  Programming* 171, 2018.

## Source

```py title="examples/12_sinkhorn_relaxation_of_wasserstein_dro.py"
--8<-- "examples/12_sinkhorn_relaxation_of_wasserstein_dro.py"
```
