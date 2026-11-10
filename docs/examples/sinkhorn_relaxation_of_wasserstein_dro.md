# Sinkhorn relaxation of Wasserstein DRO

The worst-case expectation over a Wasserstein ball on a finite support is a
linear program. Its value is exact and cheap to compute, and it is
piecewise linear in the loss vector. Along a smooth path of decisions the
robust objective therefore has kinks wherever the optimal transport plan
changes. The kinks are not a rounding artifact, and they matter in
practice, because the minimizer of a maximum of smooth functions is
typically a point where two of them cross. An outer method with a
gradient-norm stopping rule never stops there, and a fixed-step gradient
method oscillates around the kink indefinitely.

Sinkhorn DRO (Wang, Gao and Xie 2025) replaces the Wasserstein distance by
its entropic regularization. Its worst case is an exponential tilt, a
smooth function of the loss whose first and second derivatives Optora
computes exactly, and a single parameter $\epsilon$ controls how far the
ball is from the exact one. This example puts `SinkhornAmbiguitySet` next
to `WassersteinAmbiguitySet` on one problem and measures both sides of the
trade-off. It measures how fast the relaxation converges to the exact
solution as $\epsilon \to 0$, and how its smoothness helps an outer
solver.

## The entropic ball as a subclass

Optora's ambiguity sets are subclasses of
[`AmbiguitySet`](../api/core/dro_base.md). The Wasserstein ball bisects its
own piecewise-linear dual. The entropic ball is a `TiltedAmbiguitySet`, the
same base the KL ball uses, because on a finite support the Sinkhorn ball
is a mixture of KL balls with one shared budget (Wang, Gao and Xie, Lemma
3.11). Let $Q_i \propto \nu\, e^{-c_{i\cdot}/\epsilon}$ be the row of the
Gibbs kernel at support point $i$, $p$ the nominal weights, $\rho$ the
radius, and $\bar\rho$ the budget in excess of the entropic bias (see the
note below). Then

$$
\sup_{P:\, \mathcal{W}_\epsilon(\hat P, P) \le \rho} \mathbb{E}_P[\ell]
= \sup_{\gamma_1, \ldots, \gamma_n}
  \Big\{ \sum_i p_i\, \mathbb{E}_{\gamma_i}[\ell] \;:\;
  \epsilon \sum_i p_i\, \mathrm{KL}(\gamma_i \,\|\, Q_i) \le \bar\rho \Big\},
$$

and every row's maximizer is the tilt $\gamma_i \propto Q_i\,
e^{\ell / (\lambda\epsilon)}$ at one shared temperature. The set therefore
needs no solver of its own. The bracketed bisection on the tilt that
`KLAmbiguitySet` uses pins the worst case exactly, with no host
synchronization, and autograd differentiates through it. The formulation
page has the [full statement](../formulations.md#sinkhornambiguityset).

!!! note "What `radius` means for the entropic ball"

    Entropic smoothing moves the centre of the ball. The smallest Sinkhorn
    distance any distribution can have from $\hat P$ is not zero but
    $-\epsilon \sum_i p_i \log \sum_k e^{-c_{ik}/\epsilon}$, attained at
    the kernel-smoothed nominal $P^0 = p^\top Q$. The magnitude of this
    minimum is the entropic bias, and `SinkhornAmbiguitySet.radius` is the
    budget $\bar\rho$ *in excess* of it. The script prints the bias next
    to every gap. The bias vanishes with $\epsilon$ (it is
    $2.9 \times 10^{-5}$ at $\epsilon = 10^{-3}$ here), so the same number
    serves as the radius of both sets in the limit. For $\epsilon$ between
    $0.1$ and about $0.006$ the bias is more than ten times larger than
    the value gap. A ball parameterized by the raw Sinkhorn distance would
    carry an offset of the size of the bias, which is at most
    $\epsilon \log n$. The excess-budget convention removes this offset and
    makes the comparison below tight.

## The problem

The problem uses the forty log-normal demands of the regret example with
equal weight, the ground cost $|\xi - \xi'|$, the quadratic loss
$(\theta - \xi)^2$ of a scalar location decision, and the radius $0.1$.
The loss is smooth in $\theta$, so every kink in the Wasserstein objective
comes from the ambiguity set and not from the loss.

## Convergence as $\epsilon \to 0$

At the fixed decision $\theta = 1$, the exact worst case is $0.41675$. The
script evaluates the Sinkhorn value at thirteen strengths from $1$ down to
$10^{-3}$, and the table shows four of them.

| $\epsilon$ | Sinkhorn value | $\lvert V_\epsilon - V_W \rvert$ | entropic bias |
| --- | --- | --- | --- |
| $1$ | $0.34832$ | $6.8 \times 10^{-2}$ | $3.25$ |
| $10^{-1}$ | $0.42618$ | $9.4 \times 10^{-3}$ | $0.17$ |
| $10^{-2}$ | $0.41670$ | $5.6 \times 10^{-5}$ | $4.3 \times 10^{-3}$ |
| $10^{-3}$ | $0.41681$ | $5.3 \times 10^{-5}$ | $2.9 \times 10^{-5}$ |

The gap is not monotone. Below $\epsilon = 0.1$ it changes sign twice, once
between $0.03$ and $0.02$ and once between $0.01$ and $0.006$, so a log plot
of its absolute value dips at the crossings. It does obey a linear bound.
For every $\epsilon \le 0.1$ the script measures a gap of at most
$0.094\,\epsilon$ and asserts the bound $0.15\,\epsilon$. A linear rate is
consistent with the fact that, on a finite support, the entropic term of
any transport plan is bounded, so the regularization perturbs the
transport cost by $O(\epsilon)$.

The gap also shrinks uniformly over the decisions. Over a grid of $801$
decisions spanning the demand range, $\sup_\theta |V_\epsilon - V_W|$ falls
from $0.13$ at $\epsilon = 0.1$ to $0.053$ at $0.03$ and $0.013$ at
$0.01$, and the mean derivative gap falls with it, from $0.10$ to $0.037$
to $0.0097$.

<figure class="figure" markdown="span">
  ![The robust objective, its derivative and its minimizer under the Sinkhorn ball converging to the Wasserstein ones as the entropic strength vanishes](../assets/figures/sinkhorn-relaxation-light.gif#only-light){ loading=lazy }
  ![The robust objective, its derivative and its minimizer under the Sinkhorn ball converging to the Wasserstein ones as the entropic strength vanishes](../assets/figures/sinkhorn-relaxation-dark.gif#only-dark){ loading=lazy }
  <figcaption>The animation uses the site's twenty-four-scenario instance
  and sweeps $\epsilon$ from $1$ to $10^{-3}$. The left panel shows the
  robust objective in $\theta$ under both balls, with their minimizers. The
  middle panel shows the derivative, a staircase for the Wasserstein ball
  and a smooth curve for the Sinkhorn ball that approaches it. The right
  panel shows the uniform value gap and the distance between the
  minimizers as $\epsilon$ falls.</figcaption>
</figure>

## Differentiable everywhere, measured

Between neighbouring grid points, a twice differentiable function changes
its derivative by at most the step times its largest second derivative.
The script evaluates both derivatives with autograd on the $801$-point grid
(step $2.25 \times 10^{-3}$) and compares the largest increment with that
bound.

| objective | largest increment of $\mathrm{d}V/\mathrm{d}\theta$ | step $\times \max V''$ | ratio |
| --- | --- | --- | --- |
| Wasserstein | $6.5 \times 10^{-2}$ | $4.5 \times 10^{-3}$ | $14.4$ |
| Sinkhorn, $\epsilon = 0.1$ | $8.05 \times 10^{-3}$ | $8.05 \times 10^{-3}$ | $1.00$ |
| Sinkhorn, $\epsilon = 0.03$ | $9.36 \times 10^{-3}$ | $9.36 \times 10^{-3}$ | $1.00$ |
| Sinkhorn, $\epsilon = 0.01$ | $1.26 \times 10^{-2}$ | $1.26 \times 10^{-2}$ | $1.00$ |

The Sinkhorn derivative attains the bound to three digits, as a smooth
function does when its curvature peaks are wide compared with the grid
step. The Wasserstein derivative exceeds the bound fourteenfold, because
it jumps. Its autograd second derivative is $2$ everywhere, which is the
curvature of the loss under the active transport plan. The jumps have no
finite second derivative, so autograd cannot report them. The Sinkhorn
second derivative is finite, and its peaks lie near the Wasserstein kinks.
The largest peak grows from $3.6$ to $5.6$ as $\epsilon$ falls from $0.1$
to $0.01$, so the smooth curve sharpens toward the staircase.

The animation below sweeps the decision at a fixed $\epsilon = 0.02$ and
radius $0.08$ on the twenty-four-scenario instance and shows where the
jumps come from. The Wasserstein adversary is a vertex of the transport
polytope. Between kinks it does not move at all, and when $\theta$ crosses
a kink it switches vertices and relocates a block of mass within a single
frame. The Sinkhorn adversary is a tilt of the kernel rows. It moves a
little in every frame, and it moves fastest where the Wasserstein adversary
jumps.

<figure class="figure" markdown="span">
  ![The Wasserstein and Sinkhorn worst-case distributions and the derivatives they imply as the decision sweeps across the support](../assets/figures/sinkhorn-adversary-light.gif#only-light){ loading=lazy }
  ![The Wasserstein and Sinkhorn worst-case distributions and the derivatives they imply as the decision sweeps across the support](../assets/figures/sinkhorn-adversary-dark.gif#only-dark){ loading=lazy }
  <figcaption>The left panel shows the nominal, the Sinkhorn adversary
  (filled) and the Wasserstein adversary (outlined) at the current
  decision (dashed line), with the total-variation mass each adversary
  moved since the previous frame. The right panel shows the derivative of
  the robust objective under each ball, with the current decision
  marked.</figcaption>
</figure>

## The minimizers

The exact robust decision is $\theta_W = 1.2142$, and it sits on a kink.
The derivative of the Wasserstein objective jumps from
$-2.1 \times 10^{-2}$ to $+8.9 \times 10^{-4}$ across it, so the objective
rises twenty-four times faster to the left of its minimizer than to the
right.

| $\epsilon$ | Sinkhorn minimizer $\theta_\epsilon$ | $\lvert \theta_\epsilon - \theta_W \rvert$ | $V_W(\theta_\epsilon) - V_W(\theta_W)$ |
| --- | --- | --- | --- |
| $10^{-1}$ | $1.1930$ | $2.1 \times 10^{-2}$ | $1.5 \times 10^{-3}$ |
| $10^{-2}$ | $1.2177$ | $3.5 \times 10^{-3}$ | $1.5 \times 10^{-5}$ |
| $10^{-3}$ | $1.2165$ | $2.3 \times 10^{-3}$ | $7.2 \times 10^{-6}$ |

The last column is the number that matters when the relaxed decision is
deployed against the exact objective. For every $\epsilon \le 0.1$ the
script measures at most $0.022\,\epsilon$ and asserts the bound
$0.05\,\epsilon$. The middle column converges more slowly, and the flat
right flank explains why. For $\epsilon \le 10^{-2}$ the relaxed
minimizer lies to the right of the kink, where a decision can move some
distance at almost no cost in the exact objective. The location of the
minimizer is ill-conditioned, but its value is not.

## Effect of smoothness on the outer solve

`MinimaxSolver` wraps a `GradientDescent` with step $0.2$, gradient
tolerance $10^{-8}$ and a budget of $400$ steps around each ball, starting
from $\theta = 2$.

| objective | converged | iterations | final decision |
| --- | --- | --- | --- |
| Wasserstein | no | $400$ (budget) | $1.21432$ |
| Sinkhorn, $\epsilon = 0.1$ | yes | $19$ | $1.19303$ |
| Sinkhorn, $\epsilon = 0.03$ | yes | $20$ | $1.20461$ |
| Sinkhorn, $\epsilon = 0.01$ | yes | $16$ | $1.21767$ |

Unrolling sixty steps of the same iteration by hand shows what the
Wasserstein solve is doing. Over the second half of the run its gradient
magnitude stays between $1.2 \times 10^{-3}$ and $2.1 \times 10^{-2}$,
close to the magnitudes of the two one-sided slopes at the kink. The
Sinkhorn gradient at $\epsilon = 0.01$ falls to $10^{-14}$ by the last
step. The Wasserstein iterate is accurate, within $1.5 \times 10^{-4}$ of
the exact minimizer, but no gradient test can confirm this to the solver.
The relaxation is therefore useful in an outer loop. Its inner problem is
smooth, so its gradient behaves as a line search or a stopping rule
assumes, and the accuracy cost is linear in $\epsilon$ and can be
measured.

## The figure

The figure is rendered from this script when the site is built, so it
always shows the output of the current code.

<figure class="figure" markdown="span">
  ![Gaps between the Sinkhorn and Wasserstein solutions over the entropic strength, the two derivatives near the minimizer, and the gradient magnitude of a fixed-step outer solve on each objective](../assets/figures/12_sinkhorn_relaxation_of_wasserstein_dro-light.png#only-light){ loading=lazy }
  ![Gaps between the Sinkhorn and Wasserstein solutions over the entropic strength, the two derivatives near the minimizer, and the gradient magnitude of a fixed-step outer solve on each objective](../assets/figures/12_sinkhorn_relaxation_of_wasserstein_dro-dark.png#only-dark){ loading=lazy }
  <figcaption>The left panel shows the value gap, the minimizer distance
  and the exact suboptimality of the relaxed decision against $\epsilon$,
  with the entropic bias dotted. The middle panel shows the derivative of
  each objective around the Wasserstein minimizer (dashed). The right
  panel shows the gradient magnitude along sixty fixed steps on each
  objective, against the solver's tolerance.</figcaption>
</figure>

## What to take away

- `SinkhornAmbiguitySet` is the smooth counterpart of
  `WassersteinAmbiguitySet`. On a finite support it is a budgeted mixture
  of KL balls, so it reuses the tilt machinery rather than a transport
  solver, and it works with `MinimaxSolver` unchanged.
- With the excess-budget convention for its radius, the gap to the exact
  worst case is linear in $\epsilon$ with a small constant. Here the
  measured constants are below $0.1$ for the value and below $0.03$ for
  the exact suboptimality of the relaxed decision.
- The Wasserstein objective has derivative jumps, one of them at its own
  minimizer. The Sinkhorn objective is twice differentiable, and its
  curvature peaks sharpen toward those jumps as $\epsilon$ falls.
- A fixed-step gradient method with a gradient-norm stopping rule
  converges on the relaxation in about twenty steps and does not converge
  on the exact objective within $400$ steps. The exact ball is better
  served by a method that tolerates kinks, such as the derivative-sign
  bisection the script uses as its reference.

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
