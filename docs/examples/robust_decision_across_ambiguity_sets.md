# Robust decisions across geometries

Once we decide to optimize against a worst case, we still have to choose
which worst case. This example takes one small decision problem and solves
it four times, changing only the shape of the ambiguity set,

$$
\min_x \; \sup_{q \in \mathcal{Q}(\mathrm{nominal}, \rho)}
\mathbb{E}_q\!\left[(\mathrm{outcomes} - x)^2\right]
$$

with $\mathcal{Q}$ ranging over the KL, $\chi^2$, total-variation, and
Wasserstein balls. The outcomes are `[1, 2, 3, 10]` under a uniform
nominal. Three of them are ordinary values and one is an outlier, so the
experiment shows how strongly each geometry weights the 10.

## What the comparison shows

At $\rho = 0$ every ball collapses to $\{\mathrm{nominal}\}$, so all four
formulations return the same answer. This is the empirical-risk
decision, which for squared error is the nominal mean, $4.0$. The common
starting point makes the four curves comparable.

As $\rho$ grows, the three $\varphi$-divergence decisions move toward the
outlier at different rates. The radii are not on a common scale, because
each divergence measures distance in its own units, but the shape of each
curve is a property of the divergence rather than of the solver.

- **KL** moves fastest at small radii. Its worst-case distribution is an
  exponential tilt of the nominal that keeps every scenario at positive
  mass below the saturation radius. For small $\rho$ the worst-case value
  grows as
  $\mathbb{E}[\cdot] + \sqrt{2\rho \operatorname{Var}[\cdot]} + O(\rho)$,
  an expansion that `tests/dro/test_kl_dro.py` and
  `tests/dro/test_equivalences.py` check.
- **$\chi^2$** follows the same pattern with a smaller constant at the
  same radius. In the
  interior regime, where no scenario is truncated to zero mass, its worst
  case equals the variance-penalty form
  $\mathbb{E}[\cdot] + \sqrt{\rho \operatorname{Var}[\cdot]}$ (Duchi and
  Namkoong, 2021)[^dn]. Its maximizer is a truncated linear tilt, so a
  scenario can reach zero mass at a finite radius.
- **Total variation** starts slowest. Its worst case grows piecewise
  linearly in $\rho$, because it moves a block of mass from the cheapest
  scenario to the most expensive one. Its exact minimizer overtakes the
  other two between $\rho = 0.1$ and $\rho = 0.2$ and then stays at
  $5.5$, the midpoint of $1$ and $10$. There the two outcomes tie for the
  largest loss and the robust objective has a kink. The fixed-step solver
  of the script does not settle exactly at a kink within its 120
  iterations, so its printed values near $5.5$ are approximate.
- **Wasserstein** is the only geometry that depends on the distances
  between outcomes rather than only on their probabilities. It is given
  the squared-distance cost matrix, so moving mass from $1$ to $10$ costs
  $81$, while moving it from $2$ to $1$ costs $1$. Its decision moves the
  other way. Near $x = 4$ the cheapest way to raise the loss is to shift
  mass from $2$ toward $1$, and from $3$ as well at larger radii, so the
  decision drifts toward the low cluster and away from the outlier.

The last item is the main difference between the two families. A
$\varphi$-divergence depends only on the probabilities of the scenarios
and not on how far apart they are, whereas the Wasserstein distance
depends on both.

## A note on cost

Every point on every curve is a nested solve. The outer
[`MinimaxSolver`](../api/dro/minimax_solver.md) evaluates an objective
that runs the ambiguity set's own inner solve at every iteration, so the
work multiplies, and the outer iteration budget at the top of the script
is cut well below the default. The inner solves need no budget. Total
variation has a closed form, and the KL, $\chi^2$ and Wasserstein sets
are solved by a bisection with a fixed number of steps set by the dtype,
so each inner solve has a fixed cost.

The outer gradient flows correctly through `worst_case_expectation` even
though the worst-case distribution is computed under `detach`. This
follows from Danskin's theorem. The optimal value is the support function
of the ambiguity set evaluated at the loss, so its gradient with respect
to the loss is the maximizing distribution, and only the final weighted
sum of the losses needs to stay attached to the graph.

## Four landscapes, one radius

The decisions in the static figure are minimizers, and the animation
shows the objectives they minimize. The left panel draws
$\sup_q \mathbb{E}_q[(\xi - x)^2]$ as a function of $x$ for every
geometry at the current radius, with the nominal loss dashed and each
minimizer marked. The middle panel tracks the minimizers as the radius
grows, and the right panel shows the adversary of each ball at its own
decision. The minimizers come from a bisection on the derivative of the
robust objective, which gives the exact point that the fixed-step solver
of the script approaches.

The right panel also explains the direction of the Wasserstein decision.
Its adversary moves mass toward $1$, which is the cheapest transport
under the squared cost. The three $\varphi$-divergence adversaries do not
depend on distance and move mass onto the $10$. As their decisions move
toward the outlier, the loss at $1$ grows as well, so they also move mass
onto the $1$. At the total-variation decision $5.5$ the outcomes $1$ and
$10$ have equal loss.

<figure class="figure" markdown="span">
  ![The robust objective, its minimizer and its adversary under four ambiguity-set geometries as the radius grows](../assets/figures/decision-geometries-light.gif#only-light){ loading=lazy }
  ![The robust objective, its minimizer and its adversary under four ambiguity-set geometries as the radius grows](../assets/figures/decision-geometries-dark.gif#only-dark){ loading=lazy }
  <figcaption>The left panel shows the robust objective in $x$ for each
  geometry, with the nominal loss dashed and the minimizers marked. The
  middle panel shows the decisions against the radius. The right panel
  shows the worst-case distribution of each ball at its own decision,
  drawn over the nominal.</figcaption>
</figure>

## The figure

The figure is rendered from this script when the site is built, so it always shows the output of the current code.

<figure class="figure" markdown="span">
  ![Robust decision against the radius under four ambiguity-set geometries](../assets/figures/03_robust_decision_across_ambiguity_sets-light.png#only-light){ loading=lazy }
  ![Robust decision against the radius under four ambiguity-set geometries](../assets/figures/03_robust_decision_across_ambiguity_sets-dark.png#only-dark){ loading=lazy }
  <figcaption>Every geometry starts from the empirical-risk decision at radius zero. The divergence determines where the decision goes and how fast, so the $\varphi$-divergence decisions move toward the outlier and the Wasserstein decision moves the other way.</figcaption>
</figure>

## Source

```py title="examples/03_robust_decision_across_ambiguity_sets.py"
--8<-- "examples/03_robust_decision_across_ambiguity_sets.py"
```

[^dn]:
    John C. Duchi and Hongseok Namkoong, "Learning Models with Uniform
    Performance via Distributionally Robust Optimization", *Annals of
    Statistics* 49(3), 1378-1406 (2021).
    [arXiv:1810.08750](https://arxiv.org/abs/1810.08750).
