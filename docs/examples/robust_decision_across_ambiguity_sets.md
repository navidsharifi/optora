# Robust decisions across geometries

Once you accept that you should optimize against a worst case, the next
question is which worst case. This example takes one small decision problem
and solves it four times, changing only the shape of the ambiguity set:

$$
\min_x \; \sup_{q \in \mathcal{Q}(\mathrm{nominal}, \rho)}
\mathbb{E}_q\!\left[(\mathrm{outcomes} - x)^2\right]
$$

with $\mathcal{Q}$ ranging over the KL, $\chi^2$, total-variation, and
Wasserstein balls. The outcomes are `[1, 2, 3, 10]` under a uniform nominal
— three ordinary values and one outlier — because the whole point is to see
how hard each geometry leans on that 10.

## What the comparison shows

At $\rho = 0$ every ball collapses to $\{\mathrm{nominal}\}$, so all four
formulations must return the same answer: the plain empirical-risk decision,
which for squared error is just the nominal mean, $4.0$. That shared
starting point is what makes the rest of the plot comparable at all.

As $\rho$ grows, the three $\varphi$-divergence decisions are dragged
toward the outlier, at different rates, and the ordering is a property
of the divergence rather than of the solver:

- **Total variation** moves in discrete-feeling jumps — it can relocate a
  block of mass wholesale, so it reacts fast and early.
- **KL** is the most reluctant to fully abandon a scenario, since driving
  $q_i \to 0$ costs it unboundedly.
- **$\chi^2$** sits in between, and in the interior regime it matches the
  familiar $\mathbb{E}[\cdot] + \sqrt{\rho \operatorname{Var}[\cdot]}$
  variance-penalty form.
- **Wasserstein** is the only one that cares about the *geometry* of the
  outcome space rather than just the labels: it is given the squared-distance
  cost matrix, so moving mass from $1$ to $10$ costs far more than moving it
  from $3$ to $10$. It moves the other way: the cheapest way to raise the
  loss near $x = 4$ is to shift mass from $2$ and $3$ down to $1$, so its
  decision drifts toward the low cluster, away from the outlier.

That last distinction is the one I find easiest to forget. The
$\phi$-divergence family is blind to how far apart the scenarios are; it
only sees probabilities. Wasserstein is not.

## A note on cost

Every point on every curve is a nested solve — an outer
[`MinimaxSolver`](../api/dro/minimax_solver.md) whose objective internally
runs an ambiguity set's own inner solve on each iteration. Iteration counts
multiply, so the outer budget at the top of the script is cut down hard
from the defaults. The inner solves need no budget at all: all four sets
here are solver-free, so each costs a fixed, identical number of kernels.

Also worth knowing, since it looks like it should not work: the outer
gradient flows correctly through `worst_case_expectation` even though the
worst-case distribution is found under `detach`. That is Danskin's theorem
— the optimal value is a support function of the loss, so its gradient *is*
that distribution, and only the final re-weighting needs to stay attached
to the graph.

## Four landscapes, one radius

The decisions in the static figure are minimizers, and the animation
shows the objectives they minimize. The left panel draws
$\sup_q \mathbb{E}_q[(\xi - x)^2]$ in $x$ for every geometry at the
current radius, with the nominal loss dashed and each minimizer marked;
the middle panel tracks the minimizers as the radius grows; the right
panel shows the adversary each ball fields at its own decision. The
minimizers come from a bisection on the derivative of the robust
objective, the exact point the script's fixed-step solver approaches.

The right panel is also where the Wasserstein ball's odd direction stops
being odd: its adversary moves mass from $2$ and $3$ to $1$, the cheapest
transport under the squared cost, while the three $\varphi$-divergence
adversaries, blind to distance, pile mass on the $10$.

<figure class="figure" markdown="span">
  ![The robust objective, its minimizer and its adversary under four ambiguity-set geometries as the radius grows](../assets/figures/decision-geometries-light.gif#only-light){ loading=lazy }
  ![The robust objective, its minimizer and its adversary under four ambiguity-set geometries as the radius grows](../assets/figures/decision-geometries-dark.gif#only-dark){ loading=lazy }
  <figcaption>Left: the robust objective in $x$ for each geometry, with the
  nominal loss dashed and the minimizers marked. Middle: the
  decisions against the radius. Right: the worst-case distribution
  of each ball at its own decision, over the nominal.</figcaption>
</figure>

## The figure

Rendered from this script when the site is built, so it always shows the output of the current code:

<figure class="figure" markdown="span">
  ![Robust decision against the radius under four ambiguity-set geometries](../assets/figures/03_robust_decision_across_ambiguity_sets-light.png#only-light){ loading=lazy }
  ![Robust decision against the radius under four ambiguity-set geometries](../assets/figures/03_robust_decision_across_ambiguity_sets-dark.png#only-dark){ loading=lazy }
  <figcaption>Every geometry leaves the empirical-risk decision at radius zero; where it goes, and how fast, is the divergence's doing: the $\varphi$-divergences head for the outlier and the Wasserstein ball heads the other way.</figcaption>
</figure>

## Source

```py title="examples/03_robust_decision_across_ambiguity_sets.py"
--8<-- "examples/03_robust_decision_across_ambiguity_sets.py"
```
