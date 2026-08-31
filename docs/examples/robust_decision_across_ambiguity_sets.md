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

As $\rho$ grows, each robust decision $x^\star$ is dragged toward the
outlier, but at different rates, and the ordering is a property of the
divergence rather than of the solver:

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
  from $3$ to $10$.

That last distinction is the one I find easiest to forget. The
$\phi$-divergence family is blind to how far apart the scenarios are; it
only sees probabilities. Wasserstein is not.

## A note on cost

Every point on every curve is a nested solve — an outer
[`MinimaxSolver`](../api/dro/minimax_solver.md) whose objective internally
runs an ambiguity set's own dual solve on each iteration. Iteration counts
multiply, so the budgets at the top of the script are cut down hard from the
defaults, and each set gets its own step size.

!!! warning "The $\chi^2$ step size is not arbitrary"

    Its conjugate $\phi^\ast(s) = s + s^2/4$ grows faster than KL's
    `logsumexp`, and an aggressive step size makes the joint dual solve
    diverge to `NaN` silently rather than loudly. KL tolerates $0.1$ here;
    $\chi^2$ gets $0.05$ for a reason.

Also worth knowing, since it looks like it should not work: the outer
gradient flows correctly through `worst_case_expectation` even though the
inner dual variable is solved for under `detach`. That is the envelope
theorem — at the inner optimum the gradient contribution through the dual
variable vanishes, so only the final re-evaluation needs to stay attached to
the graph.

## Source

```py title="examples/03_robust_decision_across_ambiguity_sets.py"
--8<-- "examples/03_robust_decision_across_ambiguity_sets.py"
```
