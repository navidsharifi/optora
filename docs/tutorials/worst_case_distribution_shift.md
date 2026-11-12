# Worst-case distribution shift

The [`AmbiguitySet`](../api/core/dro_base.md) contract is deliberately
narrow. `worst_case_expectation` returns the scalar
$\mathbb{E}_{q^\star}[\mathrm{loss}]$ and not the worst-case distribution
$q^\star$. We think this is the right interface, because $q^\star$ is an
implementation detail of each formulation, and the sets solved through a
dual never hold it in memory. The distribution can still be recovered as
the gradient of the worst-case value with respect to the loss (Danskin's
theorem), which is how the animation below obtains it.

For total variation there is a more direct route, because the solution
is combinatorial rather than the result of an iterative solve. This
tutorial reconstructs $q^\star$ by hand and plots where the probability
mass goes.

## The closed form being reconstructed

Inside a TV ball the adversary solves a linear program with a mass
budget, and its optimal solution has a simple combinatorial structure
(Ben-Tal et al., 2013)[^btal]. Sort the scenarios by loss, then move
mass from the cheapest scenarios into the single most expensive one until
the budget

$$
\tfrac{1}{2}\sum_i |q_i - \mathrm{nominal}_i| \le \rho
$$

is used up. The cheapest scenario is drained completely before the next
one is touched. Optora implements this rule, vectorized, in
[`TotalVariationAmbiguitySet`](../api/dro/phi_dro.md). It is a sort
followed by a cumulative sum, so there is no iterative solver and no step
size to tune, and the result is exact.

The `worst_case_distribution` helper in the script reproduces the same
logic in about ten lines. The expression `cumsum(rest) - rest` gives the
mass that lies *before* each scenario in the sorted order. The shifted
form `cat([zeros(1), cumsum(rest)[:-1]])` computes the same values when
there is at least one scenario to drain. On a single-point support,
where `rest` is empty, it returns one entry instead of none.

## Cross-checking the reconstruction

A reconstruction that only looks plausible proves little, so the script
recomputes $\mathbb{E}_{q^\star}[\mathrm{loss}]$ from its own $q^\star$
and compares it with the library's `worst_case_expectation` at forty
radii. It asserts agreement below $10^{-9}$. If the two values ever
differ, either the reconstruction or the library is wrong, and the
failing assertion shows that one of them needs to be checked.

The radius sweep stops at the total movable mass
$1 - \mathrm{nominal}_{\arg\max \mathrm{loss}}$. Beyond that radius all the
mass already sits on the worst scenario and the distribution no longer
changes.

## Reading the figure

The stackplot is the main output. Each band is the probability mass of
one scenario as a function of the radius. The lowest-loss band shrinks
linearly to zero before the next band starts to shrink, while the band of
the worst scenario grows to absorb them. Each breakpoint of this staircase
is a change of the optimal basis of the linear program as the radius
varies. The picture shows directly which shift of the data the robust
objective guards against, which a scalar curve cannot show.

## The staircase, frame by frame

The stackplot compresses the sweep into one picture, and the animation
replays it. The left panel shows the nominal and the worst-case
distribution at the current radius, the middle panel draws the stackplot
up to that radius, and the right panel follows the worst-case value. The
value is piecewise linear in $\rho$, with one piece per drained scenario.
Moving a unit of mass from scenario $i$ to the worst one adds
$\ell_{\max} - \ell_i$, so the slope on each piece is the loss gap of the
scenario being drained. The slope therefore decreases at every
breakpoint, because the adversary has used up the cheapest mass and moves
on to scenarios with a higher loss.

<figure class="figure" markdown="span">
  ![The total-variation worst-case distribution, the stacked scenario masses and the worst-case value as the radius grows](../assets/figures/tv-shift-light.gif#only-light){ loading=lazy }
  ![The total-variation worst-case distribution, the stacked scenario masses and the worst-case value as the radius grows](../assets/figures/tv-shift-dark.gif#only-dark){ loading=lazy }
  <figcaption>The left panel shows the nominal and the worst-case
  distribution, with one shade per scenario. The middle panel stacks the
  scenario masses up to the current radius. The right panel shows the
  worst-case value, which is linear between the breakpoints where a
  scenario is fully drained.</figcaption>
</figure>

## The figure

The figure is rendered from this script when the site is built, so it always shows the output of the current code.

<figure class="figure" markdown="span">
  ![Probability mass of every scenario inside the total-variation ball as the radius grows](../assets/figures/02_worst_case_distribution_shift-light.png#only-light){ loading=lazy }
  ![Probability mass of every scenario inside the total-variation ball as the radius grows](../assets/figures/02_worst_case_distribution_shift-dark.png#only-dark){ loading=lazy }
  <figcaption>Each band is the mass of one scenario. The cheapest scenarios drain one at a time into the single worst one, and each breakpoint is a change of the optimal basis of the linear program.</figcaption>
</figure>

## Source

```py title="tutorials/02_worst_case_distribution_shift.py"
--8<-- "tutorials/02_worst_case_distribution_shift.py"
```

[^btal]:
    Aharon Ben-Tal, Dick den Hertog, Anja De Waegenaere, Bertrand Melenberg
    and Gijs Rennen, "Robust Solutions of Optimization Problems Affected by
    Uncertain Probabilities", *Management Science* 59(2), 341-357 (2013).
