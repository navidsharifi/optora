# Worst-case distribution shift

The [`AmbiguitySet`](../api/core/dro_base.md) contract is deliberately
narrow: `worst_case_expectation` hands back the scalar
$\mathbb{E}_{q^\star}[\mathrm{loss}]$ and nothing else. That is the right
API — the worst-case distribution $q^\star$ is an implementation detail of
each formulation, and for the dual-solved sets it never exists explicitly in
memory at all. But it does mean you never get to *look* at the adversary.

For total variation you can, because the solution is combinatorial rather
than variational. This example reconstructs $q^\star$ by hand and then plots
where the probability mass goes.

## The closed form being reconstructed

Inside a TV ball the adversary is solving a linear program with a mass
budget, and the optimal strategy is embarrassingly simple: sort the
scenarios by loss, then move mass from the cheapest scenarios into the
single most expensive one until the budget

$$
\tfrac{1}{2}\sum_i |q_i - \mathrm{nominal}_i| \le \rho
$$

runs out. Cheapest scenario first, drained completely before touching the
next. Optora implements exactly this, vectorized, in
[`TotalVariationAmbiguitySet`](../api/dro/phi_dro.md) — no iterative solver,
no step size to tune, exact.

The `worst_case_distribution` helper in the script mirrors that logic in
about ten lines. The subtle bit is `cumsum(rest) - rest`, which gives the
mass already consumed *before* each scenario; the more obvious
`cat([zeros(1), cumsum(rest)[:-1]])` is off by one when there is only a
single scenario to drain.

## Trust, but cross-check

A reconstruction that merely *looks* right is worthless, so the script
recomputes $\mathbb{E}_{q^\star}[\mathrm{loss}]$ from its own $q^\star$ and
compares it against the library's `worst_case_expectation` across forty
radii. It asserts agreement below $10^{-9}$. If the two ever drift apart,
either the reconstruction is wrong or the library is — and the assertion
tells you to go find out which.

The radius sweep stops at the total movable mass
$1 - \mathrm{nominal}_{\arg\max \mathrm{loss}}$, because past that point
everything is already piled onto the worst scenario and the plot has nothing
left to say.

## Reading the figure

The stackplot is the interesting artefact here. Each band is one scenario's
probability mass as a function of radius, and you watch the low-loss bands
collapse one at a time — never gradually and never simultaneously — while
the worst-loss band swells to absorb them. That staircase is the LP's
vertex-hopping made visible, and it is a much better intuition for "what
does robustness actually assume about the world" than any scalar curve.

## The staircase, frame by frame

The stackplot compresses forty radii into one picture. The animation
replays them: the left panel shows the nominal and the worst-case
distribution at the current radius, the middle one draws the stackplot
up to that radius, and the right one follows the worst-case value. The
value is piecewise linear in $\rho$ with one piece per drained scenario,
because moving a unit of mass from scenario $i$ to the worst one adds
$\ell_{\max} - \ell_i$, and the slope rises at every breakpoint as the
adversary runs out of cheap mass and starts on dearer scenarios.

<figure class="figure" markdown="span">
  ![The total-variation worst-case distribution, the stacked scenario masses and the worst-case value as the radius grows](../assets/figures/tv-shift-light.gif#only-light){ loading=lazy }
  ![The total-variation worst-case distribution, the stacked scenario masses and the worst-case value as the radius grows](../assets/figures/tv-shift-dark.gif#only-dark){ loading=lazy }
  <figcaption>Left: the nominal and the worst-case distribution, one shade per
  scenario. Middle: the stacked masses up to the current radius.
  Right: the worst-case value, linear between the breakpoints where
  a scenario runs dry.</figcaption>
</figure>

## The figure

Rendered from this script when the site is built, so it always shows the output of the current code:

<figure class="figure" markdown="span">
  ![Probability mass of every scenario inside the total-variation ball as the radius grows](../assets/figures/02_worst_case_distribution_shift-light.png#only-light){ loading=lazy }
  ![Probability mass of every scenario inside the total-variation ball as the radius grows](../assets/figures/02_worst_case_distribution_shift-dark.png#only-dark){ loading=lazy }
  <figcaption>Each band is one scenario's mass. The cheapest scenarios drain one at a time into the single worst one: the linear program hopping between vertices.</figcaption>
</figure>

## Source

```py title="examples/02_worst_case_distribution_shift.py"
--8<-- "examples/02_worst_case_distribution_shift.py"
```
