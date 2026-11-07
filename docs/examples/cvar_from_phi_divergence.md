# Recovering CVaR from a phi-divergence ball

Conditional value-at-risk is usually introduced as a risk measure: the
average loss over the worst $\alpha$-fraction of outcomes. It is less often
presented as what it also is, a distributionally robust objective. This
example takes that second view seriously. It builds CVaR out of the general
phi-divergence machinery in Optora, with no CVaR-specific code and no new
solver, and then checks the result against CVaR computed straight from its
definition.

Along the way it tests a shortcut that is easy to stumble on: replacing the
CVaR set with a total-variation ball of radius $(1-\alpha)/\alpha$. That
shortcut is wrong, and seeing exactly how it fails leads to the correct
relationship between CVaR and total variation.

## CVaR as a worst case

Let $p$ be a nominal distribution over $n$ scenarios with losses
$\ell_1, \dots, \ell_n$. At tail probability $\alpha \in (0, 1]$, CVaR is
the mean of the upper $\alpha$-tail of the loss. Rockafellar and Uryasev
(2002) showed that it has two equivalent forms:

$$
\mathrm{CVaR}_\alpha(\ell)
= \sup_{q:\; q_i \,\le\, p_i/\alpha} \mathbb{E}_q[\ell]
= \min_{\lambda \in \mathbb{R}} \; \lambda
  + \frac{1}{\alpha}\, \mathbb{E}_p\big[(\ell - \lambda)_+\big].
$$

The first is a worst case. The adversary may reweight the scenarios, but no
likelihood ratio $q_i / p_i$ may exceed $1/\alpha$. The best it can do is
raise the ratio of every scenario in the upper tail to that cap and set the
rest to zero. The second form is a one-dimensional convex program whose
minimizer is the value-at-risk.

## The generator that turns a phi-ball into CVaR

A phi-divergence ball is $\{q : D_\phi(q \,\|\, p) \le \rho\}$ with
$D_\phi(q \,\|\, p) = \sum_i p_i \, \phi(q_i / p_i)$. Ben-Tal et al. (2013)
derived its dual,

$$
\sup_{q:\, D_\phi(q \,\|\, p) \le \rho} \mathbb{E}_q[\ell]
= \inf_{\eta > 0,\, \lambda \in \mathbb{R}}
  \eta \rho + \lambda + \eta\, \mathbb{E}_p\!\left[
  \phi^*\!\left(\frac{\ell - \lambda}{\eta}\right)\right],
$$

which is the formula `PhiAmbiguitySet` solves. To recover CVaR, choose the
indicator generator

$$
\phi_\alpha(t) =
\begin{cases}
0, & 0 \le t \le 1/\alpha, \\
+\infty, & \text{otherwise}.
\end{cases}
$$

The divergence $D_{\phi_\alpha}(q \,\|\, p)$ is zero when every likelihood
ratio stays below $1/\alpha$ and infinite otherwise. The ball is therefore
the CVaR set for every radius $\rho \ge 0$. The conjugate of this generator
is $\phi_\alpha^*(s) = (s)_+ / \alpha$, which is positively homogeneous, so
$\eta\, \phi_\alpha^*((\ell - \lambda)/\eta) = (\ell - \lambda)_+/\alpha$.
At $\rho = 0$ the multiplier $\eta$ cancels completely, and the dual becomes
the Rockafellar-Uryasev formula above. The third panel of the figure below
shows this: the dual objective drawn at $\eta = 1$ (lines) and
$\eta = 10$ (circles) is the same curve, and its minimum, marked with a
diamond, sits at the CVaR level.

In code, the conjugate is already in the library as `cvar_conjugate`, and
the generator is three lines:

```py
def cvar_generator(ratio, alpha):
    return torch.zeros_like(ratio).masked_fill(ratio > 1.0 / alpha, torch.inf)

ambiguity_set = PhiAmbiguitySet(
    nominal,
    divergence=PhiDivergence(functools.partial(cvar_generator, alpha=alpha)),
    radius=0.0,
    phi_conjugate=functools.partial(cvar_conjugate, alpha=alpha),
)
for step_size in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
    ambiguity_set.dual_solver = GradientDescent(step_size, max_iter=1500, tol=1e-12)
    value = ambiguity_set.worst_case_expectation(loss.expand(len(alphas), -1))
```

Here `alphas` holds 25 levels between $1$ and $0.01$ and
`alpha = alphas.unsqueeze(-1)`, so every level is solved in one batched
call. The loop over step sizes is explained in the next section.

!!! note "A zero radius is solved, not short-circuited"

    Most ambiguity sets in Optora return the nominal expectation at the
    float radius `0.0` without solving anything. That shortcut is correct
    for a generator that vanishes only at $t = 1$, because the ball of
    radius zero then contains only the nominal. The indicator generator
    vanishes on the whole interval $[0, 1/\alpha]$, so its ball of radius
    zero is the full CVaR set. `PhiAmbiguitySet` with a `dual_solver`
    therefore solves its dual at every radius, zero included; only its
    exact tilt path, which needs a differentiable conjugate, keeps the
    shortcut.

## Solving a dual that is not smooth

The Rockafellar-Uryasev objective is piecewise linear in $\lambda$, with a
kink at every scenario loss. Gradient descent with a fixed step size cannot
settle exactly at a kink. It oscillates around the minimizer, and the error
is proportional to the step. The script deals with this in the standard
way for subgradient methods: it shrinks the step in stages, from $10^{-2}$
to $10^{-6}$, and each stage starts from the point where the previous one
stopped (the set keeps that warm start between calls).

Each stage also returns an upper bound on CVaR, by weak duality, since any
feasible $(\eta, \lambda)$ gives a dual value at least as large as the
primal optimum. The middle panel of the figure shows the relative gap
closing from above, about one order of magnitude per stage. After the last
stage the largest relative error over all 25 levels is about
$2 \times 10^{-7}$, and the script checks that it stays below $10^{-6}$.

## Checking against the definition

The reference value is CVaR computed directly from its definition as a tail
mean (Acerbi and Tasche 2002). Sort the scenarios by loss, take each one
with its full nominal mass from the top until the cumulative mass reaches
$\alpha$, give the boundary scenario whatever mass is left, and divide by
$\alpha$. The nominal is a uniform draw from the simplex over 400
log-normal losses, so the boundary scenario is split at almost every level
and no round number hides an off-by-one error.

The script compares three values:

| Route | Code | Agreement with the tail mean |
| --- | --- | --- |
| Phi-divergence dual | `PhiAmbiguitySet` + `cvar_conjugate` + `GradientDescent` | about $10^{-7}$ relative, from above |
| Closed form | `CVaRAmbiguitySet` | machine precision |
| Definition | tail mean, written out in the script | reference |

It also reads the worst-case distribution $q^\star$ from the closed form,
as the gradient of the value with respect to the loss (Danskin's theorem),
and checks two things: no likelihood ratio exceeds $1/\alpha$, and the
total-variation distance of $q^\star$ from $p$ is at most $1 - \alpha$.
That second number is the subject of the next section.

## The total-variation shortcut, and why it fails

The number $(1-\alpha)/\alpha$ does appear in the CVaR set. Rewriting the
constraint $q_i \le p_i/\alpha$ as

$$
\frac{q_i - p_i}{p_i} \le \frac{1 - \alpha}{\alpha}
$$

shows that it bounds how far each likelihood ratio may rise above one. This
is a pointwise constraint, a bound in the sup norm on $q/p - 1$. Total
variation, $\tfrac{1}{2}\sum_i |q_i - p_i|$, limits something different:
the total ($L^1$) amount of mass that moves. Using one budget in place of
the other gives the wrong set. For $\alpha \le 1/2$ the budget
$(1-\alpha)/\alpha$ is at least one, the total-variation ball covers the
whole simplex, and its worst case is simply the largest loss. The script
confirms this at every such level.

The correct link is an identity. A total-variation adversary with budget
$\varepsilon$ removes mass $\varepsilon$ from the cheapest scenarios and
places all of it on the single most expensive one. What remains in place
is the upper $(1-\varepsilon)$-tail of $p$, so

$$
\sup_{q:\, \mathrm{TV}(q, p) \le \varepsilon} \mathbb{E}_q[\ell]
= \varepsilon \max_i \ell_i
  + (1 - \varepsilon)\, \mathrm{CVaR}_{1-\varepsilon}(\ell),
\qquad 0 \le \varepsilon \le 1.
$$

Setting $\varepsilon = 1 - \alpha$ gives

$$
\sup_{q:\, \mathrm{TV}(q, p) \le 1-\alpha} \mathbb{E}_q[\ell]
= \alpha\, \mathrm{CVaR}_\alpha(\ell) + (1 - \alpha) \max_i \ell_i .
$$

So the right radius is $1 - \alpha$, and even then the total-variation
worst case is not CVaR but a mixture of CVaR and the maximum loss. The
script verifies this identity to machine precision and then inverts it,
recovering CVaR from `TotalVariationAmbiguitySet` as
$\big(\mathrm{TV}_{1-\alpha} - (1-\alpha)\max_i \ell_i\big) / \alpha$, with
a relative error near $10^{-13}$.

There is also a geometric reason the radius $1 - \alpha$ is natural. If
$q_i \le p_i / \alpha$, then wherever $q_i \ge p_i$ we have
$q_i - p_i \le (1 - \alpha) q_i$, and summing over those scenarios gives
$\mathrm{TV}(q, p) \le 1 - \alpha$. The CVaR set always lies inside the
total-variation ball of radius $1 - \alpha$. It is a strictly smaller set,
which is why its worst case is smaller.

## Watching the two sets

The first animation draws both sets on the probability simplex for three
scenarios as $\alpha$ decreases from one. The filled region is the CVaR
set, the solid outline is the total-variation ball of radius $1 - \alpha$,
and the dashed outline is the ball of radius $(1-\alpha)/\alpha$. The CVaR
set never leaves the solid ball. The dashed ball grows much faster, covers
the whole triangle once $\alpha$ falls below about one half, and its worst
case (the hollow diamond) is pinned to the vertex with the largest loss
from then on. On the right, the hollow circles are computed from the CVaR
curve alone as $\alpha\,\mathrm{CVaR}_\alpha + (1-\alpha)\max \ell$, and
they fall exactly on the total-variation curve.

<figure class="figure" markdown="span">
  ![The CVaR set inside the total-variation ball of radius one minus alpha on the probability simplex, as alpha decreases](../assets/figures/cvar-tv-simplex-light.gif#only-light){ loading=lazy }
  ![The CVaR set inside the total-variation ball of radius one minus alpha on the probability simplex, as alpha decreases](../assets/figures/cvar-tv-simplex-dark.gif#only-dark){ loading=lazy }
  <figcaption>Left: the CVaR set (filled), the total-variation ball of
  radius $1-\alpha$ (solid) and of radius $(1-\alpha)/\alpha$ (dashed),
  with the worst case of each. Right: the three worst-case values over
  $\alpha$; the circles are $\alpha\,\mathrm{CVaR}_\alpha +
  (1-\alpha)\max\ell$.</figcaption>
</figure>

The second animation shows the same comparison as likelihood ratios on the
twelve-scenario instance used across this site. CVaR raises every scenario
in the upper tail to the same ratio, $1/\alpha$ (the dashed line), and sets
the rest to zero. The total-variation adversary with the same reach leaves
the tail at ratio one and stacks all the removed mass on one scenario,
whose ratio climbs far past the CVaR cap. Neither moves more than
$1 - \alpha$ of mass; they spend it in very different places.

<figure class="figure" markdown="span">
  ![Likelihood ratios of the CVaR and total-variation worst cases for each scenario as alpha decreases](../assets/figures/cvar-tv-ratio-light.gif#only-light){ loading=lazy }
  ![Likelihood ratios of the CVaR and total-variation worst cases for each scenario as alpha decreases](../assets/figures/cvar-tv-ratio-dark.gif#only-dark){ loading=lazy }
  <figcaption>Likelihood ratio $q^\star_i / p_i$ of each scenario, sorted
  by loss, on a log scale. Left: CVaR caps every ratio at $1/\alpha$.
  Right: the total-variation ball of radius $1-\alpha$ puts its whole
  surplus on the worst scenario.</figcaption>
</figure>

## The figure

Rendered from this script when the site is built, so it always shows the
output of the current code:

<figure class="figure" markdown="span">
  ![CVaR from three routes against the total-variation shortcut, the closing dual gap, and the dual objective at two values of eta](../assets/figures/10_cvar_from_phi_divergence-light.png#only-light){ loading=lazy }
  ![CVaR from three routes against the total-variation shortcut, the closing dual gap, and the dual objective at two values of eta](../assets/figures/10_cvar_from_phi_divergence-dark.png#only-dark){ loading=lazy }
  <figcaption>Left: the tail mean, `CVaRAmbiguitySet` and the
  phi-divergence dual coincide, while both total-variation balls sit well
  above them. Middle: the relative gap of the dual after each step-size
  stage. Right: the dual objective in $\lambda$ for three levels at
  $\eta = 1$ (lines) and $\eta = 10$ (circles); the diamond marks the
  minimum, at the CVaR level.</figcaption>
</figure>

## What to take away

- CVaR is the phi-divergence ball of the indicator generator
  $\phi_\alpha$, and `PhiAmbiguitySet` with `cvar_conjugate` computes it
  with the general dual and no new solver.
- The dual is piecewise linear, so a fixed step size only gets close. A
  short schedule of shrinking, warm-started steps reaches $10^{-7}$, and
  every intermediate value is a valid upper bound.
- A total-variation ball of radius $(1-\alpha)/\alpha$ is not the CVaR
  set. The radius $1-\alpha$ is the right one, and the resulting worst case
  equals $\alpha\,\mathrm{CVaR}_\alpha + (1-\alpha)\max\ell$, from which
  CVaR can be recovered exactly.
- For real work, use `CVaRAmbiguitySet`. It gives the same value in closed
  form, exactly and in one pass. The point of this example is that the
  general machinery agrees with it.

## References

- Aharon Ben-Tal, Dick den Hertog, Anja De Waegenaere, Bertrand Melenberg
  and Gijs Rennen, "Robust Solutions of Optimization Problems Affected by
  Uncertain Probabilities", *Management Science* 59(2), 2013.
- R. Tyrrell Rockafellar and Stanislav Uryasev, "Conditional Value-at-Risk
  for General Loss Distributions", *Journal of Banking and Finance* 26(7),
  2002.
- Carlo Acerbi and Dirk Tasche, "On the Coherence of Expected Shortfall",
  *Journal of Banking and Finance* 26(7), 2002.

## Source

```py title="examples/10_cvar_from_phi_divergence.py"
--8<-- "examples/10_cvar_from_phi_divergence.py"
```
