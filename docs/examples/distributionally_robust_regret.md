# Robust regret against robust loss

Distributionally robust optimization protects a decision against every
distribution in a ball around the sample. The protection has a price. The
adversary is paid for every unit of loss it can produce, including loss
that no decision could have avoided, and the robust decision retreats
accordingly. Fiechtner and Blanchet (2025) propose to charge the adversary
only for the loss that a better decision would have avoided: minimize the
worst-case *regret* rather than the worst-case loss. They prove that, under
smoothness, this leaves the empirical-risk decision unchanged to first
order in the radius, while ordinary Wasserstein DRO moves it by an amount
proportional to the radius.

This example builds their distributionally robust regret optimization
(DRRO) on top of `WassersteinAmbiguitySet` and `MinimaxSolver`, measures
both claims, and compares the outcome with Wasserstein DRO on three losses,
one of which is the paper's own newsvendor.

## Two kinds of regret

Write $\ell(\theta, \xi)$ for the loss of a decision $\theta$ when the
uncertain quantity takes the value $\xi$, and $P$ for a distribution of
$\xi$. The regret of $\theta$ under $P$ is its expected loss in excess of
the best decision for that distribution,

$$
R(\theta, P) = \mathbb{E}_P[\ell(\theta, \xi)]
  - \min_{\beta} \mathbb{E}_P[\ell(\beta, \xi)] .
$$

The benchmark $\beta$ is chosen knowing $P$ but before $\xi$ is drawn, which
the paper calls *ex-ante* regret. DRRO minimizes its worst case over a
Wasserstein ball $B_\delta$ around the empirical distribution $\hat P$:

$$
\min_{\theta} \; \sup_{P \in B_\delta} R(\theta, P) .
$$

There is a simpler notion, *ex-post* regret, where the benchmark is chosen
after seeing $\xi$: the per-scenario quantity
$\ell(\theta, \xi) - \min_\beta \ell(\beta, \xi)$. The hindsight term does
not depend on $\theta$, so this is the original loss shifted scenario by
scenario, and the shifted loss drops into `MinimaxSolver` unchanged. The
script tests both, and the difference between them turns out to matter.

!!! note "Which regret the paper means"

    The theorems below are about ex-ante regret. Ex-post regret is
    included because it is the natural first thing to try, and because for
    two of the three losses it changes nothing at all: when every scenario
    can be fit exactly, the hindsight term is zero and ex-post DRRO is
    Wasserstein DRO under another name. The script asserts that.

## Ex-ante regret as a worst-case expectation

The inner minimum over $\beta$ is a minimum of linear functionals of $P$,
so its negative is a supremum, and the two suprema can be exchanged:

$$
\sup_{P \in B_\delta} R(\theta, P)
= \sup_{\beta} \; \sup_{P \in B_\delta}
  \mathbb{E}_P\big[\ell(\theta, \xi) - \ell(\beta, \xi)\big] .
$$

The inner supremum is now the worst-case expectation of an ordinary
per-scenario loss, the difference $\ell(\theta, \cdot) - \ell(\beta, \cdot)$
for a fixed pair of decisions. That is exactly what
`WassersteinAmbiguitySet.worst_case_expectation` computes, so the only new
ingredient is the outer maximization over the benchmark. This is the
representation the paper works with (its Section 3 writes
$R_\delta(\theta) = \sup_\beta F_\delta(\theta, \beta)$), and it is why no
new solver is needed.

In the script the "wrapper" is therefore two lines: subtract the loss of
the benchmark from the loss of the decision, and hand the difference to the
ambiguity set. The outer maximization is where the work is, and the paper
is candid that in general it is hard (evaluating the regret exactly is
NP-hard for max-affine losses). Here the decision is a scalar, which keeps
it tractable.

## The problems

Forty demands are drawn from a log-normal distribution and given equal
weight. Three losses are studied:

| Loss | Form | Why it is here |
| --- | --- | --- |
| quadratic | $(\theta - \xi)^2$ | The paper's Theorem 4: the empirical-risk decision is DRRO-optimal at every radius. |
| LINEX (Varian 1975) | $e^{a(\theta - \xi)} - a(\theta - \xi) - 1$ | A smooth, asymmetric loss outside the quadratic family, for Theorem 3. |
| newsvendor | $b\theta - s\min(\theta, \xi)$ | The paper's own example. Piecewise linear, so the smoothness assumption fails. |

The estimation losses use the ground cost $|\xi - \xi'|$. Under it the
set of means the adversary can reach is the interval
$[\bar\xi - \delta, \bar\xi + \delta]$, symmetric about the sample mean,
which is what makes the quadratic result exact. The newsvendor uses
$(\xi - \xi')^2$, which makes many short moves cheaper than one long one.
With the absolute cost the newsvendor adversary's best moves shuffle mass
among low-demand scenarios, which raises the worst-case loss at the full
rate without ever crossing the order quantity, and no decision changes
until the radius is large; that is a true property of a discrete support,
but it hides what the comparison is about.

## Solving each policy

Every decision is a scalar and every objective is convex in it, so each
outer problem is solved by bisection on the sign of its derivative. This is
exact to floating-point precision, including at a kink, which matters
because the DRRO objective has one exactly where its minimum is. The
derivative of a worst-case expectation is the expected derivative of the
loss under the worst-case distribution, and that distribution is the
gradient of the worst case with respect to the loss (Danskin's theorem), so
`torch.autograd` returns it together with the value.

The benchmark maximization starts from the best candidate on either side of
$\theta$ in a small grid and then alternates best responses: the adversary
answers $\beta$ with its worst-case distribution, and $\beta$ answers with
the empirical-risk minimizer under that distribution. Each round can only
raise the regret. For the newsvendor the candidates are the scenarios
themselves, which is exact for a piecewise-linear loss, and no rounds are
needed. Every inner solution is verified against a dense benchmark grid,
and the script asserts the gap is below $10^{-9}$.

Wasserstein DRO and ex-post DRRO are ordinary `MinimaxProblem`s, with the
loss or the shifted loss as `loss_fn`. The script solves them by the same
bisection for a like-for-like comparison and cross-checks against
`MinimaxSolver` at one radius; the two agree to $1.3 \times 10^{-2}$,
which is the oscillation a fixed step of $10^{-2}$ leaves around the kink
of a piecewise-linear objective. Ex-ante DRRO is not a `MinimaxProblem`,
because the supremum over the benchmark sits outside the worst-case
expectation rather than inside a per-scenario loss.

## The quadratic case: pinned to the empirical decision

For the quadratic loss, the regret under $P$ is
$(\theta - \mathbb{E}_P[\xi])^2$: it depends on $P$ only through its mean.
The robust regret is therefore the squared distance from $\theta$ to the
farther end of the reachable-mean interval, and its minimizer is the
midpoint of that interval. With the absolute ground cost the interval is
symmetric about the sample mean until one of its ends reaches the edge of
the support, so the DRRO decision *is* the empirical-risk decision. The
script finds the two agree to the last bit at every radius, and that the
robust regret is exactly $\delta^2$, as the paper's constant
$\kappa$ predicts for this loss. Wasserstein DRO, on the same ball, moves
by exactly $\delta$.

The animation shows why. The robust regret is the larger of two parabolas,
one for the adversary that pushes the mean up and one for the adversary
that pushes it down, and the two cross exactly at the sample mean. Raising
the radius lifts the crossing point but does not move it. The worst-case
loss has no such symmetry: its adversary piles mass on whichever end of the
support is farther from $\theta$, and the minimizer drifts toward the
middle of the support.

<figure class="figure" markdown="span">
  ![The worst-case loss and the robust regret of a quadratic loss as functions of the decision, with their adversaries and minimizers, as the radius grows](../assets/figures/regret-landscape-light.gif#only-light){ loading=lazy }
  ![The worst-case loss and the robust regret of a quadratic loss as functions of the decision, with their adversaries and minimizers, as the radius grows](../assets/figures/regret-landscape-dark.gif#only-dark){ loading=lazy }
  <figcaption>Left: the nominal and the adversarial distributions at each
  policy's own decision. Middle: the nominal loss, the worst-case loss and
  the robust regret in $\theta$; the pale curves are the two regret
  branches. Right: the decisions as the radius grows.</figcaption>
</figure>

## A smooth loss: first order gone, second order left

The LINEX loss is smooth and strictly convex but not quadratic, so Theorem
3 is the relevant result: the DRRO decision is $\theta_0 + o(\delta)$ and
its value is $\kappa\delta^2 + o(\delta^2)$, against $O(\delta)$ for both
under Wasserstein DRO. The script fits log-log slopes on the five smallest
radii, $10^{-3}$ to $10^{-2}$:

| Quantity | Wasserstein DRO | ex-ante DRRO |
| --- | --- | --- |
| $\lvert\theta_\delta - \theta_{\mathrm{ERM}}\rvert$ | slope $0.98$ | slope $2.52$ |
| value above the empirical risk | slope $0.95$ (price of robustness) | slope $1.93$ (robust regret) |

At $\delta = 10^{-2}$ the DRO decision has moved by $2.0 \times 10^{-2}$
and the DRRO decision by $6.7 \times 10^{-4}$, thirty times less. The
slopes at the two largest radii bend away from these values because the
pair of scenarios the adversary trades between changes there and the
constant in front of $\delta$ changes with it; the theorems are statements
about $\delta \to 0$.

## The newsvendor: which way each policy moves

The paper introduces DRRO with a newsvendor and reports, at one radius,
that DRO orders less than the empirical-risk policy while DRRO orders
more. The same ordering appears here across the whole sweep once the
radius is above $10^{-2}$: DRO orders less, both regret policies order
more, and from $3 \times 10^{-2}$ on the ex-ante policy is the one that
moves least. At $\delta = 0.1$ the order quantities relative to the
empirical-risk policy are $-0.23$ (DRO), $+0.13$ (ex-post) and
$+0.06$ (ex-ante), on an empirical order of $1.22$.

The ex-post wrapper has a concrete meaning for this loss. Its hindsight
term is $-(s - b)\,\xi$, the profit of ordering exactly the demand, and
subtracting it turns the profit form of the loss into the classical cost
form $b(\theta - \xi)_+ + (s - b)(\xi - \theta)_+$. The profit-form
adversary is rewarded for low demand on its own, whichever quantity was
ordered, so DRO orders less; the cost-form adversary is rewarded only for
demand on the wrong side of the order, so it attacks both sides and the
decision moves the other way.

Two things differ from the smooth cases. The decisions move in steps,
because the minimizer of a piecewise-linear objective on a finite sample
is always one of the scenarios. And at small radii the ex-ante decision
wanders slightly below the empirical one before settling above it; with a
piecewise-linear loss the assumptions of Theorem 3 do not hold, and the
paper notes that the regret policy "can become more or less aggressive as
the radius grows, depending on the loss profile".

<figure class="figure" markdown="span">
  ![The adversarial demand distribution of each newsvendor policy and the four order quantities as the radius grows](../assets/figures/regret-newsvendor-light.gif#only-light){ loading=lazy }
  ![The adversarial demand distribution of each newsvendor policy and the four order quantities as the radius grows](../assets/figures/regret-newsvendor-dark.gif#only-dark){ loading=lazy }
  <figcaption>Left: the nominal demand and the worst-case demand each
  policy faces at its own order quantity (dashed). Right: order quantities
  against the radius; the DRO policy steps down, the regret policies step
  up.</figcaption>
</figure>

## The figure

Rendered from this script when the site is built, so it always shows the
output of the current code:

<figure class="figure" markdown="span">
  ![Newsvendor order quantities, decision deviations and values of Wasserstein DRO against distributionally robust regret, over the radius](../assets/figures/11_distributionally_robust_regret-light.png#only-light){ loading=lazy }
  ![Newsvendor order quantities, decision deviations and values of Wasserstein DRO against distributionally robust regret, over the radius](../assets/figures/11_distributionally_robust_regret-dark.png#only-dark){ loading=lazy }
  <figcaption>Left: newsvendor order quantities. Middle: distance of each
  decision from the empirical-risk decision, with reference slopes one and
  two; the quadratic DRRO decision is not drawn because its distance is
  zero. Right: the price of robustness of Wasserstein DRO against the
  robust regret of DRRO.</figcaption>
</figure>

## What to take away

- Ex-ante robust regret is $\sup_\beta$ of a worst-case expectation of a
  loss difference, so `WassersteinAmbiguitySet` evaluates it with a
  two-line wrapper and no new solver; the maximization over the benchmark
  is the part that costs effort.
- Ex-post regret is a per-scenario shift of the loss and plugs into
  `MinimaxSolver` as is. For losses whose scenarios can be fit exactly it
  is the original problem; for the newsvendor it is the cost form.
- On a quadratic loss the DRRO decision is the empirical-risk decision at
  every radius, to the last bit. On a smooth loss it moves at second order
  where Wasserstein DRO moves at first order.
- On the paper's newsvendor the regret policies order more and DRO orders
  less, and the ex-ante policy stays closest to the empirical decision.
- Optora's Wasserstein ball reweights the sample rather than moving its
  points, so decisions on a piecewise-linear loss change in steps. The
  statements of the theorems carry over; the staircase is the price of a
  finite support.

## References

- Lukas-Benedikt Fiechtner and Jose Blanchet, "Wasserstein Distributionally
  Robust Regret Optimization", arXiv:2504.10796, 2025.
- Hal R. Varian, "A Bayesian Approach to Real Estate Assessment", in
  *Studies in Bayesian Econometrics and Statistics in Honor of Leonard J.
  Savage*, North-Holland, 1975.
- Peyman Mohajerin Esfahani and Daniel Kuhn, "Data-Driven Distributionally
  Robust Optimization Using the Wasserstein Metric", *Mathematical
  Programming* 171, 2018.

## Source

```py title="examples/11_distributionally_robust_regret.py"
--8<-- "examples/11_distributionally_robust_regret.py"
```
