---
icon: lucide/sigma
---

# Mathematical formulations and design

This page holds the full mathematical background and design rationale for
`optora`'s distributionally robust optimization (DRO) core: the problem
`optora` solves, the convex duals underneath each ambiguity set, and the
design principles the public API follows. The root `README.md` stays a
short, code-first landing page on purpose — this page is where the depth
lives. For runnable code that exercises this math end to end, with plots,
see the `examples/` directory in the repository root (GitHub-only, not
part of the published package — see `examples/README.md`).

## Why distributionally robust optimization

If you minimize the average loss over your training data (empirical risk
minimization), you're implicitly trusting that your training data is a
faithful picture of the distribution you actually care about. That's often
a shaky assumption — you might have too few samples, the world might have
shifted since you collected the data, or someone might be actively trying
to fool your model. DRO's answer to this is refreshingly blunt: instead of
optimizing against your best guess at the distribution, optimize against
the *worst* distribution within some plausible neighborhood of it.

$$
\min_x \; \sup_{q \in \mathrm{ambiguity\_set}(\mathrm{nominal}, \mathrm{radius})} \mathbb{E}_q[\ell(x, \xi)]
$$

That neighborhood — the *ambiguity set* — is defined by capping some
statistical divergence $D$ between a candidate distribution $q$ and your
nominal (reference) distribution at a radius you choose:

$$
\mathrm{ambiguity\_set}(\mathrm{nominal}, \mathrm{radius}) = \{\, q : D(q \,\|\, \mathrm{nominal}) \le \mathrm{radius} \,\}
$$

Optora's whole architecture is basically this formula, typed out as code: a
`Divergence` is `D`, an `AmbiguitySet` bundles a `Divergence` with a
`nominal` distribution and a `radius`, and a `Solver` does the actual
numerical work of finding the worst case (or, one level up, solving the
full minimax problem over both the decision and the distribution).

## Core abstractions

| Contract | Location | Responsibility |
| --- | --- | --- |
| `Divergence` | `optora.core.divergence_base` | Callable `forward(p, q) -> Tensor` computing a nonnegative discrepancy between two distributions, zero exactly when `p == q`. |
| `AmbiguitySet` | `optora.core.dro_base` | Holds a `nominal` distribution, a `Divergence`, and a `radius`; requires `worst_case_expectation(loss) -> Tensor` from subclasses. |
| `Solver[ProblemT, ResultT]` | `optora.core.solver_base` | Generic numerical method with a single `solve(problem) -> result` method, so each algorithm defines its own problem/result dataclasses instead of a one-size-fits-all signature. |

## Formulations implemented today

Every ambiguity set below reduces its inner `sup` over candidate
distributions `q` to a tractable convex dual or an exact closed form, so
none of them needs to search over the full space of candidate
distributions directly.

### `KLAmbiguitySet`

The KL-ball. The inner supremum has a classic one-dimensional convex dual
(Hu and Hong, 2013; Ben-Tal et al., 2013):

$$
\sup_{q:\, D_{\mathrm{KL}}(q \,\|\, \mathrm{nominal}) \,\le\, \mathrm{radius}} \mathbb{E}_q[\mathrm{loss}]
= \inf_{\eta > 0} \; \eta \cdot \mathrm{radius}
    + \eta \log \mathbb{E}_{\mathrm{nominal}}\!\left[\exp\!\left(\frac{\mathrm{loss}}{\eta}\right)\right]
$$

Solved over `log(eta)` rather than `eta` itself, so the unconstrained
`GradientDescent` solver can't wander into `eta <= 0`. `radius == 0`
returns `E_nominal[loss]` exactly, skipping the numerical solve.

### Dual-solved sets: loss scale and saturation

`KLAmbiguitySet` and `ChiSquareAmbiguitySet` (and any other
`PhiAmbiguitySet`) share two guarantees, implemented once in
`optora.core.dro_base.DualAmbiguitySet`.

**The dual is solved on a standardized loss.** The set of candidate
distributions does not depend on `loss`, so for $a > 0$

$$
\sup_q \mathbb{E}_q[a\,\ell + b] = a \sup_q \mathbb{E}_q[\ell] + b .
$$

The dual's curvature grows with the squared loss spread, so a step size
tuned on an order-one loss diverges on a wide one (`eta` overflows and
the objective becomes `nan`). The dual is therefore minimized on
$(\ell - \min\ell)/(\max\ell - \min\ell)$ with a detached shift and scale.
For any fixed constants this is the same function of `loss`, so values
and gradients are unchanged; only the conditioning is.

!!! note "Dual-solver step sizes are relative to a unit-spread loss"
    Because the solver iterates on the standardized loss, a step size
    means the same thing at every loss scale: order one (up to about 2)
    for KL, and at most about 0.3 for the stiffer chi-square dual. A step
    tuned on a raw loss with spread $S$ should be multiplied by roughly
    $S$. `initial_log_eta` and `initial_lam` are still read in the units of
    the raw loss when given; left at `None` they start at `eta` equal to the
    loss spread (and `lam` equal to the minimum loss), the same standardized
    point at every scale.

**Saturation returns `max(loss)` exactly.** Once the radius reaches the
divergence of the distribution concentrated on the highest-loss scenarios,
that distribution lies in the set and the worst case is $\max\ell$. With
$P^\star$ the nominal mass on those scenarios, the thresholds are
$-\log P^\star$ for KL and $1/P^\star - 1$ for chi-square. Past them the
dual has no minimizer (its infimum is approached only as `eta` goes to
zero), so no solver converges; the sets return $\max\ell$ directly, using
device reductions only. The gradient there is the nominal restricted to
those scenarios and renormalized, which lies inside the set.

### `PhiAmbiguitySet`, `ChiSquareAmbiguitySet`, `TotalVariationAmbiguitySet`

The general phi-divergence-ball case, plus two named instances.
`PhiAmbiguitySet` generalizes the KL-DRO dual above to any phi-divergence
(Ben-Tal et al., 2013; Duchi, Glynn, and Namkoong, 2021; Duchi and
Namkoong, 2021), at the cost of a second dual variable `lam`:

$$
\sup_{q:\, D_\phi(q \,\|\, \mathrm{nominal}) \,\le\, \mathrm{radius}} \mathbb{E}_q[\mathrm{loss}]
= \inf_{\substack{\eta > 0 \\ \lambda}} \;
    \eta \cdot \mathrm{radius} + \lambda
    + \eta \, \mathbb{E}_{\mathrm{nominal}}\!\left[\phi^*\!\left(\frac{\mathrm{loss} - \lambda}{\eta}\right)\right]
$$

where `phi*` is `phi`'s convex conjugate. `ChiSquareAmbiguitySet` plugs in
the closed-form chi-square conjugate, smooth everywhere. Total
variation's conjugate is *not* smooth everywhere (it has a hard
boundary), so `TotalVariationAmbiguitySet` skips the dual entirely and
computes the worst case directly from a closed-form combinatorial
solution: sort the scenarios by loss and shift probability mass, from
the cheapest ones, onto the single worst-case scenario until the
total-variation budget is used up.

### `WassersteinAmbiguitySet`

The Wasserstein-ball case, for candidates sharing the nominal
distribution's support with a given pairwise ground cost. This also
reduces to a clean one-dimensional dual (Mohajerin Esfahani and Kuhn,
2018; Blanchet and Murthy, 2019; Gao and Kleywegt, 2022):

$$
\sup_{q:\, W_c(q, \mathrm{nominal}) \,\le\, \mathrm{radius}} \mathbb{E}_q[\mathrm{loss}]
= \inf_{\gamma \ge 0} \; \gamma \cdot \mathrm{radius}
    + \mathbb{E}_{\mathrm{nominal}}\!\left[\max_j \big(\mathrm{loss}_j - \gamma \cdot \mathrm{cost}(\cdot, j)\big)\right]
$$

`gamma`'s optimum can sit exactly at the boundary `gamma = 0` (once the
radius is generous enough to move all the mass to the worst scenario).
The dual is convex but *piecewise linear* in `gamma`, so a fixed-step
gradient method oscillates around its kink and never converges; its
derivative is nevertheless monotone, so `WassersteinAmbiguitySet` takes
no dual solver at all and instead bisects that derivative's sign change
inside a closed-form bracket, reaching the exact minimizer in a fixed,
dtype-determined number of steps. It uses a `SinkhornDivergence`
internally only as an approximate `contains(...)` membership check; the
worst-case expectation itself is solved exactly, not through the entropic
approximation.

### `MinimaxSolver`

Wires any of the ambiguity sets above together with an outer solver
(`GradientDescent` by default) to solve the *full* DRO problem, decision
variable and all:

$$
\min_x \; \sup_{q:\, \mathrm{divergence}(q, \mathrm{nominal}) \,\le\, \mathrm{radius}} \mathbb{E}_q[\mathrm{loss\_fn}(x)]
$$

Every ambiguity set above already turns the inner "sup over q" into
something differentiable in `x` (a dual objective, or an exact closed
form), so `MinimaxSolver` only has to minimize
`x -> ambiguity_set.worst_case_expectation(loss_fn(x))` — an ordinary
scalar objective. Gradients still flow correctly through `x` even though
each ambiguity set solves its own dual variable "under the hood," which
follows from the envelope theorem.

Every piece above is covered by pytest tests checked against known
closed-form results, independent grid-search cross-checks, convergence
limits, and monotonicity properties, and the whole package is
type-checked under mypy's strict mode.

## Design principles

- **Small, component-oriented public API.** No package-level workflow
  dispatchers (`minimize(...)`, `train(...)`) that hide the method being
  studied; researchers instantiate the exact algorithm they want.
- **Mathematical objects separated from numerical solvers.** A divergence,
  an ambiguity set, and a solver are independent, individually reusable
  components.
- **Every extension point is subclassable.** `Divergence`, `AmbiguitySet`,
  and `Solver` are ABCs, not a closed enumeration — Optora ships reference
  implementations, not the full space of methods.
- **PyTorch-native and accelerator-friendly.** Tensor dtype and device are
  preserved throughout; the numerical core avoids unnecessary host
  synchronizations and scalar round-trips so it stays friendly to future
  GPU-heavy and batched workloads.
- **Tested against known mathematics.** Every solver and divergence is
  checked against closed-form solutions, convergence limits, or established
  monotonicity properties, not only smoke tests.

## Roadmap and future horizon

Optora's scope is deliberately narrow: ambiguity sets, divergences, and the
minimax solve that connects them. That core is complete end to end. What's
next is a set of extensions that only make sense once they're actually
needed, not day-one scope:

- A dedicated risk-functional layer, once more than one risk measure (CVaR,
  entropic risk, and so on) needs to sit on top of the worst-case objective.
- Jointly-learned, differentiable ambiguity sets, once the current static
  ones (fixed `nominal`, fixed `radius`) feel limiting.
- Causality-constrained DRO for sequential decision problems.

Further out: stochastic and differentiable-backend solvers, optimal
transport as a first-class object, and reinforcement learning under model
uncertainty. See `progress/architecture.md` in the repository for the full,
living build-order and extension map (not part of the published package).
