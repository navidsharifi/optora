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

```text
minimize_x   sup_{q in ambiguity_set(nominal, radius)}  E_q[ loss(x, xi) ]
```

That neighborhood — the *ambiguity set* — is defined by capping some
statistical divergence `D` between a candidate distribution `q` and your
nominal (reference) distribution at a radius you choose:

```text
ambiguity_set(nominal, radius) = { q : D(q || nominal) <= radius }
```

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

- **`KLAmbiguitySet`** — the KL-ball. The inner supremum has a classic
  one-dimensional convex dual (Hu and Hong, 2013; Ben-Tal et al., 2013):

  ```text
  sup_{q: D_KL(q || nominal) <= radius} E_q[loss]
      = inf_{eta > 0} eta * radius + eta * log E_nominal[exp(loss / eta)]
  ```

  Solved over `log(eta)` rather than `eta` itself, so the unconstrained
  `GradientDescent` solver can't wander into `eta <= 0`. `radius == 0`
  returns `E_nominal[loss]` exactly, skipping the numerical solve.

- **`PhiAmbiguitySet`, `ChiSquareAmbiguitySet`, `TotalVariationAmbiguitySet`**
  — the general phi-divergence-ball case, plus two named instances.
  `PhiAmbiguitySet` generalizes the KL-DRO dual above to any phi-divergence
  (Ben-Tal et al., 2013; Duchi, Glynn, and Namkoong, 2021; Duchi and
  Namkoong, 2021), at the cost of a second dual variable `lam`:

  ```text
  sup_{q: D_phi(q||nominal) <= radius} E_q[loss]
      = inf_{eta > 0, lam} eta * radius + lam + eta * E_nominal[phi*((loss - lam) / eta)]
  ```

  where `phi*` is `phi`'s convex conjugate. `ChiSquareAmbiguitySet` plugs in
  the closed-form chi-square conjugate, smooth everywhere. Total
  variation's conjugate is *not* smooth everywhere (it has a hard
  boundary), so `TotalVariationAmbiguitySet` skips the dual entirely and
  computes the worst case directly from a closed-form combinatorial
  solution: sort the scenarios by loss and shift probability mass, from
  the cheapest ones, onto the single worst-case scenario until the
  total-variation budget is used up.

- **`WassersteinAmbiguitySet`** — the Wasserstein-ball case, for candidates
  sharing the nominal distribution's support with a given pairwise ground
  cost. This also reduces to a clean one-dimensional dual (Mohajerin
  Esfahani and Kuhn, 2018; Blanchet and Murthy, 2019; Gao and Kleywegt,
  2022):

  ```text
  sup_{q: W_c(q, nominal) <= radius} E_q[loss]
      = inf_{gamma >= 0} gamma * radius + E_nominal[max_j (loss_j - gamma * cost(., j))]
  ```

  `gamma`'s optimum can sit exactly at the boundary `gamma = 0` (once the
  radius is generous enough to move all the mass to the worst scenario),
  so this one is reparameterized with a `clamp` instead of an exponential.
  It uses a `SinkhornDivergence` internally only as an approximate
  `contains(...)` membership check; the worst-case expectation itself is
  solved exactly, not through the entropic approximation.

- **`MinimaxSolver`** — wires any of the ambiguity sets above together with
  an outer solver (`GradientDescent` by default) to solve the *full* DRO
  problem, decision variable and all:

  ```text
  min_x sup_{q: divergence(q, nominal) <= radius} E_q[loss_fn(x)]
  ```

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
