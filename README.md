# Optora

**A small, composable, PyTorch-native library for distributionally robust optimization (DRO).**

Status: pre-alpha (`v0.0.3`) &middot; Python 3.10+ &middot; PyTorch &ge;2.2 &middot; MIT License

I started building Optora because most of the DRO code I ran into while
reading papers lived inside one-off experiment scripts, hard-wired to
whatever setup that particular paper used. What I wanted instead was a
small set of pieces — a divergence, an ambiguity set, a solver — that I
could actually reuse and recombine when trying to implement a method from
a different paper the following week. Optora is that attempt: a deliberately
narrow toolkit rather than a framework that tries to make decisions for
you. There's no `fit()`, no `train()`, no string like `method="wasserstein"`
that silently picks an algorithm on your behalf. You build the pieces you
need and wire them together yourself, which is exactly what you want when
you're trying to understand (and extend) the actual math.

Everything is written against PyTorch tensors, so gradients flow through
the whole pipeline and nothing here should get in the way of eventually
running on a GPU.

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

There are exactly three base contracts, all living in `optora.core`, and
everything else in the library is an implementation of one of them. If
you're implementing something new — a divergence from a paper you're
reading, a different solver — you subclass one of these three instead of
looking for some closed registry of "supported methods."

| Contract | Location | Responsibility |
| --- | --- | --- |
| `Divergence` | `optora.core.divergence_base` | Callable `__call__(p, q) -> Tensor` computing a nonnegative discrepancy between two distributions, zero exactly when `p == q`. |
| `AmbiguitySet` | `optora.core.dro_base` | Holds a `nominal` distribution, a `Divergence`, and a `radius`; requires `worst_case_expectation(loss) -> Tensor` from subclasses. |
| `Solver[ProblemT, ResultT]` | `optora.core.solver_base` | Generic numerical method with a single `solve(problem) -> result` method, so each algorithm defines its own problem/result dataclasses instead of a one-size-fits-all signature. |

## What is implemented today

The full pipeline — ambiguity sets, the divergences underneath them, the
solvers underneath those, and a minimax solver that ties everything
together — is implemented end to end. This is the whole planned "DRO
core"; what's left is future, deliberately deferred, extensions (more on
that below).

### Divergences (`optora.divergences`)

These are just the geometry: a way of measuring how far one distribution
is from another. Every ambiguity set below is built on top of one of these.

- `KLDivergence` — the Kullback-Leibler divergence `D_KL(p || q)`.
- `PhiDivergence` — the general f-divergence
  `D_phi(p || q) = sum_i q_i * phi(p_i / q_i)`, parameterized by whatever
  convex generator `phi` you hand it. This is the one to subclass if you
  want a divergence that isn't already here.
- `ChiSquareDivergence` and `TotalVariationDivergence` — two named
  `PhiDivergence` instances, with the standard chi-square and
  total-variation generators plugged in.
- `SinkhornDivergence` — the debiased entropic Wasserstein (Sinkhorn)
  divergence over a ground-cost matrix you supply, computed via Sinkhorn's
  fixed-point iteration. It's "debiased" because plain entropic optimal
  transport isn't zero when you compare a distribution to itself — we
  subtract off that self-transport bias so it actually satisfies the
  `Divergence` contract.

### Solvers (`optora.solvers`)

The numerical engines that everything else is built on.

- `GradientDescent` — plain fixed-step-size gradient descent on a
  differentiable objective, using `torch.autograd.grad` for the gradient at
  every step. It supports warm starts structurally: hand it a previous
  result's `point` as the next problem's `initial_point` and it picks up
  from there.
- `SaddlePointSolver` — primal-dual gradient ascent-descent for
  `min_x max_y objective(x, y)`, which is the shape of the general DRO
  minimax problem if you wanted to solve it directly over the raw candidate
  distribution instead of through a dual (see `MinimaxSolver` below for why
  we usually don't). It takes an optional projection step to keep the dual
  variable feasible — for instance, projected back onto an ambiguity set.

### DRO formulations (`optora.dro`)

This is where a divergence, a nominal distribution, and a radius actually
turn into something you can call `.worst_case_expectation(loss)` on. Every
one of these reduces the inner "sup over q" to something you can solve
directly, so no ambiguity set here needs to search over the full space of
candidate distributions.

- **`KLAmbiguitySet`** — the KL-ball. The inner supremum has a classic
  one-dimensional convex dual (Hu and Hong, 2013; Ben-Tal et al., 2013):

  ```text
  sup_{q: D_KL(q || nominal) <= radius} E_q[loss]
      = inf_{eta > 0} eta * radius + eta * log E_nominal[exp(loss / eta)]
  ```

  We solve this over `log(eta)` rather than `eta` itself, so the ordinary
  unconstrained `GradientDescent` solver can't accidentally wander into
  `eta <= 0`. When `radius == 0`, the ball collapses to a point and we just
  return `E_nominal[loss]` exactly, skipping the numerical solve entirely.

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
  the closed-form chi-square conjugate, which happens to be smooth
  everywhere, so the joint dual solve behaves nicely. Total variation's
  conjugate is *not* smooth everywhere (it has a hard boundary), which
  makes unconstrained gradient descent unreliable there — so
  `TotalVariationAmbiguitySet` skips the dual entirely and instead computes
  the worst case directly from a closed-form combinatorial solution:
  sort the scenarios by loss and shift probability mass, starting from the
  cheapest ones, onto the single worst-case scenario until you've used up
  your total-variation budget.

- **`WassersteinAmbiguitySet`** — the Wasserstein-ball case, for candidates
  sharing the nominal distribution's support with a given pairwise ground
  cost. This one also reduces to a clean one-dimensional dual (Mohajerin
  Esfahani and Kuhn, 2018; Blanchet and Murthy, 2019; Gao and Kleywegt,
  2022):

  ```text
  sup_{q: W_c(q, nominal) <= radius} E_q[loss]
      = inf_{gamma >= 0} gamma * radius + E_nominal[max_j (loss_j - gamma * cost(., j))]
  ```

  Unlike the KL and phi-divergence duals above, `gamma`'s optimum can sit
  exactly at the boundary `gamma = 0` (once the radius is generous enough
  to move all the mass to the worst scenario), so this one is
  reparameterized with a `clamp` instead of an exponential — that lets
  plain gradient descent actually land on the boundary rather than just
  creep toward it forever. It uses a `SinkhornDivergence` internally, but
  only as an approximate membership check (`contains(...)`); the worst-case
  expectation itself is solved exactly, not through the entropic
  approximation.

- **`MinimaxSolver`** — wires any of the ambiguity sets above together with
  an outer solver (a `GradientDescent` by default) to solve the *full* DRO
  problem, decision variable and all:

  ```text
  min_x sup_{q: divergence(q, nominal) <= radius} E_q[loss_fn(x)]
  ```

  The trick that makes this simple is that every ambiguity set above has
  already turned the inner "sup over q" into something differentiable in
  `x` (a dual objective, or an exact closed form), so `MinimaxSolver` really
  only has to minimize `x -> ambiguity_set.worst_case_expectation(loss_fn(x))`
  — an ordinary scalar objective — rather than reimplementing a generic
  primal-dual scheme over the whole distribution. Gradients still flow
  correctly through `x` even though each ambiguity set solves its own dual
  variable "under the hood," which follows from the envelope theorem.

Every piece above is covered by pytest tests checked against known
closed-form results, independent grid-search cross-checks, convergence
limits, and monotonicity properties (not just smoke tests), and the whole
package is type-checked under mypy's strict mode.

## Quickstart

Optora is pre-alpha and not yet published on PyPI; install it from a local
clone in editable mode:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,docs]"
```

Compute the worst-case expected loss over a KL-ball around a nominal
distribution:

```python
import torch
from optora.dro import KLAmbiguitySet

nominal = torch.tensor([0.25, 0.25, 0.25, 0.25])
loss = torch.tensor([0.0, 1.0, 2.0, 5.0])

ambiguity_set = KLAmbiguitySet(nominal=nominal, radius=0.1)
worst_case_loss = ambiguity_set.worst_case_expectation(loss)
```

Use a divergence as a standalone building block:

```python
import torch
from optora.divergences import ChiSquareDivergence, KLDivergence

p = torch.tensor([0.5, 0.3, 0.2])
q = torch.tensor([0.4, 0.4, 0.2])

kl_divergence = KLDivergence()
chi_square_divergence = ChiSquareDivergence()

kl_divergence(p, q), chi_square_divergence(p, q)
```

Solve an arbitrary differentiable objective directly with `GradientDescent`:

```python
import torch
from optora.solvers import GradientDescent, GradientDescentProblem

solver = GradientDescent(step_size=0.1, max_iter=500, tol=1e-8)
problem = GradientDescentProblem(
    objective=lambda x: (x - 3.0) ** 2,
    initial_point=torch.tensor(0.0),
)
result = solver.solve(problem)
```

Now put it all together: find the decision `x` that minimizes the
worst-case squared error against a KL-ball of plausible distributions over
four observed outcomes. This is `MinimaxSolver` doing the whole DRO
problem, not just the inner worst-case-expectation step:

```python
import torch
from optora.dro import KLAmbiguitySet, MinimaxProblem, MinimaxSolver

nominal = torch.tensor([0.25, 0.25, 0.25, 0.25])
outcomes = torch.tensor([1.0, 2.0, 3.0, 10.0])

ambiguity_set = KLAmbiguitySet(nominal=nominal, radius=0.1)
problem = MinimaxProblem(
    ambiguity_set=ambiguity_set,
    loss_fn=lambda x: (outcomes - x) ** 2,
    initial_point=torch.tensor(0.0),
)

result = MinimaxSolver().solve(problem)
robust_decision = result.point
```

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
minimax solve that connects them. That core is now done end to end — every
box in the pipeline above is implemented and tested — so I'm treating it as
a complete v0 rather than something with an "immediate build order" left to
finish.

What's next is a set of extensions that only make sense once you actually
need them, not day-one scope:

- A dedicated risk-functional layer, once more than one risk measure (CVaR,
  entropic risk, and so on) needs to sit on top of the worst-case objective.
- Jointly-learned, differentiable ambiguity sets, once the current static
  ones (fixed `nominal`, fixed `radius`) feel limiting.
- Causality-constrained DRO for sequential decision problems — the hardest
  of the three, saved for last on purpose.

Further out, and not something I'm designing toward yet: stochastic and
differentiable-backend solvers, optimal transport as a first-class object
rather than something folded into `divergences/wasserstein.py`, and
reinforcement learning under model uncertainty. These are the kind of
long-horizon questions that motivated building this in the first place, but
I'd rather let the core prove itself first than sketch out five more
subpackages before anyone's used the first one.

## Development

Optora targets Python 3.10+ and PyTorch 2.2+. From an activated virtual
environment in the repository root:

```powershell
pip install -e ".[dev,docs]"
pytest
ruff check .
ruff format --check .
mypy
```

## License

Optora is distributed under the MIT License; see [`LICENSE`](../LICENSE)
for the full text.
