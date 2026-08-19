# Optora

**A composable, PyTorch-native library for distributionally robust optimization (DRO).**

Status: pre-alpha (`v0.0.3`) &middot; Python 3.10+ &middot; PyTorch &ge;2.2 &middot; MIT License

Optora gives researchers and practitioners the primitives to formulate,
solve, and study optimization problems that must remain robust under
distributional uncertainty. Rather than shipping a closed set of
algorithms behind a single `fit`/`train` entry point, Optora exposes small,
explicit, composable components — divergences, ambiguity sets, and
solvers — that can be combined directly or subclassed to implement new
methods from a paper. It is built to sit comfortably underneath academic
research code: every abstraction is a Python `ABC` a researcher can extend,
and every numerical routine is written against PyTorch tensors so
computation stays differentiable and GPU-ready end to end.

## Why distributionally robust optimization

Empirical risk minimization optimizes performance against a single
estimated distribution of the data. When that estimate is wrong — due to
finite samples, distribution shift, or adversarial perturbation — the
resulting decision can perform poorly. Distributionally robust optimization
instead optimizes against the worst distribution within a plausible
neighborhood of a nominal (reference) distribution:

```text
minimize_x   sup_{q in ambiguity_set(nominal, radius)}  E_q[ loss(x, xi) ]
```

The neighborhood, or *ambiguity set*, is defined by bounding a statistical
divergence `D` between a candidate distribution `q` and the nominal
distribution:

```text
ambiguity_set(nominal, radius) = { q : D(q || nominal) <= radius }
```

Optora's architecture mirrors this formulation directly: a `Divergence`
defines `D`, an `AmbiguitySet` wraps a `Divergence` with a `nominal`
distribution and a `radius`, and a `Solver` carries out the inner
maximization (or the joint minimax problem) numerically.

## Core abstractions

Optora fixes exactly one abstract base contract per architectural layer, in
`optora.core`. Every concrete implementation elsewhere in the library
satisfies one of these contracts, and researchers are expected to subclass
them directly for new methods rather than working around a closed
algorithm registry.

| Contract | Location | Responsibility |
| --- | --- | --- |
| `Divergence` | `optora.core.divergence_base` | Callable `__call__(p, q) -> Tensor` computing a nonnegative discrepancy between two distributions, zero exactly when `p == q`. |
| `AmbiguitySet` | `optora.core.dro_base` | Holds a `nominal` distribution, a `Divergence`, and a `radius`; requires `worst_case_expectation(loss) -> Tensor` from subclasses. |
| `Solver[ProblemT, ResultT]` | `optora.core.solver_base` | Generic numerical method with a single `solve(problem) -> result` method, so each algorithm defines its own problem/result dataclasses instead of a one-size-fits-all signature. |

## What is implemented today

**Divergences** (`optora.divergences`) — the geometry of an ambiguity set:

- `KLDivergence` — Kullback-Leibler divergence `D_KL(p || q)`.
- `PhiDivergence` — general f-divergence `D_phi(p || q) = sum_i q_i * phi(p_i / q_i)` for any caller-supplied convex generator `phi`.
- `ChiSquareDivergence` and `TotalVariationDivergence` — `PhiDivergence` instances with the standard chi-square and total-variation generators.
- `SinkhornDivergence` — debiased entropic Wasserstein (Sinkhorn) divergence, computed via Sinkhorn's fixed-point iteration over a caller-supplied ground-cost matrix, with self-transport bias removed so it satisfies the `Divergence` contract exactly.

**Solvers** (`optora.solvers`) — the numerical machinery:

- `GradientDescent` — fixed-step-size gradient descent on a differentiable objective, using `torch.autograd.grad` each step, with structural support for warm-starting from a previous solve's iterate.
- `SaddlePointSolver` — primal-dual gradient ascent-descent for `min_x max_y objective(x, y)`, the shape of the joint DRO minimax problem, with an optional projection step to keep the dual (distributional) iterate feasible.

**DRO formulations** (`optora.dro`) — divergence and solver wired together into a trainable ambiguity set:

- `KLAmbiguitySet` — KL-constrained ambiguity set (KL-DRO). Computes the worst-case expected loss via the classical convex dual (Hu and Hong, 2013; Ben-Tal et al., 2013),

  ```text
  sup_{q: D_KL(q || nominal) <= radius} E_q[loss]
      = inf_{eta > 0} eta * radius + eta * log E_nominal[exp(loss / eta)]
  ```

  reducing the inner maximization to a one-dimensional convex minimization over `log(eta)`, solved with an injectable `Solver` (a tuned `GradientDescent` by default). The zero-radius case is handled exactly, without a numerical solve.

Every implementation above is covered by pytest tests checked against known
closed-form results, convergence limits, and monotonicity properties, and
the whole package is type-checked under mypy's strict mode.

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

Optora's current scope is deliberately narrow: ambiguity sets, divergences,
and the minimax solve that connects them — a complete, well-tested DRO
core rather than a broad grab-bag of methods. The immediate build order is:

1. `optora.dro.phi_dro` — chi-square and total-variation DRO formulations built on `PhiDivergence`.
2. `optora.dro.wasserstein_dro` — Wasserstein-DRO built on `SinkhornDivergence`.
3. `optora.dro.minimax_solver` — the general wiring of an `AmbiguitySet` and a `Solver` (e.g. `SaddlePointSolver`) into a single trainable minimax objective.

Once that DRO core is solid, several directions are natural, deliberately
deferred extensions rather than day-one scope: a dedicated risk-functional
layer once more than one risk measure is needed; jointly-learned
(differentiable) ambiguity sets; and causality-constrained DRO for
sequential decision problems. Longer-horizon research directions —
stochastic and differentiable-backend methods, optimal transport as a
first-class object, and reinforcement learning under model uncertainty —
motivate the library's overall trajectory without being committed
near-term milestones.

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
