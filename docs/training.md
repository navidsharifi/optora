---
icon: material/school-outline
---

# Training with `torch.optim`

Optora's ambiguity sets are `nn.Module` subclasses whose
`worst_case_expectation(loss)` is an ordinary differentiable function of
`loss`. Nothing else is needed to train a model robustly: wrap the
per-scenario loss, call `.backward()`, and step any `torch.optim`
optimizer.

```python
import torch
from torch import nn

from optora import GradientDescent, KLAmbiguitySet

model = nn.Linear(num_features, 1)
nominal = torch.full((num_scenarios,), 1.0 / num_scenarios)

ambiguity_set = KLAmbiguitySet(nominal=nominal, radius=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

for _ in range(num_steps):
    optimizer.zero_grad()
    per_scenario_loss = (model(features).squeeze(-1) - targets) ** 2
    value = ambiguity_set.worst_case_expectation(per_scenario_loss)
    value.backward()
    optimizer.step()
```

!!! note "`loss` is per scenario, not reduced"

    `worst_case_expectation` expects one loss entry per support point of
    `nominal`, shape `(..., n)`, and performs the reduction itself: it
    returns the expectation under the worst distribution in the ambiguity
    set instead of the mean under the nominal one. Do not call `.mean()`
    before passing the loss in — that would discard exactly the
    per-scenario structure the ambiguity set reasons about.

## Why this is a correct gradient

The inner supremum is not differentiated through. Each ambiguity set
locates the maximizer $q^\star$ in a **detached** inner solve, then
reports $\sum_i q^\star_i \ell_i$ with the still-attached `loss`. By
Danskin's theorem that is already the exact gradient: the optimal value is
a support function of the loss, so

$$
\nabla_\theta\, V(\theta)
= \sum_{i} q^\star_i \, \nabla_\theta\, \ell_i(\theta),
\qquad
q^\star = \arg\max_{q \in \mathcal{Q}} \; \mathbb{E}_q[\ell(\theta)],
$$

with $q^\star$ held fixed. The derivative of $q^\star$ with respect to
$\theta$ contributes nothing, because $q^\star$ already maximizes the
inner problem. This has three practical consequences:

<div class="grid cards" markdown>

-   :material-check-decagram:{ .lg .middle } __No gradient leakage__

    ---

    The gradient depends only on where the inner solve landed, never on
    the path it took to get there.

-   :material-memory:{ .lg .middle } __Constant memory__

    ---

    Inner iterations are never retained for backward, so the inner solve
    costs time but not autograd memory.

-   :material-alert-outline:{ .lg .middle } __Accuracy follows the inner solve__

    ---

    The identity holds *at* the maximizer. `KLAmbiguitySet` and
    `ChiSquareAmbiguitySet` bisect to the dtype's precision on every
    call; a `PhiAmbiguitySet` with a loose inner `tol` biases the outer
    gradient, so tighten it before blaming the optimizer.

</div>

## Reading off the worst-case distribution

Because $\partial V / \partial \ell_i = q^\star_i$, the worst-case
distribution is an autograd by-product — no separate API is needed:

```python
loss = (model(features).squeeze(-1) - targets) ** 2
loss.retain_grad()
ambiguity_set.worst_case_expectation(loss).backward()
worst_case_distribution = loss.grad
```

## Second derivatives

Hessians and Hessian-vector products of the worst case with respect to
`loss` are exact too: `create_graph=True`,
`torch.autograd.functional.hessian`, and `torch.func.hessian` all work.

=== "KL and chi-square"

    The worst case is curved in `loss`, so a detached $q^\star$ alone would
    report a zero Hessian. A recorded backward pass instead rebuilds
    $q^\star$ from the solved tilt by one implicit-function step, which has
    exactly the Jacobian $\partial q^\star / \partial \ell$. A plain
    `.backward()` never pays for that step.

=== "Total variation and Wasserstein"

    The worst case is piecewise linear in `loss`, so its Hessian is zero
    wherever it exists, which is exactly what autograd reports.

!!! warning "Unsupported second-order paths"

    - `torch.func.jacfwd(torch.func.jacfwd(...))` on `KLAmbiguitySet` or
      `ChiSquareAmbiguitySet`: PyTorch does not differentiate a custom
      forward-mode rule a second time. Use `torch.func.hessian`
      (forward over reverse) or reverse over reverse.
    - `PhiAmbiguitySet`: the first derivative is exact once the inner dual
      solve has converged, but the second derivative ignores how the dual
      optimum moves with `loss`.

## Non-differentiable points

The worst case $V(\ell) = \sup_{q \in \mathcal{U}} \langle q, \ell \rangle$
is convex in $\ell$ and differentiable exactly where the maximizer is
unique. Where several distributions attain the supremum (a *kink*),
autograd returns one of them. That is always a subgradient: a feasible
distribution $g$ with $\langle g, \ell \rangle = V(\ell)$, so

$$
V(\ell + d) \ge V(\ell) + \langle g, d \rangle
\quad \text{for every } d,
\qquad
-V'(\ell; -d) \le \langle g, d \rangle \le V'(\ell; d),
$$

where $V'(\ell; d)$ is the one-sided directional derivative. Every
subgradient is a valid descent signal for a subgradient method or
`torch.optim`; which one is returned is fixed per set:

| Set | Kinks in `loss` | Gradient returned at a kink |
| --- | --- | --- |
| `KLAmbiguitySet` | Tied highest losses once the radius saturates the set (it then contains the distribution concentrated on them) | The tied mass split in proportion to `nominal` |
| `ChiSquareAmbiguitySet` | As KL. Also not twice differentiable where a scenario enters or leaves the support of $q^\star$; the gradient stays continuous there | As KL |
| `TotalVariationAmbiguitySet` | Ties among the highest losses, and ties among the lowest-loss scenarios that mass is drained from | All reallocated mass on one tied scenario, chosen by sort order |
| `WassersteinAmbiguitySet` | Ties in a support point's best response $\ell_j - \gamma^\star c_{ij}$, including coincident support points with equal losses | The convex combination of the two extreme transport plans that spends the radius exactly |

`tests/dro/test_gradients.py` checks each of these kinks by one-sided
finite differences, alongside `gradcheck` and `gradgradcheck` in float64 at
smooth points.

## Practical notes

| Topic | Guidance |
| --- | --- |
| Inner cost | `KLAmbiguitySet` and `ChiSquareAmbiguitySet` locate the worst case by a bisection whose trip count is fixed by the dtype, so every step costs the same and there is nothing to tune or warm-start. |
| Minibatches | Pass a batched loss of shape `(..., n)`; the inner problems decouple across leading dimensions and are bisected jointly in one call. |
| Device and dtype | `ambiguity_set.to(device)` moves `nominal` and every cached buffer; the returned value follows the loss's dtype and device. |
| Differentiable inputs | Only `loss`. Treat `nominal` and a tensor `radius` as constants: gradients with respect to them are not supported and are not meaningful. |
| Zero radius | `radius=0.0` makes the objective exactly the nominal-weighted empirical risk, which is the natural non-robust baseline to compare against. For `WassersteinAmbiguitySet` this needs a metric `cost` (`cost[i, j] > 0` for `i != j`); a zero-cost pair of distinct support points still lets mass move between them at zero radius. |

## When to use `MinimaxSolver` instead

[`MinimaxSolver`](api/dro/minimax_solver.md) drives the same composed
objective with an Optora `Solver` over a single decision tensor. Prefer
it for a low-dimensional decision variable solved to convergence with
reported diagnostics; prefer the loop above when the decision variable is
an `nn.Module`'s parameters, since `torch.optim` already owns that state.
Both minimize the identical objective and reach the same optimum.

For a complete training run held against an ERM baseline and two controls,
see [Training under subpopulation shift](examples/subpopulation_shift_training.md).
