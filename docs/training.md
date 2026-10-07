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

## Practical notes

| Topic | Guidance |
| --- | --- |
| Inner cost | `KLAmbiguitySet` and `ChiSquareAmbiguitySet` locate the worst case by a bisection whose trip count is fixed by the dtype, so every step costs the same and there is nothing to tune or warm-start. |
| Minibatches | Pass a batched loss of shape `(..., n)`; the inner problems decouple across leading dimensions and are bisected jointly in one call. |
| Device and dtype | `ambiguity_set.to(device)` moves `nominal` and every cached buffer; the returned value follows the loss's dtype and device. |
| Zero radius | `radius=0.0` makes the objective exactly the nominal-weighted empirical risk, which is the natural non-robust baseline to compare against. |

## When to use `MinimaxSolver` instead

[`MinimaxSolver`](api/dro/minimax_solver.md) drives the same composed
objective with an Optora `Solver` over a single decision tensor. Prefer
it for a low-dimensional decision variable solved to convergence with
reported diagnostics; prefer the loop above when the decision variable is
an `nn.Module`'s parameters, since `torch.optim` already owns that state.
Both minimize the identical objective and reach the same optimum.

For a complete training run held against an ERM baseline and two controls,
see [Training under subpopulation shift](examples/subpopulation_shift_training.md).
