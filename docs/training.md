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

ambiguity_set = KLAmbiguitySet(
    nominal=nominal,
    radius=0.1,
    dual_solver=GradientDescent(step_size=0.05, max_iter=500, tol=1e-9),
)
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
reduces its inner problem to a convex dual, solves it in a **detached**
inner solve, and then re-evaluates the dual objective at that fixed dual
optimum with the still-attached `loss`. By the envelope theorem the
derivative of the dual variable with respect to the model parameters
contributes nothing, because the dual objective is stationary in its own
dual variable at the optimum:

$$
\nabla_\theta\, V(\theta)
= \nabla_\theta\, g\bigl(\eta^\star(\theta), \theta\bigr)
  + \underbrace{\partial_\eta\, g\bigl(\eta^\star(\theta), \theta\bigr)}_{=\,0}
    \nabla_\theta\, \eta^\star(\theta)
= \nabla_\theta\, g\bigl(\eta^\star, \theta\bigr).
$$

Equivalently, in primal terms the gradient is the worst-case
distribution's expectation of the per-scenario gradients,

$$
\nabla_\theta\, V(\theta)
= \sum_{i} q^\star_i \, \nabla_\theta\, \ell_i(\theta),
\qquad
q^\star = \arg\max_{q \in \mathcal{Q}} \; \mathbb{E}_q[\ell(\theta)],
$$

with $q^\star$ held fixed. This has three practical consequences:

<div class="grid cards" markdown>

-   :material-check-decagram:{ .lg .middle } __No gradient leakage__

    ---

    The gradient does not depend on the inner solver's start point,
    step size, or iteration count, only on where it converged.

-   :material-memory:{ .lg .middle } __Constant memory__

    ---

    Inner iterations are never retained for backward, so inner
    `max_iter` costs time but not autograd memory.

-   :material-alert-outline:{ .lg .middle } __Accuracy follows the inner solve__

    ---

    The identity holds *at* the dual optimum. A loose inner `tol`
    biases the outer gradient, so tighten it before blaming the
    optimizer.

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
| Warm starts | Reuse one ambiguity set across steps. Consecutive solves warm-start from the previous dual optimum, so later steps converge in far fewer inner iterations at the same optimum. |
| Switching problems | Call `reset_warm_start()` before evaluating an unrelated loss, so a stale dual point does not slow the next solve. |
| Minibatches | Pass a batched loss of shape `(..., n)`; the duals decouple across leading dimensions and are solved jointly in one call. |
| Device and dtype | `ambiguity_set.to(device)` moves `nominal` and every cached buffer; the returned value follows the loss's dtype and device. |
| Zero radius | `radius=0.0` makes the objective exactly the nominal-weighted empirical risk, which is the natural non-robust baseline to compare against. |

## When to use `MinimaxSolver` instead

[`MinimaxSolver`](api/dro/minimax_solver.md) drives the same composed
objective with an Optora `Solver` over a single decision tensor. Prefer
it for a low-dimensional decision variable solved to convergence with
reported diagnostics; prefer the loop above when the decision variable is
an `nn.Module`'s parameters, since `torch.optim` already owns that state.
Both minimize the identical objective and reach the same optimum.
