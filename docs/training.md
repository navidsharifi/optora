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

=== "KL, chi-square, and Sinkhorn"

    The worst case is curved in `loss`, so a detached $q^\star$ alone would
    report a zero Hessian. A recorded backward pass instead rebuilds
    $q^\star$ from the solved tilt by one implicit-function step, which has
    exactly the Jacobian $\partial q^\star / \partial \ell$, and the same
    step in `nominal` and `radius` gives the mixed derivatives. A plain
    `.backward()` never pays for that step.

=== "MMD"

    The worst case is curved in `loss`. A zero-valued term built from the
    implicit function theorem on the optimality conditions over the
    maximizer's support supplies the Jacobian $\partial q^\star / \partial
    \ell$, so the Hessian is exact once the interior-point solve has
    converged. It is built only when `loss` requires grad.

=== "Total variation and Wasserstein"

    The worst case is piecewise linear in `loss`, so its Hessian is zero
    wherever it exists, which is exactly what autograd reports.

=== "General phi-divergence"

    `PhiAmbiguitySet` re-evaluates its dual objective $F$ at the detached
    optimum $x^\star$, which alone would miss how $x^\star$ moves with the
    loss. It also adds a term that is zero in value and in first
    derivative but whose Hessian is the implicit-function correction
    $-F_{\ell x} F_{xx}^{-1} F_{x \ell}$, built from the dual's own Hessian
    at the optimum. Second derivatives are then exact once the dual solve
    has converged, including the mixed ones with `nominal` and `radius`.
    The term costs a few passes over the dual objective per call, against
    the solve's many, and is skipped when nothing requires grad.

    `torch.func` transforms work too: inside one the inner
    `GradientDescent` differentiates the dual with `torch.func` rather than
    `torch.autograd`. Under `torch.func.vmap` a batched convergence flag
    has no single value to read, so every mapped element runs the dual
    solver's whole `max_iter` budget; keep it modest there.

!!! warning "Unsupported second-order paths"

    - `torch.func.jacfwd(torch.func.jacfwd(...))` on `KLAmbiguitySet`,
      `ChiSquareAmbiguitySet`, or `SinkhornAmbiguitySet`: PyTorch does not
      differentiate a custom forward-mode rule a second time, so the result
      is silently wrong rather than an error. Use `torch.func.hessian`
      (forward over reverse) or reverse over reverse.

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
`torch.optim`; which one is returned is fixed per set, and is the same on
every device:

| Set | Kinks in `loss` | Gradient returned at a kink |
| --- | --- | --- |
| `KLAmbiguitySet` | Tied highest losses once the radius saturates the set (it then contains the distribution concentrated on them) | The tied mass split in proportion to `nominal` |
| `ChiSquareAmbiguitySet` | As KL. Also not twice differentiable where a scenario enters or leaves the support of $q^\star$; the gradient stays continuous there | As KL |
| `TotalVariationAmbiguitySet` | Ties among the highest losses, and ties among the lowest-loss scenarios that mass is drained from | Ties broken by support index: tied low-loss scenarios drained in index order, all reallocated mass on the last tied maximum |
| `CVaRAmbiguitySet` | Tied losses that straddle the boundary of the upper $\alpha$-tail | The tail filled in support-index order among the tied scenarios |
| `SinkhornAmbiguitySet` | As KL, with ties among the highest losses on the reference measure's support | Each kernel row's tied mass split in proportion to that row |
| `MMDAmbiguitySet` | Where a scenario enters or leaves the support of $q^\star$, and tied highest losses once the radius saturates the set; the gradient stays continuous at the first | The interior-point maximizer, which splits tied mass by the central path |
| `WassersteinAmbiguitySet` | Ties in a support point's best response $\ell_j - \gamma^\star c_{ij}$, including coincident support points with equal losses | The convex combination of the two extreme transport plans that spends the radius exactly |

`tests/dro/test_gradients.py` checks each of these kinks by one-sided
finite differences, alongside `gradcheck` and `gradgradcheck` in float64 at
smooth points.

## Gradients with respect to `nominal` and `radius`

`nominal` and a tensor `radius` (or a tensor `alpha` for
`CVaRAmbiguitySet`) are differentiable too, for example to learn a
reweighting of the training data through `nominal = torch.softmax(logits,
dim=-1)` or to tune the radius. The inner solve again stays out of the
graph: by the envelope theorem, the derivatives are read off the
Lagrangian

$$
L(q, \lambda; \ell, p, \rho) = \langle q, \ell \rangle
    - \lambda \big( D(q \,\|\, p) - \rho \big)
$$

at the saddle point $(q^\star, \lambda^\star)$ the set already solved for:

$$
\frac{\partial V}{\partial \rho} = \lambda^\star \ge 0,
\qquad
\frac{\partial V}{\partial p} = \frac{\partial L}{\partial p}
    \Big|_{(q^\star, \lambda^\star)}.
$$

The multiplier $\lambda^\star$ is the marginal value of robustness: how
much the worst case rises per unit of radius. It falls as the radius
grows and vanishes once the radius saturates the set. For KL, chi-square
and Sinkhorn it is unbounded as the radius vanishes, where the worst case
grows like $\sqrt{\rho}$.

!!! note "Only directions that keep `nominal` a distribution are defined"

    The worst case is defined for distributions only, so its derivative
    with respect to `nominal` is defined only along directions $d$ with
    $\sum_i d_i = 0$. The component along the all-ones direction is
    implementation-defined; a softmax or any other map onto the simplex
    discards it.

The worst case is concave in $p$ and in $\rho$ (for CVaR, in the
likelihood-ratio bound $1/\alpha$ rather than in $\rho$), so at a kink,
such as a total
variation radius that drains a scenario exactly, the returned derivative is
a supergradient: one of the one-sided derivatives.

Second derivatives that involve `nominal` and `radius`, mixed ones with
`loss` included, are exact on the curved sets. `KLAmbiguitySet`,
`ChiSquareAmbiguitySet` and `SinkhornAmbiguitySet` get them from the same
implicit-function step that gives their loss Hessian, which takes `nominal`
and `radius` as inputs too, so a recorded backward pass rebuilds the
maximizer and both envelope derivatives with their exact Jacobians.
`PhiAmbiguitySet` gets them from its dual objective, so, like its loss
derivatives, they are exact once its inner dual solve has converged. On
the total-variation, CVaR, Wasserstein, and MMD sets derivatives with
respect to `nominal` and `radius` are first order: second derivatives that
involve them are not supported.
`tests/dro/test_parameter_gradients.py` checks every set against `gradcheck`
and against finite differences of independent NumPy duals.

## Minibatches from a population

The pattern above treats `nominal` as the whole training distribution. When
the data are instead a stream of i.i.d. minibatches from an unknown
population $P_0$, running an ambiguity set on each minibatch's uniform
empirical distribution $\hat P_m$ is a **biased** estimate of the
population worst case: the supremum is nonlinear in the sample, so

$$
\mathbb{E}\Big[\sup_{D(q \,\|\, \hat P_m) \le \rho} \mathbb{E}_q[\ell]\Big]
\neq \sup_{D(q \,\|\, P_0) \le \rho} \mathbb{E}_q[\ell],
$$

and its gradient is not an unbiased stochastic gradient (for KL, it is the
log-sum-exp of a subsample). [`optora.dro.minibatch`](api/dro/minibatch.md)
provides the two standard remedies:

=== "Multilevel Monte Carlo"

    `MultilevelMonteCarloEstimator` keeps the exact solve of any
    nominal-only ambiguity set and removes its bias up to the largest batch
    size $n_0 2^L$, at an expected batch size of $n_0 L / (1 - 2^{-L})$
    (Blanchet and Glynn 2015; Levy et al. 2020). The batch size is drawn
    per step:

    ```python
    import functools

    from optora import KLAmbiguitySet, MultilevelMonteCarloEstimator

    estimator = MultilevelMonteCarloEstimator(
        functools.partial(KLAmbiguitySet, radius=0.1),
        base_batch_size=32,
        max_level=4,
    )

    for _ in range(num_steps):
        optimizer.zero_grad()
        features, targets = sample(estimator.sample_batch_size())
        per_sample_loss = (model(features).squeeze(-1) - targets) ** 2
        estimator(per_sample_loss).backward()
        optimizer.step()
    ```

=== "Stochastic dual"

    `StochasticPhiDual` keeps the phi-divergence dual variables
    $(\log \eta, \lambda)$ as learnable parameters. For fixed dual
    variables the dual is an expectation over $P_0$, so its minibatch mean
    is exactly unbiased with a fixed batch size; the dual variables train
    with the model (Levy et al. 2020; Jin et al. 2021):

    ```python
    import torch

    from optora import StochasticPhiDual
    from optora.dro import kl_conjugate

    dual = StochasticPhiDual(radius=0.1, phi_conjugate=kl_conjugate)
    optimizer = torch.optim.Adam([*model.parameters(), *dual.parameters()])

    for features, targets in loader:
        optimizer.zero_grad()
        per_sample_loss = (model(features).squeeze(-1) - targets) ** 2
        dual(per_sample_loss).backward()
        optimizer.step()
    ```

    `optora.dro.kl_conjugate` is the KL conjugate $e^s - 1$, continued
    linearly past a quarter of the dtype's exponent range so that $\lambda$
    lagging a loss spike gives a large finite gradient instead of
    `inf`, which `torch.expm1` would turn into `nan` dual variables.
    Use `optora.dro.chi_square_conjugate` for the chi-square ball. For
    CVaR at level $\alpha$, pass `radius=0.0` and
    `functools.partial(optora.dro.cvar_conjugate, alpha=alpha)`: the dual
    then reduces to the Rockafellar-Uryasev objective
    $\lambda + \mathbb{E}[(\ell - \lambda)_+] / \alpha$, and $\eta$ drops
    out. Start
    the dual variables on the scale of the loss, for example at the exact
    dual optimum of a pilot minibatch: with KL's exponential conjugate,
    the default start can still produce gradients large enough to stall
    Adam.

| | `MultilevelMonteCarloEstimator` | `StochasticPhiDual` |
| --- | --- | --- |
| Unbiased for | The plug-in worst case at $n_0 2^L$ samples | The population dual at the current $(\eta, \lambda)$ |
| Batch size | Random, $n_0 2^J$ | Fixed, any |
| Ambiguity sets | Any set built from a nominal alone (KL, chi-square, total variation, CVaR, general phi) | Phi-divergence balls with a finite conjugate (KL, chi-square, CVaR) |
| Per-step cost | Three exact solves | One pass over the minibatch, no inner solve |

[Minibatch DRO against the full batch](examples/minibatch_dro_training.md)
measures the plug-in's bias and trains with all three estimators against
the exact full-batch optimum.

## Practical notes

| Topic | Guidance |
| --- | --- |
| Inner cost | `KLAmbiguitySet` and `ChiSquareAmbiguitySet` locate the worst case by a bisection whose trip count is fixed by the dtype, so every step costs the same and there is nothing to tune or warm-start. |
| Batched losses | Pass a batched loss of shape `(..., n)`; the inner problems decouple across leading dimensions and are bisected jointly in one call. For minibatches drawn from a population, see [Minibatches from a population](#minibatches-from-a-population). |
| Device and dtype | `ambiguity_set.to(device)` moves `nominal` and every cached buffer; the returned value follows the loss's dtype and device. Half-precision losses (including under `torch.autocast`) are solved in float32 and rounded once; see [Numerical behaviour](numerics.md). |
| Debugging | `validate=True` checks `nominal`, a tensor `radius`, and every `loss` for non-finite or invalid values, at one host synchronization per call. |
| Differentiable inputs | `loss`, `nominal`, and a tensor `radius` (`alpha` for CVaR); see [Gradients with respect to `nominal` and `radius`](#gradients-with-respect-to-nominal-and-radius). The extra derivatives are computed only when `nominal` or the radius requires grad. |
| Choosing the radius | `optora.dro.kl_radius`, `chi_square_radius`, `phi_divergence_radius` and `wasserstein_radius` turn a confidence level and a sample size into a radius under which the robust value is an asymptotically exact upper confidence bound on the true risk; see [Calibrating the ambiguity radius](examples/calibrated_ambiguity_radius.md). |
| Zero radius | `radius=0.0` makes the objective exactly the nominal-weighted empirical risk, which is the natural non-robust baseline to compare against. For `WassersteinAmbiguitySet` this needs a metric `cost` (`cost[i, j] > 0` for `i != j`); a zero-cost pair of distinct support points still lets mass move between them at zero radius. |

## When to use `MinimaxSolver` instead

[`MinimaxSolver`](api/dro/minimax_solver.md) drives the same composed
objective with an Optora `Solver` over a single decision tensor. Prefer
it for a low-dimensional decision variable solved to convergence with
reported diagnostics; prefer the loop above when the decision variable is
an `nn.Module`'s parameters, since `torch.optim` already owns that state.
Both minimize the identical objective and reach the same optimum.

`MinimaxSolver` with a `GradientDescent` outer solver, and
`SaddlePointSolver`, compose with `torch.func` transforms. For example,
`torch.func.vmap` over a tensor radius solves one robust problem per radius
in a single call. Under `vmap` a batched convergence flag has no single
value to read, so every mapped problem runs the solver's whole `max_iter`
budget, frozen once it converges. Each iterate is detached, so an
enclosing `torch.func.grad` sees the solution as a constant rather than
differentiating through the solver's steps.

For a complete training run held against an ERM baseline and two controls,
see [Training under subpopulation shift](examples/subpopulation_shift_training.md).
