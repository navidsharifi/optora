---
icon: material/school-outline
---

# Training with `torch.optim`

Optora's ambiguity sets are `nn.Module` subclasses whose
`worst_case_expectation(loss)` is an ordinary differentiable function of
`loss`. Robust training therefore needs no dedicated driver. We pass the
per-scenario loss to the ambiguity set, call `.backward()` on the result,
and step any `torch.optim` optimizer.

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
    `nominal`, with shape `(..., n)`, and performs the reduction itself. It
    returns the expectation under the worst-case distribution in the
    ambiguity set instead of the mean under the nominal one. Do not call
    `.mean()` before passing the loss in, because that would discard the
    per-scenario structure the ambiguity set acts on.

## Why this is a correct gradient

The inner supremum is not differentiated through. Each ambiguity set
locates the maximizer $q^\star$ in a **detached** inner solve and then
reports $\sum_i q^\star_i \ell_i$ with the still-attached `loss`. By
Danskin's theorem this already gives the exact gradient. The optimal value
is the support function of the ambiguity set evaluated at the loss, so

$$
\nabla_\theta\, V(\theta)
= \sum_{i} q^\star_i \, \nabla_\theta\, \ell_i(\theta),
\qquad
q^\star = \arg\max_{q \in \mathcal{Q}} \; \mathbb{E}_q[\ell(\theta)],
$$

with $q^\star$ held fixed. The derivative of $q^\star$ with respect to
$\theta$ contributes nothing, because $q^\star$ already maximizes the
inner problem. The identity has the following practical consequences.

<div class="grid cards" markdown>

-   :material-check-decagram:{ .lg .middle } __Independent of the solver path__

    ---

    The gradient depends only on the point where the inner solve ended,
    not on the iterates that led there.

-   :material-memory:{ .lg .middle } __Constant memory__

    ---

    Inner iterations are never retained for the backward pass, so the
    inner solve adds run time but no autograd memory.

-   :material-alert-outline:{ .lg .middle } __Accuracy follows the inner solve__

    ---

    The identity holds *at* the maximizer. `KLAmbiguitySet` and
    `ChiSquareAmbiguitySet` bisect to the precision of the dtype on every
    call, and so does `PhiAmbiguitySet` without a `dual_solver`. With a
    `dual_solver` and a loose `tol`, the inexact maximizer biases the
    outer gradient, so tighten the tolerance before suspecting the outer
    optimizer.

</div>

## Reading off the worst-case distribution

Because $\partial V / \partial \ell_i = q^\star_i$, the worst-case
distribution is a by-product of autograd, and no separate API is needed.

```python
loss = (model(features).squeeze(-1) - targets) ** 2
loss.retain_grad()
ambiguity_set.worst_case_expectation(loss).backward()
worst_case_distribution = loss.grad
```

This by-product also shows what a robust training run does. In the
animation below, Adam fits a linear model to data drawn from the same
two-group model as in
[Training under subpopulation shift](examples/subpopulation_shift_training.md),
once by empirical risk minimization and once on the KL worst case.

<figure class="figure" markdown="span">
  ![Weight paths, exact group risks and the adversary's minority mass along ERM and KL-DRO training](assets/figures/training-dynamics-light.gif#only-light){ loading=lazy }
  ![Weight paths, exact group risks and the adversary's minority mass along ERM and KL-DRO training](assets/figures/training-dynamics-dark.gif#only-dark){ loading=lazy }
  <figcaption>The left panel shows the weight path of each model between
  the majority and minority slopes. The middle panel shows the exact risk
  on each group at every step. The right panel shows the total worst-case
  mass that the KL adversary places on minority samples, read from
  <code>loss.grad</code>, against their nominal share.</figcaption>
</figure>

## Second derivatives

Hessians and Hessian-vector products of the worst case with respect to
`loss` are exact as well. They are available through `create_graph=True`,
`torch.autograd.functional.hessian` and `torch.func.hessian`.

=== "KL, chi-square, and Sinkhorn"

    The worst case has nonzero curvature in `loss`, so a detached
    $q^\star$ alone would report a zero Hessian. A recorded backward pass
    therefore rebuilds $q^\star$ from the solved tilt by one
    implicit-function step, whose Jacobian is exactly
    $\partial q^\star / \partial \ell$. The same step, taken in `nominal`
    and `radius`, gives the mixed derivatives. A plain `.backward()` does
    not compute this step.

=== "MMD"

    The worst case has nonzero curvature in `loss`. A zero-valued term,
    built from the implicit function theorem applied to the optimality
    conditions on the support of the maximizer, supplies the Jacobian
    $\partial q^\star / \partial \ell$. The Hessian is therefore exact
    once the interior-point solve has converged. The term is built only
    when `loss` requires grad or the call runs inside a `torch.func`
    transform.

=== "Total variation and Wasserstein"

    The worst case is piecewise linear in `loss`, so its Hessian is zero
    wherever it exists, and autograd reports exactly that.

=== "General phi-divergence"

    Without a `dual_solver`, `PhiAmbiguitySet` is a tilted set like KL and
    chi-square and has the same exact implicit-function Hessian. Its inner
    normalizing level gets its derivative from one Newton step of its own.
    With a `dual_solver`, it re-evaluates its dual objective $F$ at the
    detached optimum $x^\star$, which alone would miss how $x^\star$ moves
    with the loss. It therefore adds a term whose value and first
    derivative are zero but whose Hessian is the implicit-function
    correction $-F_{\ell x} F_{xx}^{-1} F_{x \ell}$, built from the
    Hessian of the dual at the optimum. Second derivatives are then exact
    once the dual solve has converged, including the mixed derivatives with
    `nominal` and `radius`. The term needs a few passes over the dual
    objective per call, compared with the many passes of the solve, and it
    is skipped when nothing requires grad.

    `torch.func` transforms also work. Inside one, the inner
    `GradientDescent` differentiates the dual with `torch.func` rather than
    `torch.autograd`. Under `torch.func.vmap` a batched convergence flag
    has no single value to read, so every mapped element runs the whole
    `max_iter` budget of the dual solver. A modest budget is advisable
    there.

!!! warning "Unsupported second-order paths"

    - `torch.func.jacfwd(torch.func.jacfwd(...))` is not supported on
      `KLAmbiguitySet`, `ChiSquareAmbiguitySet`, `SinkhornAmbiguitySet`,
      or `PhiAmbiguitySet` without a `dual_solver`. PyTorch does not
      differentiate a custom forward-mode rule a second time, so the result
      is silently wrong rather than an error. Use `torch.func.hessian`
      (forward over reverse) or reverse over reverse instead.

## Non-differentiable points

The worst case $V(\ell) = \sup_{q \in \mathcal{U}} \langle q, \ell \rangle$
is convex in $\ell$ and differentiable exactly where the maximizer is
unique. Where several distributions attain the supremum, at a *kink*,
autograd returns one of them. This is always a subgradient, namely a
feasible distribution $g$ with $\langle g, \ell \rangle = V(\ell)$, so

$$
V(\ell + d) \ge V(\ell) + \langle g, d \rangle
\quad \text{for every } d,
\qquad
-V'(\ell; -d) \le \langle g, d \rangle \le V'(\ell; d),
$$

where $V'(\ell; d)$ is the one-sided directional derivative. A
subgradient method accepts any subgradient, so a `torch.optim` step stays
well defined at a kink. Which subgradient is returned is fixed for each
set and is the same on every device.

| Set | Kinks in `loss` | Gradient returned at a kink |
| --- | --- | --- |
| `KLAmbiguitySet` | Tied highest losses once the radius saturates the set (the set then contains the distribution concentrated on them) | The tied mass split in proportion to `nominal` |
| `ChiSquareAmbiguitySet` | As for KL. The worst case is also not twice differentiable where a scenario enters or leaves the support of $q^\star$, although the gradient stays continuous there | As for KL |
| `TotalVariationAmbiguitySet` | Ties among the highest losses, and ties among the lowest-loss scenarios that mass is drained from | Ties are broken by support index. Tied low-loss scenarios are drained in index order, and all reallocated mass goes to the last tied maximum |
| `CVaRAmbiguitySet` | Tied losses that straddle the boundary of the upper $\alpha$-tail | The tail is filled in support-index order among the tied scenarios |
| `SinkhornAmbiguitySet` | As for KL, with ties among the highest losses on the support of the reference measure | The tied mass of each kernel row is split in proportion to that row |
| `MMDAmbiguitySet` | Tied highest losses once the radius saturates the set. The worst case is also not twice differentiable where a scenario enters or leaves the support of $q^\star$, although the gradient stays continuous there | The maximizer of the interior-point method, which splits tied mass according to the central path |
| `WassersteinAmbiguitySet` | Ties in the best response $\ell_j - \gamma^\star c_{ij}$ of a support point, including coincident support points with equal losses | The convex combination of the two extreme transport plans that spends the radius exactly |

`tests/dro/test_gradients.py` checks each of these kinks by one-sided
finite differences, and checks smooth points with `gradcheck` and
`gradgradcheck` in float64.

## Gradients with respect to `nominal` and `radius`

`nominal` and a tensor `radius` (or a tensor `alpha` for
`CVaRAmbiguitySet`) are differentiable as well. This allows, for example,
learning a reweighting of the training data through
`nominal = torch.softmax(logits, dim=-1)`, or tuning the radius. The inner
solve again stays out of the graph. By the envelope theorem, the
derivatives are read off the Lagrangian

$$
L(q, \lambda; \ell, p, \rho) = \langle q, \ell \rangle
    - \lambda \big( D(q \,\|\, p) - \rho \big)
$$

at the saddle point $(q^\star, \lambda^\star)$ that the set has already
solved for, which gives

$$
\frac{\partial V}{\partial \rho} = \lambda^\star \ge 0,
\qquad
\frac{\partial V}{\partial p} = \frac{\partial L}{\partial p}
    \Big|_{(q^\star, \lambda^\star)}.
$$

The multiplier $\lambda^\star$ is the shadow price of the radius
constraint, namely the rate at which the worst case increases per unit of
radius. It does not increase as the radius grows, and it vanishes once the
radius saturates the set. For KL, chi-square and Sinkhorn it diverges as
the radius tends to zero on a loss that is not constant, because the
worst case grows like $\sqrt{\rho}$ there.

!!! note "Only directions that keep `nominal` a distribution are defined"

    The worst case is defined only for distributions, so its derivative
    with respect to `nominal` is defined only along directions $d$ with
    $\sum_i d_i = 0$. The component along the all-ones direction is
    implementation-defined. A softmax, or any other map onto the simplex,
    discards it.

The worst case is concave in $p$ and in $\rho$ (for CVaR, in the
likelihood-ratio bound $1/\alpha$ rather than in $\rho$). At a kink, such
as a total-variation radius that drains a scenario exactly, the returned
derivative is therefore a supergradient, namely one of the one-sided
derivatives.

Second derivatives that involve `nominal` and `radius`, including the mixed
derivatives with `loss`, are exact on the curved sets. `KLAmbiguitySet`,
`ChiSquareAmbiguitySet` and `SinkhornAmbiguitySet` obtain them from the
implicit-function step that gives their loss Hessian. That step also takes
`nominal` and `radius` as inputs, so a recorded backward pass rebuilds the
maximizer and both envelope derivatives with their exact Jacobians.
`PhiAmbiguitySet` obtains them in the same way without a `dual_solver`.
With one, it obtains them from its dual objective, and they are exact once
that solve has converged. On the total-variation, CVaR, Wasserstein and
MMD sets, the derivatives with respect to `nominal` and `radius` are first
order only, and second derivatives that involve them are not supported.
`tests/dro/test_parameter_gradients.py` checks every set against
`gradcheck` and against finite differences of independent NumPy duals.

## Minibatches from a population

The pattern above treats `nominal` as the whole training distribution.
When the data instead arrive as a stream of i.i.d. minibatches from an
unknown population $P_0$, running an ambiguity set on the uniform
empirical distribution $\hat P_m$ of each minibatch gives a **biased**
estimate of the population worst case. The supremum is nonlinear in the
sample, so

$$
\mathbb{E}\Big[\sup_{D(q \,\|\, \hat P_m) \le \rho} \mathbb{E}_q[\ell]\Big]
\neq \sup_{D(q \,\|\, P_0) \le \rho} \mathbb{E}_q[\ell],
$$

and the gradient of this plug-in estimate is not an unbiased stochastic
gradient (for KL, the plug-in is the log-sum-exp of a subsample).
[`optora.dro.minibatch`](api/dro/minibatch.md) provides the two standard
remedies.

=== "Multilevel Monte Carlo"

    `MultilevelMonteCarloEstimator` keeps the exact solve of any ambiguity
    set built from a nominal alone. It reduces the bias to that of a
    single solve on $n_0 2^L$ samples, at an expected batch size of
    $n_0 L / (1 - 2^{-L})$ (Blanchet and Glynn 2015; Levy et al. 2020).
    The batch size is drawn at every step.

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
    is an unbiased estimate at any fixed batch size. The dual variables
    are trained together with the model (Levy et al. 2020; Jin et al.
    2021).

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
    along its tangent beyond $s = c$, where $c$ is a quarter of the
    logarithm of the largest finite value of the dtype. When $\lambda$
    lags behind a loss spike, this gives a large but finite gradient
    instead of the `inf` of `torch.expm1`, which would turn the dual
    variables into `nan`. Use `optora.dro.chi_square_conjugate` for the
    chi-square ball. For CVaR at level $\alpha$, pass `radius=0.0` and
    `functools.partial(optora.dro.cvar_conjugate, alpha=alpha)`. The dual
    then reduces to the Rockafellar-Uryasev objective
    $\lambda + \mathbb{E}[(\ell - \lambda)_+] / \alpha$ (Rockafellar and
    Uryasev 2000), and $\eta$ drops out. Start the dual variables on the
    scale of the loss, for example at the exact dual optimum of a pilot
    minibatch. With the exponential conjugate of KL, the default start can
    still produce gradients large enough to stall Adam.

| | `MultilevelMonteCarloEstimator` | `StochasticPhiDual` |
| --- | --- | --- |
| Unbiased for | The plug-in worst case at $n_0 2^L$ samples | The population dual at the current $(\eta, \lambda)$ |
| Batch size | Random, $n_0 2^J$ | Fixed, any |
| Ambiguity sets | Any set built from a nominal alone (KL, chi-square, total variation, CVaR, general phi) | Phi-divergence balls with a finite conjugate (KL, chi-square, CVaR) |
| Per-step cost | Four exact solves, of which the two half-batch solves run as one batched call | One pass over the minibatch, no inner solve |

[Minibatch DRO against the full batch](examples/minibatch_dro_training.md)
measures the bias of the plug-in estimate, trains with all three
estimators and compares each with the exact full-batch optimum.

## Group DRO

When the training data come from known groups $g = 1, \dots, G$, for
example the combinations of a label and a spurious attribute, group DRO
minimizes the risk of the worst group, $\min_\theta \max_g R_g(\theta)$
(Sagawa et al. 2020). [`optora.dro.group_dro`](api/dro/group_dro.md) adds
two components. The function `group_losses(loss, groups, num_groups)`
averages per-sample losses within each group, and `OnlineGroupDRO` keeps
the group weights for minibatch training. Over the group losses, the
maximum over groups is the worst case of an existing ambiguity set
centred on the group proportions.

=== "Exact, on a fixed dataset"

    ```python
    import torch

    from optora import CVaRAmbiguitySet
    from optora.dro import group_losses

    counts = torch.bincount(groups, minlength=num_groups)
    worst_group = CVaRAmbiguitySet(counts / counts.sum(), alpha=0.0)

    for _ in range(num_steps):
        optimizer.zero_grad()
        per_sample_loss = (model(features).squeeze(-1) - targets) ** 2
        group_loss = group_losses(per_sample_loss, groups, num_groups)
        worst_group.worst_case_expectation(group_loss).backward()
        optimizer.step()
    ```

    `CVaRAmbiguitySet` at `alpha=0.0` equals the maximum over the groups
    present, and its level interpolates to ERM at `alpha=1.0`.
    `TotalVariationAmbiguitySet` at radius one is the maximum over every
    group. `KLAmbiguitySet` and `ChiSquareAmbiguitySet` around the
    proportions give worst cases between the mixture risk and the
    worst-group risk. They reach the worst group $g^\star$ once the radius
    reaches $\log(1/p_{g^\star})$ and $1/p_{g^\star} - 1$, respectively.

=== "Online, on minibatches"

    ```python
    import torch

    from optora import OnlineGroupDRO
    from optora.dro import group_losses

    adversary = OnlineGroupDRO(torch.full((num_groups,), 1.0 / num_groups), step_size=0.01)

    for features, targets, groups in loader:
        optimizer.zero_grad()
        per_sample_loss = (model(features).squeeze(-1) - targets) ** 2
        group_loss = group_losses(per_sample_loss, groups, num_groups)
        adversary.step(group_loss)
        adversary(group_loss).backward()
        optimizer.step()
    ```

    The maximum of the minibatch group means overestimates the risk of the
    worst group, because the expectation of a maximum is at least the
    maximum of the expectations. `OnlineGroupDRO` instead implements
    Algorithm 1 of Sagawa et al. Its `step` method moves the group weights
    by exponentiated-gradient ascent, $q_g \propto q_g e^{\eta \hat\ell_g}$,
    and `forward` returns the weighted loss $\sum_g q_g \hat\ell_g$ that the
    model descends. Call `step` first, as above, and outside any
    `torch.func` transform. `forward` is a pure function and composes with
    all of them.

| Topic | Guidance |
| --- | --- |
| Regularization | Necessary when the model can fit the training set. Once the loss of every training group is near zero, the adversary has nothing left to upweight, and group DRO does no better than ERM on the worst group (Sagawa et al. 2020). Use a strong L2 penalty or early stopping, as [Group DRO under a spurious correlation](examples/group_dro_training.md) shows. |
| Absent groups | `group_losses` assigns loss zero to a group without samples, so `step` leaves its unnormalized weight unchanged, which is how Algorithm 1 treats a group that was not sampled. Sampling a fixed number of examples per group keeps every group in every minibatch. |
| Generalization adjustment | Sagawa et al. add $C / \sqrt{n_g}$ to the loss of each group to favour small groups. Pass the adjusted losses to `step` and the raw losses to `forward`. The adjustment does not depend on $\theta$, so it changes the weights but not the form of the gradient. |
| Many models at once | Initial weights of shape `(..., G)` give one adversary per batch element of the losses, and a tensor `step_size` sweeps several step sizes in one module. |
| Reading the weights | After $T$ steps on fixed group losses the weights are the exponential tilt $q_0 e^{T\eta\ell} / Z$, which is the worst case of a KL ball around $q_0$. The iterates therefore follow the KL-DRO worst-case path toward the worst group. |

## Practical notes

| Topic | Guidance |
| --- | --- |
| Inner cost | `KLAmbiguitySet` and `ChiSquareAmbiguitySet` locate the worst case by a bisection whose trip count is fixed by the dtype. Every step therefore has the same cost, and there is no tolerance to tune and no warm start to manage. |
| Batched losses | Pass a batched loss of shape `(..., n)`. The inner problems decouple across the leading dimensions and are bisected jointly in one call. For minibatches drawn from a population, see [Minibatches from a population](#minibatches-from-a-population). |
| Device and dtype | `ambiguity_set.to(device)` moves `nominal` and every cached buffer. The returned value is on the device of `loss` and has the promoted dtype of `loss` and `nominal`. Half-precision inputs, including those produced under `torch.autocast`, are solved in float32, and the result is rounded once. See [Numerical behaviour](numerics.md). |
| Debugging | `validate=True` checks `nominal` and a tensor `radius` for invalid values at construction, and checks every `loss` for non-finite values at each call. The per-call check costs one host synchronization. |
| Differentiable inputs | These are `loss`, `nominal` and a tensor `radius` (`alpha` for CVaR), as described in [Gradients with respect to `nominal` and `radius`](#gradients-with-respect-to-nominal-and-radius). The extra derivatives are computed only when `nominal` or the radius requires grad. |
| Choosing the radius | `optora.dro.kl_radius`, `chi_square_radius` and `phi_divergence_radius` turn a confidence level and a sample size into a radius under which the robust value of the phi-divergence ball is an asymptotically exact upper confidence bound on the true risk. `wasserstein_radius` does the same for a quadratic transport cost from the losses and their input gradients, calibrated for transport over the whole input space. See [Calibrating the ambiguity radius](examples/calibrated_ambiguity_radius.md). |
| Zero radius | `radius=0.0` makes the objective equal to the nominal-weighted empirical risk, which is the natural non-robust baseline. For `WassersteinAmbiguitySet` this requires a metric `cost` (`cost[i, j] > 0` for `i != j`), because a zero-cost pair of distinct support points still lets mass move between them at zero radius. `PhiAmbiguitySet` with a `dual_solver` solves its dual at a zero radius as at any other radius, since a generator may vanish on an interval (the CVaR indicator). For an exact baseline, use it without a `dual_solver`, or use a dedicated set. `MMDAmbiguitySet` returns the nominal expectation at zero radius only when its Gram matrix is positive definite. Duplicate support points make the Gram matrix singular, mass can then move between them at zero radius, and the set solves that face of the simplex instead. |

## When to use `MinimaxSolver` instead

[`MinimaxSolver`](api/dro/minimax_solver.md) minimizes the same composed
objective with an Optora `Solver` over a single decision tensor. It is the
better choice for a low-dimensional decision variable that should be
solved to convergence with reported diagnostics. The loop above is the
better choice when the decision variable consists of the parameters of an
`nn.Module`, since `torch.optim` already manages that state. Both minimize
the same objective, and `tests/dro/test_training_loop.py` checks that they
reach the same optimum.

`MinimaxSolver` with a `GradientDescent` outer solver, and
`SaddlePointSolver`, compose with `torch.func` transforms. For example,
`torch.func.vmap` over a tensor radius solves one robust problem per radius
in a single call. Under `vmap` a batched convergence flag has no single
value to read, so every mapped problem runs the whole `max_iter` budget of
the solver, with its iterate frozen once it converges. Each iterate is
detached, so an enclosing `torch.func.grad` treats the solution as a
constant and does not differentiate through the steps of the solver.

For a complete training run compared with an ERM baseline and two
controls, see
[Training under subpopulation shift](examples/subpopulation_shift_training.md).
