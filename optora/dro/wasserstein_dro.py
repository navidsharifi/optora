"""Wasserstein-distance-constrained ambiguity set (Wasserstein-DRO)."""

import torch

from optora.core.dro_base import DualAmbiguitySet
from optora.core.solver_base import (
    MinimizationProblem,
    MinimizationResult,
    Solver,
)
from optora.divergences.wasserstein import SinkhornDivergence


class WassersteinAmbiguitySet(DualAmbiguitySet):
    r"""Wasserstein-distance-constrained ambiguity set for Wasserstein-DRO.

    Bounds every candidate distribution `q`, sharing `nominal`'s finite
    support with pairwise ground cost `cost`, by the type-1 Wasserstein
    distance $W_c(q, \mathrm{nominal}) \le \mathrm{radius}$, where $W_c$ is
    the optimal-transport cost of moving `nominal` to `q` under `cost`. The
    worst-case expected loss over this set admits an exact strong-duality
    reformulation (Mohajerin Esfahani and Kuhn 2018; Blanchet and Murthy
    2019; Gao and Kleywegt 2022), specialized to a finite shared support:

    $$
    \sup_{q:\, W_c(q, \mathrm{nominal}) \,\le\, \mathrm{radius}}
        \mathbb{E}_q[\mathrm{loss}]
    = \inf_{\gamma \ge 0} \;
        \gamma \cdot \mathrm{radius}
        + \mathbb{E}_{\mathrm{nominal}}\!\left[
            \max_j \big(\mathrm{loss}_j - \gamma \cdot \mathrm{cost}(\cdot, j)\big)
        \right]
    $$

    This follows from LP duality on the transportation polytope: the
    primal is

    $$
    \max_{\pi \ge 0} \sum_{ij} \pi_{ij}\, \mathrm{loss}_j
    \quad \text{s.t.} \quad
    \sum_j \pi_{ij} = \mathrm{nominal}_i, \qquad
    \sum_{ij} \pi_{ij}\, \mathrm{cost}_{ij} \le \mathrm{radius}
    $$

    (the column marginal, which defines the candidate `q`, is left free).
    Dualizing the budget constraint with multiplier $\gamma \ge 0$ and the
    row constraints with free multipliers, and maximizing out $\pi$
    pointwise, yields exactly the one-dimensional convex dual above.
    `dual_solver` minimizes this objective over an unconstrained
    `gamma_raw`, reparameterized as
    $\gamma = \max(\mathrm{gamma\_raw}, 0)$
    rather than $\exp(\log(\eta))$ (the reparameterization used by
    `KLAmbiguitySet` and `PhiAmbiguitySet`): unlike those formulations' dual
    variable, which must stay strictly positive, $\gamma$ ranges over the
    closed half-line $[0, \infty)$ and its optimum is genuinely attained at
    $\gamma = 0$ once `radius` is large enough to move all nominal mass onto
    the single highest-loss support point (projection lets unconstrained
    `GradientDescent` reach that boundary exactly, rather than only approach
    it asymptotically). The projection is written as a `torch.where` on
    `gamma_raw >= 0` rather than as `torch.clamp`, so that the default
    starting point $\mathrm{gamma\_raw} = 0$ carries the one-sided
    derivative of the dual at $\gamma = 0^+$: `torch.clamp` returns a zero
    subgradient on its boundary, which would stall gradient descent at the
    initial point.

    `divergence` is fixed to a `SinkhornDivergence` over `cost` (the only
    Wasserstein-type divergence implemented in `optora.divergences`), so
    `contains(...)` checks the entropy-regularized Sinkhorn divergence as a
    fast, differentiable approximation of exact Wasserstein-ball
    membership. `worst_case_expectation` does not depend on this
    approximation: it solves the exact (non-regularized) dual above
    directly from `cost`, since that dual has a clean closed form and does
    not need entropic relaxation for tractability.

    Attributes:
        nominal: Reference distribution the ambiguity set is centered on.
        divergence: `SinkhornDivergence` instance approximating distance
            from `nominal` for `contains(...)`.
        radius: Nonnegative bound on the Wasserstein distance of any
            distribution inside the ambiguity set from `nominal`, either a
            float or a tensor of radii evaluated as one batch.
        cost: Square, nonnegative pairwise ground cost matrix between the
            shared support points of `nominal` and any candidate
            distribution.
        dual_solver: Solver minimizing the dual objective over `gamma_raw`.
        initial_dual_point: Value of `gamma_raw` the first dual solve starts
            from; later solves warm-start from the previous optimum (see
            `optora.core.dro_base.DualAmbiguitySet`).
    """

    cost: torch.Tensor

    def __init__(
        self,
        nominal: torch.Tensor,
        cost: torch.Tensor,
        radius: float | torch.Tensor,
        epsilon: float = 0.1,
        sinkhorn_max_iter: int = 100,
        sinkhorn_tol: float = 1e-6,
        eps: float = 1e-12,
        dual_solver: Solver[MinimizationProblem, MinimizationResult] | None = None,
        initial_gamma: float = 0.0,
        validate: bool = False,
    ) -> None:
        """Initialize the Wasserstein-DRO ambiguity set.

        Args:
            nominal: Reference distribution the ambiguity set is centered
                on, a nonnegative tensor that sums to one along its last
                dimension.
            cost: Square, nonnegative pairwise ground cost matrix between
                the shared support points of `nominal` and any candidate
                distribution, shape `(n, n)` where `n = nominal.shape[-1]`.
            radius: Nonnegative bound on the Wasserstein distance of any
                distribution inside the ambiguity set from `nominal`. A
                tensor radius is broadcast against the batch shape of
                `worst_case_expectation`'s `loss`, evaluating a sweep of
                radii in one solve.
            epsilon: Positive entropic regularization strength for the
                `SinkhornDivergence` used by `contains(...)`.
            sinkhorn_max_iter: Maximum number of Sinkhorn scaling
                iterations, passed through to `SinkhornDivergence`.
            sinkhorn_tol: Sinkhorn scaling convergence tolerance, passed
                through to `SinkhornDivergence`.
            eps: Small positive constant used to clamp Sinkhorn scaling
                denominators away from zero, passed through to
                `SinkhornDivergence`.
            dual_solver: Solver minimizing the dual objective over
                `gamma_raw`. Required when evaluating a positive-radius set.
            initial_gamma: Value of `gamma_raw` the first dual solve starts
                from. Later calls warm-start from the previous solve's
                optimum unless `reset_warm_start()` is called.
            validate: Whether to check that `cost` is nonnegative, passed
                through to `SinkhornDivergence`, and that `nominal` is a
                valid probability distribution. The checks read reductions
                over `cost` and `nominal` on the host, which blocks until
                the device has produced them, so they are opt-in and off by
                default to keep construction asynchronous.

        Raises:
            ValueError: If `radius` is a negative float, if `cost` is not a
                square 2D tensor, if `cost`'s size does not match
                `nominal`'s support size, or if `validate` is set and
                `cost` contains negative entries or `nominal` is not a
                valid probability distribution.
        """
        # A scalar nominal has no support size to compare against; let
        # AmbiguitySet reject it with its own message.
        if nominal.ndim > 0 and cost.shape[-1] != nominal.shape[-1]:
            raise ValueError(
                "cost must have shape (n, n) matching nominal's support "
                f"size {nominal.shape[-1]}, got cost shape {tuple(cost.shape)}."
            )
        super().__init__(
            nominal=nominal,
            divergence=SinkhornDivergence(
                cost=cost,
                epsilon=epsilon,
                max_iter=sinkhorn_max_iter,
                tol=sinkhorn_tol,
                eps=eps,
                validate=validate,
            ),
            radius=radius,
            dual_solver=dual_solver,
            initial_dual_point=torch.tensor(
                initial_gamma, dtype=nominal.dtype, device=nominal.device
            ),
            validate=validate,
        )
        self.register_buffer("cost", cost)

    def worst_case_expectation(self, loss: torch.Tensor) -> torch.Tensor:
        """Compute the worst-case expected loss over the Wasserstein ambiguity set.

        Args:
            loss: Per-scenario loss values of shape `(..., n)`, one trailing
                entry per element of `nominal`'s support. Leading dimensions
                are a batch of independent loss vectors, each solved with
                its own dual variable `gamma` in a single joint solve.

        Returns:
            A tensor of shape `(...)` holding the worst-case expected loss:
            the exact `sum(nominal * loss)` when `radius` is zero (the
            ambiguity set then contains only `nominal`), otherwise the
            convex dual objective evaluated at the `gamma` found by
            `dual_solver`.

        Raises:
            ValueError: If `loss`'s trailing dimension does not match
                `nominal`'s support size, or its batch shape does not
                broadcast against `nominal` and `radius`.
            RuntimeError: If `radius` is positive and `dual_solver` is
                `None`.
        """
        batch_shape = self._batch_shape(loss)
        if self._radius_is_zero:
            return self._nominal_expectation(loss, batch_shape)

        def dual_objective(gamma_raw: torch.Tensor) -> torch.Tensor:
            # `where` keeps the one-sided derivative at gamma_raw = 0, where
            # `clamp` reports a zero subgradient and stalls gradient descent.
            gamma = torch.where(
                gamma_raw >= 0.0, gamma_raw, torch.zeros_like(gamma_raw)
            )
            shifted = loss.unsqueeze(-2) - gamma[..., None, None] * self.cost
            row_max = torch.amax(shifted, dim=-1)
            return gamma * self.radius + torch.sum(self.nominal * row_max, dim=-1)

        return self._solve_dual(dual_objective, batch_shape)
