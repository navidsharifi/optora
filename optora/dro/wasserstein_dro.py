"""Wasserstein-distance-constrained ambiguity set (Wasserstein-DRO)."""

import torch

from optora.core.dro_base import AmbiguitySet
from optora.core.solver_base import (
    MinimizationProblem,
    MinimizationResult,
    Solver,
)
from optora.divergences.wasserstein import SinkhornDivergence


class WassersteinAmbiguitySet(AmbiguitySet):
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
    $\gamma = \mathrm{clamp}(\mathrm{gamma\_raw}, \min=0)$
    rather than $\exp(\log(\eta))$ (the reparameterization used by
    `KLAmbiguitySet` and `PhiAmbiguitySet`): unlike those formulations' dual
    variable, which must stay strictly positive, $\gamma$ ranges over the
    closed half-line $[0, \infty)$ and its optimum is genuinely attained at
    $\gamma = 0$ once `radius` is large enough to move all nominal mass onto
    the single highest-loss support point (`clamp` lets unconstrained
    `GradientDescent` reach that boundary exactly, rather than only approach
    it asymptotically).

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
        radius: Nonnegative scalar bounding the Wasserstein distance of any
            distribution inside the ambiguity set from `nominal`.
        cost: Square, nonnegative pairwise ground cost matrix between the
            shared support points of `nominal` and any candidate
            distribution.
        dual_solver: Solver minimizing the dual objective over `gamma_raw`.
        initial_gamma: Initial value of `gamma_raw` passed to
            `dual_solver` for each `worst_case_expectation` call.
    """

    cost: torch.Tensor

    def __init__(
        self,
        nominal: torch.Tensor,
        cost: torch.Tensor,
        radius: float,
        epsilon: float = 0.1,
        sinkhorn_max_iter: int = 100,
        sinkhorn_tol: float = 1e-6,
        eps: float = 1e-12,
        dual_solver: Solver[MinimizationProblem, MinimizationResult] | None = None,
        initial_gamma: float = 0.0,
    ) -> None:
        """Initialize the Wasserstein-DRO ambiguity set.

        Args:
            nominal: Reference distribution the ambiguity set is centered
                on, a nonnegative tensor that sums to one along its last
                dimension.
            cost: Square, nonnegative pairwise ground cost matrix between
                the shared support points of `nominal` and any candidate
                distribution, shape `(n, n)` where `n = nominal.shape[-1]`.
            radius: Nonnegative scalar bounding the Wasserstein distance of
                any distribution inside the ambiguity set from `nominal`.
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
            initial_gamma: Initial value of `gamma_raw` passed to
                `dual_solver` for each `worst_case_expectation` call.

        Raises:
            ValueError: If `radius` is negative, if `cost` is not a square
                2D tensor, if `cost` contains negative entries, or if
                `cost`'s size does not match `nominal`'s support size.
        """
        if cost.shape[-1] != nominal.shape[-1]:
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
            ),
            radius=radius,
        )
        self.register_buffer("cost", cost)
        self.dual_solver = dual_solver
        self.initial_gamma = initial_gamma

    def worst_case_expectation(self, loss: torch.Tensor) -> torch.Tensor:
        """Compute the worst-case expected loss over the Wasserstein ambiguity set.

        Args:
            loss: Per-scenario loss values, one entry per element of
                `nominal`'s support.

        Returns:
            A scalar tensor holding the worst-case expected loss: the exact
            `sum(nominal * loss)` when `radius` is zero (the ambiguity set
            then contains only `nominal`), otherwise the convex dual
            objective evaluated at the `gamma` found by `dual_solver`.

        Raises:
            ValueError: If `loss` does not have the same shape as
                `nominal`.
        """
        if loss.shape != self.nominal.shape:
            raise ValueError(
                f"loss must have the same shape as nominal, got {tuple(loss.shape)} "
                f"and {tuple(self.nominal.shape)}."
            )
        if self.radius == 0.0:
            return torch.sum(self.nominal * loss)

        def dual_objective(gamma_raw: torch.Tensor) -> torch.Tensor:
            gamma = torch.clamp(gamma_raw, min=0.0)
            shifted = loss.unsqueeze(-2) - gamma * self.cost
            row_max = torch.amax(shifted, dim=-1)
            return gamma * self.radius + torch.sum(self.nominal * row_max)

        problem = MinimizationProblem(
            objective=dual_objective,
            initial_point=torch.tensor(
                self.initial_gamma, dtype=loss.dtype, device=loss.device
            ),
        )
        if self.dual_solver is None:
            raise RuntimeError(
                "dual_solver is required to evaluate a positive-radius "
                "WassersteinAmbiguitySet."
            )
        result = self.dual_solver.solve(problem)
        return dual_objective(result.point)
