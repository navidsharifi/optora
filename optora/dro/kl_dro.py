"""KL-divergence-constrained ambiguity set (KL-DRO)."""

import torch

from optora.core.dro_base import AmbiguitySet
from optora.core.solver_base import Solver
from optora.divergences.kl import KLDivergence
from optora.solvers.gradient_descent import (
    GradientDescent,
    GradientDescentProblem,
    GradientDescentResult,
)


class KLAmbiguitySet(AmbiguitySet):
    r"""KL-divergence-constrained ambiguity set for KL-DRO.

    Bounds every candidate distribution `q` by
    `D_KL(q || nominal) <= radius`. The worst-case expected loss over this
    set admits a convex dual (Hu and Hong 2013; Ben-Tal et al. 2013):

        sup_{q: D_KL(q || nominal) <= radius} E_q[loss]
            = inf_{eta > 0} eta * radius + eta * log E_nominal[exp(loss / eta)]

    reducing the worst-case expectation to a one-dimensional convex
    minimization over the dual variable `eta`. `dual_solver` solves this
    minimization over `log(eta)` rather than `eta` directly, so the
    unconstrained `GradientDescent` solver keeps `eta` strictly positive
    throughout the iteration. This formulation needs no optimal-transport
    machinery, making it the simplest DRO formulation to build (see
    `progress/architecture.md`).

    Attributes:
        nominal: Reference distribution the ambiguity set is centered on.
        divergence: `KLDivergence` instance measuring distance from
            `nominal`.
        radius: Nonnegative scalar bounding the KL divergence of any
            distribution inside the ambiguity set from `nominal`.
        eps: Small positive constant used to clamp `nominal` away from zero
            before taking the logarithm inside the dual objective.
        dual_solver: Solver minimizing the dual objective over `log(eta)`.
        initial_log_eta: Initial value of `log(eta)` passed to
            `dual_solver` for each `worst_case_expectation` call.
    """

    def __init__(
        self,
        nominal: torch.Tensor,
        radius: float,
        eps: float = 1e-12,
        dual_solver: Solver[GradientDescentProblem, GradientDescentResult]
        | None = None,
        initial_log_eta: float = 0.0,
    ) -> None:
        """Initialize the KL-DRO ambiguity set.

        Args:
            nominal: Reference distribution the ambiguity set is centered
                on, a nonnegative tensor that sums to one along its last
                dimension.
            radius: Nonnegative scalar bounding the KL divergence of any
                distribution inside the ambiguity set from `nominal`.
            eps: Small positive constant used to clamp `nominal` away from
                zero before taking the logarithm inside the dual objective,
                and passed through to the underlying `KLDivergence`.
            dual_solver: Solver minimizing the dual objective over
                `log(eta)`. Defaults to a `GradientDescent` instance tuned
                for this reparameterization.
            initial_log_eta: Initial value of `log(eta)` passed to
                `dual_solver` for each `worst_case_expectation` call.

        Raises:
            ValueError: If `radius` is negative or `eps` is not positive.
        """
        super().__init__(
            nominal=nominal, divergence=KLDivergence(eps=eps), radius=radius
        )
        self.eps = eps
        self.dual_solver: Solver[GradientDescentProblem, GradientDescentResult] = (
            dual_solver
            if dual_solver is not None
            else GradientDescent(step_size=0.1, max_iter=2000, tol=1e-9)
        )
        self.initial_log_eta = initial_log_eta

    def worst_case_expectation(self, loss: torch.Tensor) -> torch.Tensor:
        """Compute the worst-case expected loss over the KL ambiguity set.

        Args:
            loss: Per-scenario loss values, one entry per element of
                `nominal`'s support.

        Returns:
            A scalar tensor holding the worst-case expected loss: the exact
            `sum(nominal * loss)` when `radius` is zero (the ambiguity set
            then contains only `nominal`), otherwise the convex dual
            objective evaluated at the `log(eta)` found by `dual_solver`.

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

        log_nominal = torch.log(torch.clamp(self.nominal, min=self.eps))

        def dual_objective(log_eta: torch.Tensor) -> torch.Tensor:
            eta = torch.exp(log_eta)
            log_mgf = torch.logsumexp(log_nominal + loss / eta, dim=-1)
            return eta * self.radius + eta * log_mgf

        problem = GradientDescentProblem(
            objective=dual_objective,
            initial_point=torch.tensor(
                self.initial_log_eta, dtype=loss.dtype, device=loss.device
            ),
        )
        result = self.dual_solver.solve(problem)
        return dual_objective(result.point)
