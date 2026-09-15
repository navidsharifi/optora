"""KL-divergence-constrained ambiguity set (KL-DRO)."""

import torch

from optora.core.dro_base import DualAmbiguitySet
from optora.core.solver_base import (
    MinimizationProblem,
    MinimizationResult,
    Solver,
)
from optora.divergences.kl import KLDivergence


class KLAmbiguitySet(DualAmbiguitySet):
    r"""KL-divergence-constrained ambiguity set for KL-DRO.

    Bounds every candidate distribution `q` by
    $D_{\mathrm{KL}}(q \,\|\, \mathrm{nominal}) \le \mathrm{radius}$. The
    worst-case expected loss over this set admits a convex dual (Hu and Hong
    2013; Ben-Tal et al. 2013):

    $$
    \sup_{q:\, D_{\mathrm{KL}}(q \,\|\, \mathrm{nominal}) \,\le\, \mathrm{radius}}
        \mathbb{E}_q[\mathrm{loss}]
    = \inf_{\eta > 0} \; \eta \cdot \mathrm{radius}
        + \eta \log
        \mathbb{E}_{\mathrm{nominal}}\!\left[\exp\!\left(\frac{\mathrm{loss}}{\eta}\right)\right]
    $$

    reducing the worst-case expectation to a one-dimensional convex
    minimization over the dual variable $\eta$. `dual_solver` solves this
    minimization over $\log(\eta)$ rather than $\eta$ directly, so the
    unconstrained `GradientDescent` solver keeps $\eta$ strictly positive
    throughout the iteration. This formulation needs no optimal-transport
    machinery, making it the simplest DRO formulation to build (see
    `progress/architecture.md`).

    Attributes:
        nominal: Reference distribution the ambiguity set is centered on.
        divergence: `KLDivergence` instance measuring distance from
            `nominal`.
        radius: Nonnegative bound on the KL divergence of any distribution
            inside the ambiguity set from `nominal`, either a float or a
            tensor of radii evaluated as one batch.
        eps: Small positive constant used to clamp `nominal` away from zero
            before taking the logarithm inside the dual objective.
        dual_solver: Solver minimizing the dual objective over `log(eta)`.
        initial_dual_point: Value of `log(eta)` the first dual solve starts
            from; later solves warm-start from the previous optimum (see
            `optora.core.dro_base.DualAmbiguitySet`).
        log_nominal: Elementwise logarithm of the clamped `nominal`, a
            constant of the dual objective cached once rather than
            recomputed on every call.
    """

    log_nominal: torch.Tensor

    def __init__(
        self,
        nominal: torch.Tensor,
        radius: float | torch.Tensor,
        eps: float = 1e-12,
        dual_solver: Solver[MinimizationProblem, MinimizationResult] | None = None,
        initial_log_eta: float = 0.0,
    ) -> None:
        """Initialize the KL-DRO ambiguity set.

        Args:
            nominal: Reference distribution the ambiguity set is centered
                on, a nonnegative tensor that sums to one along its last
                dimension.
            radius: Nonnegative bound on the KL divergence of any
                distribution inside the ambiguity set from `nominal`. A
                tensor radius is broadcast against the batch shape of
                `worst_case_expectation`'s `loss`, evaluating a sweep of
                radii in one solve.
            eps: Small positive constant used to clamp `nominal` away from
                zero before taking the logarithm inside the dual objective,
                and passed through to the underlying `KLDivergence`.
            dual_solver: Solver minimizing the dual objective over
                `log(eta)`. Required when evaluating a positive-radius set.
            initial_log_eta: Value of `log(eta)` the first dual solve starts
                from. Later calls warm-start from the previous solve's
                optimum unless `reset_warm_start()` is called.

        Raises:
            ValueError: If `radius` is a negative float or `eps` is not
                positive.
        """
        super().__init__(
            nominal=nominal,
            divergence=KLDivergence(eps=eps),
            radius=radius,
            dual_solver=dual_solver,
            initial_dual_point=torch.tensor(
                initial_log_eta, dtype=nominal.dtype, device=nominal.device
            ),
        )
        self.eps = eps
        self.register_buffer(
            "log_nominal",
            torch.log(torch.clamp(nominal, min=eps)),
            persistent=False,
        )

    def worst_case_expectation(self, loss: torch.Tensor) -> torch.Tensor:
        """Compute the worst-case expected loss over the KL ambiguity set.

        Args:
            loss: Per-scenario loss values of shape `(..., n)`, one trailing
                entry per element of `nominal`'s support. Leading dimensions
                are a batch of independent loss vectors, each solved with
                its own dual variable `log(eta)` in a single joint solve.

        Returns:
            A tensor of shape `(...)` holding the worst-case expected loss:
            the exact `sum(nominal * loss)` when `radius` is zero (the
            ambiguity set then contains only `nominal`), otherwise the
            convex dual objective evaluated at the `log(eta)` found by
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

        def dual_objective(log_eta: torch.Tensor) -> torch.Tensor:
            eta = torch.exp(log_eta)
            log_mgf = torch.logsumexp(
                self.log_nominal + loss / eta.unsqueeze(-1), dim=-1
            )
            return eta * self.radius + eta * log_mgf

        return self._solve_dual(dual_objective, batch_shape)
