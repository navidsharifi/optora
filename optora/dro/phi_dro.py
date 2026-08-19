"""Phi-divergence-constrained ambiguity sets (phi-DRO)."""

from collections.abc import Callable

import torch

from optora.core.dro_base import AmbiguitySet
from optora.core.solver_base import Solver
from optora.divergences.f_divergence import (
    ChiSquareDivergence,
    PhiDivergence,
    TotalVariationDivergence,
)
from optora.solvers.gradient_descent import (
    GradientDescent,
    GradientDescentProblem,
    GradientDescentResult,
)


def _chi_square_conjugate(scaled_shift: torch.Tensor) -> torch.Tensor:
    r"""Convex conjugate of the chi-square generator `phi(t) = (t - 1)^2`, `t >= 0`.

    `phi*(s) = sup_{t >= 0} (s * t - phi(t))`. The unconstrained maximizer
    `t = s / 2 + 1` is nonnegative whenever `s >= -2`, giving the interior
    branch `s + s^2 / 4`; otherwise the constrained maximizer sits at the
    boundary `t = 0`, giving the constant `-1`. The two branches agree at
    `s = -2`, and both have zero derivative there, so `phi*` is continuously
    differentiable everywhere on the real line.

    Args:
        scaled_shift: Elementwise dual argument `s = (loss - lam) / eta`.

    Returns:
        `phi*` evaluated elementwise on `scaled_shift`.
    """
    interior = scaled_shift + scaled_shift**2 / 4.0
    boundary = torch.full_like(scaled_shift, -1.0)
    return torch.where(scaled_shift >= -2.0, interior, boundary)


class PhiAmbiguitySet(AmbiguitySet):
    r"""Phi-divergence-constrained ambiguity set solved via its convex dual.

    Bounds every candidate distribution `q` by `D_phi(q || nominal) <=
    radius` for a general convex generator `phi` (see
    `optora.divergences.f_divergence.PhiDivergence`). The worst-case expected
    loss over this set admits a convex dual (Ben-Tal et al. 2013; Duchi,
    Glynn, and Namkoong 2021; Duchi and Namkoong 2021):

        sup_{q: D_phi(q||nominal) <= radius} E_q[loss]
            = inf_{eta > 0, lam in R}
                eta * radius + lam + eta * E_nominal[phi*((loss - lam) / eta)]

    where `phi*` is the convex (Legendre-Fenchel) conjugate of `phi`
    restricted to its effective domain `t >= 0`. This generalizes the
    `KLAmbiguitySet` dual to an arbitrary phi-divergence at the cost of a
    second dual variable `lam`; setting `phi(t) = t * log(t) - t + 1` (whose
    conjugate is `phi*(s) = exp(s) - 1`) recovers the KL-DRO dual exactly.
    `dual_solver` minimizes this joint objective over `(log(eta), lam)`
    rather than `(eta, lam)` directly, so the unconstrained `GradientDescent`
    solver keeps `eta` strictly positive throughout the iteration.

    This base class assumes `phi_conjugate` is finite everywhere on the real
    line (true for, for example, the chi-square generator's conjugate used
    by `ChiSquareAmbiguitySet`). Phi-divergences whose conjugate has a hard
    finite feasibility boundary (for example total variation, whose
    conjugate is `+inf` past a threshold) are not solved robustly by this
    unconstrained joint dual, since gradient descent can step past the
    boundary into a region of infinite objective value; `TotalVariationAmbiguitySet`
    instead computes its worst-case expectation from a dedicated closed
    form.

    Attributes:
        nominal: Reference distribution the ambiguity set is centered on.
        divergence: `PhiDivergence` instance measuring distance from
            `nominal`.
        radius: Nonnegative scalar bounding the phi-divergence of any
            distribution inside the ambiguity set from `nominal`.
        phi_conjugate: Convex (Legendre-Fenchel) conjugate of
            `divergence.phi`, finite everywhere on the real line.
        dual_solver: Solver minimizing the dual objective over
            `(log(eta), lam)`.
        initial_log_eta: Initial value of `log(eta)` passed to
            `dual_solver` for each `worst_case_expectation` call.
        initial_lam: Initial value of `lam` passed to `dual_solver` for
            each `worst_case_expectation` call.
    """

    def __init__(
        self,
        nominal: torch.Tensor,
        divergence: PhiDivergence,
        radius: float,
        phi_conjugate: Callable[[torch.Tensor], torch.Tensor],
        dual_solver: Solver[GradientDescentProblem, GradientDescentResult]
        | None = None,
        initial_log_eta: float = 0.0,
        initial_lam: float = 0.0,
    ) -> None:
        """Initialize the phi-divergence ambiguity set.

        Args:
            nominal: Reference distribution the ambiguity set is centered
                on, a nonnegative tensor that sums to one along its last
                dimension.
            divergence: `PhiDivergence` instance measuring distance from
                `nominal`.
            radius: Nonnegative scalar bounding the phi-divergence of any
                distribution inside the ambiguity set from `nominal`.
            phi_conjugate: Convex conjugate of `divergence.phi`, finite
                everywhere on the real line.
            dual_solver: Solver minimizing the dual objective over
                `(log(eta), lam)`. Defaults to a `GradientDescent` instance
                tuned for this reparameterization.
            initial_log_eta: Initial value of `log(eta)` passed to
                `dual_solver` for each `worst_case_expectation` call.
            initial_lam: Initial value of `lam` passed to `dual_solver` for
                each `worst_case_expectation` call.

        Raises:
            ValueError: If `radius` is negative.
        """
        super().__init__(nominal=nominal, divergence=divergence, radius=radius)
        self.phi_conjugate = phi_conjugate
        self.dual_solver: Solver[GradientDescentProblem, GradientDescentResult] = (
            dual_solver
            if dual_solver is not None
            else GradientDescent(step_size=0.05, max_iter=5000, tol=1e-9)
        )
        self.initial_log_eta = initial_log_eta
        self.initial_lam = initial_lam

    def worst_case_expectation(self, loss: torch.Tensor) -> torch.Tensor:
        """Compute the worst-case expected loss over the phi-divergence ambiguity set.

        Args:
            loss: Per-scenario loss values, one entry per element of
                `nominal`'s support.

        Returns:
            A scalar tensor holding the worst-case expected loss: the exact
            `sum(nominal * loss)` when `radius` is zero (the ambiguity set
            then contains only `nominal`), otherwise the convex dual
            objective evaluated at the `(log(eta), lam)` found by
            `dual_solver`.

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

        def dual_objective(params: torch.Tensor) -> torch.Tensor:
            eta = torch.exp(params[0])
            lam = params[1]
            conjugate_term = torch.sum(
                self.nominal * self.phi_conjugate((loss - lam) / eta)
            )
            return eta * self.radius + lam + eta * conjugate_term

        initial_point = torch.stack(
            [
                torch.tensor(
                    self.initial_log_eta, dtype=loss.dtype, device=loss.device
                ),
                torch.tensor(self.initial_lam, dtype=loss.dtype, device=loss.device),
            ]
        )
        problem = GradientDescentProblem(
            objective=dual_objective, initial_point=initial_point
        )
        result = self.dual_solver.solve(problem)
        return dual_objective(result.point)


class ChiSquareAmbiguitySet(PhiAmbiguitySet):
    r"""Chi-square-divergence-constrained ambiguity set for chi-square-DRO.

    Fixes `divergence` to a `ChiSquareDivergence` and `phi_conjugate` to the
    closed-form conjugate of `phi(t) = (t - 1)^2`, which is finite and
    continuously differentiable everywhere on the real line (see
    `_chi_square_conjugate`), making the joint `PhiAmbiguitySet` dual solve
    over `(log(eta), lam)` numerically well-behaved.

    In the interior regime where no candidate distribution is pushed to the
    boundary `q_i = 0`, the dual optimum over `lam` reduces to
    `lam = E_nominal[loss]`, and the dual optimum over `eta` reduces to
    `eta = sqrt(Var_nominal(loss) / (4 * radius))`, giving the well-known
    closed form (Duchi and Namkoong 2021):

        sup_{q: D_chi2(q||nominal) <= radius} E_q[loss]
            = E_nominal[loss] + sqrt(radius * Var_nominal(loss))

    `worst_case_expectation` still solves the general dual rather than this
    closed form directly, since the closed form only holds away from the
    boundary regime.
    """

    def __init__(
        self,
        nominal: torch.Tensor,
        radius: float,
        eps: float = 1e-12,
        dual_solver: Solver[GradientDescentProblem, GradientDescentResult]
        | None = None,
        initial_log_eta: float = 0.0,
        initial_lam: float = 0.0,
    ) -> None:
        """Initialize the chi-square ambiguity set.

        Args:
            nominal: Reference distribution the ambiguity set is centered
                on, a nonnegative tensor that sums to one along its last
                dimension.
            radius: Nonnegative scalar bounding the chi-square divergence of
                any distribution inside the ambiguity set from `nominal`.
            eps: Small positive constant used to clamp `nominal` away from
                zero before dividing, passed through to the underlying
                `ChiSquareDivergence`.
            dual_solver: Solver minimizing the dual objective over
                `(log(eta), lam)`. Defaults to a `GradientDescent` instance
                tuned for this reparameterization.
            initial_log_eta: Initial value of `log(eta)` passed to
                `dual_solver` for each `worst_case_expectation` call.
            initial_lam: Initial value of `lam` passed to `dual_solver` for
                each `worst_case_expectation` call.

        Raises:
            ValueError: If `radius` is negative or `eps` is not positive.
        """
        super().__init__(
            nominal=nominal,
            divergence=ChiSquareDivergence(eps=eps),
            radius=radius,
            phi_conjugate=_chi_square_conjugate,
            dual_solver=dual_solver,
            initial_log_eta=initial_log_eta,
            initial_lam=initial_lam,
        )


class TotalVariationAmbiguitySet(AmbiguitySet):
    r"""Total-variation-constrained ambiguity set for total-variation-DRO.

    Bounds every candidate distribution `q` by
    `D_TV(q || nominal) = 0.5 * sum_i |q_i - nominal_i| <= radius`. Unlike
    `PhiAmbiguitySet`, `worst_case_expectation` is computed from a direct
    closed form rather than the general convex dual, because total
    variation's conjugate has a hard finite feasibility boundary that is
    not well suited to unconstrained gradient-based dual optimization (see
    `PhiAmbiguitySet`).

    The worst-case expectation is instead the value of a linear program over
    the simplex intersected with the total-variation ball, whose optimal
    solution has a simple combinatorial structure (Ben-Tal et al. 2013):
    starting from `nominal`, reallocate mass, in ascending order of loss,
    from the lowest-loss scenarios to the single highest-loss scenario,
    until the reallocated mass reaches `radius` (or every scenario but the
    highest-loss one has been fully drained, whichever happens first). This
    is implemented as a sort followed by a cumulative-sum sweep rather than
    an iterative solve, so it is both exact and free of solver tuning.

    Attributes:
        nominal: Reference distribution the ambiguity set is centered on.
        divergence: `TotalVariationDivergence` instance measuring distance
            from `nominal`.
        radius: Nonnegative scalar bounding the total variation distance of
            any distribution inside the ambiguity set from `nominal`.
    """

    def __init__(
        self,
        nominal: torch.Tensor,
        radius: float,
        eps: float = 1e-12,
    ) -> None:
        """Initialize the total variation ambiguity set.

        Args:
            nominal: Reference distribution the ambiguity set is centered
                on, a nonnegative tensor that sums to one along its last
                dimension.
            radius: Nonnegative scalar bounding the total variation distance
                of any distribution inside the ambiguity set from `nominal`.
            eps: Small positive constant used to clamp the reference
                distribution away from zero before dividing, passed through
                to the underlying `TotalVariationDivergence`.

        Raises:
            ValueError: If `radius` is negative or `eps` is not positive.
        """
        super().__init__(
            nominal=nominal,
            divergence=TotalVariationDivergence(eps=eps),
            radius=radius,
        )

    def worst_case_expectation(self, loss: torch.Tensor) -> torch.Tensor:
        """Compute the worst-case expected loss over the total variation ambiguity set.

        Args:
            loss: Per-scenario loss values, one entry per element of
                `nominal`'s support.

        Returns:
            A scalar tensor holding the worst-case expected loss: the exact
            `sum(nominal * loss)` when `radius` is zero (the ambiguity set
            then contains only `nominal`), otherwise the closed-form value
            of the mass-reallocation linear program described in the class
            docstring.

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

        sorted_loss, sort_index = torch.sort(loss)
        sorted_nominal = self.nominal[sort_index]

        rest_nominal = sorted_nominal[:-1]
        rest_loss = sorted_loss[:-1]
        max_loss = sorted_loss[-1]

        mass_available_before = torch.cumsum(rest_nominal, dim=0) - rest_nominal
        remaining_budget = torch.clamp(self.radius - mass_available_before, min=0.0)
        reallocated_mass = torch.minimum(remaining_budget, rest_nominal)

        return torch.sum(self.nominal * loss) + torch.sum(
            reallocated_mass * (max_loss - rest_loss)
        )
