"""Phi-divergence-constrained ambiguity sets (phi-DRO)."""

from collections.abc import Callable

import torch

from optora.core.dro_base import AmbiguitySet, DualAmbiguitySet
from optora.core.solver_base import (
    MinimizationProblem,
    MinimizationResult,
    Solver,
)
from optora.divergences.f_divergence import (
    ChiSquareDivergence,
    PhiDivergence,
    TotalVariationDivergence,
)


def _chi_square_conjugate(scaled_shift: torch.Tensor) -> torch.Tensor:
    r"""Convex conjugate of the chi-square generator $\phi(t) = (t-1)^2$, $t \ge 0$.

    $\phi^*(s) = \sup_{t \ge 0} \big(s\,t - \phi(t)\big)$. The unconstrained
    maximizer $t = s/2 + 1$ is nonnegative whenever $s \ge -2$, giving the
    interior branch $s + s^2/4$; otherwise the constrained maximizer sits at
    the boundary $t = 0$, giving the constant $-1$. The two branches agree
    at $s = -2$, and both have zero derivative there, so $\phi^*$ is
    continuously differentiable everywhere on the real line.

    Args:
        scaled_shift: Elementwise dual argument
            $s = (\mathrm{loss} - \lambda) / \eta$.

    Returns:
        $\phi^*$ evaluated elementwise on `scaled_shift`.
    """
    interior = scaled_shift + scaled_shift**2 / 4.0
    boundary = torch.full_like(scaled_shift, -1.0)
    return torch.where(scaled_shift >= -2.0, interior, boundary)


class PhiAmbiguitySet(DualAmbiguitySet):
    r"""Phi-divergence-constrained ambiguity set solved via its convex dual.

    Bounds every candidate distribution `q` by
    $D_\phi(q \,\|\, \mathrm{nominal}) \le \mathrm{radius}$ for a general
    convex generator $\phi$ (see
    `optora.divergences.f_divergence.PhiDivergence`). The worst-case expected
    loss over this set admits a convex dual (Ben-Tal et al. 2013; Duchi,
    Glynn, and Namkoong 2021; Duchi and Namkoong 2021):

    $$
    \sup_{q:\, D_\phi(q \,\|\, \mathrm{nominal}) \,\le\, \mathrm{radius}}
        \mathbb{E}_q[\mathrm{loss}]
    = \inf_{\substack{\eta > 0 \\ \lambda \in \mathbb{R}}}
        \eta \cdot \mathrm{radius} + \lambda
        + \eta \, \mathbb{E}_{\mathrm{nominal}}\!\left[
            \phi^*\!\left(\frac{\mathrm{loss} - \lambda}{\eta}\right)
        \right]
    $$

    where $\phi^*$ is the convex (Legendre-Fenchel) conjugate of $\phi$
    restricted to its effective domain $t \ge 0$. This generalizes the
    `KLAmbiguitySet` dual to an arbitrary phi-divergence at the cost of a
    second dual variable $\lambda$; setting $\phi(t) = t \log t - t + 1$
    (whose conjugate is $\phi^*(s) = \exp(s) - 1$) recovers the KL-DRO dual
    exactly. `dual_solver` minimizes this joint objective over
    $(\log(\eta), \lambda)$ rather than $(\eta, \lambda)$ directly, so the
    unconstrained `GradientDescent` solver keeps $\eta$ strictly positive
    throughout the iteration.

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
        radius: Nonnegative bound on the phi-divergence of any distribution
            inside the ambiguity set from `nominal`, either a float or a
            tensor of radii evaluated as one batch.
        phi_conjugate: Convex (Legendre-Fenchel) conjugate of
            `divergence.phi`, finite everywhere on the real line.
        dual_solver: Solver minimizing the dual objective over
            `(log(eta), lam)`.
        initial_dual_point: Values of `(log(eta), lam)` the first dual solve
            starts from; later solves warm-start from the previous optimum
            (see `optora.core.dro_base.DualAmbiguitySet`).
    """

    def __init__(
        self,
        nominal: torch.Tensor,
        divergence: PhiDivergence,
        radius: float | torch.Tensor,
        phi_conjugate: Callable[[torch.Tensor], torch.Tensor],
        dual_solver: Solver[MinimizationProblem, MinimizationResult] | None = None,
        initial_log_eta: float = 0.0,
        initial_lam: float = 0.0,
        validate: bool = False,
    ) -> None:
        """Initialize the phi-divergence ambiguity set.

        Args:
            nominal: Reference distribution the ambiguity set is centered
                on, a nonnegative tensor that sums to one along its last
                dimension.
            divergence: `PhiDivergence` instance measuring distance from
                `nominal`.
            radius: Nonnegative bound on the phi-divergence of any
                distribution inside the ambiguity set from `nominal`. A
                tensor radius is broadcast against the batch shape of
                `worst_case_expectation`'s `loss`.
            phi_conjugate: Convex conjugate of `divergence.phi`, finite
                everywhere on the real line.
            dual_solver: Solver minimizing the dual objective over
                `(log(eta), lam)`. Required when evaluating a positive-radius
                set.
            initial_log_eta: Value of `log(eta)` the first dual solve starts
                from. Later calls warm-start from the previous solve's
                optimum unless `reset_warm_start()` is called.
            initial_lam: Value of `lam` the first dual solve starts from,
                warm-started on later calls alongside `initial_log_eta`.
            validate: Whether to check that `nominal` is nonnegative and
                sums to one. The check synchronizes with the device, so it
                is opt-in and off by default.

        Raises:
            ValueError: If `radius` is a negative float, or if `validate`
                is set and `nominal` is not a valid probability
                distribution.
        """
        super().__init__(
            nominal=nominal,
            divergence=divergence,
            radius=radius,
            dual_solver=dual_solver,
            initial_dual_point=torch.tensor(
                [initial_log_eta, initial_lam],
                dtype=nominal.dtype,
                device=nominal.device,
            ),
            validate=validate,
        )
        self.phi_conjugate = phi_conjugate

    def worst_case_expectation(self, loss: torch.Tensor) -> torch.Tensor:
        """Compute the worst-case expected loss over the phi-divergence ambiguity set.

        Args:
            loss: Per-scenario loss values of shape `(..., n)`, one trailing
                entry per element of `nominal`'s support. Leading dimensions
                are a batch of independent loss vectors, each solved with
                its own dual pair `(log(eta), lam)` in a single joint solve.

        Returns:
            A tensor of shape `(...)` holding the worst-case expected loss:
            the exact `sum(nominal * loss)` when `radius` is zero (the
            ambiguity set then contains only `nominal`), otherwise the
            convex dual objective evaluated at the `(log(eta), lam)` found
            by `dual_solver`.

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

        def dual_objective(params: torch.Tensor) -> torch.Tensor:
            eta = torch.exp(params[..., 0])
            lam = params[..., 1]
            scaled_shift = (loss - lam.unsqueeze(-1)) / eta.unsqueeze(-1)
            conjugate_term = torch.sum(
                self.nominal * self.phi_conjugate(scaled_shift), dim=-1
            )
            return eta * self.radius + lam + eta * conjugate_term

        return self._solve_dual(dual_objective, batch_shape)


class ChiSquareAmbiguitySet(PhiAmbiguitySet):
    r"""Chi-square-divergence-constrained ambiguity set for chi-square-DRO.

    Fixes `divergence` to a `ChiSquareDivergence` and `phi_conjugate` to the
    closed-form conjugate of $\phi(t) = (t-1)^2$, which is finite and
    continuously differentiable everywhere on the real line (see
    `_chi_square_conjugate`), making the joint `PhiAmbiguitySet` dual solve
    over $(\log(\eta), \lambda)$ numerically well-behaved.

    In the interior regime where no candidate distribution is pushed to the
    boundary $q_i = 0$, the dual optimum over $\lambda$ reduces to
    $\lambda = \mathbb{E}_{\mathrm{nominal}}[\mathrm{loss}]$, and the dual
    optimum over $\eta$ reduces to
    $\eta = \sqrt{\mathrm{Var}_{\mathrm{nominal}}(\mathrm{loss})
        / (4\,\mathrm{radius})}$,
    giving the well-known closed form (Duchi and Namkoong 2021):

    $$
    \sup_{q:\, D_{\chi^2}(q \,\|\, \mathrm{nominal}) \,\le\, \mathrm{radius}}
        \mathbb{E}_q[\mathrm{loss}]
    = \mathbb{E}_{\mathrm{nominal}}[\mathrm{loss}]
        + \sqrt{\mathrm{radius} \cdot \mathrm{Var}_{\mathrm{nominal}}(\mathrm{loss})}
    $$

    `worst_case_expectation` still solves the general dual rather than this
    closed form directly, since the closed form only holds away from the
    boundary regime.
    """

    def __init__(
        self,
        nominal: torch.Tensor,
        radius: float | torch.Tensor,
        eps: float = 1e-12,
        dual_solver: Solver[MinimizationProblem, MinimizationResult] | None = None,
        initial_log_eta: float = 0.0,
        initial_lam: float = 0.0,
        validate: bool = False,
    ) -> None:
        """Initialize the chi-square ambiguity set.

        Args:
            nominal: Reference distribution the ambiguity set is centered
                on, a nonnegative tensor that sums to one along its last
                dimension.
            radius: Nonnegative bound on the chi-square divergence of any
                distribution inside the ambiguity set from `nominal`. A
                tensor radius is broadcast against the batch shape of
                `worst_case_expectation`'s `loss`.
            eps: Small positive constant used to clamp `nominal` away from
                zero before dividing, passed through to the underlying
                `ChiSquareDivergence`.
            dual_solver: Solver minimizing the dual objective over
                `(log(eta), lam)`. Required when evaluating a positive-radius
                set.
            initial_log_eta: Value of `log(eta)` the first dual solve starts
                from. Later calls warm-start from the previous solve's
                optimum unless `reset_warm_start()` is called.
            initial_lam: Value of `lam` the first dual solve starts from,
                warm-started on later calls alongside `initial_log_eta`.
            validate: Whether to check that `nominal` is nonnegative and
                sums to one. The check synchronizes with the device, so it
                is opt-in and off by default.

        Raises:
            ValueError: If `radius` is a negative float, if `eps` is not
                positive, or if `validate` is set and `nominal` is not a
                valid probability distribution.
        """
        super().__init__(
            nominal=nominal,
            divergence=ChiSquareDivergence(eps=eps),
            radius=radius,
            phi_conjugate=_chi_square_conjugate,
            dual_solver=dual_solver,
            initial_log_eta=initial_log_eta,
            initial_lam=initial_lam,
            validate=validate,
        )


class TotalVariationAmbiguitySet(AmbiguitySet):
    r"""Total-variation-constrained ambiguity set for total-variation-DRO.

    Bounds every candidate distribution `q` by

    $$
    D_{\mathrm{TV}}(q \,\|\, \mathrm{nominal})
        = \frac{1}{2} \sum_i |q_i - \mathrm{nominal}_i| \le \mathrm{radius}.
    $$

    Unlike
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
        radius: Nonnegative bound on the total variation distance of any
            distribution inside the ambiguity set from `nominal`, either a
            float or a tensor of radii evaluated as one batch.
    """

    def __init__(
        self,
        nominal: torch.Tensor,
        radius: float | torch.Tensor,
        eps: float = 1e-12,
        validate: bool = False,
    ) -> None:
        """Initialize the total variation ambiguity set.

        Args:
            nominal: Reference distribution the ambiguity set is centered
                on, a nonnegative tensor that sums to one along its last
                dimension.
            radius: Nonnegative bound on the total variation distance of any
                distribution inside the ambiguity set from `nominal`. A
                tensor radius is broadcast against the batch shape of
                `worst_case_expectation`'s `loss`, which turns a radius
                sweep into one vectorized evaluation of the closed form.
            eps: Small positive constant used to clamp the reference
                distribution away from zero before dividing, passed through
                to the underlying `TotalVariationDivergence`.
            validate: Whether to check that `nominal` is nonnegative and
                sums to one. The check synchronizes with the device, so it
                is opt-in and off by default.

        Raises:
            ValueError: If `radius` is a negative float, if `eps` is not
                positive, or if `validate` is set and `nominal` is not a
                valid probability distribution.
        """
        super().__init__(
            nominal=nominal,
            divergence=TotalVariationDivergence(eps=eps),
            radius=radius,
            validate=validate,
        )

    def worst_case_expectation(self, loss: torch.Tensor) -> torch.Tensor:
        """Compute the worst-case expected loss over the total variation ambiguity set.

        Args:
            loss: Per-scenario loss values of shape `(..., n)`, one trailing
                entry per element of `nominal`'s support. Leading dimensions
                are a batch of independent loss vectors; the closed form is
                evaluated over the whole batch at once, with no solver and
                no Python-level loop over batch elements.

        Returns:
            A tensor of shape `(...)` holding the worst-case expected loss:
            the exact `sum(nominal * loss)` when `radius` is zero (the
            ambiguity set then contains only `nominal`), otherwise the
            closed-form value of the mass-reallocation linear program
            described in the class docstring.

        Raises:
            ValueError: If `loss`'s trailing dimension does not match
                `nominal`'s support size, or its batch shape does not
                broadcast against `nominal` and `radius`.
        """
        batch_shape = self._batch_shape(loss)
        if self._radius_is_zero:
            return self._nominal_expectation(loss, batch_shape)

        sorted_loss, sort_index = torch.sort(loss, dim=-1)
        sorted_nominal = self.nominal.expand_as(loss).gather(-1, sort_index)

        rest_nominal = sorted_nominal[..., :-1]
        rest_loss = sorted_loss[..., :-1]
        max_loss = sorted_loss[..., -1:]

        radius = self.radius
        budget = radius.unsqueeze(-1) if isinstance(radius, torch.Tensor) else radius
        mass_available_before = torch.cumsum(rest_nominal, dim=-1) - rest_nominal
        remaining_budget = torch.clamp(budget - mass_available_before, min=0.0)
        reallocated_mass = torch.minimum(remaining_budget, rest_nominal)

        nominal_expectation = torch.sum(self.nominal * loss, dim=-1)
        reallocation_gain = torch.sum(reallocated_mass * (max_loss - rest_loss), dim=-1)
        return (nominal_expectation + reallocation_gain).expand(batch_shape)
