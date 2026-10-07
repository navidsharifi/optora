"""Phi-divergence-constrained ambiguity sets (phi-DRO)."""

from collections.abc import Callable

import torch

from optora.core.dro_base import AmbiguitySet, DualAmbiguitySet, TiltedAmbiguitySet
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
    line (true for, for example, `_chi_square_conjugate`). Phi-divergences
    whose conjugate has a hard finite feasibility boundary (for example
    total variation, whose conjugate is `+inf` past a threshold) are not
    solved robustly by this unconstrained joint dual, since gradient
    descent can step past the boundary into a region of infinite objective
    value; `TotalVariationAmbiguitySet` instead computes its worst-case
    expectation from a dedicated closed form.

    A fixed-step first-order solver on this joint dual is the fallback for
    a generator optora does not recognize, not the preferred route. It is
    sensitive to the support size and to the radius: the minimizer runs off
    to infinity as the radius vanishes, and a large support can push the
    iterate into a region the conjugate overflows. `KLAmbiguitySet` and
    `ChiSquareAmbiguitySet` therefore do not use it, and instead pin their
    known maximizer with the bracketed bisection of
    `optora.core.dro_base.TiltedAmbiguitySet`. Prefer them over an
    equivalent `PhiAmbiguitySet`, and keep a general `phi_conjugate`'s
    radius well inside the saturation threshold.

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
            starts from, in the units of the raw loss (`eta` and `lam` have
            the units of `loss`); later solves warm-start from the previous
            optimum (see `optora.core.dro_base.DualAmbiguitySet`).
    """

    def __init__(
        self,
        nominal: torch.Tensor,
        divergence: PhiDivergence,
        radius: float | torch.Tensor,
        phi_conjugate: Callable[[torch.Tensor], torch.Tensor],
        dual_solver: Solver[MinimizationProblem, MinimizationResult] | None = None,
        initial_log_eta: float | None = None,
        initial_lam: float | None = None,
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
                from, in the units of the raw loss (`eta` has the units of
                `loss`). Later calls warm-start from the previous solve's
                optimum unless `reset_warm_start()` is called.
            initial_lam: Value of `lam` the first dual solve starts from, in
                the units of the raw loss, warm-started on later calls
                alongside `initial_log_eta`. Give both or neither: when
                both are `None`, the default, the solve starts at `eta`
                equal to the loss spread and `lam` equal to the minimum
                loss, which is the same point whatever the scale or offset
                of the loss.
            validate: Whether to check that `nominal` is nonnegative and
                sums to one. The check synchronizes with the device, so it
                is opt-in and off by default.

        Raises:
            ValueError: If `radius` is a negative float, if only one of
                `initial_log_eta` and `initial_lam` is given, or if
                `validate` is set and `nominal` is not a valid probability
                distribution.
        """
        if (initial_log_eta is None) != (initial_lam is None):
            raise ValueError(
                "initial_log_eta and initial_lam must be given together or not at all."
            )
        super().__init__(
            nominal=nominal,
            divergence=divergence,
            radius=radius,
            dual_solver=dual_solver,
            initial_dual_point=torch.tensor(
                [0.0, 0.0]
                if initial_log_eta is None
                else [initial_log_eta, initial_lam],
                dtype=nominal.dtype,
                device=nominal.device,
            ),
            initial_in_raw_units=initial_log_eta is not None,
            validate=validate,
        )
        self.phi_conjugate = phi_conjugate

    def _dual_objective(
        self,
        point: torch.Tensor,
        loss: torch.Tensor,
        radius: float | torch.Tensor,
    ) -> torch.Tensor:
        r"""Evaluate the phi dual at `point = (log(eta), lam)` on the standardized loss.

        Args:
            point: Dual variables `(log(eta), lam)` of shape
                `batch_shape + (2,)`.
            loss: Standardized loss of shape `(..., n)`.
            radius: Phi-divergence radius, broadcastable against
                `batch_shape`.

        Returns:
            $\eta \cdot \mathrm{radius} + \lambda + \eta\,
            \mathbb{E}_{\mathrm{nominal}}[\phi^*((\mathrm{loss} - \lambda)
            / \eta)]$ per batch element.
        """
        eta = torch.exp(point[..., 0])
        lam = point[..., 1]
        scaled_shift = (loss - lam.unsqueeze(-1)) / eta.unsqueeze(-1)
        conjugate_term = torch.sum(
            self.nominal * self.phi_conjugate(scaled_shift), dim=-1
        )
        return eta * radius + lam + eta * conjugate_term

    def _to_standard_units(
        self, point: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """Map `(log(eta), lam)` to `(log(eta / scale), (lam - shift) / scale)`."""
        log_eta, lam = point.unbind(-1)
        return torch.stack((log_eta - torch.log(scale), (lam - shift) / scale), dim=-1)

    def _from_standard_units(
        self, point: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """Map `(log(eta / scale), (lam - shift) / scale)` back to `(log(eta), lam)`."""
        log_eta, lam = point.unbind(-1)
        return torch.stack((log_eta + torch.log(scale), lam * scale + shift), dim=-1)


class ChiSquareAmbiguitySet(TiltedAmbiguitySet):
    r"""Chi-square-divergence-constrained ambiguity set for chi-square-DRO.

    Bounds every candidate distribution `q` by

    $$
    D_{\chi^2}(q \,\|\, \mathrm{nominal})
        = \sum_i \frac{(q_i - \mathrm{nominal}_i)^2}{\mathrm{nominal}_i}
        \le \mathrm{radius},
    $$

    the phi-divergence generated by $\phi(t) = (t-1)^2$. Unlike
    `PhiAmbiguitySet`, `worst_case_expectation` is not computed by
    minimizing the general convex dual over $(\eta, \lambda)$, because the
    chi-square maximizer is known in closed form up to a single scalar: the
    KKT conditions give

    $$
    q^\star_i \propto \mathrm{nominal}_i \,
        (\mathrm{loss}_i - c)_+,
    $$

    a *linear* tilt of the nominal truncated at a cut level $c$
    (Duchi and Namkoong 2021). Its divergence rises monotonically as $c$
    rises, from $0$ as $c \to -\infty$ to $1/P^\star - 1$ as $c \to \max
    \mathrm{loss}$, where $P^\star$ is the nominal mass on the highest-loss
    scenarios, so the radius constraint pins $c$ by one monotone scalar
    root solve. See `optora.core.dro_base.TiltedAmbiguitySet` for the
    bracketed bisection that runs it.

    In the interior regime where no scenario is truncated to $q_i = 0$, the
    cut level drops below $\min \mathrm{loss}$ and the value collapses to
    the well-known closed form (Duchi and Namkoong 2021):

    $$
    \sup_{q:\, D_{\chi^2}(q \,\|\, \mathrm{nominal}) \,\le\, \mathrm{radius}}
        \mathbb{E}_q[\mathrm{loss}]
    = \mathbb{E}_{\mathrm{nominal}}[\mathrm{loss}]
        + \sqrt{\mathrm{radius} \cdot \mathrm{Var}_{\mathrm{nominal}}(\mathrm{loss})}.
    $$

    The bisection reproduces that regime without special-casing it, and
    keeps working in the truncated regime the closed form does not cover.

    Attributes:
        nominal: Reference distribution the ambiguity set is centered on.
        divergence: `ChiSquareDivergence` instance measuring distance from
            `nominal`.
        radius: Nonnegative bound on the chi-square divergence of any
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
            validate=validate,
        )

    def _tilted_distribution(
        self, gap: torch.Tensor, tilt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Evaluate the truncated linear tilt and its chi-square divergence.

        The cut level is parameterized by its reciprocal depth below the
        maximum loss, so the unnormalized weight is
        $w = (1 - a \cdot \mathrm{gap})_+$ for a rate $a \ge 0$ rather than
        $(\mathrm{loss} - c)_+$. The two differ by the positive factor
        $1/a$, which the normalization cancels, but the reciprocal form is
        exactly `1` on the highest-loss scenarios for every rate, so the
        saturated end of the path is reached without dividing by a
        vanishing depth.

        Writing $Z = \mathbb{E}_{\mathrm{nominal}}[w]$, the divergence is
        $\mathbb{E}_{\mathrm{nominal}}[(w - Z)^2] / Z^2$: substituting
        $q = \mathrm{nominal}\, w / Z$ into $\sum_i (q_i -
        \mathrm{nominal}_i)^2 / \mathrm{nominal}_i$ and expanding about
        $Z$ cancels the cross term, because $w$ has nominal mean $Z$ by
        construction. Differencing the weights before squaring rather than
        after is what keeps a vanishing radius accurate: both
        $\mathbb{E}_{\mathrm{nominal}}[w^2]$ and $Z^2$ approach one as the
        rate vanishes, and their difference would lose every significant
        digit of an $O(a^2)$ answer.

        Args:
            gap: Standardized shortfall below the maximum loss, of shape
                `(..., n)`.
            tilt: Path parameter in $[0, 1)$ of shape `batch_shape`.

        Returns:
            The tilted distribution and its chi-square divergence from
            `nominal`.
        """
        rate = self._tilt_rate(tilt)
        weight = torch.clamp(1.0 - rate * gap, min=0.0)
        normalizer = torch.sum(self.nominal * weight, dim=-1, keepdim=True)
        distribution = self.nominal * weight / normalizer
        deviation = weight - normalizer
        divergence = (
            torch.sum(self.nominal * deviation**2, dim=-1, keepdim=True) / normalizer**2
        ).squeeze(-1)
        return distribution, torch.clamp(divergence, min=0.0)


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
