"""Tests for the phi-divergence ambiguity sets in `optora.dro.phi_dro`.

Covers `PhiAmbiguitySet`, `ChiSquareAmbiguitySet`,
`TotalVariationAmbiguitySet`, and `CVaRAmbiguitySet`.
"""

import math

import pytest
import torch

from optora.core.convergence import ConvergenceStatus
from optora.core.dro_base import AmbiguitySet, TiltedAmbiguitySet
from optora.core.solver_base import (
    MinimizationProblem,
    MinimizationResult,
    Solver,
)
from optora.divergences.f_divergence import PhiDivergence
from optora.divergences.max_divergence import MaxDivergence
from optora.dro.kl_dro import KLAmbiguitySet
from optora.dro.phi_dro import (
    ChiSquareAmbiguitySet,
    CVaRAmbiguitySet,
    PhiAmbiguitySet,
    TotalVariationAmbiguitySet,
    chi_square_conjugate,
    kl_conjugate,
)
from optora.solvers.gradient_descent import GradientDescent
from tests.dro.duality import (
    cressie_read_dual_minimum,
    cressie_read_generator,
)
from tests.dro.instances import INSTANCES, Instance
from tests.dro.tolerances import feasibility_bound


class _RecordingSolver(Solver[MinimizationProblem, MinimizationResult]):
    """Fake dual solver returning a fixed `(log(eta), lam)` for deterministic checks."""

    def __init__(self, log_eta: float, lam: float) -> None:
        self.log_eta = log_eta
        self.lam = lam
        self.received_problem: MinimizationProblem | None = None

    def solve(self, problem: MinimizationProblem) -> MinimizationResult:
        self.received_problem = problem
        point = torch.tensor([self.log_eta, self.lam], dtype=torch.float64)
        return MinimizationResult(
            point=point,
            value=problem.objective(point),
            status=ConvergenceStatus(torch.tensor(True), torch.tensor(0)),
        )


def _kl_generator(ratio: torch.Tensor) -> torch.Tensor:
    """Convex generator `phi(t) = t * log(t) - t + 1` reproducing `D_KL(p || q)`."""
    return ratio * torch.log(ratio) - ratio + 1.0


def _kl_conjugate(scaled_shift: torch.Tensor) -> torch.Tensor:
    """Convex conjugate `phi*(s) = exp(s) - 1` of `_kl_generator`."""
    return torch.exp(scaled_shift) - 1.0


def _grid_search_chi_square_dual_minimum(
    nominal: torch.Tensor, loss: torch.Tensor, radius: float
) -> torch.Tensor:
    """Independent fine-grid cross-check of the chi-square-DRO dual minimum."""
    eta_grid = torch.logspace(-3, 3, steps=601, dtype=nominal.dtype)
    lam_grid = torch.linspace(
        float(loss.min()) - 5.0,
        float(loss.max()) + 5.0,
        steps=1201,
        dtype=nominal.dtype,
    )
    eta = eta_grid.view(-1, 1, 1)
    lam = lam_grid.view(1, -1, 1)
    scaled = (loss.view(1, 1, -1) - lam) / eta
    conjugate_term = torch.sum(
        nominal.view(1, 1, -1) * chi_square_conjugate(scaled), dim=-1
    )
    dual_values = (
        eta.squeeze(-1) * radius + lam.squeeze(-1) + eta.squeeze(-1) * conjugate_term
    )
    return dual_values.min()


# Tied losses on a support large enough (n >= 33) that PyTorch's unstable
# CPU sort reorders equal keys. A total-variation radius of 0.3 ends the
# drain inside the loss-0 tie group, and a CVaR level of 0.1 ends the tail
# inside the loss-2 tie group.
TIED_NOMINAL = torch.linspace(1.0, 2.0, 40, dtype=torch.float64)
TIED_NOMINAL = TIED_NOMINAL / TIED_NOMINAL.sum()
TIED_LOSS = (torch.arange(40) % 3).to(torch.float64)

# An index-proportional perturbation far below the unit loss gap, which
# turns every tie into a strict order without changing any other.
TIE_BREAK = 1e-6 * torch.arange(40, dtype=torch.float64)


def _maximizer(ambiguity_set: AmbiguitySet, loss: torch.Tensor) -> torch.Tensor:
    """Return the gradient of the worst case at `loss`, its maximizer."""
    differentiable = loss.clone().requires_grad_(True)
    (gradient,) = torch.autograd.grad(
        ambiguity_set.worst_case_expectation(differentiable), differentiable
    )
    return gradient


# --- PhiAmbiguitySet (generic base) ---------------------------------------


def test_zero_radius_returns_exact_expectation() -> None:
    nominal = torch.tensor([0.3, 0.3, 0.4])
    loss = torch.tensor([1.0, 2.0, 3.0])
    ambiguity_set = PhiAmbiguitySet(
        nominal,
        divergence=PhiDivergence(phi=_kl_generator),
        radius=0.0,
        phi_conjugate=_kl_conjugate,
    )

    result = ambiguity_set.worst_case_expectation(loss)

    assert torch.allclose(result, torch.sum(nominal * loss), atol=1e-6)


def test_worst_case_expectation_uses_custom_dual_solver() -> None:
    nominal = torch.tensor([0.5, 0.5], dtype=torch.float64)
    loss = torch.tensor([0.0, 2.0], dtype=torch.float64)
    fake_solver = _RecordingSolver(log_eta=0.5, lam=0.3)
    ambiguity_set = PhiAmbiguitySet(
        nominal,
        divergence=PhiDivergence(phi=_kl_generator),
        radius=0.3,
        phi_conjugate=_kl_conjugate,
        dual_solver=fake_solver,
    )

    result = ambiguity_set.worst_case_expectation(loss)

    # The solver iterates on the loss standardized to unit spread, so its
    # returned point is read in those units and the dual value is rescaled
    # by the spread (2.0 here) with no shift (the minimum loss is 0.0).
    assert fake_solver.received_problem is not None
    eta = torch.exp(torch.tensor(0.5, dtype=torch.float64))
    lam = torch.tensor(0.3, dtype=torch.float64)
    standardized = loss / 2.0
    expected = 2.0 * (
        eta * 0.3
        + lam
        + eta * torch.sum(nominal * _kl_conjugate((standardized - lam) / eta))
    )
    assert torch.allclose(result, expected, atol=1e-6)


def _cressie_read_set(
    nominal: torch.Tensor, radius: float, power: float
) -> PhiAmbiguitySet:
    """Build the Cressie-Read ball of order `power` through the exact path."""

    def generator(ratio: torch.Tensor) -> torch.Tensor:
        return (ratio**power - power * ratio + power - 1.0) / (power * (power - 1.0))

    def conjugate(shift: torch.Tensor) -> torch.Tensor:
        base = torch.clamp((power - 1.0) * shift + 1.0, min=0.0)
        return (base ** (power / (power - 1.0)) - 1.0) / power

    return PhiAmbiguitySet(
        nominal,
        divergence=PhiDivergence(phi=generator),
        radius=radius,
        phi_conjugate=conjugate,
    )


@pytest.mark.parametrize("power", [1.5, 3.0])
@pytest.mark.parametrize("radius", [1e-4, 0.05, 0.3, 1e3])
@pytest.mark.parametrize("instance", INSTANCES, ids=lambda instance: instance.name)
def test_cressie_read_matches_the_independent_dual_minimum(
    instance: Instance, radius: float, power: float
) -> None:
    # No closed form exists for this family: the reference minimizes the
    # dual by nested bounded searches in NumPy and SciPy.
    nominal, loss, _ = instance.tensors()

    value = _cressie_read_set(nominal, radius, power).worst_case_expectation(loss)

    expected = cressie_read_dual_minimum(
        instance.nominal_array, instance.loss_array, radius, power
    )
    spread = float(loss.max() - loss.min())
    assert float(value) == pytest.approx(expected, abs=1e-9 * (1.0 + spread))


@pytest.mark.parametrize("power", [1.5, 3.0])
@pytest.mark.parametrize("radius", [1e-4, 0.05, 0.3])
def test_cressie_read_maximizer_is_feasible_and_attains_the_value(
    radius: float, power: float
) -> None:
    generator = torch.Generator().manual_seed(7)
    nominal = torch.softmax(torch.randn(9, generator=generator, dtype=torch.float64), 0)
    loss = torch.randn(9, generator=generator, dtype=torch.float64).requires_grad_()

    value = _cressie_read_set(nominal, radius, power).worst_case_expectation(loss)
    (candidate,) = torch.autograd.grad(value, loss)

    ratio = (candidate / nominal).numpy()
    divergence = float(nominal.numpy() @ cressie_read_generator(ratio, power))
    assert float(candidate.min()) >= 0.0
    assert float(candidate.sum()) == pytest.approx(1.0, abs=1e-12)
    assert divergence <= radius * (1.0 + 1e-9) + 1e-15
    assert divergence == pytest.approx(radius, rel=1e-6)
    assert float(candidate @ loss.detach()) == pytest.approx(float(value), abs=1e-12)


def test_cressie_read_saturates_to_the_maximum_loss_exactly() -> None:
    nominal = torch.tensor([0.1, 0.3, 0.4, 0.2], dtype=torch.float64)
    loss = torch.tensor(
        [[1.0, -2.0, 0.5, 3.0], [4.0, 4.0, 0.0, -1.0]], dtype=torch.float64
    )

    value = _cressie_read_set(nominal, 1e6, 1.5).worst_case_expectation(loss)

    assert torch.equal(value, loss.amax(-1))


@pytest.mark.parametrize("radius", [1e-8, 1e-3, 0.1, 0.5, 1e3])
def test_without_a_dual_solver_the_kl_generator_is_solved_exactly(
    radius: float,
) -> None:
    nominal = torch.tensor([0.1, 0.25, 0.4, 0.25], dtype=torch.float64)
    loss = torch.tensor(
        [[2.0, -1.0, 0.5, 1.5], [0.0, 3.0, 3.0, -2.0]], dtype=torch.float64
    )
    general = PhiAmbiguitySet(
        nominal,
        divergence=PhiDivergence(phi=_kl_generator),
        radius=radius,
        phi_conjugate=_kl_conjugate,
    )

    result = general.worst_case_expectation(loss)

    expected = KLAmbiguitySet(nominal, radius=radius).worst_case_expectation(loss)
    torch.testing.assert_close(result, expected, rtol=0.0, atol=1e-12)


def test_matches_kl_ambiguity_set_when_using_kl_generator_and_conjugate() -> None:
    nominal = torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    radius = 0.2

    generic = PhiAmbiguitySet(
        nominal,
        divergence=PhiDivergence(phi=_kl_generator),
        radius=radius,
        phi_conjugate=_kl_conjugate,
        dual_solver=GradientDescent(step_size=0.05, max_iter=20000, tol=1e-11),
    )
    kl_specific = KLAmbiguitySet(nominal, radius=radius)

    generic_result = generic.worst_case_expectation(loss)
    kl_result = kl_specific.worst_case_expectation(loss)

    assert torch.allclose(generic_result, kl_result, atol=1e-4)


def test_mismatched_loss_shape_raises_value_error() -> None:
    nominal = torch.tensor([0.5, 0.5])
    loss = torch.tensor([1.0, 2.0, 3.0])
    ambiguity_set = PhiAmbiguitySet(
        nominal,
        divergence=PhiDivergence(phi=_kl_generator),
        radius=0.1,
        phi_conjugate=_kl_conjugate,
    )

    with pytest.raises(ValueError):
        ambiguity_set.worst_case_expectation(loss)


def test_negative_radius_raises_value_error() -> None:
    nominal = torch.tensor([0.5, 0.5])

    with pytest.raises(ValueError):
        PhiAmbiguitySet(
            nominal,
            divergence=PhiDivergence(phi=_kl_generator),
            radius=-1.0,
            phi_conjugate=_kl_conjugate,
        )


def test_is_an_ambiguity_set_instance() -> None:
    nominal = torch.tensor([0.5, 0.5])

    ambiguity_set = PhiAmbiguitySet(
        nominal,
        divergence=PhiDivergence(phi=_kl_generator),
        radius=0.1,
        phi_conjugate=_kl_conjugate,
    )

    assert isinstance(ambiguity_set, AmbiguitySet)


def test_initial_dual_point_pairs_the_configured_log_eta_and_lam() -> None:
    # Both are in raw loss units, so on a loss of spread 2.0 and minimum 0.0
    # the solver receives (log(eta / 2), lam / 2).
    nominal = torch.tensor([0.5, 0.5], dtype=torch.float64)
    loss = torch.tensor([0.0, 2.0], dtype=torch.float64)
    fake_solver = _RecordingSolver(log_eta=0.5, lam=0.3)
    ambiguity_set = PhiAmbiguitySet(
        nominal,
        divergence=PhiDivergence(phi=_kl_generator),
        radius=0.3,
        phi_conjugate=_kl_conjugate,
        dual_solver=fake_solver,
        initial_log_eta=-0.75,
        initial_lam=1.5,
    )

    ambiguity_set.worst_case_expectation(loss)

    assert fake_solver.received_problem is not None
    assert torch.allclose(
        fake_solver.received_problem.initial_point,
        torch.tensor([-0.75 - math.log(2.0), 1.5 / 2.0], dtype=torch.float64),
    )


def test_repeated_calls_warm_start_from_the_previous_dual_optimum() -> None:
    nominal = torch.tensor([0.5, 0.5], dtype=torch.float64)
    loss = torch.tensor([0.0, 2.0], dtype=torch.float64)
    fake_solver = _RecordingSolver(log_eta=0.5, lam=0.3)
    ambiguity_set = PhiAmbiguitySet(
        nominal,
        divergence=PhiDivergence(phi=_kl_generator),
        radius=0.3,
        phi_conjugate=_kl_conjugate,
        dual_solver=fake_solver,
        initial_log_eta=-0.75,
        initial_lam=1.5,
    )

    ambiguity_set.worst_case_expectation(loss)
    ambiguity_set.worst_case_expectation(loss)

    # Same loss, so the same standardization: the previous optimum comes back
    # as the starting point (up to the float round trip through raw units).
    assert fake_solver.received_problem is not None
    assert torch.allclose(
        fake_solver.received_problem.initial_point,
        torch.tensor([0.5, 0.3], dtype=torch.float64),
    )

    ambiguity_set.reset_warm_start()
    ambiguity_set.worst_case_expectation(loss)

    assert torch.allclose(
        fake_solver.received_problem.initial_point,
        torch.tensor([-0.75 - math.log(2.0), 1.5 / 2.0], dtype=torch.float64),
    )


def test_repeated_chi_square_calls_are_deterministic() -> None:
    # The chi-square set is bisected rather than dual-solved, so it keeps no
    # warm start and a repeat solve must reproduce the first one exactly.
    nominal = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64)
    base = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    losses = [base * (1.0 + 0.05 * step) for step in range(3)]

    reused = ChiSquareAmbiguitySet(nominal, radius=0.2)
    reused_values = [reused.worst_case_expectation(loss) for loss in losses]
    fresh_values = [
        ChiSquareAmbiguitySet(nominal, radius=0.2).worst_case_expectation(loss)
        for loss in losses
    ]

    for repeated, fresh in zip(reused_values, fresh_values, strict=True):
        assert repeated == fresh


# --- ChiSquareAmbiguitySet -------------------------------------------------


def test_chi_square_zero_radius_returns_exact_expectation() -> None:
    nominal = torch.tensor([0.3, 0.3, 0.4])
    loss = torch.tensor([1.0, 2.0, 3.0])
    ambiguity_set = ChiSquareAmbiguitySet(nominal, radius=0.0)

    result = ambiguity_set.worst_case_expectation(loss)

    assert torch.allclose(result, torch.sum(nominal * loss), atol=1e-6)


def test_chi_square_calibrated_radius_gives_the_normal_interval() -> None:
    """The Duchi-Namkoong calibration must reproduce the textbook normal interval.

    Setting `radius = chi2_{1, 1 - alpha} / n` on the uniform empirical
    distribution of `n` observations makes the ambiguity set an
    asymptotically valid `1 - alpha` confidence region, which in the
    interior regime means the worst-case expectation is exactly the upper
    end of the normal confidence interval,
    `mean + z_{1 - alpha/2} * std / sqrt(n)`.
    """
    num_samples = 64
    quantiles = (torch.arange(num_samples, dtype=torch.float64) + 0.5) / num_samples
    observations = torch.special.ndtri(quantiles)
    nominal = torch.full_like(observations, 1.0 / num_samples)
    normal_quantile = torch.special.ndtri(torch.tensor(0.975, dtype=torch.float64))
    radius = float(normal_quantile**2) / num_samples
    ambiguity_set = ChiSquareAmbiguitySet(nominal, radius=radius)

    result = ambiguity_set.worst_case_expectation(observations)

    standard_error = torch.std(observations, unbiased=False) / num_samples**0.5
    expected = torch.mean(observations) + normal_quantile * standard_error
    assert torch.allclose(result, expected, atol=1e-12)


def test_chi_square_matches_grid_search_over_dual_variables() -> None:
    nominal = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0, 2.0, 5.0], dtype=torch.float64)
    radius = 0.2
    ambiguity_set = ChiSquareAmbiguitySet(nominal, radius=radius)

    result = ambiguity_set.worst_case_expectation(loss)

    reference = _grid_search_chi_square_dual_minimum(nominal, loss, radius)
    assert torch.allclose(result, reference, atol=1e-2)


def test_chi_square_reaches_max_loss_for_large_radius() -> None:
    nominal = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0, 2.0, 5.0], dtype=torch.float64)
    ambiguity_set = ChiSquareAmbiguitySet(nominal, radius=5.0)

    result = ambiguity_set.worst_case_expectation(loss)

    assert result == loss.max()


def test_chi_square_worst_case_expectation_is_monotonic_in_radius() -> None:
    nominal = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    radii = torch.logspace(-6, 1, steps=200, dtype=torch.float64)

    values = ChiSquareAmbiguitySet(nominal, radius=radii).worst_case_expectation(loss)

    assert torch.all(torch.diff(values) >= 0.0)


def test_chi_square_worst_case_expectation_is_at_least_nominal_expectation() -> None:
    nominal = torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    ambiguity_set = ChiSquareAmbiguitySet(nominal, radius=0.1)

    result = ambiguity_set.worst_case_expectation(loss)

    assert result >= torch.sum(nominal * loss)


def test_chi_square_worst_case_expectation_does_not_exceed_max_loss() -> None:
    nominal = torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    ambiguity_set = ChiSquareAmbiguitySet(nominal, radius=0.1)

    result = ambiguity_set.worst_case_expectation(loss)

    assert result <= loss.max()


def test_chi_square_mismatched_loss_shape_raises_value_error() -> None:
    nominal = torch.tensor([0.5, 0.5])
    loss = torch.tensor([1.0, 2.0, 3.0])
    ambiguity_set = ChiSquareAmbiguitySet(nominal, radius=0.1)

    with pytest.raises(ValueError):
        ambiguity_set.worst_case_expectation(loss)


def test_chi_square_negative_radius_raises_value_error() -> None:
    nominal = torch.tensor([0.5, 0.5])

    with pytest.raises(ValueError):
        ChiSquareAmbiguitySet(nominal, radius=-1.0)


def test_chi_square_invalid_eps_raises_value_error() -> None:
    nominal = torch.tensor([0.5, 0.5])

    with pytest.raises(ValueError):
        ChiSquareAmbiguitySet(nominal, radius=0.1, eps=0.0)
    with pytest.raises(ValueError):
        ChiSquareAmbiguitySet(nominal, radius=0.1, eps=-1.0)


def test_chi_square_contains_uses_chi_square_divergence_and_radius() -> None:
    nominal = torch.tensor([0.5, 0.5])
    ambiguity_set = ChiSquareAmbiguitySet(nominal, radius=0.05)

    assert ambiguity_set.contains(nominal)
    assert not ambiguity_set.contains(torch.tensor([0.95, 0.05]))


def test_chi_square_is_a_tilted_and_ambiguity_set_instance() -> None:
    nominal = torch.tensor([0.5, 0.5])
    ambiguity_set = ChiSquareAmbiguitySet(nominal, radius=0.1)

    assert isinstance(ambiguity_set, TiltedAmbiguitySet)
    assert isinstance(ambiguity_set, AmbiguitySet)


@pytest.mark.parametrize("support_size", [10, 50, 100, 1000])
@pytest.mark.parametrize("fraction", [0.1, 0.5, 0.9, 1.0, 2.0])
def test_chi_square_on_a_large_support_is_finite_and_bounded(
    support_size: int, fraction: float
) -> None:
    # Issue #51. With the repo's old dual solver this returned values above
    # `max(loss)` from n = 10 and `nan` from n = 50, because a fixed-step
    # first-order solve overshot into a region the conjugate overflowed.
    nominal = torch.full((support_size,), 1.0 / support_size, dtype=torch.float64)
    loss = torch.linspace(0.0, 1.0, support_size, dtype=torch.float64)
    radius = fraction * (support_size - 1.0)

    result = ChiSquareAmbiguitySet(nominal, radius=radius).worst_case_expectation(loss)

    assert torch.isfinite(result)
    assert result <= loss.max()
    assert result >= torch.sum(nominal * loss)
    maximum = float(loss.max())
    if fraction > 1.0:
        assert float(result) == maximum
    elif fraction == 1.0:
        # Exactly at saturation the feasibility test compares two values
        # equal in exact arithmetic; reduction order may cost a few ulps.
        assert float(result) >= maximum - 4 * math.ulp(maximum)


@pytest.mark.parametrize("radius", [1e-14, 1e-10, 1e-6])
def test_chi_square_at_a_vanishing_radius_matches_the_closed_form(
    radius: float,
) -> None:
    # Issue #49. In the interior regime the worst case is exactly
    # `mean + sqrt(radius * variance)` however small the radius is, so the
    # excess over the nominal expectation must stay accurate relative to
    # itself rather than to the loss scale.
    nominal = torch.tensor([0.5, 0.2, 0.2, 0.05, 0.05], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0, 2.0, 3.0, 10.0], dtype=torch.float64)
    mean = torch.sum(nominal * loss)
    variance = torch.sum(nominal * (loss - mean) ** 2)

    result = ChiSquareAmbiguitySet(nominal, radius=radius).worst_case_expectation(loss)

    excess = float(result - mean)
    expected = float(torch.sqrt(radius * variance))
    assert abs(excess - expected) <= 1e-6 * expected


@pytest.mark.parametrize("fraction", [1e-12, 1e-6, 0.01, 0.5, 0.9, 0.999])
def test_chi_square_returned_distribution_stays_inside_the_radius(
    fraction: float,
) -> None:
    nominal = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0, 2.0, 5.0], dtype=torch.float64)
    saturation = 1.0 / float(nominal[loss >= loss.max()].sum()) - 1.0
    radius = fraction * saturation
    ambiguity_set = ChiSquareAmbiguitySet(nominal, radius=radius)
    differentiable = loss.clone().requires_grad_(True)

    ambiguity_set.worst_case_expectation(differentiable).backward()

    assert differentiable.grad is not None
    divergence = ambiguity_set.divergence(differentiable.grad, nominal)
    assert float(divergence) <= feasibility_bound(radius, nominal.numel())


# --- TotalVariationAmbiguitySet --------------------------------------------


def test_total_variation_zero_radius_returns_exact_expectation() -> None:
    nominal = torch.tensor([0.3, 0.3, 0.4])
    loss = torch.tensor([1.0, 2.0, 3.0])
    ambiguity_set = TotalVariationAmbiguitySet(nominal, radius=0.0)

    result = ambiguity_set.worst_case_expectation(loss)

    assert torch.allclose(result, torch.sum(nominal * loss), atol=1e-6)


def test_total_variation_matches_hand_computed_two_scenario_example() -> None:
    nominal = torch.tensor([0.5, 0.5], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0], dtype=torch.float64)
    ambiguity_set = TotalVariationAmbiguitySet(nominal, radius=0.2)

    result = ambiguity_set.worst_case_expectation(loss)

    assert torch.allclose(result, torch.tensor(0.7, dtype=torch.float64), atol=1e-6)


def test_total_variation_matches_hand_computed_three_scenario_example() -> None:
    nominal = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    ambiguity_set = TotalVariationAmbiguitySet(nominal, radius=0.15)

    result = ambiguity_set.worst_case_expectation(loss)

    assert torch.allclose(result, torch.tensor(1.6, dtype=torch.float64), atol=1e-6)


def test_total_variation_worst_case_expectation_is_attained_by_candidate() -> None:
    nominal = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    ambiguity_set = TotalVariationAmbiguitySet(nominal, radius=0.15)
    candidate = torch.tensor([0.05, 0.3, 0.65], dtype=torch.float64)

    result = ambiguity_set.worst_case_expectation(loss)

    assert ambiguity_set.divergence(candidate, nominal) <= feasibility_bound(
        ambiguity_set.radius, nominal.numel()
    )
    assert torch.allclose(result, torch.sum(candidate * loss), atol=1e-6)


def test_total_variation_reaches_max_loss_for_radius_covering_whole_simplex() -> None:
    nominal = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0, 2.0, 5.0], dtype=torch.float64)
    ambiguity_set = TotalVariationAmbiguitySet(nominal, radius=1.0)

    result = ambiguity_set.worst_case_expectation(loss)

    assert torch.allclose(result, torch.tensor(5.0, dtype=torch.float64), atol=1e-6)


def test_total_variation_worst_case_expectation_is_monotonic_in_radius() -> None:
    nominal = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)

    small = TotalVariationAmbiguitySet(nominal, radius=0.05).worst_case_expectation(
        loss
    )
    large = TotalVariationAmbiguitySet(nominal, radius=0.5).worst_case_expectation(loss)

    assert large >= small - 1e-6


def test_total_variation_worst_case_expectation_is_at_least_nominal_expectation() -> (
    None
):
    nominal = torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    ambiguity_set = TotalVariationAmbiguitySet(nominal, radius=0.1)

    result = ambiguity_set.worst_case_expectation(loss)

    assert result >= torch.sum(nominal * loss) - 1e-6


def test_total_variation_worst_case_expectation_does_not_exceed_max_loss() -> None:
    nominal = torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    ambiguity_set = TotalVariationAmbiguitySet(nominal, radius=0.5)

    result = ambiguity_set.worst_case_expectation(loss)

    assert result <= loss.max() + 1e-6


def test_total_variation_mismatched_loss_shape_raises_value_error() -> None:
    nominal = torch.tensor([0.5, 0.5])
    loss = torch.tensor([1.0, 2.0, 3.0])
    ambiguity_set = TotalVariationAmbiguitySet(nominal, radius=0.1)

    with pytest.raises(ValueError):
        ambiguity_set.worst_case_expectation(loss)


def test_total_variation_negative_radius_raises_value_error() -> None:
    nominal = torch.tensor([0.5, 0.5])

    with pytest.raises(ValueError):
        TotalVariationAmbiguitySet(nominal, radius=-1.0)


def test_total_variation_invalid_eps_raises_value_error() -> None:
    nominal = torch.tensor([0.5, 0.5])

    with pytest.raises(ValueError):
        TotalVariationAmbiguitySet(nominal, radius=0.1, eps=0.0)
    with pytest.raises(ValueError):
        TotalVariationAmbiguitySet(nominal, radius=0.1, eps=-1.0)


def test_total_variation_contains_uses_total_variation_divergence_and_radius() -> None:
    nominal = torch.tensor([0.5, 0.5])
    ambiguity_set = TotalVariationAmbiguitySet(nominal, radius=0.05)

    assert ambiguity_set.contains(nominal)
    assert not ambiguity_set.contains(torch.tensor([0.95, 0.05]))


def test_total_variation_is_an_ambiguity_set_instance() -> None:
    nominal = torch.tensor([0.5, 0.5])

    assert isinstance(TotalVariationAmbiguitySet(nominal, radius=0.1), AmbiguitySet)


def test_total_variation_breaks_ties_as_if_a_later_index_had_a_larger_loss() -> None:
    # Issue #59: at a kink the subgradient is the maximizer of the
    # index-perturbed loss, whose maximizer is unique, on every device.
    ambiguity_set = TotalVariationAmbiguitySet(TIED_NOMINAL, radius=0.3)

    tied = _maximizer(ambiguity_set, TIED_LOSS)
    perturbed = _maximizer(ambiguity_set, TIED_LOSS + TIE_BREAK)

    assert torch.allclose(tied, perturbed, rtol=0.0, atol=1e-12)


# --- CVaRAmbiguitySet --------------------------------------------------------

CVAR_NOMINAL = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
CVAR_LOSS = torch.tensor([0.0, 1.0, 2.0, 5.0], dtype=torch.float64)


def test_cvar_at_level_one_returns_the_exact_nominal_expectation() -> None:
    ambiguity_set = CVaRAmbiguitySet(CVAR_NOMINAL, alpha=1.0)

    result = ambiguity_set.worst_case_expectation(CVAR_LOSS)

    assert ambiguity_set.radius == 0.0
    assert torch.equal(result, torch.sum(CVAR_NOMINAL * CVAR_LOSS))


@pytest.mark.parametrize(
    ("alpha", "expected"),
    [(0.7, (0.3 * 2.0 + 0.4 * 5.0) / 0.7), (0.5, 4.4), (0.4, 5.0), (0.2, 5.0)],
)
def test_cvar_matches_hand_computed_upper_tail_means(
    alpha: float, expected: float
) -> None:
    result = CVaRAmbiguitySet(CVAR_NOMINAL, alpha=alpha).worst_case_expectation(
        CVAR_LOSS
    )

    assert float(result) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("tail_size", [1, 3, 5, 8])
def test_cvar_on_a_uniform_nominal_is_the_mean_of_the_largest_losses(
    tail_size: int,
) -> None:
    loss = torch.tensor([0.3, -1.2, 4.0, 2.2, 0.9, -0.4, 3.1, 1.7], dtype=torch.float64)
    nominal = torch.full((8,), 1.0 / 8.0, dtype=torch.float64)

    result = CVaRAmbiguitySet(nominal, alpha=tail_size / 8.0).worst_case_expectation(
        loss
    )

    largest = torch.sort(loss, descending=True).values[:tail_size]
    assert float(result) == pytest.approx(float(largest.mean()), abs=1e-12)


def test_cvar_at_level_zero_is_the_maximum_loss_on_the_nominal_support() -> None:
    nominal = torch.tensor([0.0, 0.3, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([9.0, 1.0, 2.0, 5.0], dtype=torch.float64)
    ambiguity_set = CVaRAmbiguitySet(nominal, alpha=0.0)

    result = ambiguity_set.worst_case_expectation(loss)

    assert ambiguity_set.radius == math.inf
    assert float(result) == 5.0


@pytest.mark.parametrize("scale", [1.0, 1e-3, 1e-12, 1e-30])
def test_cvar_at_level_zero_is_bitwise_the_maximum_at_any_loss_scale(
    dtype: torch.dtype, scale: float
) -> None:
    # The floored level is finfo.tiny: weighting a loss below one by it
    # before normalizing went subnormal, ~1e-5 relative off in float32 at a
    # loss of 2e-3 (found by the group-DRO example).
    nominal = torch.tensor([0.0, 0.3, 0.3, 0.4], dtype=dtype)
    loss = torch.tensor([9.0, 0.3, 0.9, 0.2], dtype=dtype) * scale
    ambiguity_set = CVaRAmbiguitySet(nominal, alpha=0.0)

    result = ambiguity_set.worst_case_expectation(loss)

    assert torch.equal(result, loss[2])


def test_cvar_gradient_is_the_tail_maximizer() -> None:
    loss = CVAR_LOSS.clone().requires_grad_(True)

    CVaRAmbiguitySet(CVAR_NOMINAL, alpha=0.5).worst_case_expectation(loss).backward()

    assert loss.grad is not None
    expected = torch.tensor([0.0, 0.0, 0.2, 0.8], dtype=torch.float64)
    torch.testing.assert_close(loss.grad, expected, rtol=0.0, atol=1e-15)


def test_cvar_derivative_in_alpha_is_the_quantile_gap_over_alpha() -> None:
    # d CVaR / d alpha = (VaR - CVaR) / alpha at a level strictly inside a
    # scenario's mass: here VaR = 2 and CVaR = 4.4 at alpha = 0.5.
    alpha = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)

    value = CVaRAmbiguitySet(CVAR_NOMINAL, alpha=alpha).worst_case_expectation(
        CVAR_LOSS
    )
    (derivative,) = torch.autograd.grad(value, alpha)

    assert float(derivative) == pytest.approx((2.0 - 4.4) / 0.5, abs=1e-12)


def test_cvar_is_nonincreasing_in_alpha() -> None:
    levels = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)

    values = CVaRAmbiguitySet(CVAR_NOMINAL, alpha=levels).worst_case_expectation(
        CVAR_LOSS
    )

    assert bool(torch.all(values[1:] <= values[:-1] + 1e-12))
    assert float(values[0]) == 5.0
    assert float(values[-1]) == pytest.approx(float(CVAR_NOMINAL @ CVAR_LOSS))


def test_cvar_is_translation_equivariant_and_positively_homogeneous() -> None:
    ambiguity_set = CVaRAmbiguitySet(CVAR_NOMINAL, alpha=0.35)
    base = ambiguity_set.worst_case_expectation(CVAR_LOSS)

    shifted = ambiguity_set.worst_case_expectation(CVAR_LOSS + 3.0)
    scaled = ambiguity_set.worst_case_expectation(2.5 * CVAR_LOSS)

    torch.testing.assert_close(shifted, base + 3.0, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(scaled, 2.5 * base, rtol=0.0, atol=1e-12)


def test_cvar_radius_is_the_max_divergence_radius_of_the_level() -> None:
    levels = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64)

    scalar = CVaRAmbiguitySet(CVAR_NOMINAL, alpha=0.25)
    sweep = CVaRAmbiguitySet(CVAR_NOMINAL, alpha=levels)

    assert scalar.alpha == 0.25
    assert scalar.radius == pytest.approx(math.log(4.0))
    assert isinstance(scalar.divergence, MaxDivergence)
    buffers = dict(sweep.named_buffers())
    assert buffers["alpha"] is levels
    torch.testing.assert_close(
        buffers["radius"],
        torch.tensor([0.0, math.log(2.0), math.inf], dtype=torch.float64),
    )


def test_cvar_contains_exactly_the_likelihood_ratio_ball() -> None:
    ambiguity_set = CVaRAmbiguitySet(CVAR_NOMINAL, alpha=0.5)

    inside = torch.tensor([0.0, 0.0, 0.2, 0.8], dtype=torch.float64)
    outside = torch.tensor([0.0, 0.0, 0.1, 0.9], dtype=torch.float64)

    assert ambiguity_set.contains(CVAR_NOMINAL)
    assert ambiguity_set.contains(inside)
    assert not ambiguity_set.contains(outside)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, math.nan, -math.inf])
def test_cvar_rejects_a_float_level_outside_the_unit_interval(alpha: float) -> None:
    with pytest.raises(ValueError, match=r"alpha must lie in \[0, 1\]"):
        CVaRAmbiguitySet(CVAR_NOMINAL, alpha=alpha)


@pytest.mark.parametrize(
    "alpha",
    [
        torch.tensor(-0.1, dtype=torch.float64),
        torch.tensor([0.5, 1.5], dtype=torch.float64),
        torch.tensor([0.5, math.nan], dtype=torch.float64),
    ],
    ids=["negative", "above_one", "nan_entry"],
)
def test_cvar_validate_rejects_a_tensor_level_outside_the_unit_interval(
    alpha: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match=r"alpha must lie in \[0, 1\]"):
        CVaRAmbiguitySet(CVAR_NOMINAL, alpha=alpha, validate=True)


@pytest.mark.parametrize(
    "alpha",
    [
        torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64),
        torch.empty(0, dtype=torch.float64),
    ],
    ids=["closed_interval", "empty"],
)
def test_cvar_validate_accepts_a_tensor_level_in_the_unit_interval(
    alpha: torch.Tensor,
) -> None:
    ambiguity_set = CVaRAmbiguitySet(CVAR_NOMINAL, alpha=alpha, validate=True)

    assert ambiguity_set.alpha is alpha


def test_cvar_accepts_an_invalid_tensor_level_without_validation() -> None:
    alpha = torch.tensor(1.5, dtype=torch.float64)

    ambiguity_set = CVaRAmbiguitySet(CVAR_NOMINAL, alpha=alpha)

    assert ambiguity_set.alpha is alpha


def test_cvar_validate_rejects_a_non_finite_loss() -> None:
    ambiguity_set = CVaRAmbiguitySet(CVAR_NOMINAL, alpha=0.5, validate=True)
    loss = torch.tensor([0.0, math.nan, 2.0, 5.0], dtype=torch.float64)

    with pytest.raises(ValueError, match="loss must be finite"):
        ambiguity_set.worst_case_expectation(loss)


def test_cvar_invalid_eps_raises_value_error() -> None:
    with pytest.raises(ValueError, match="eps"):
        CVaRAmbiguitySet(CVAR_NOMINAL, alpha=0.5, eps=0.0)


def test_cvar_is_an_ambiguity_set_instance() -> None:
    assert isinstance(CVaRAmbiguitySet(CVAR_NOMINAL, alpha=0.5), AmbiguitySet)


def test_cvar_fills_tied_losses_in_support_index_order() -> None:
    # The tail boundary falls inside the tie group at loss 2, so the
    # partially filled scenario must be chosen by index, not by the sort.
    ambiguity_set = CVaRAmbiguitySet(TIED_NOMINAL, alpha=0.1)

    tied = _maximizer(ambiguity_set, TIED_LOSS)
    perturbed = _maximizer(ambiguity_set, TIED_LOSS - TIE_BREAK)

    assert torch.allclose(tied, perturbed, rtol=0.0, atol=1e-12)


def _kl_conjugate_limit(dtype: torch.dtype) -> float:
    """Return a quarter of `dtype`'s logarithmic range: where `kl_conjugate` bends."""
    return 0.25 * math.log(torch.finfo(dtype).max)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_kl_conjugate_is_expm1_up_to_half_the_exponent_range(
    dtype: torch.dtype,
) -> None:
    limit = _kl_conjugate_limit(dtype)
    scaled_shift = torch.tensor(
        [-1e4, -40.0, -1.0, -1e-9, 0.0, 1e-9, 1.0, 20.0, limit], dtype=dtype
    )

    torch.testing.assert_close(
        kl_conjugate(scaled_shift), torch.expm1(scaled_shift), rtol=0.0, atol=0.0
    )


def test_kl_conjugate_is_the_conjugate_of_the_ratio_capped_generator() -> None:
    # Past the limit the supremum of s t - phi(t) over 0 <= t <= T sits at
    # the cap T = exp(limit), where phi(T) = T * limit - T + 1.
    limit = _kl_conjugate_limit(torch.float64)
    cap = math.exp(limit)
    scaled_shift = torch.tensor([limit + 0.5, limit + 10.0, 2e3], dtype=torch.float64)
    expected = scaled_shift * cap - (cap * limit - cap + 1.0)

    torch.testing.assert_close(kl_conjugate(scaled_shift), expected, rtol=1e-13, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_kl_conjugate_stays_finite_and_continuously_differentiable(
    dtype: torch.dtype,
) -> None:
    limit = _kl_conjugate_limit(dtype)
    scaled_shift = torch.tensor(
        [limit - 1e-6, limit, limit + 1e-6, limit + 1.0, limit + 1e3, 1e6], dtype=dtype
    ).requires_grad_(True)

    value = kl_conjugate(scaled_shift)
    (slope,) = torch.autograd.grad(value.sum(), scaled_shift)

    assert torch.isfinite(value).all()
    assert torch.isfinite(slope).all()
    torch.testing.assert_close(
        slope, torch.full_like(slope, math.exp(limit)), rtol=2e-3, atol=0.0
    )
    assert torch.all(value[1:] >= value[:-1])
