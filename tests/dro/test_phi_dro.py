"""Tests for `PhiAmbiguitySet`, `ChiSquareAmbiguitySet`, and `TotalVariationAmbiguitySet`."""  # noqa: E501

import pytest
import torch

from optora.core.dro_base import AmbiguitySet
from optora.core.solver_base import Solver
from optora.divergences.f_divergence import PhiDivergence
from optora.dro.kl_dro import KLAmbiguitySet
from optora.dro.phi_dro import (
    ChiSquareAmbiguitySet,
    PhiAmbiguitySet,
    TotalVariationAmbiguitySet,
    _chi_square_conjugate,
)
from optora.solvers.gradient_descent import (
    GradientDescent,
    GradientDescentProblem,
    GradientDescentResult,
)


class _RecordingSolver(Solver[GradientDescentProblem, GradientDescentResult]):
    """Fake dual solver returning a fixed `(log(eta), lam)` for deterministic checks."""

    def __init__(self, log_eta: float, lam: float) -> None:
        self.log_eta = log_eta
        self.lam = lam
        self.received_problem: GradientDescentProblem | None = None

    def solve(self, problem: GradientDescentProblem) -> GradientDescentResult:
        self.received_problem = problem
        point = torch.tensor([self.log_eta, self.lam], dtype=torch.float64)
        return GradientDescentResult(
            point=point,
            value=problem.objective(point),
            converged=True,
            num_iterations=0,
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
        nominal.view(1, 1, -1) * _chi_square_conjugate(scaled), dim=-1
    )
    dual_values = (
        eta.squeeze(-1) * radius + lam.squeeze(-1) + eta.squeeze(-1) * conjugate_term
    )
    return dual_values.min()


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

    assert fake_solver.received_problem is not None
    eta = torch.exp(torch.tensor(0.5, dtype=torch.float64))
    lam = torch.tensor(0.3, dtype=torch.float64)
    expected = (
        eta * 0.3 + lam + eta * torch.sum(nominal * _kl_conjugate((loss - lam) / eta))
    )
    assert torch.allclose(result, expected, atol=1e-6)


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
    kl_specific = KLAmbiguitySet(
        nominal,
        radius=radius,
        dual_solver=GradientDescent(step_size=0.1, max_iter=20000, tol=1e-11),
    )

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


# --- ChiSquareAmbiguitySet -------------------------------------------------


def test_chi_square_zero_radius_returns_exact_expectation() -> None:
    nominal = torch.tensor([0.3, 0.3, 0.4])
    loss = torch.tensor([1.0, 2.0, 3.0])
    ambiguity_set = ChiSquareAmbiguitySet(nominal, radius=0.0)

    result = ambiguity_set.worst_case_expectation(loss)

    assert torch.allclose(result, torch.sum(nominal * loss), atol=1e-6)


def test_chi_square_matches_mean_plus_sqrt_radius_variance_in_interior_regime() -> None:
    nominal = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    radius = 0.05
    solver = GradientDescent(step_size=0.05, max_iter=20000, tol=1e-11)
    ambiguity_set = ChiSquareAmbiguitySet(nominal, radius=radius, dual_solver=solver)

    result = ambiguity_set.worst_case_expectation(loss)

    mean = torch.sum(nominal * loss)
    variance = torch.sum(nominal * (loss - mean) ** 2)
    expected = mean + torch.sqrt(torch.tensor(radius, dtype=torch.float64) * variance)
    assert torch.allclose(result, expected, atol=1e-4)


def test_chi_square_matches_grid_search_over_dual_variables() -> None:
    nominal = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0, 2.0, 5.0], dtype=torch.float64)
    radius = 0.2
    solver = GradientDescent(step_size=0.05, max_iter=5000, tol=1e-9)
    ambiguity_set = ChiSquareAmbiguitySet(nominal, radius=radius, dual_solver=solver)

    result = ambiguity_set.worst_case_expectation(loss)

    reference = _grid_search_chi_square_dual_minimum(nominal, loss, radius)
    assert torch.allclose(result, reference, atol=1e-2)


def test_chi_square_reaches_max_loss_for_large_radius() -> None:
    nominal = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0, 2.0, 5.0], dtype=torch.float64)
    solver = GradientDescent(step_size=0.01, max_iter=50000, tol=1e-10)
    ambiguity_set = ChiSquareAmbiguitySet(nominal, radius=5.0, dual_solver=solver)

    result = ambiguity_set.worst_case_expectation(loss)

    assert torch.allclose(result, torch.tensor(5.0, dtype=torch.float64), atol=1e-2)


def test_chi_square_worst_case_expectation_is_monotonic_in_radius() -> None:
    nominal = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)

    small = ChiSquareAmbiguitySet(
        nominal,
        radius=0.02,
        dual_solver=GradientDescent(step_size=0.05, max_iter=5000, tol=1e-9),
    ).worst_case_expectation(loss)
    large = ChiSquareAmbiguitySet(
        nominal,
        radius=0.2,
        dual_solver=GradientDescent(step_size=0.05, max_iter=5000, tol=1e-9),
    ).worst_case_expectation(loss)

    assert large >= small - 1e-6


def test_chi_square_worst_case_expectation_is_at_least_nominal_expectation() -> None:
    nominal = torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    solver = GradientDescent(step_size=0.05, max_iter=5000, tol=1e-9)
    ambiguity_set = ChiSquareAmbiguitySet(nominal, radius=0.1, dual_solver=solver)

    result = ambiguity_set.worst_case_expectation(loss)

    assert result >= torch.sum(nominal * loss) - 1e-4


def test_chi_square_worst_case_expectation_does_not_exceed_max_loss() -> None:
    nominal = torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    solver = GradientDescent(step_size=0.05, max_iter=5000, tol=1e-9)
    ambiguity_set = ChiSquareAmbiguitySet(nominal, radius=0.1, dual_solver=solver)

    result = ambiguity_set.worst_case_expectation(loss)

    assert result <= loss.max() + 1e-3


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


def test_chi_square_is_a_phi_and_ambiguity_set_instance() -> None:
    nominal = torch.tensor([0.5, 0.5])
    ambiguity_set = ChiSquareAmbiguitySet(nominal, radius=0.1)

    assert isinstance(ambiguity_set, PhiAmbiguitySet)
    assert isinstance(ambiguity_set, AmbiguitySet)


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

    assert ambiguity_set.divergence(candidate, nominal) <= ambiguity_set.radius + 1e-9
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
