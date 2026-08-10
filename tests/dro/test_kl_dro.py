"""Tests for `KLAmbiguitySet`."""

import pytest
import torch

from optora.core.dro_base import AmbiguitySet
from optora.core.solver_base import Solver
from optora.dro.kl_dro import KLAmbiguitySet
from optora.solvers.gradient_descent import (
    GradientDescent,
    GradientDescentProblem,
    GradientDescentResult,
)


class _RecordingSolver(Solver[GradientDescentProblem, GradientDescentResult]):
    """Fake dual solver returning a fixed `log(eta)` for deterministic checks."""

    def __init__(self, log_eta: float) -> None:
        self.log_eta = log_eta
        self.received_problem: GradientDescentProblem | None = None

    def solve(self, problem: GradientDescentProblem) -> GradientDescentResult:
        self.received_problem = problem
        point = torch.tensor(self.log_eta, dtype=torch.float32)
        return GradientDescentResult(
            point=point,
            value=problem.objective(point),
            converged=True,
            num_iterations=0,
        )


def _grid_search_dual_minimum(
    nominal: torch.Tensor, loss: torch.Tensor, radius: float
) -> torch.Tensor:
    """Independent fine-grid cross-check of the KL-DRO dual formula's minimum."""
    log_nominal = torch.log(nominal)
    eta_grid = torch.logspace(-3, 3, steps=20001, dtype=nominal.dtype)
    log_mgf = torch.logsumexp(
        log_nominal.unsqueeze(0) + loss.unsqueeze(0) / eta_grid.unsqueeze(1),
        dim=-1,
    )
    dual_values = eta_grid * radius + eta_grid * log_mgf
    return dual_values.min()


def test_zero_radius_returns_exact_expectation() -> None:
    nominal = torch.tensor([0.3, 0.3, 0.4])
    loss = torch.tensor([1.0, 2.0, 3.0])
    ambiguity_set = KLAmbiguitySet(nominal, radius=0.0)

    result = ambiguity_set.worst_case_expectation(loss)

    assert torch.allclose(result, torch.sum(nominal * loss), atol=1e-6)


def test_worst_case_expectation_uses_custom_dual_solver() -> None:
    nominal = torch.tensor([0.5, 0.5])
    loss = torch.tensor([0.0, 2.0])
    fake_solver = _RecordingSolver(log_eta=1.0)
    ambiguity_set = KLAmbiguitySet(nominal, radius=0.3, dual_solver=fake_solver)

    result = ambiguity_set.worst_case_expectation(loss)

    assert fake_solver.received_problem is not None
    eta = torch.exp(torch.tensor(1.0))
    expected = eta * 0.3 + eta * torch.logsumexp(
        torch.log(nominal) + loss / eta, dim=-1
    )
    assert torch.allclose(result, expected, atol=1e-6)


def test_worst_case_expectation_matches_grid_search_over_dual_variable() -> None:
    nominal = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0, 2.0, 5.0], dtype=torch.float64)
    radius = 0.2
    solver = GradientDescent(step_size=0.1, max_iter=20000, tol=1e-10)
    ambiguity_set = KLAmbiguitySet(nominal, radius=radius, dual_solver=solver)

    result = ambiguity_set.worst_case_expectation(loss)

    reference = _grid_search_dual_minimum(nominal, loss, radius)
    assert torch.allclose(result, reference, atol=1e-3)


def test_worst_case_expectation_reaches_max_loss_for_large_radius() -> None:
    nominal = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0, 2.0, 5.0], dtype=torch.float64)
    solver = GradientDescent(step_size=0.1, max_iter=20000, tol=1e-10)
    ambiguity_set = KLAmbiguitySet(nominal, radius=2.0, dual_solver=solver)

    result = ambiguity_set.worst_case_expectation(loss)

    assert torch.allclose(result, torch.tensor(5.0, dtype=torch.float64), atol=1e-2)


def test_worst_case_expectation_is_monotonic_in_radius() -> None:
    nominal = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)

    small = KLAmbiguitySet(
        nominal,
        radius=0.05,
        dual_solver=GradientDescent(step_size=0.1, max_iter=5000, tol=1e-9),
    ).worst_case_expectation(loss)
    large = KLAmbiguitySet(
        nominal,
        radius=0.5,
        dual_solver=GradientDescent(step_size=0.1, max_iter=5000, tol=1e-9),
    ).worst_case_expectation(loss)

    assert large >= small - 1e-6


def test_worst_case_expectation_is_at_least_nominal_expectation() -> None:
    nominal = torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    solver = GradientDescent(step_size=0.1, max_iter=5000, tol=1e-9)
    ambiguity_set = KLAmbiguitySet(nominal, radius=0.1, dual_solver=solver)

    result = ambiguity_set.worst_case_expectation(loss)

    assert result >= torch.sum(nominal * loss) - 1e-4


def test_worst_case_expectation_does_not_exceed_max_loss() -> None:
    nominal = torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    solver = GradientDescent(step_size=0.1, max_iter=5000, tol=1e-9)
    ambiguity_set = KLAmbiguitySet(nominal, radius=0.5, dual_solver=solver)

    result = ambiguity_set.worst_case_expectation(loss)

    assert result <= loss.max() + 1e-2


def test_mismatched_loss_shape_raises_value_error() -> None:
    nominal = torch.tensor([0.5, 0.5])
    loss = torch.tensor([1.0, 2.0, 3.0])
    ambiguity_set = KLAmbiguitySet(nominal, radius=0.1)

    with pytest.raises(ValueError):
        ambiguity_set.worst_case_expectation(loss)


def test_negative_radius_raises_value_error() -> None:
    nominal = torch.tensor([0.5, 0.5])

    with pytest.raises(ValueError):
        KLAmbiguitySet(nominal, radius=-1.0)


def test_invalid_eps_raises_value_error() -> None:
    nominal = torch.tensor([0.5, 0.5])

    with pytest.raises(ValueError):
        KLAmbiguitySet(nominal, radius=0.1, eps=0.0)
    with pytest.raises(ValueError):
        KLAmbiguitySet(nominal, radius=0.1, eps=-1.0)


def test_contains_uses_kl_divergence_and_radius() -> None:
    nominal = torch.tensor([0.5, 0.5])
    ambiguity_set = KLAmbiguitySet(nominal, radius=0.05)

    assert ambiguity_set.contains(nominal)
    assert not ambiguity_set.contains(torch.tensor([0.95, 0.05]))


def test_is_an_ambiguity_set_instance() -> None:
    nominal = torch.tensor([0.5, 0.5])

    assert isinstance(KLAmbiguitySet(nominal, radius=0.1), AmbiguitySet)
