"""Tests for `WassersteinAmbiguitySet`."""

import pytest
import torch

from optora.core.dro_base import AmbiguitySet
from optora.core.solver_base import (
    MinimizationProblem,
    MinimizationResult,
    Solver,
)
from optora.divergences.wasserstein import SinkhornDivergence
from optora.dro.wasserstein_dro import WassersteinAmbiguitySet
from optora.solvers.gradient_descent import GradientDescent

TWO_POINT_COST = torch.tensor([[0.0, 2.0], [2.0, 0.0]], dtype=torch.float64)

FOUR_POINT_COST = torch.tensor(
    [
        [0.0, 1.0, 2.0, 3.0],
        [1.0, 0.0, 1.0, 2.0],
        [2.0, 1.0, 0.0, 1.0],
        [3.0, 2.0, 1.0, 0.0],
    ],
    dtype=torch.float64,
)


class _RecordingSolver(Solver[MinimizationProblem, MinimizationResult]):
    """Fake dual solver returning a fixed `gamma_raw` for deterministic checks."""

    def __init__(self, gamma_raw: float) -> None:
        self.gamma_raw = gamma_raw
        self.received_problem: MinimizationProblem | None = None

    def solve(self, problem: MinimizationProblem) -> MinimizationResult:
        self.received_problem = problem
        point = torch.tensor(self.gamma_raw, dtype=problem.initial_point.dtype)
        return MinimizationResult(
            point=point,
            value=problem.objective(point),
            converged=True,
            num_iterations=0,
        )


def _grid_search_wasserstein_dual_minimum(
    nominal: torch.Tensor, loss: torch.Tensor, cost: torch.Tensor, radius: float
) -> torch.Tensor:
    """Independent fine-grid cross-check of the Wasserstein-DRO dual minimum."""
    gamma_grid = torch.linspace(0.0, 10.0, steps=400_001, dtype=nominal.dtype)
    shifted = loss.view(1, 1, -1) - gamma_grid.view(-1, 1, 1) * cost.view(
        1, *cost.shape
    )
    row_max = torch.amax(shifted, dim=-1)
    dual_values = gamma_grid * radius + torch.sum(nominal.view(1, -1) * row_max, dim=-1)
    return dual_values.min()


# --- Core dual solve behavior ----------------------------------------------


def test_zero_radius_returns_exact_expectation() -> None:
    nominal = torch.tensor([0.3, 0.3, 0.4])
    loss = torch.tensor([1.0, 2.0, 3.0])
    cost = torch.tensor(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]],
    )
    ambiguity_set = WassersteinAmbiguitySet(nominal, cost=cost, radius=0.0)

    result = ambiguity_set.worst_case_expectation(loss)

    assert torch.allclose(result, torch.sum(nominal * loss), atol=1e-6)


def test_worst_case_expectation_uses_custom_dual_solver() -> None:
    nominal = torch.tensor([0.5, 0.5], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0], dtype=torch.float64)
    fake_solver = _RecordingSolver(gamma_raw=0.5)
    ambiguity_set = WassersteinAmbiguitySet(
        nominal, cost=TWO_POINT_COST, radius=0.3, dual_solver=fake_solver
    )

    result = ambiguity_set.worst_case_expectation(loss)

    assert fake_solver.received_problem is not None
    gamma = torch.tensor(0.5, dtype=torch.float64)
    shifted = loss.unsqueeze(-2) - gamma * TWO_POINT_COST
    row_max = torch.amax(shifted, dim=-1)
    expected = gamma * 0.3 + torch.sum(nominal * row_max)
    assert torch.allclose(result, expected, atol=1e-6)


def test_positive_radius_requires_an_explicit_dual_solver() -> None:
    ambiguity_set = WassersteinAmbiguitySet(
        torch.tensor([0.5, 0.5]), cost=TWO_POINT_COST, radius=0.1
    )

    with pytest.raises(RuntimeError, match="dual_solver is required"):
        ambiguity_set.worst_case_expectation(
            torch.tensor([0.0, 1.0], dtype=torch.float64)
        )


def test_dual_reparameterization_clamps_negative_gamma_to_zero() -> None:
    nominal = torch.tensor([0.5, 0.5], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0], dtype=torch.float64)
    fake_solver = _RecordingSolver(gamma_raw=-3.0)
    ambiguity_set = WassersteinAmbiguitySet(
        nominal, cost=TWO_POINT_COST, radius=0.3, dual_solver=fake_solver
    )

    result = ambiguity_set.worst_case_expectation(loss)

    assert torch.allclose(result, loss.max(), atol=1e-6)


def test_dual_objective_keeps_one_sided_gradient_at_the_zero_boundary() -> None:
    # The nonnegativity projection must expose the right derivative of the dual
    # at gamma = 0, the default starting point: `torch.clamp` reports a zero
    # subgradient there, which would stall gradient descent immediately.
    nominal = torch.tensor([0.5, 0.5], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0], dtype=torch.float64)
    fake_solver = _RecordingSolver(gamma_raw=0.0)
    ambiguity_set = WassersteinAmbiguitySet(
        nominal, cost=TWO_POINT_COST, radius=0.3, dual_solver=fake_solver
    )

    ambiguity_set.worst_case_expectation(loss)

    assert fake_solver.received_problem is not None
    gamma_raw = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    (gradient,) = torch.autograd.grad(
        fake_solver.received_problem.objective(gamma_raw), gamma_raw
    )
    expected = 0.3 - torch.sum(nominal * TWO_POINT_COST[:, 1])
    assert torch.allclose(gradient, expected, atol=1e-12)


def test_worst_case_expectation_matches_grid_search_over_dual_variable() -> None:
    nominal = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0, 2.0, 5.0], dtype=torch.float64)
    radius = 0.2
    solver = GradientDescent(step_size=0.01, max_iter=20000, tol=1e-10)
    ambiguity_set = WassersteinAmbiguitySet(
        nominal, cost=FOUR_POINT_COST, radius=radius, dual_solver=solver
    )

    result = ambiguity_set.worst_case_expectation(loss)

    reference = _grid_search_wasserstein_dual_minimum(
        nominal, loss, FOUR_POINT_COST, radius
    )
    assert torch.allclose(result, reference, atol=1e-3)


# --- Hand-computed closed-form examples -------------------------------------


def test_matches_hand_computed_two_point_example_below_threshold() -> None:
    # nominal=[0.5,0.5], loss=[0,1], cost=[[0,2],[2,0]]: moving mass m at cost
    # 2*m per unit reallocated from point 0 to point 1 under budget 0.3 gives
    # m = 0.15, so Q = [0.35, 0.65] and E_Q[loss] = 0.65 (verified by hand and
    # against a fine grid search over gamma).
    nominal = torch.tensor([0.5, 0.5], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0], dtype=torch.float64)
    solver = GradientDescent(step_size=0.01, max_iter=20000, tol=1e-10)
    ambiguity_set = WassersteinAmbiguitySet(
        nominal, cost=TWO_POINT_COST, radius=0.3, dual_solver=solver
    )

    result = ambiguity_set.worst_case_expectation(loss)

    assert torch.allclose(result, torch.tensor(0.65, dtype=torch.float64), atol=1e-4)


def test_matches_hand_computed_two_point_example_at_threshold() -> None:
    # The budget needed to move all mass to the max-loss point is
    # sum_i p_i * cost_{i, argmax(loss)} = 0.5 * 2 + 0.5 * 0 = 1.0; at exactly
    # that radius the dual optimum sits at gamma = 0 and the worst case is the
    # max loss.
    nominal = torch.tensor([0.5, 0.5], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0], dtype=torch.float64)
    solver = GradientDescent(step_size=0.01, max_iter=20000, tol=1e-10)
    ambiguity_set = WassersteinAmbiguitySet(
        nominal, cost=TWO_POINT_COST, radius=1.0, dual_solver=solver
    )

    result = ambiguity_set.worst_case_expectation(loss)

    assert torch.allclose(result, torch.tensor(1.0, dtype=torch.float64), atol=1e-6)


def test_worst_case_expectation_is_attained_by_explicit_transport_plan() -> None:
    # The coupling pi = [[0.35, 0.15], [0.0, 0.5]] has row marginal
    # nominal=[0.5,0.5], column marginal Q=[0.35,0.65], and exact transport
    # cost sum(pi * cost) = 0.15 * 2 = 0.3 = radius, so Q lies exactly on the
    # boundary of the ambiguity set and attains the worst-case expectation.
    nominal = torch.tensor([0.5, 0.5], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0], dtype=torch.float64)
    candidate = torch.tensor([0.35, 0.65], dtype=torch.float64)
    coupling = torch.tensor([[0.35, 0.15], [0.0, 0.5]], dtype=torch.float64)

    assert torch.allclose(coupling.sum(dim=1), nominal, atol=1e-9)
    assert torch.allclose(coupling.sum(dim=0), candidate, atol=1e-9)
    exact_transport_cost = torch.sum(coupling * TWO_POINT_COST)
    assert exact_transport_cost <= 0.3 + 1e-9

    solver = GradientDescent(step_size=0.01, max_iter=20000, tol=1e-10)
    ambiguity_set = WassersteinAmbiguitySet(
        nominal, cost=TWO_POINT_COST, radius=0.3, dual_solver=solver
    )
    result = ambiguity_set.worst_case_expectation(loss)

    assert torch.allclose(result, torch.sum(candidate * loss), atol=1e-4)


def test_worst_case_expectation_reaches_max_loss_for_large_radius() -> None:
    nominal = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0, 2.0, 5.0], dtype=torch.float64)
    solver = GradientDescent(step_size=0.01, max_iter=20000, tol=1e-10)
    ambiguity_set = WassersteinAmbiguitySet(
        nominal, cost=FOUR_POINT_COST, radius=2.0, dual_solver=solver
    )

    result = ambiguity_set.worst_case_expectation(loss)

    assert torch.allclose(result, torch.tensor(5.0, dtype=torch.float64), atol=1e-6)


# --- General bounds and monotonicity ----------------------------------------


def test_worst_case_expectation_is_monotonic_in_radius() -> None:
    nominal = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0, 2.0, 5.0], dtype=torch.float64)
    solver = GradientDescent(step_size=0.01, max_iter=20000, tol=1e-10)

    small = WassersteinAmbiguitySet(
        nominal, cost=FOUR_POINT_COST, radius=0.05, dual_solver=solver
    ).worst_case_expectation(loss)
    large = WassersteinAmbiguitySet(
        nominal, cost=FOUR_POINT_COST, radius=0.5, dual_solver=solver
    ).worst_case_expectation(loss)

    assert large >= small - 1e-6


def test_worst_case_expectation_is_at_least_nominal_expectation() -> None:
    nominal = torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    cost = torch.tensor(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]], dtype=torch.float64
    )
    solver = GradientDescent(step_size=0.01, max_iter=20000, tol=1e-10)
    ambiguity_set = WassersteinAmbiguitySet(
        nominal, cost=cost, radius=0.1, dual_solver=solver
    )

    result = ambiguity_set.worst_case_expectation(loss)

    assert result >= torch.sum(nominal * loss) - 1e-4


def test_worst_case_expectation_does_not_exceed_max_loss() -> None:
    nominal = torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    cost = torch.tensor(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]], dtype=torch.float64
    )
    solver = GradientDescent(step_size=0.01, max_iter=20000, tol=1e-10)
    ambiguity_set = WassersteinAmbiguitySet(
        nominal, cost=cost, radius=0.5, dual_solver=solver
    )

    result = ambiguity_set.worst_case_expectation(loss)

    assert result <= loss.max() + 1e-4


# --- Validation --------------------------------------------------------------


def test_mismatched_loss_shape_raises_value_error() -> None:
    nominal = torch.tensor([0.5, 0.5])
    loss = torch.tensor([1.0, 2.0, 3.0])
    ambiguity_set = WassersteinAmbiguitySet(
        nominal, cost=torch.tensor([[0.0, 1.0], [1.0, 0.0]]), radius=0.1
    )

    with pytest.raises(ValueError):
        ambiguity_set.worst_case_expectation(loss)


def test_negative_radius_raises_value_error() -> None:
    nominal = torch.tensor([0.5, 0.5])

    with pytest.raises(ValueError):
        WassersteinAmbiguitySet(
            nominal, cost=torch.tensor([[0.0, 1.0], [1.0, 0.0]]), radius=-1.0
        )


def test_cost_size_mismatched_with_nominal_raises_value_error() -> None:
    nominal = torch.tensor([0.3, 0.3, 0.4])

    with pytest.raises(ValueError):
        WassersteinAmbiguitySet(
            nominal, cost=torch.tensor([[0.0, 1.0], [1.0, 0.0]]), radius=0.1
        )


def test_non_square_cost_raises_value_error() -> None:
    nominal = torch.tensor([0.5, 0.5])

    with pytest.raises(ValueError):
        WassersteinAmbiguitySet(nominal, cost=torch.zeros(2, 3), radius=0.1)


def test_negative_cost_raises_value_error() -> None:
    nominal = torch.tensor([0.5, 0.5])

    with pytest.raises(ValueError):
        WassersteinAmbiguitySet(
            nominal, cost=torch.tensor([[0.0, -1.0], [-1.0, 0.0]]), radius=0.1
        )


def test_sinkhorn_parameters_are_passed_through_to_divergence() -> None:
    nominal = torch.tensor([0.5, 0.5])
    cost = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    ambiguity_set = WassersteinAmbiguitySet(
        nominal,
        cost=cost,
        radius=0.1,
        epsilon=0.02,
        sinkhorn_max_iter=50,
        sinkhorn_tol=1e-4,
        eps=1e-10,
    )

    assert isinstance(ambiguity_set.divergence, SinkhornDivergence)
    assert ambiguity_set.divergence.epsilon == 0.02
    assert ambiguity_set.divergence.max_iter == 50
    assert ambiguity_set.divergence.tol == 1e-4
    assert ambiguity_set.divergence.eps == 1e-10


# --- Containment and type checks ---------------------------------------------


def test_contains_uses_sinkhorn_divergence_and_radius() -> None:
    nominal = torch.tensor([0.5, 0.5])
    cost = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    ambiguity_set = WassersteinAmbiguitySet(nominal, cost=cost, radius=0.05)

    assert ambiguity_set.contains(nominal)
    assert not ambiguity_set.contains(torch.tensor([0.95, 0.05]))


def test_is_an_ambiguity_set_instance() -> None:
    nominal = torch.tensor([0.5, 0.5])
    cost = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    assert isinstance(
        WassersteinAmbiguitySet(nominal, cost=cost, radius=0.1), AmbiguitySet
    )
