"""Tests for `MinimaxSolver`."""

import pytest
import torch

from optora.core.solver_base import (
    MinimizationProblem,
    MinimizationResult,
    Solver,
)
from optora.dro.kl_dro import KLAmbiguitySet
from optora.dro.minimax_solver import MinimaxProblem, MinimaxResult, MinimaxSolver
from optora.dro.phi_dro import ChiSquareAmbiguitySet, TotalVariationAmbiguitySet
from optora.dro.wasserstein_dro import WassersteinAmbiguitySet
from optora.solvers.gradient_descent import GradientDescent


class _RecordingSolver(Solver[MinimizationProblem, MinimizationResult]):
    """Fake outer solver returning fixed results for deterministic checks."""

    def __init__(
        self, point: float, value: float, converged: bool, num_iterations: int
    ) -> None:
        self.point = point
        self.value = value
        self.converged = converged
        self.num_iterations = num_iterations
        self.received_problem: MinimizationProblem | None = None

    def solve(self, problem: MinimizationProblem) -> MinimizationResult:
        self.received_problem = problem
        dtype = problem.initial_point.dtype
        return MinimizationResult(
            point=torch.tensor(self.point, dtype=dtype),
            value=torch.tensor(self.value, dtype=dtype),
            converged=self.converged,
            num_iterations=self.num_iterations,
        )


# --- Wiring and delegation ---------------------------------------------------


def test_solve_delegates_to_the_injected_outer_solver() -> None:
    nominal = torch.tensor([0.5, 0.5])
    targets = torch.tensor([1.0, 5.0])
    ambiguity_set = KLAmbiguitySet(nominal, radius=0.0)

    def loss_fn(x: torch.Tensor) -> torch.Tensor:
        return (x - targets) ** 2

    fake_solver = _RecordingSolver(
        point=2.5, value=1.23, converged=True, num_iterations=7
    )
    initial_point = torch.tensor(0.5)
    problem = MinimaxProblem(
        ambiguity_set=ambiguity_set, loss_fn=loss_fn, initial_point=initial_point
    )

    result = MinimaxSolver(solver=fake_solver).solve(problem)

    assert fake_solver.received_problem is not None
    assert torch.equal(fake_solver.received_problem.initial_point, initial_point)
    probe = torch.tensor(3.0)
    expected_objective = ambiguity_set.worst_case_expectation(loss_fn(probe))
    actual_objective = fake_solver.received_problem.objective(probe)
    assert torch.allclose(actual_objective, expected_objective)
    assert torch.allclose(result.point, torch.tensor(2.5))
    assert torch.allclose(result.value, torch.tensor(1.23))
    assert result.converged is True
    assert result.num_iterations == 7


def test_solver_is_required() -> None:
    with pytest.raises(TypeError):
        MinimaxSolver()  # type: ignore[call-arg]


def test_is_a_solver_instance() -> None:
    assert isinstance(MinimaxSolver(GradientDescent()), Solver)


def test_mismatched_loss_fn_shape_raises_value_error() -> None:
    nominal = torch.tensor([0.5, 0.5])
    ambiguity_set = KLAmbiguitySet(nominal, radius=0.1)

    def bad_loss_fn(x: torch.Tensor) -> torch.Tensor:
        return torch.stack([x, x, x])

    problem = MinimaxProblem(
        ambiguity_set=ambiguity_set,
        loss_fn=bad_loss_fn,
        initial_point=torch.tensor(0.0),
    )

    with pytest.raises(ValueError):
        MinimaxSolver(GradientDescent()).solve(problem)


# --- Differentiability of the composed objective (envelope theorem) --------


def test_worst_case_expectation_gradient_matches_finite_differences() -> None:
    # `MinimaxSolver` relies on `worst_case_expectation` staying differentiable
    # with respect to a decision variable the loss depends on, even though the
    # ambiguity set solves its own dual variable through a detached inner
    # `dual_solver`. Verify this envelope-theorem property directly.
    nominal = torch.tensor([0.5, 0.5], dtype=torch.float64)
    targets = torch.tensor([1.0, 5.0], dtype=torch.float64)
    ambiguity_set = KLAmbiguitySet(
        nominal,
        radius=0.2,
        dual_solver=GradientDescent(step_size=0.1, max_iter=500, tol=1e-9),
    )

    def objective(x: torch.Tensor) -> torch.Tensor:
        return ambiguity_set.worst_case_expectation((x - targets) ** 2)

    x = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    value = objective(x)
    (grad,) = torch.autograd.grad(value, x)

    eps = 1e-4
    finite_difference = (
        objective(torch.tensor(eps, dtype=torch.float64))
        - objective(torch.tensor(-eps, dtype=torch.float64))
    ) / (2 * eps)

    assert torch.allclose(grad, finite_difference, atol=1e-3)


# --- Exact closed-form robust decisions --------------------------------------


def test_zero_radius_converges_to_weighted_least_squares_solution() -> None:
    # With radius=0, worst_case_expectation(loss) is exactly sum(nominal *
    # loss), so the robust problem is plain weighted least squares, whose
    # minimizer is the nominal-weighted mean of the targets.
    nominal = torch.tensor([0.3, 0.7], dtype=torch.float64)
    targets = torch.tensor([1.0, 5.0], dtype=torch.float64)
    ambiguity_set = KLAmbiguitySet(nominal, radius=0.0)

    def loss_fn(x: torch.Tensor) -> torch.Tensor:
        return (x - targets) ** 2

    problem = MinimaxProblem(
        ambiguity_set=ambiguity_set,
        loss_fn=loss_fn,
        initial_point=torch.tensor(0.0, dtype=torch.float64),
    )
    solver = MinimaxSolver(
        solver=GradientDescent(step_size=0.05, max_iter=3000, tol=1e-9)
    )

    result = solver.solve(problem)

    expected = torch.sum(nominal * targets)
    assert result.converged
    assert torch.allclose(result.point, expected, atol=1e-3)


def test_total_variation_ambiguity_shifts_decision_to_symmetric_midpoint() -> None:
    # nominal=[0.8,0.2], targets=[0,10], radius=0.3: on the side of the
    # midpoint x=5 closer to target 0, the reallocatable mass caps at
    # min(radius, 0.8)=0.3, giving worst case 0.5*loss0+0.5*loss1; on the
    # side closer to target 10, it caps at min(radius, 0.2)=0.2, giving
    # worst case loss0 alone. Both pieces are minimized at the symmetric
    # midpoint x=5 (value 25), verified below both analytically (hard-coded)
    # and against an independent grid search.
    nominal = torch.tensor([0.8, 0.2], dtype=torch.float64)
    targets = torch.tensor([0.0, 10.0], dtype=torch.float64)
    ambiguity_set = TotalVariationAmbiguitySet(nominal, radius=0.3)

    def loss_fn(x: torch.Tensor) -> torch.Tensor:
        return (x - targets) ** 2

    problem = MinimaxProblem(
        ambiguity_set=ambiguity_set,
        loss_fn=loss_fn,
        initial_point=torch.tensor(0.0, dtype=torch.float64),
    )
    solver = MinimaxSolver(
        solver=GradientDescent(step_size=0.01, max_iter=3000, tol=1e-9)
    )

    result = solver.solve(problem)

    assert result.converged
    assert torch.allclose(
        result.point, torch.tensor(5.0, dtype=torch.float64), atol=1e-3
    )
    assert torch.allclose(
        result.value, torch.tensor(25.0, dtype=torch.float64), atol=1e-3
    )

    grid = torch.linspace(-5.0, 15.0, 2001, dtype=torch.float64)
    grid_values = torch.stack(
        [ambiguity_set.worst_case_expectation(loss_fn(x)) for x in grid]
    )
    best = torch.argmin(grid_values)
    assert torch.allclose(grid[best], result.point, atol=0.02)
    assert torch.allclose(grid_values[best], result.value, atol=1e-2)


# --- Robustification against a naive (radius=0) decision --------------------


def test_kl_ambiguity_radius_improves_worst_case_over_naive_decision() -> None:
    nominal = torch.tensor([0.3, 0.7], dtype=torch.float64)
    targets = torch.tensor([1.0, 5.0], dtype=torch.float64)
    ambiguity_set = KLAmbiguitySet(
        nominal,
        radius=0.3,
        dual_solver=GradientDescent(step_size=0.3, max_iter=150, tol=1e-8),
    )

    def loss_fn(x: torch.Tensor) -> torch.Tensor:
        return (x - targets) ** 2

    initial_point = torch.tensor(0.0, dtype=torch.float64)
    problem = MinimaxProblem(
        ambiguity_set=ambiguity_set, loss_fn=loss_fn, initial_point=initial_point
    )
    solver = MinimaxSolver(
        solver=GradientDescent(step_size=0.1, max_iter=150, tol=1e-6)
    )

    result = solver.solve(problem)

    # Self-consistency: the reported value matches a fresh evaluation.
    recomputed = ambiguity_set.worst_case_expectation(loss_fn(result.point))
    assert torch.allclose(recomputed, result.value, atol=1e-6)

    # The worst-case expectation of any loss lies between its nominal
    # expectation and its maximum.
    loss_at_result = loss_fn(result.point)
    assert result.value >= torch.sum(nominal * loss_at_result) - 1e-4
    assert result.value <= loss_at_result.max() + 1e-4

    # The robust decision does at least as well, in the worst case, as the
    # naive (radius=0) weighted-mean decision evaluated under the same
    # ambiguity set.
    naive_point = torch.sum(nominal * targets)
    naive_value = ambiguity_set.worst_case_expectation(loss_fn(naive_point))
    assert result.value <= naive_value + 1e-6
    assert not torch.allclose(result.point, initial_point)


# --- Generic wiring across ambiguity-set types -------------------------------


def test_generic_wiring_for_chi_square_ambiguity_set() -> None:
    nominal = torch.tensor([0.5, 0.5], dtype=torch.float64)
    targets = torch.tensor([1.0, 5.0], dtype=torch.float64)
    ambiguity_set = ChiSquareAmbiguitySet(
        nominal,
        radius=0.3,
        dual_solver=GradientDescent(step_size=0.02, max_iter=150, tol=1e-7),
    )

    def loss_fn(x: torch.Tensor) -> torch.Tensor:
        return (x - targets) ** 2

    initial_point = torch.tensor(0.0, dtype=torch.float64)
    problem = MinimaxProblem(
        ambiguity_set=ambiguity_set, loss_fn=loss_fn, initial_point=initial_point
    )
    solver = MinimaxSolver(
        solver=GradientDescent(step_size=0.02, max_iter=150, tol=1e-5)
    )

    result = solver.solve(problem)

    recomputed = ambiguity_set.worst_case_expectation(loss_fn(result.point))
    assert torch.allclose(recomputed, result.value, atol=1e-6)
    assert result.value >= torch.sum(nominal * loss_fn(result.point)) - 1e-4
    assert not torch.allclose(result.point, initial_point)


def test_generic_wiring_for_wasserstein_ambiguity_set() -> None:
    nominal = torch.tensor([0.5, 0.5], dtype=torch.float64)
    targets = torch.tensor([1.0, 5.0], dtype=torch.float64)
    cost = torch.tensor([[0.0, 2.0], [2.0, 0.0]], dtype=torch.float64)
    ambiguity_set = WassersteinAmbiguitySet(
        nominal,
        cost=cost,
        radius=0.3,
        dual_solver=GradientDescent(step_size=0.02, max_iter=150, tol=1e-7),
    )

    def loss_fn(x: torch.Tensor) -> torch.Tensor:
        return (x - targets) ** 2

    initial_point = torch.tensor(0.0, dtype=torch.float64)
    problem = MinimaxProblem(
        ambiguity_set=ambiguity_set, loss_fn=loss_fn, initial_point=initial_point
    )
    solver = MinimaxSolver(
        solver=GradientDescent(step_size=0.02, max_iter=150, tol=1e-5)
    )

    result = solver.solve(problem)

    recomputed = ambiguity_set.worst_case_expectation(loss_fn(result.point))
    assert torch.allclose(recomputed, result.value, atol=1e-6)
    assert result.value >= torch.sum(nominal * loss_fn(result.point)) - 1e-4
    assert not torch.allclose(result.point, initial_point)


def test_minimax_result_is_a_plain_dataclass_of_the_expected_shape() -> None:
    nominal = torch.tensor([0.5, 0.5])
    targets = torch.tensor([1.0, 5.0])
    ambiguity_set = KLAmbiguitySet(nominal, radius=0.0)

    def loss_fn(x: torch.Tensor) -> torch.Tensor:
        return (x - targets) ** 2

    problem = MinimaxProblem(
        ambiguity_set=ambiguity_set,
        loss_fn=loss_fn,
        initial_point=torch.tensor(0.0),
    )

    result = MinimaxSolver(GradientDescent()).solve(problem)

    assert isinstance(result, MinimaxResult)
    assert isinstance(result.point, torch.Tensor)
    assert isinstance(result.value, torch.Tensor)
    assert isinstance(result.converged, bool)
    assert isinstance(result.num_iterations, int)
