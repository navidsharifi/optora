"""Tests for `SaddlePointSolver`."""

import pytest
import torch

from optora.core.solver_base import Solver
from optora.solvers.saddle_point import SaddlePointProblem, SaddlePointSolver


def test_converges_to_saddle_point_of_separable_quadratic() -> None:
    solver = SaddlePointSolver(
        primal_step_size=0.1, dual_step_size=0.1, max_iter=1000, tol=1e-8
    )
    problem = SaddlePointProblem(
        objective=lambda x, y: torch.sum((x - 1.0) ** 2) - torch.sum((y - 5.0) ** 2),
        primal_initial_point=torch.zeros(1),
        dual_initial_point=torch.zeros(1),
    )

    result = solver.solve(problem)

    assert torch.allclose(result.primal_point, torch.tensor([1.0]), atol=1e-3)
    assert torch.allclose(result.dual_point, torch.tensor([5.0]), atol=1e-3)


def test_dual_projection_keeps_dual_point_feasible() -> None:
    solver = SaddlePointSolver(
        primal_step_size=0.1, dual_step_size=0.1, max_iter=500, tol=1e-8
    )
    problem = SaddlePointProblem(
        objective=lambda x, y: torch.sum((x - 1.0) ** 2) - torch.sum((y - 5.0) ** 2),
        primal_initial_point=torch.zeros(1),
        dual_initial_point=torch.zeros(1),
        dual_projection=lambda y: torch.clamp(y, max=2.0),
    )

    result = solver.solve(problem)

    assert torch.allclose(result.dual_point, torch.tensor([2.0]), atol=1e-3)


def test_stops_at_max_iter_when_not_converged() -> None:
    solver = SaddlePointSolver(
        primal_step_size=0.1, dual_step_size=0.1, max_iter=1, tol=1e-12
    )
    problem = SaddlePointProblem(
        objective=lambda x, y: torch.sum((x - 1.0) ** 2) - torch.sum((y - 5.0) ** 2),
        primal_initial_point=torch.zeros(1),
        dual_initial_point=torch.zeros(1),
    )

    result = solver.solve(problem)

    assert not result.converged
    assert result.num_iterations == 1


def test_invalid_primal_step_size_raises_value_error() -> None:
    with pytest.raises(ValueError):
        SaddlePointSolver(primal_step_size=0.0)
    with pytest.raises(ValueError):
        SaddlePointSolver(primal_step_size=-1.0)


def test_invalid_dual_step_size_raises_value_error() -> None:
    with pytest.raises(ValueError):
        SaddlePointSolver(dual_step_size=0.0)
    with pytest.raises(ValueError):
        SaddlePointSolver(dual_step_size=-1.0)


def test_invalid_max_iter_raises_value_error() -> None:
    with pytest.raises(ValueError):
        SaddlePointSolver(max_iter=0)


def test_invalid_tol_raises_value_error() -> None:
    with pytest.raises(ValueError):
        SaddlePointSolver(tol=0.0)


def test_is_a_solver_instance() -> None:
    assert isinstance(SaddlePointSolver(), Solver)
