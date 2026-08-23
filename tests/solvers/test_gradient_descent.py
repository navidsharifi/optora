"""Tests for `GradientDescent`."""

import pytest
import torch

from optora.core.solver_base import Solver
from optora.solvers.gradient_descent import (
    GradientDescent,
    GradientDescentProblem,
)


def test_converges_to_minimizer_of_quadratic() -> None:
    solver = GradientDescent(step_size=0.1, max_iter=1000, tol=1e-8)
    minimizer = torch.tensor([3.0, -2.0], dtype=torch.float64)
    problem = GradientDescentProblem(
        objective=lambda x: torch.sum((x - minimizer) ** 2),
        initial_point=torch.zeros(2, dtype=torch.float64),
    )

    result = solver.solve(problem)

    assert result.converged
    assert torch.allclose(result.point, minimizer, atol=1e-3)
    assert torch.allclose(result.value, torch.zeros((), dtype=torch.float64), atol=1e-3)


def test_warm_start_resumes_from_previous_result() -> None:
    solver = GradientDescent(step_size=0.1, max_iter=1, tol=1e-8)
    minimizer = torch.tensor([3.0, -2.0])
    objective = lambda x: torch.sum((x - minimizer) ** 2)  # noqa: E731
    first = solver.solve(
        GradientDescentProblem(objective=objective, initial_point=torch.zeros(2))
    )

    warm_started = solver.solve(
        GradientDescentProblem(objective=objective, initial_point=first.point)
    )
    from_scratch = solver.solve(
        GradientDescentProblem(objective=objective, initial_point=torch.zeros(2))
    )

    assert torch.sum((warm_started.point - minimizer) ** 2) < torch.sum(
        (from_scratch.point - minimizer) ** 2
    )


def test_stops_at_max_iter_when_not_converged() -> None:
    solver = GradientDescent(step_size=0.1, max_iter=1, tol=1e-12)
    minimizer = torch.tensor([3.0, -2.0])
    problem = GradientDescentProblem(
        objective=lambda x: torch.sum((x - minimizer) ** 2),
        initial_point=torch.zeros(2),
    )

    result = solver.solve(problem)

    assert not result.converged
    assert result.num_iterations == 1


def test_objective_running_a_nested_inner_solve_does_not_raise() -> None:
    # Regression test: an objective that itself runs another autograd-based
    # `GradientDescent` solve (as `AmbiguitySet.worst_case_expectation` does
    # for KL/phi/Wasserstein-DRO) used to crash the outer solver's post-loop
    # final-value computation with "element 0 of tensors does not require
    # grad", because that computation ran under `torch.no_grad()`, which
    # also disables autograd for the nested inner solve.
    inner_solver = GradientDescent(step_size=0.5, max_iter=20, tol=1e-10)

    def objective(x: torch.Tensor) -> torch.Tensor:
        inner_result = inner_solver.solve(
            GradientDescentProblem(
                objective=lambda y: (y - 1.0) ** 2,
                initial_point=torch.zeros(()),
            )
        )
        return x**2 + inner_result.value

    solver = GradientDescent(step_size=0.1, max_iter=200, tol=1e-8)
    problem = GradientDescentProblem(
        objective=objective, initial_point=torch.tensor(5.0)
    )

    result = solver.solve(problem)

    assert result.converged
    assert torch.allclose(result.point, torch.zeros(()), atol=1e-3)


def test_invalid_step_size_raises_value_error() -> None:
    with pytest.raises(ValueError):
        GradientDescent(step_size=0.0)
    with pytest.raises(ValueError):
        GradientDescent(step_size=-1.0)


def test_invalid_max_iter_raises_value_error() -> None:
    with pytest.raises(ValueError):
        GradientDescent(max_iter=0)


def test_invalid_tol_raises_value_error() -> None:
    with pytest.raises(ValueError):
        GradientDescent(tol=0.0)


def test_is_a_solver_instance() -> None:
    assert isinstance(GradientDescent(), Solver)
