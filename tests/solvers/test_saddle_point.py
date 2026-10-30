"""Tests for `SaddlePointSolver`."""

from collections.abc import Callable
from typing import Any

import pytest
import torch

from optora.core.solver_base import MinimizationProblem, Solver
from optora.solvers.gradient_descent import GradientDescent
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


def test_invalid_check_interval_raises_value_error() -> None:
    with pytest.raises(ValueError, match="check_interval must be positive"):
        SaddlePointSolver(check_interval=0)


@pytest.mark.parametrize("check_interval", [1, 7, 1000])
def test_check_interval_does_not_change_the_result(check_interval: int) -> None:
    problem = SaddlePointProblem(
        objective=lambda x, y: torch.sum((x - 1.0) ** 2) - torch.sum((y - 5.0) ** 2),
        primal_initial_point=torch.zeros(1, dtype=torch.float64),
        dual_initial_point=torch.zeros(1, dtype=torch.float64),
    )
    reference = SaddlePointSolver(
        primal_step_size=0.1, dual_step_size=0.1, max_iter=500, tol=1e-8
    ).solve(problem)

    result = SaddlePointSolver(
        primal_step_size=0.1,
        dual_step_size=0.1,
        max_iter=500,
        tol=1e-8,
        check_interval=check_interval,
    ).solve(problem)

    assert torch.equal(result.primal_point, reference.primal_point)
    assert torch.equal(result.dual_point, reference.dual_point)
    assert result.converged == reference.converged
    assert result.num_iterations == reference.num_iterations


def test_does_not_synchronize_on_every_iteration(
    host_sync_counter: Callable[[], Any],
) -> None:
    # Regression test: see the matching test in `tests/solvers/
    # test_gradient_descent.py`.
    solver = SaddlePointSolver(
        primal_step_size=0.1,
        dual_step_size=0.1,
        max_iter=50,
        tol=1e-30,
        check_interval=10,
    )
    problem = SaddlePointProblem(
        objective=lambda x, y: torch.sum((x - 1.0) ** 2) - torch.sum((y - 5.0) ** 2),
        primal_initial_point=torch.zeros(1),
        dual_initial_point=torch.zeros(1),
    )

    with host_sync_counter() as syncs:
        solver.solve(problem)

    assert len(syncs) <= 50 // 10 + 1


def test_no_objective_evaluations_after_convergence() -> None:
    # Regression test: see the matching test in `tests/solvers/
    # test_gradient_descent.py`. The post-loop final-value computation used
    # to re-evaluate the objective at iterates convergence had already
    # frozen.
    calls: list[None] = []

    def objective(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        calls.append(None)
        return torch.sum((x - 1.0) ** 2) - torch.sum((y - 5.0) ** 2)

    solver = SaddlePointSolver(
        primal_step_size=0.1,
        dual_step_size=0.1,
        max_iter=500,
        tol=1e-8,
        check_interval=1,
    )
    problem = SaddlePointProblem(
        objective=objective,
        primal_initial_point=torch.zeros(1, dtype=torch.float64),
        dual_initial_point=torch.zeros(1, dtype=torch.float64),
    )

    result = solver.solve(problem)

    assert result.converged
    assert len(calls) == result.num_iterations + 1


@pytest.mark.parametrize("tol", [1e-8, 1e-30])
def test_value_is_the_objective_at_the_returned_iterates(tol: float) -> None:
    # Covers both the converged path, where the value is reused from the
    # last frozen iteration, and the unconverged path, where it is computed
    # after the loop.
    def objective(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sum((x - 1.0) ** 2) - torch.sum((y - 5.0) ** 2)

    solver = SaddlePointSolver(
        primal_step_size=0.1,
        dual_step_size=0.1,
        max_iter=60,
        tol=tol,
        check_interval=1,
    )
    problem = SaddlePointProblem(
        objective=objective,
        primal_initial_point=torch.zeros(1, dtype=torch.float64),
        dual_initial_point=torch.zeros(1, dtype=torch.float64),
    )

    result = solver.solve(problem)

    assert torch.allclose(
        result.value, objective(result.primal_point, result.dual_point)
    )


def test_objective_running_a_nested_inner_solve_does_not_raise() -> None:
    # Regression test: see the matching test in `tests/solvers/
    # test_gradient_descent.py` for why the post-loop final-value
    # computation must not disable autograd.
    inner_solver = GradientDescent(step_size=0.5, max_iter=20, tol=1e-10)

    def objective(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        inner_result = inner_solver.solve(
            MinimizationProblem(
                objective=lambda z: (z - 1.0) ** 2,
                initial_point=torch.zeros(()),
            )
        )
        return (
            torch.sum((x - 1.0) ** 2) - torch.sum((y - 5.0) ** 2) + inner_result.value
        )

    solver = SaddlePointSolver(
        primal_step_size=0.1, dual_step_size=0.1, max_iter=200, tol=1e-8
    )
    problem = SaddlePointProblem(
        objective=objective,
        primal_initial_point=torch.zeros(1),
        dual_initial_point=torch.zeros(1),
    )

    result = solver.solve(problem)

    assert torch.allclose(result.primal_point, torch.tensor([1.0]), atol=1e-3)
    assert torch.allclose(result.dual_point, torch.tensor([5.0]), atol=1e-3)


def test_unconverged_objective_running_a_nested_inner_solve_does_not_raise() -> None:
    # The converged path reuses the last in-loop value, so only an
    # unconverged solve still reaches the post-loop evaluation guarded by
    # the regression test above. Keep that path covered.
    inner_solver = GradientDescent(step_size=0.5, max_iter=20, tol=1e-10)

    def objective(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        inner_result = inner_solver.solve(
            MinimizationProblem(
                objective=lambda z: (z - 1.0) ** 2,
                initial_point=torch.zeros(()),
            )
        )
        return (
            torch.sum((x - 1.0) ** 2) - torch.sum((y - 5.0) ** 2) + inner_result.value
        )

    solver = SaddlePointSolver(
        primal_step_size=0.1, dual_step_size=0.1, max_iter=5, tol=1e-30
    )
    problem = SaddlePointProblem(
        objective=objective,
        primal_initial_point=torch.zeros(1),
        dual_initial_point=torch.zeros(1),
    )

    result = solver.solve(problem)

    assert not result.converged


def test_objective_independent_of_the_primal_point_raises_value_error() -> None:
    # A missing gradient must never be silently replaced by zeros: that
    # would report `converged=True` at iteration 1 on iterates the solver
    # never optimized.
    solver = SaddlePointSolver(primal_step_size=0.1, dual_step_size=0.1, max_iter=10)
    problem = SaddlePointProblem(
        objective=lambda x, y: -torch.sum((y - 5.0) ** 2),
        primal_initial_point=torch.zeros(1),
        dual_initial_point=torch.zeros(1),
    )

    with pytest.raises(ValueError, match="does not depend on the primal point"):
        solver.solve(problem)


def test_objective_independent_of_the_dual_point_raises_value_error() -> None:
    solver = SaddlePointSolver(primal_step_size=0.1, dual_step_size=0.1, max_iter=10)
    problem = SaddlePointProblem(
        objective=lambda x, y: torch.sum((x - 1.0) ** 2),
        primal_initial_point=torch.zeros(1),
        dual_initial_point=torch.zeros(1),
    )

    with pytest.raises(ValueError, match="does not depend on the dual point"):
        solver.solve(problem)


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


def _coupled_saddle(
    center: torch.Tensor, cap: float | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve min_x max_y ||x - c||^2 + <x, y> - ||y + c||^2 from the origin.

    Its saddle point is x = 6c / 5, y = -2c / 5; a `cap` projects the dual
    iterate onto y <= cap after every ascent step.
    """
    problem = SaddlePointProblem(
        objective=lambda x, y: (
            torch.sum((x - center) ** 2)
            + torch.sum(x * y)
            - torch.sum((y + center) ** 2)
        ),
        primal_initial_point=torch.zeros_like(center),
        dual_initial_point=torch.zeros_like(center),
        dual_projection=None if cap is None else lambda y: torch.clamp(y, max=cap),
    )
    solver = SaddlePointSolver(
        primal_step_size=0.1, dual_step_size=0.1, max_iter=2000, tol=1e-12
    )
    result = solver.solve(problem)
    return result.primal_point, result.dual_point


CENTERS = torch.tensor([[1.0, -2.0], [0.5, 3.0], [0.0, 0.0]], dtype=torch.float64)


def test_solves_inside_torch_func_vmap() -> None:
    # Each mapped element converges on its own and is frozen there, so the
    # solution is the one an independent solve would return.
    primal, dual = torch.func.vmap(_coupled_saddle)(CENTERS)

    torch.testing.assert_close(primal, 1.2 * CENTERS, rtol=0.0, atol=1e-10)
    torch.testing.assert_close(dual, -0.4 * CENTERS, rtol=0.0, atol=1e-10)
    for index, center in enumerate(CENTERS):
        alone = _coupled_saddle(center)
        torch.testing.assert_close(primal[index], alone[0], rtol=0.0, atol=1e-12)
        torch.testing.assert_close(dual[index], alone[1], rtol=0.0, atol=1e-12)


def test_solves_inside_torch_func_vmap_with_a_dual_projection() -> None:
    primal, dual = torch.func.vmap(lambda center: _coupled_saddle(center, cap=-0.5))(
        CENTERS
    )

    assert bool(torch.all(dual <= -0.5))
    for index, center in enumerate(CENTERS):
        alone = _coupled_saddle(center, cap=-0.5)
        torch.testing.assert_close(primal[index], alone[0], rtol=0.0, atol=1e-12)
        torch.testing.assert_close(dual[index], alone[1], rtol=0.0, atol=1e-12)


def test_solves_inside_torch_func_grad_without_differentiating_its_steps() -> None:
    # Every iterate is detached, so the returned point is a constant to the
    # enclosing transform and the gradient below is the point itself.
    center = CENTERS[0]
    expected, _ = _coupled_saddle(center)

    gradient = torch.func.grad(
        lambda scale: torch.sum(scale * _coupled_saddle(center)[0])
    )(torch.ones(2, dtype=torch.float64))

    torch.testing.assert_close(gradient, expected)


def test_inside_a_transform_an_ignored_iterate_has_a_zero_gradient() -> None:
    # torch.func cannot tell an ignored iterate from a stationary one, so the
    # primal point stays where it started instead of raising.
    def solve(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        problem = SaddlePointProblem(
            objective=lambda x, y: -torch.sum((y - target) ** 2),
            primal_initial_point=torch.ones(1, dtype=torch.float64),
            dual_initial_point=torch.zeros(1, dtype=torch.float64),
        )
        solver = SaddlePointSolver(
            primal_step_size=0.1, dual_step_size=0.1, max_iter=500, tol=1e-12
        )
        result = solver.solve(problem)
        return result.primal_point, result.dual_point

    targets = torch.tensor([[5.0], [-1.0]], dtype=torch.float64)

    primal, dual = torch.func.vmap(solve)(targets)

    torch.testing.assert_close(primal, torch.ones(2, 1, dtype=torch.float64))
    torch.testing.assert_close(dual, targets, rtol=0.0, atol=1e-10)
