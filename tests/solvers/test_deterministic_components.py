"""Tests for explicit deterministic solver components."""

import torch

import optora as opt
from optora.solvers.deterministic import BFGS, GradientDescent


def quadratic(x: torch.Tensor) -> torch.Tensor:
    """Convex quadratic test objective."""
    return (x[0] - 1.0) ** 2 + 10.0 * (x[1] + 2.0) ** 2


def quadratic_grad(x: torch.Tensor) -> torch.Tensor:
    """Gradient of ``quadratic``."""
    return torch.stack((2.0 * (x[0] - 1.0), 20.0 * (x[1] + 2.0)))


def test_bfgs_solves_explicit_problem() -> None:
    objective = opt.Objective(fun=quadratic, grad=quadratic_grad)
    problem = opt.UnconstrainedOptimizationProblem(
        objective=objective,
        initial_point=torch.tensor([0.0, 0.0], dtype=torch.float64),
    )

    result = BFGS().solve(problem)

    assert result.success
    torch.testing.assert_close(
        result.x,
        torch.tensor([1.0, -2.0], dtype=torch.float64),
        atol=1e-5,
        rtol=1e-5,
    )


def test_gradient_descent_solves_problem_with_finite_differences() -> None:
    objective = opt.Objective(fun=quadratic)
    problem = opt.UnconstrainedOptimizationProblem(
        objective=objective,
        initial_point=torch.tensor([0.0, 0.0], dtype=torch.float64),
    )
    solver = GradientDescent(
        max_iter=5000,
        tol=1e-5,
        step_size=0.05,
        line_search=False,
    )

    result = solver.solve(problem)

    assert result.success
    torch.testing.assert_close(
        result.x,
        torch.tensor([1.0, -2.0], dtype=torch.float64),
        atol=1e-4,
        rtol=1e-4,
    )


def test_solver_preserves_initial_tensor_dtype() -> None:
    initial_point = torch.tensor([0.0, 0.0], dtype=torch.float32)
    objective = opt.Objective(fun=quadratic, grad=quadratic_grad)
    problem = opt.UnconstrainedOptimizationProblem(
        objective=objective,
        initial_point=initial_point,
    )

    result = BFGS().solve(problem)

    assert result.x.dtype == torch.float32
    assert result.jac is not None
    assert result.jac.dtype == torch.float32


def test_package_exposes_components_not_minimize_dispatcher() -> None:
    assert opt.__version__ == "0.1.0"
    assert not hasattr(opt, "minimize")
