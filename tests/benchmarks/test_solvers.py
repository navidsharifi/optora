"""Performance guard for the iterative solvers in `optora.solvers`.

Each solve runs to `max_iter` with an unreachable tolerance, so it pays
exactly one host synchronization per `check_interval` iterations and none
for its diagnostics, which stay on the device until read.
"""

from collections.abc import Callable
from typing import Any

import torch

from optora.core.solver_base import MinimizationProblem
from optora.solvers.gradient_descent import GradientDescent
from optora.solvers.saddle_point import SaddlePointProblem, SaddlePointSolver

DIMENSION = 1024
MAX_ITER = 500
CHECK_INTERVAL = 50


def test_gradient_descent(guarded_benchmark: Callable[..., Any]) -> None:
    minimizer = torch.linspace(-1.0, 1.0, DIMENSION)
    problem = MinimizationProblem(
        objective=lambda x: torch.sum((x - minimizer) ** 2),
        initial_point=torch.zeros(DIMENSION),
    )
    solver = GradientDescent(
        step_size=0.1, max_iter=MAX_ITER, tol=1e-30, check_interval=CHECK_INTERVAL
    )

    result = guarded_benchmark(
        lambda: solver.solve(problem), host_syncs=MAX_ITER // CHECK_INTERVAL
    )

    assert result.point.shape == (DIMENSION,)


def test_saddle_point(guarded_benchmark: Callable[..., Any]) -> None:
    primal_target = torch.linspace(-1.0, 1.0, DIMENSION)
    dual_target = torch.linspace(1.0, 2.0, DIMENSION)
    problem = SaddlePointProblem(
        objective=lambda x, y: (
            torch.sum((x - primal_target) ** 2) - torch.sum((y - dual_target) ** 2)
        ),
        primal_initial_point=torch.zeros(DIMENSION),
        dual_initial_point=torch.zeros(DIMENSION),
    )
    solver = SaddlePointSolver(
        primal_step_size=0.1,
        dual_step_size=0.1,
        max_iter=MAX_ITER,
        tol=1e-30,
        check_interval=CHECK_INTERVAL,
    )

    result = guarded_benchmark(
        lambda: solver.solve(problem), host_syncs=MAX_ITER // CHECK_INTERVAL
    )

    assert result.primal_point.shape == (DIMENSION,)
