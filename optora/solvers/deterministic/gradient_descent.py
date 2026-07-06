"""Gradient descent with optional Armijo backtracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from optora.core.objective import Objective
from optora.core.problem import UnconstrainedOptimizationProblem
from optora.core.result import OptimizationResult
from optora.solvers.base import Solver
from optora.solvers.line_search import backtracking_armijo


@dataclass(slots=True)
class GradientDescent(Solver):
    """Steepest descent for smooth unconstrained objectives."""

    step_size: float = 1.0
    line_search: bool = True

    def solve(self, problem: UnconstrainedOptimizationProblem) -> OptimizationResult:
        objective: Objective = problem.objective
        x = problem.initial_tensor()
        history: list[dict[str, Any]] = []
        f = objective.value(x)

        for iteration in range(self.max_iter + 1):
            grad = objective.gradient(x)
            grad_norm = torch.linalg.vector_norm(grad)

            if self.store_history:
                history.append(
                    {"iteration": iteration, "fun": f, "grad_norm": grad_norm}
                )

            if bool((grad_norm <= self.tol).detach().item()):
                return OptimizationResult(
                    x=x,
                    fun=f,
                    jac=grad,
                    nit=iteration,
                    nfev=objective.nfev,
                    njev=objective.njev,
                    success=True,
                    status=0,
                    message="Converged: gradient norm is below tolerance.",
                    history=history,
                )

            if iteration == self.max_iter:
                break

            direction = -grad
            if self.line_search:
                step, f = backtracking_armijo(
                    objective.value,
                    x,
                    direction,
                    grad,
                    initial_step=self.step_size,
                )
            else:
                step = self.step_size
                f = objective.value(x + step * direction)
            x = x + step * direction

        return OptimizationResult(
            x=x,
            fun=f,
            jac=objective.gradient(x),
            nit=self.max_iter,
            nfev=objective.nfev,
            njev=objective.njev,
            success=False,
            status=1,
            message="Stopped: maximum iterations reached.",
            history=history,
        )
