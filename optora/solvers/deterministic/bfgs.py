"""BFGS quasi-Newton method."""

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
class BFGS(Solver):
    """Inverse-Hessian BFGS for smooth unconstrained objectives."""

    initial_step: float = 1.0
    curvature_tol: float = 1e-12

    def solve(self, problem: UnconstrainedOptimizationProblem) -> OptimizationResult:
        objective: Objective = problem.objective
        x = problem.initial_tensor()
        n = x.numel()
        inverse_hessian = torch.eye(n, dtype=x.dtype, device=x.device)
        history: list[dict[str, Any]] = []

        f = objective.value(x)
        grad = objective.gradient(x)

        for iteration in range(self.max_iter + 1):
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

            direction = -(inverse_hessian @ grad.reshape(-1)).reshape(x.shape)

            is_descent_direction = torch.dot(
                direction.reshape(-1),
                grad.reshape(-1),
            )
            if bool((is_descent_direction >= 0.0).detach().item()):
                direction = -grad
                inverse_hessian = torch.eye(n, dtype=x.dtype, device=x.device)

            step, f_next = backtracking_armijo(
                objective.value,
                x,
                direction,
                grad,
                initial_step=self.initial_step,
            )
            x_next = x + step * direction
            grad_next = objective.gradient(x_next)

            s = (x_next - x).reshape(-1)
            y = (grad_next - grad).reshape(-1)
            curvature = torch.dot(y, s)

            if bool((curvature > self.curvature_tol).detach().item()):
                rho = torch.reciprocal(curvature)
                identity = torch.eye(n, dtype=x.dtype, device=x.device)
                sy = torch.outer(s, y)
                ys = torch.outer(y, s)
                ss = torch.outer(s, s)
                inverse_hessian = (identity - rho * sy) @ inverse_hessian @ (
                    identity - rho * ys
                ) + rho * ss

            x = x_next
            f = f_next
            grad = grad_next

        return OptimizationResult(
            x=x,
            fun=f,
            jac=grad,
            nit=self.max_iter,
            nfev=objective.nfev,
            njev=objective.njev,
            success=False,
            status=1,
            message="Stopped: maximum iterations reached.",
            history=history,
        )
