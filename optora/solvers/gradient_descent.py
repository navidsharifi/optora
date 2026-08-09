"""Gradient descent solver for differentiable objectives, with warm starts."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from optora.core.solver_base import Solver


@dataclass(frozen=True)
class GradientDescentProblem:
    """Differentiable minimization problem solved by `GradientDescent`.

    Attributes:
        objective: Differentiable scalar-valued function of a single tensor
            argument.
        initial_point: Starting point for the iteration. Passing a previous
            solve's `GradientDescentResult.point` here warm-starts the
            solver from that solution instead of from scratch, which is
            useful for the repeated inner-loop solves a DRO ambiguity set
            or minimax solver runs as its outer state changes slightly
            between calls.
    """

    objective: Callable[[torch.Tensor], torch.Tensor]
    initial_point: torch.Tensor


@dataclass(frozen=True)
class GradientDescentResult:
    """Outcome of a `GradientDescent` solve.

    Attributes:
        point: Final iterate.
        value: Objective value at `point`.
        converged: Whether the gradient norm fell below `tol` before
            `max_iter` steps were exhausted.
        num_iterations: Number of gradient steps actually performed.
    """

    point: torch.Tensor
    value: torch.Tensor
    converged: bool
    num_iterations: int


class GradientDescent(Solver[GradientDescentProblem, GradientDescentResult]):
    """Fixed-step-size gradient descent for a differentiable objective.

    Repeatedly steps the current point against the objective's gradient,
    `point <- point - step_size * grad(objective)(point)`, until the
    gradient norm falls below `tol` or `max_iter` steps are exhausted.
    `optora.dro` formulations use this as an inner-loop solver, for example
    to compute the dual variable of a phi-divergence ambiguity set's
    worst-case expectation.

    Attributes:
        step_size: Positive learning rate applied to each gradient step.
        max_iter: Maximum number of gradient steps.
        tol: Convergence tolerance on the gradient norm.
    """

    def __init__(
        self,
        step_size: float = 1e-2,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        """Initialize the gradient descent solver.

        Args:
            step_size: Positive learning rate applied to each gradient
                step.
            max_iter: Maximum number of gradient steps.
            tol: Convergence tolerance on the gradient norm.

        Raises:
            ValueError: If `step_size`, `max_iter`, or `tol` are not
                positive.
        """
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}.")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {max_iter}.")
        if tol <= 0:
            raise ValueError(f"tol must be positive, got {tol}.")
        self.step_size = step_size
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, problem: GradientDescentProblem) -> GradientDescentResult:
        """Minimize `problem.objective` starting from `problem.initial_point`.

        Args:
            problem: Objective and initial point to solve from.

        Returns:
            A `GradientDescentResult` holding the final iterate and
            convergence diagnostics.
        """
        point = problem.initial_point.detach().clone().requires_grad_(True)
        converged = False
        num_iterations = 0
        for iteration in range(self.max_iter):
            num_iterations = iteration + 1
            value = problem.objective(point)
            (grad,) = torch.autograd.grad(value, point, allow_unused=True)
            if grad is None:
                grad = torch.zeros_like(point)
            if torch.linalg.vector_norm(grad) < self.tol:
                converged = True
                break
            with torch.no_grad():
                point = point - self.step_size * grad
            point = point.detach().requires_grad_(True)
        with torch.no_grad():
            final_value = problem.objective(point)
        return GradientDescentResult(
            point=point.detach(),
            value=final_value.detach(),
            converged=converged,
            num_iterations=num_iterations,
        )
