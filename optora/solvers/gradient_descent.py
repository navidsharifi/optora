"""Gradient descent solver for differentiable objectives, with warm starts."""

import torch

from optora.core.solver_base import (
    MinimizationProblem,
    MinimizationResult,
    Solver,
    require_gradient,
)


class GradientDescent(Solver[MinimizationProblem, MinimizationResult]):
    r"""Fixed-step-size gradient descent for a differentiable objective.

    Repeatedly steps the current point against the objective's gradient,

    $$
    \mathrm{point} \leftarrow \mathrm{point}
        - \mathrm{step\_size} \cdot \nabla\, \mathrm{objective}(\mathrm{point}),
    $$

    until the gradient norm falls below `tol` or `max_iter` steps are
    exhausted.
    Solves `MinimizationProblem`, the shared unconstrained-minimization
    contract, so it can be injected wherever a solver of that problem class
    is expected — for example as the inner dual solver of an `optora.dro`
    ambiguity set.

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

    def solve(self, problem: MinimizationProblem) -> MinimizationResult:
        """Minimize `problem.objective` starting from `problem.initial_point`.

        Args:
            problem: Objective and initial point to solve from.

        Returns:
            A `MinimizationResult` holding the final iterate and
            convergence diagnostics.

        Raises:
            ValueError: If `problem.objective` does not depend on its
                argument through autograd.
        """
        point = problem.initial_point.detach().clone().requires_grad_(True)
        converged = False
        num_iterations = 0
        for iteration in range(self.max_iter):
            num_iterations = iteration + 1
            value = problem.objective(point)
            (raw_grad,) = torch.autograd.grad(value, point, allow_unused=True)
            grad = require_gradient(raw_grad, "the point")
            if torch.linalg.vector_norm(grad) < self.tol:
                converged = True
                break
            with torch.no_grad():
                point = point - self.step_size * grad
            point = point.detach().requires_grad_(True)
        # Not wrapped in `torch.no_grad()`: an objective composed from an
        # `AmbiguitySet.worst_case_expectation` runs its own inner
        # autograd-based dual solve, which needs autograd enabled here too.
        final_value = problem.objective(point).detach()
        return MinimizationResult(
            point=point.detach(),
            value=final_value,
            converged=converged,
            num_iterations=num_iterations,
        )
