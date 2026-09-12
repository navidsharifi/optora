"""Gradient descent solver for differentiable objectives, with warm starts."""

import torch

from optora.core.convergence import (
    DEFAULT_CHECK_INTERVAL,
    ConvergenceTracker,
    validate_check_interval,
)
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

    The gradient-norm test is evaluated on the iterate's device and read
    back to the host only every `check_interval` steps; steps taken after
    convergence are frozen, so the solution does not depend on that
    interval. See `optora.core.convergence.ConvergenceTracker`.

    Attributes:
        step_size: Positive learning rate applied to each gradient step.
        max_iter: Maximum number of gradient steps.
        tol: Convergence tolerance on the gradient norm.
        check_interval: Number of steps between host reads of the
            convergence flag.
    """

    def __init__(
        self,
        step_size: float = 1e-2,
        max_iter: int = 1000,
        tol: float = 1e-6,
        check_interval: int = DEFAULT_CHECK_INTERVAL,
    ) -> None:
        """Initialize the gradient descent solver.

        Args:
            step_size: Positive learning rate applied to each gradient
                step.
            max_iter: Maximum number of gradient steps.
            tol: Convergence tolerance on the gradient norm.
            check_interval: Number of steps between host reads of the
                convergence flag. Raise it to trade redundant frozen steps
                for fewer device synchronizations.

        Raises:
            ValueError: If `step_size`, `max_iter`, `tol`, or
                `check_interval` are not positive.
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
        self.check_interval = validate_check_interval(check_interval)

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
        tracker = ConvergenceTracker(self.tol, self.check_interval, point)
        for iteration in range(self.max_iter):
            value = problem.objective(point)
            (raw_grad,) = torch.autograd.grad(value, point, allow_unused=True)
            grad = require_gradient(raw_grad, "the point")
            with torch.no_grad():
                frozen = tracker.update(torch.linalg.vector_norm(grad))
                point = torch.where(frozen, point, point - self.step_size * grad)
            point = point.detach().requires_grad_(True)
            if tracker.should_stop(iteration):
                break
        # Not wrapped in `torch.no_grad()`: an objective composed from an
        # `AmbiguitySet.worst_case_expectation` runs its own inner
        # autograd-based dual solve, which needs autograd enabled here too.
        final_value = problem.objective(point).detach()
        converged, num_iterations = tracker.to_host()
        return MinimizationResult(
            point=point.detach(),
            value=final_value,
            converged=converged,
            num_iterations=num_iterations,
        )
