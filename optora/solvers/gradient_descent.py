"""Gradient descent solver for differentiable objectives, with warm starts."""

from collections.abc import Callable

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
    interval. Keep `check_interval=1` when the objective is expensive, as
    it is when this solver is the outer solver of an `optora.dro.MinimaxSolver`
    and every step costs an inner dual solve. See
    `optora.core.convergence.ConvergenceTracker`. The returned diagnostics
    stay on the device as well (see
    `optora.core.convergence.ConvergenceStatus`), so a solve whose
    `converged` and `num_iterations` nobody reads — every inner dual solve
    of an `optora.dro` ambiguity set — never synchronizes for them.

    The point is a single tensor of any shape, and the stopping rule is the
    norm of the whole gradient. A batch of independent problems stacked into
    one point (as an `optora.dro` ambiguity set does for a batched loss) is
    therefore stopped jointly: iteration continues until every element is
    stationary. Per-element early exit is deliberately not offered, since
    retiring elements individually needs a host-side read of a per-element
    mask on every step, which is exactly the synchronization
    `ConvergenceTracker` exists to avoid. The frozen-step rule makes the
    extra steps exact no-ops for elements that already converged.

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
                for fewer device synchronizations, but only when a step is
                cheap relative to a synchronization.

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

    def _evaluate(
        self,
        objective: Callable[[torch.Tensor], torch.Tensor],
        point: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the objective and its gradient at a differentiable point.

        Args:
            objective: Differentiable scalar-valued objective.
            point: Iterate the objective is evaluated at, requiring grad.

        Returns:
            The objective value and its gradient with respect to `point`.

        Raises:
            ValueError: If `objective` does not depend on `point` through
                autograd.
        """
        value = objective(point)
        (raw_grad,) = torch.autograd.grad(value, point, allow_unused=True)
        return value, require_gradient(raw_grad, "the point")

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
        # Each iteration steps with the gradient of the previous evaluation
        # and then evaluates the new iterate, so the last evaluation is
        # always taken at the point returned below. Evaluating after the
        # loop instead would need the convergence flag on the host, one
        # synchronization per solve, which nesting multiplies by the outer
        # iteration count.
        value, grad = self._evaluate(problem.objective, point)
        frozen = tracker.start(torch.linalg.vector_norm(grad))
        for iteration in range(self.max_iter):
            with torch.no_grad():
                point = torch.where(frozen, point, point - self.step_size * grad)
            point = point.detach().requires_grad_(True)
            value, grad = self._evaluate(problem.objective, point)
            frozen = tracker.update(torch.linalg.vector_norm(grad))
            if tracker.should_stop(iteration):
                break
        return MinimizationResult(
            point=point.detach(),
            value=value.detach(),
            status=tracker.status(),
        )
