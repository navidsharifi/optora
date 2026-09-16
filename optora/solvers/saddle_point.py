"""Primal-dual saddle-point solver for the DRO minimax problem."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from optora.core.convergence import (
    DEFAULT_CHECK_INTERVAL,
    ConvergenceDiagnostics,
    ConvergenceStatus,
    ConvergenceTracker,
    validate_check_interval,
)
from optora.core.solver_base import Solver, require_gradient


@dataclass(frozen=True)
class SaddlePointProblem:
    r"""Minimax problem solved by `SaddlePointSolver`.

    Represents $\min_x \max_y \mathrm{objective}(x, y)$, the shape of the DRO
    minimax problem once a decision variable `x` and an ambiguity set over
    distributions `y` are both in play.

    Attributes:
        objective: Differentiable scalar-valued function
            `objective(primal_point, dual_point)`, minimized over its first
            argument and maximized over its second. It must depend on both
            arguments through autograd; the solver rejects an objective
            whose value is disconnected from either iterate rather than
            treating it as stationary in that variable.
        primal_initial_point: Starting point for the primal (minimizing)
            variable.
        dual_initial_point: Starting point for the dual (maximizing)
            variable.
        dual_projection: Callable applied to the dual iterate after every
            ascent step to keep it inside its feasible set, for example
            projection onto the probability simplex or onto a DRO
            ambiguity set. Defaults to the identity (unconstrained ascent)
            when `None`.
    """

    objective: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    primal_initial_point: torch.Tensor
    dual_initial_point: torch.Tensor
    dual_projection: Callable[[torch.Tensor], torch.Tensor] | None = None


@dataclass(frozen=True)
class SaddlePointResult(ConvergenceDiagnostics):
    """Outcome of a `SaddlePointSolver` solve.

    Attributes:
        primal_point: Final primal (minimizing) iterate.
        dual_point: Final dual (maximizing) iterate.
        value: Objective value at `(primal_point, dual_point)`.
        status: Convergence diagnostics of the solve, exposed on the host
            as `converged` and `num_iterations` by `ConvergenceDiagnostics`
            and read back from the device only when one of those is
            accessed. Convergence is the combined primal/dual gradient norm
            falling below `tol` before `max_iter` iterations are exhausted.
    """

    primal_point: torch.Tensor
    dual_point: torch.Tensor
    value: torch.Tensor
    status: ConvergenceStatus


class SaddlePointSolver(Solver[SaddlePointProblem, SaddlePointResult]):
    r"""Primal-dual gradient ascent-descent for a minimax problem.

    Solves $\min_x \max_y \mathrm{objective}(x, y)$ by alternating, at every
    iteration, a gradient descent step on the primal variable $x$ and a
    gradient ascent step on the dual variable $y$:

    $$
    x \leftarrow x - \mathrm{primal\_step\_size} \cdot \nabla_x \mathrm{objective}(x, y)
    $$

    $$
    y \leftarrow \mathrm{dual\_projection}\big(
        y + \mathrm{dual\_step\_size} \cdot \nabla_y \mathrm{objective}(x, y)
    \big)
    $$

    `dual_projection` keeps `y` feasible after each ascent step. This is
    the generic minimax solve `optora.dro.minimax_solver` uses to train a
    decision variable against a DRO ambiguity set: `x` is the decision
    variable, `y` ranges over distributions inside the ambiguity set, and
    `dual_projection` enforces that constraint (for example simplex
    projection or an `AmbiguitySet`-specific projection).

    The gradient-norm test is evaluated on the iterates' device and read
    back to the host only every `check_interval` iterations; iterations
    taken after convergence are frozen, so the solution does not depend on
    that interval. Keep `check_interval=1` when the objective is expensive,
    for example when each iteration costs an `optora.dro` inner dual solve.
    See `optora.core.convergence.ConvergenceTracker`. The returned
    diagnostics stay on the device as well (see
    `optora.core.convergence.ConvergenceStatus`), so a solve nobody
    inspects never synchronizes for them.

    Attributes:
        primal_step_size: Positive learning rate for the descent step on
            the primal variable.
        dual_step_size: Positive learning rate for the ascent step on the
            dual variable.
        max_iter: Maximum number of ascent-descent iterations.
        tol: Convergence tolerance on the combined primal/dual gradient
            norm.
        check_interval: Number of iterations between host reads of the
            convergence flag.
    """

    def __init__(
        self,
        primal_step_size: float = 1e-2,
        dual_step_size: float = 1e-2,
        max_iter: int = 1000,
        tol: float = 1e-6,
        check_interval: int = DEFAULT_CHECK_INTERVAL,
    ) -> None:
        """Initialize the saddle-point solver.

        Args:
            primal_step_size: Positive learning rate for the descent step
                on the primal variable.
            dual_step_size: Positive learning rate for the ascent step on
                the dual variable.
            max_iter: Maximum number of ascent-descent iterations.
            tol: Convergence tolerance on the combined primal/dual gradient
                norm.
            check_interval: Number of iterations between host reads of the
                convergence flag. Raise it to trade redundant frozen
                iterations for fewer device synchronizations, but only when
                an iteration is cheap relative to a synchronization.

        Raises:
            ValueError: If `primal_step_size`, `dual_step_size`,
                `max_iter`, `tol`, or `check_interval` are not positive.
        """
        if primal_step_size <= 0:
            raise ValueError(
                f"primal_step_size must be positive, got {primal_step_size}."
            )
        if dual_step_size <= 0:
            raise ValueError(f"dual_step_size must be positive, got {dual_step_size}.")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {max_iter}.")
        if tol <= 0:
            raise ValueError(f"tol must be positive, got {tol}.")
        self.primal_step_size = primal_step_size
        self.dual_step_size = dual_step_size
        self.max_iter = max_iter
        self.tol = tol
        self.check_interval = validate_check_interval(check_interval)

    @staticmethod
    def _evaluate(
        objective: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        primal_point: torch.Tensor,
        dual_point: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate the objective and both gradients at differentiable iterates.

        Args:
            objective: Differentiable scalar-valued minimax objective.
            primal_point: Primal iterate, requiring grad.
            dual_point: Dual iterate, requiring grad.

        Returns:
            The objective value and its gradients with respect to the
            primal and the dual iterate.

        Raises:
            ValueError: If `objective` does not depend on both iterates
                through autograd.
        """
        value = objective(primal_point, dual_point)
        raw_primal_grad, raw_dual_grad = torch.autograd.grad(
            value, (primal_point, dual_point), allow_unused=True
        )
        return (
            value,
            require_gradient(raw_primal_grad, "the primal point"),
            require_gradient(raw_dual_grad, "the dual point"),
        )

    def solve(self, problem: SaddlePointProblem) -> SaddlePointResult:
        """Find a saddle point of `problem.objective`.

        Args:
            problem: Objective, initial primal/dual points, and optional
                dual feasibility projection to solve from.

        Returns:
            A `SaddlePointResult` holding the final primal/dual iterates
            and convergence diagnostics.

        Raises:
            ValueError: If `problem.objective` does not depend on both of
                its arguments through autograd.
        """
        dual_projection = problem.dual_projection or (lambda point: point)
        primal_point = (
            problem.primal_initial_point.detach().clone().requires_grad_(True)
        )
        dual_point = (
            dual_projection(problem.dual_initial_point.detach().clone())
            .detach()
            .requires_grad_(True)
        )
        tracker = ConvergenceTracker(self.tol, self.check_interval, primal_point)
        # Each iteration steps with the gradients of the previous evaluation
        # and then evaluates the new iterates, so the last evaluation is
        # always taken at the iterates returned below. Evaluating after the
        # loop instead would need the convergence flag on the host, one
        # synchronization per solve, which nesting multiplies by the outer
        # iteration count.
        value, primal_grad, dual_grad = self._evaluate(
            problem.objective, primal_point, dual_point
        )
        frozen = tracker.start(
            torch.linalg.vector_norm(primal_grad) + torch.linalg.vector_norm(dual_grad)
        )
        for iteration in range(self.max_iter):
            with torch.no_grad():
                primal_point = torch.where(
                    frozen,
                    primal_point,
                    primal_point - self.primal_step_size * primal_grad,
                )
                dual_point = torch.where(
                    frozen,
                    dual_point,
                    dual_projection(dual_point + self.dual_step_size * dual_grad),
                )
            primal_point = primal_point.detach().requires_grad_(True)
            dual_point = dual_point.detach().requires_grad_(True)
            value, primal_grad, dual_grad = self._evaluate(
                problem.objective, primal_point, dual_point
            )
            frozen = tracker.update(
                torch.linalg.vector_norm(primal_grad)
                + torch.linalg.vector_norm(dual_grad)
            )
            if tracker.should_stop(iteration):
                break
        return SaddlePointResult(
            primal_point=primal_point.detach(),
            dual_point=dual_point.detach(),
            value=value.detach(),
            status=tracker.status(),
        )
