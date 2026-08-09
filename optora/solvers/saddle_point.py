"""Primal-dual saddle-point solver for the DRO minimax problem."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from optora.core.solver_base import Solver


@dataclass(frozen=True)
class SaddlePointProblem:
    """Minimax problem solved by `SaddlePointSolver`.

    Represents `min_x max_y objective(x, y)`, the shape of the DRO minimax
    problem once a decision variable `x` and an ambiguity set over
    distributions `y` are both in play.

    Attributes:
        objective: Differentiable scalar-valued function
            `objective(primal_point, dual_point)`, minimized over its first
            argument and maximized over its second.
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
class SaddlePointResult:
    """Outcome of a `SaddlePointSolver` solve.

    Attributes:
        primal_point: Final primal (minimizing) iterate.
        dual_point: Final dual (maximizing) iterate.
        value: Objective value at `(primal_point, dual_point)`.
        converged: Whether the combined primal/dual gradient norm fell
            below `tol` before `max_iter` iterations were exhausted.
        num_iterations: Number of ascent-descent iterations actually
            performed.
    """

    primal_point: torch.Tensor
    dual_point: torch.Tensor
    value: torch.Tensor
    converged: bool
    num_iterations: int


class SaddlePointSolver(Solver[SaddlePointProblem, SaddlePointResult]):
    r"""Primal-dual gradient ascent-descent for a minimax problem.

    Solves `min_x max_y objective(x, y)` by alternating, at every
    iteration, a gradient descent step on the primal variable `x` and a
    gradient ascent step on the dual variable `y`:

        x <- x - primal_step_size * grad_x objective(x, y)
        y <- dual_projection(y + dual_step_size * grad_y objective(x, y))

    `dual_projection` keeps `y` feasible after each ascent step. This is
    the generic minimax solve `optora.dro.minimax_solver` uses to train a
    decision variable against a DRO ambiguity set: `x` is the decision
    variable, `y` ranges over distributions inside the ambiguity set, and
    `dual_projection` enforces that constraint (for example simplex
    projection or an `AmbiguitySet`-specific projection).

    Attributes:
        primal_step_size: Positive learning rate for the descent step on
            the primal variable.
        dual_step_size: Positive learning rate for the ascent step on the
            dual variable.
        max_iter: Maximum number of ascent-descent iterations.
        tol: Convergence tolerance on the combined primal/dual gradient
            norm.
    """

    def __init__(
        self,
        primal_step_size: float = 1e-2,
        dual_step_size: float = 1e-2,
        max_iter: int = 1000,
        tol: float = 1e-6,
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

        Raises:
            ValueError: If `primal_step_size`, `dual_step_size`,
                `max_iter`, or `tol` are not positive.
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

    def solve(self, problem: SaddlePointProblem) -> SaddlePointResult:
        """Find a saddle point of `problem.objective`.

        Args:
            problem: Objective, initial primal/dual points, and optional
                dual feasibility projection to solve from.

        Returns:
            A `SaddlePointResult` holding the final primal/dual iterates
            and convergence diagnostics.
        """
        dual_projection = problem.dual_projection or (lambda point: point)
        primal_point = problem.primal_initial_point.detach().clone()
        dual_point = dual_projection(problem.dual_initial_point.detach().clone())
        converged = False
        num_iterations = 0
        for iteration in range(self.max_iter):
            num_iterations = iteration + 1
            primal_point = primal_point.detach().requires_grad_(True)
            dual_point = dual_point.detach().requires_grad_(True)
            value = problem.objective(primal_point, dual_point)
            primal_grad, dual_grad = torch.autograd.grad(
                value, (primal_point, dual_point), allow_unused=True
            )
            if primal_grad is None:
                primal_grad = torch.zeros_like(primal_point)
            if dual_grad is None:
                dual_grad = torch.zeros_like(dual_point)
            grad_norm = torch.linalg.vector_norm(
                primal_grad
            ) + torch.linalg.vector_norm(dual_grad)
            if grad_norm < self.tol:
                converged = True
                break
            with torch.no_grad():
                primal_point = primal_point - self.primal_step_size * primal_grad
                dual_point = dual_projection(
                    dual_point + self.dual_step_size * dual_grad
                )
        with torch.no_grad():
            final_value = problem.objective(primal_point, dual_point)
        return SaddlePointResult(
            primal_point=primal_point.detach(),
            dual_point=dual_point.detach(),
            value=final_value.detach(),
            converged=converged,
            num_iterations=num_iterations,
        )
