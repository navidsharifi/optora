"""Shared contracts for numerical solvers and the problems they solve."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

ProblemT = TypeVar("ProblemT")
ResultT = TypeVar("ResultT")


class Solver(ABC, Generic[ProblemT, ResultT]):
    """Numerical method that transforms an optimization problem into a result.

    Subclasses implement a specific iterative algorithm (for example
    gradient descent on a differentiable objective, or primal-dual
    ascent-descent on a DRO minimax problem). `Solver` is generic in the
    problem and result types so each subclass can pair itself with whatever
    problem description its algorithm needs (an objective and an initial
    point, a primal-dual pair of objectives, and so on) instead of forcing
    every algorithm through one fixed set of arguments.

    Problem and result types are keyed to the *mathematical problem class*,
    not to the algorithm: every solver of unconstrained differentiable
    minimization consumes `MinimizationProblem` and returns
    `MinimizationResult`, so callers such as `optora.dro` ambiguity sets can
    depend on `Solver[MinimizationProblem, MinimizationResult]` and accept
    any solver of that problem class.
    """

    @abstractmethod
    def solve(self, problem: ProblemT) -> ResultT:
        """Run the solver on `problem` and return its outcome.

        Args:
            problem: Description of the optimization problem to solve.

        Returns:
            The result of running this solver on `problem`.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class MinimizationProblem:
    """Unconstrained minimization of a differentiable scalar objective.

    Attributes:
        objective: Differentiable scalar-valued function of a single tensor
            argument. It must depend on that argument through autograd;
            solvers reject an objective whose value is disconnected from
            the point rather than treating it as stationary.
        initial_point: Starting point for the iteration. Passing a previous
            solve's `MinimizationResult.point` here warm-starts the solver
            from that solution instead of from scratch, which is useful for
            the repeated inner-loop solves a DRO ambiguity set or minimax
            solver runs as its outer state changes slightly between calls.
    """

    objective: Callable[[torch.Tensor], torch.Tensor]
    initial_point: torch.Tensor


@dataclass(frozen=True)
class MinimizationResult:
    """Outcome of solving a `MinimizationProblem`.

    Attributes:
        point: Final iterate.
        value: Objective value at `point`.
        converged: Whether the solver's convergence criterion was met before
            its iteration budget was exhausted.
        num_iterations: Number of iterations actually performed.
    """

    point: torch.Tensor
    value: torch.Tensor
    converged: bool
    num_iterations: int


def require_gradient(gradient: torch.Tensor | None, variable: str) -> torch.Tensor:
    """Return a gradient produced with `allow_unused=True`, rejecting `None`.

    `torch.autograd.grad(..., allow_unused=True)` returns `None` when the
    objective's value is disconnected from the differentiated variable.
    Substituting a zero gradient there would make a solver report immediate
    convergence at a point it never optimized, so Optora treats a missing
    gradient as a violated problem contract instead of a stationary point.

    Args:
        gradient: Gradient returned by `torch.autograd.grad` for `variable`.
        variable: Human-readable name of the differentiated variable, used
            in the error message.

    Returns:
        `gradient`, guaranteed not to be `None`.

    Raises:
        ValueError: If `gradient` is `None`.
    """
    if gradient is None:
        raise ValueError(
            f"objective does not depend on {variable} through autograd, so no "
            "gradient is available; pass an objective that is a differentiable "
            "function of it."
        )
    return gradient
