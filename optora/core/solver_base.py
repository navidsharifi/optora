"""Shared contract for numerical solvers used across optora."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

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
