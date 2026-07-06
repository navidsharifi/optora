"""Base solver interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from optora.core.problem import UnconstrainedOptimizationProblem
from optora.core.result import OptimizationResult


@dataclass(slots=True)
class Solver(ABC):
    """Base interface for explicit algorithm components.

    Solvers are method implementations, not top-level workflows. A researcher
    constructs a problem object, chooses a solver class that corresponds to the
    method under study, and calls ``solve`` explicitly.

    Args:
        max_iter: Maximum number of iterations.
        tol: Stationarity tolerance used by the method.
        store_history: Whether to retain per-iteration diagnostics.
    """

    max_iter: int = 1000
    tol: float = 1e-6
    store_history: bool = False

    def __post_init__(self) -> None:
        """Validate common solver parameters."""
        if self.max_iter < 0:
            msg = "max_iter must be nonnegative"
            raise ValueError(msg)
        if self.tol <= 0.0:
            msg = "tol must be positive"
            raise ValueError(msg)

    @abstractmethod
    def solve(
        self,
        problem: UnconstrainedOptimizationProblem,
    ) -> OptimizationResult:
        """Solve an explicit mathematical problem."""
