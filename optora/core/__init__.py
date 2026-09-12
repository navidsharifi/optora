"""Core ABC contracts shared across optora packages."""

from optora.core.convergence import ConvergenceTracker, validate_check_interval
from optora.core.divergence_base import Divergence
from optora.core.dro_base import AmbiguitySet
from optora.core.solver_base import (
    MinimizationProblem,
    MinimizationResult,
    Solver,
    require_gradient,
)

__all__ = [
    "AmbiguitySet",
    "ConvergenceTracker",
    "Divergence",
    "MinimizationProblem",
    "MinimizationResult",
    "Solver",
    "require_gradient",
    "validate_check_interval",
]
