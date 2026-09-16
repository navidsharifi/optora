"""Core ABC contracts shared across optora packages."""

from optora.core.convergence import (
    ConvergenceDiagnostics,
    ConvergenceStatus,
    ConvergenceTracker,
    validate_check_interval,
)
from optora.core.divergence_base import Divergence
from optora.core.dro_base import AmbiguitySet, DualAmbiguitySet
from optora.core.solver_base import (
    MinimizationProblem,
    MinimizationResult,
    Solver,
    require_gradient,
)

__all__ = [
    "AmbiguitySet",
    "ConvergenceDiagnostics",
    "ConvergenceStatus",
    "ConvergenceTracker",
    "Divergence",
    "DualAmbiguitySet",
    "MinimizationProblem",
    "MinimizationResult",
    "Solver",
    "require_gradient",
    "validate_check_interval",
]
