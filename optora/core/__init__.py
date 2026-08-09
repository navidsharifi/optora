"""Core ABC contracts shared across optora packages."""

from optora.core.divergence_base import Divergence
from optora.core.dro_base import AmbiguitySet
from optora.core.solver_base import Solver

__all__ = ["AmbiguitySet", "Divergence", "Solver"]
