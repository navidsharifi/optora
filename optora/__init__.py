"""Optora public API.

The main mathematical nouns are re-exported here so that
``from optora import KLAmbiguitySet, MinimaxSolver`` works. Subpackage imports
such as ``from optora.dro import KLAmbiguitySet`` remain equally valid, and the
subpackages stay the home of everything not listed in ``__all__`` (convergence
diagnostics, helper functions). No workflow functions live at the root; run an
algorithm with ``Solver.solve(problem)``.
"""

from optora._version import __version__
from optora.core import (
    AmbiguitySet,
    Divergence,
    DualAmbiguitySet,
    MinimizationProblem,
    MinimizationResult,
    Solver,
)
from optora.divergences import (
    ChiSquareDivergence,
    KLDivergence,
    PhiDivergence,
    SinkhornDivergence,
    TotalVariationDivergence,
)
from optora.dro import (
    ChiSquareAmbiguitySet,
    KLAmbiguitySet,
    MinimaxProblem,
    MinimaxResult,
    MinimaxSolver,
    PhiAmbiguitySet,
    TotalVariationAmbiguitySet,
    WassersteinAmbiguitySet,
)
from optora.solvers import (
    GradientDescent,
    SaddlePointProblem,
    SaddlePointResult,
    SaddlePointSolver,
)

__all__ = [
    "AmbiguitySet",
    "ChiSquareAmbiguitySet",
    "ChiSquareDivergence",
    "Divergence",
    "DualAmbiguitySet",
    "GradientDescent",
    "KLAmbiguitySet",
    "KLDivergence",
    "MinimaxProblem",
    "MinimaxResult",
    "MinimaxSolver",
    "MinimizationProblem",
    "MinimizationResult",
    "PhiAmbiguitySet",
    "PhiDivergence",
    "SaddlePointProblem",
    "SaddlePointResult",
    "SaddlePointSolver",
    "SinkhornDivergence",
    "Solver",
    "TotalVariationAmbiguitySet",
    "TotalVariationDivergence",
    "WassersteinAmbiguitySet",
    "__version__",
]
