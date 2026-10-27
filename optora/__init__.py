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
    TiltedAmbiguitySet,
)
from optora.divergences import (
    ChiSquareDivergence,
    EntropicTransportDivergence,
    KLDivergence,
    MaxDivergence,
    PhiDivergence,
    SinkhornDivergence,
    TotalVariationDivergence,
)
from optora.dro import (
    ChiSquareAmbiguitySet,
    CVaRAmbiguitySet,
    KLAmbiguitySet,
    MinimaxProblem,
    MinimaxResult,
    MinimaxSolver,
    MultilevelMonteCarloEstimator,
    PhiAmbiguitySet,
    SinkhornAmbiguitySet,
    StochasticPhiDual,
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
    "CVaRAmbiguitySet",
    "ChiSquareAmbiguitySet",
    "ChiSquareDivergence",
    "Divergence",
    "DualAmbiguitySet",
    "EntropicTransportDivergence",
    "GradientDescent",
    "KLAmbiguitySet",
    "KLDivergence",
    "MaxDivergence",
    "MinimaxProblem",
    "MinimaxResult",
    "MinimaxSolver",
    "MinimizationProblem",
    "MinimizationResult",
    "MultilevelMonteCarloEstimator",
    "PhiAmbiguitySet",
    "PhiDivergence",
    "SaddlePointProblem",
    "SaddlePointResult",
    "SaddlePointSolver",
    "SinkhornAmbiguitySet",
    "SinkhornDivergence",
    "Solver",
    "StochasticPhiDual",
    "TiltedAmbiguitySet",
    "TotalVariationAmbiguitySet",
    "TotalVariationDivergence",
    "WassersteinAmbiguitySet",
    "__version__",
]
