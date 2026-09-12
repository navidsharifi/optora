"""Minimax solver wiring a divergence-based ambiguity set and a solver together."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from optora.core.dro_base import AmbiguitySet
from optora.core.solver_base import (
    MinimizationProblem,
    MinimizationResult,
    Solver,
)


@dataclass(frozen=True)
class MinimaxProblem:
    r"""Distributionally robust minimization problem solved by `MinimaxSolver`.

    Represents the DRO minimax problem

    $$
    \min_x \;
    \sup_{q:\, \mathrm{divergence}(q, \mathrm{nominal}) \,\le\, \mathrm{radius}}
        \mathbb{E}_q[\mathrm{loss\_fn}(x)]
    $$

    Every `AmbiguitySet` subclass (`KLAmbiguitySet`, `PhiAmbiguitySet`,
    `ChiSquareAmbiguitySet`, `TotalVariationAmbiguitySet`,
    `WassersteinAmbiguitySet`) already reduces its inner supremum over
    candidate distributions `q` to a tractable convex dual objective or an
    exact closed form via `worst_case_expectation`. `MinimaxProblem`
    therefore only needs to describe the remaining outer minimization over
    the decision variable `x`.

    Attributes:
        ambiguity_set: Divergence-based ambiguity set whose
            `worst_case_expectation` computes the inner supremum over
            candidate distributions.
        loss_fn: Differentiable per-scenario loss as a function of the
            decision variable, returning a tensor shaped like
            `ambiguity_set.nominal` (one entry per support point).
        initial_point: Starting point for the decision variable.
    """

    ambiguity_set: AmbiguitySet
    loss_fn: Callable[[torch.Tensor], torch.Tensor]
    initial_point: torch.Tensor


@dataclass(frozen=True)
class MinimaxResult:
    """Outcome of a `MinimaxSolver` solve.

    Attributes:
        point: Final decision-variable iterate.
        value: Worst-case expected loss at `point`, i.e.
            `ambiguity_set.worst_case_expectation(loss_fn(point))`.
        converged: Whether the outer solver's convergence criterion was met
            before its iteration budget was exhausted.
        num_iterations: Number of outer iterations actually performed.
    """

    point: torch.Tensor
    value: torch.Tensor
    converged: bool
    num_iterations: int


class MinimaxSolver(Solver[MinimaxProblem, MinimaxResult]):
    r"""Solves the DRO minimax problem by delegating to an ambiguity set's dual.

    Wires together a divergence-based `AmbiguitySet` and an (outer) `Solver`
    to solve

    $$
    \min_x \;
    \sup_{q:\, \mathrm{divergence}(q, \mathrm{nominal}) \,\le\, \mathrm{radius}}
        \mathbb{E}_q[\mathrm{loss\_fn}(x)]
    $$

    Because every `AmbiguitySet` subclass already reformulates its inner
    supremum as a tractable convex dual objective (or an exact closed form)
    via `worst_case_expectation(loss)`, the composed function
    `x -> ambiguity_set.worst_case_expectation(loss_fn(x))` is itself a
    plain differentiable scalar objective of `x`, so the remaining work is
    an ordinary minimization over `x`. `worst_case_expectation` stays
    differentiable with respect to `x` by the envelope theorem: each
    ambiguity set that needs an inner dual solve (`KLAmbiguitySet`,
    `PhiAmbiguitySet`, `ChiSquareAmbiguitySet`, `WassersteinAmbiguitySet`)
    solves its own dual variable via a detached inner `dual_solver`, then
    re-evaluates the dual objective at that (detached) optimum with the
    still-attached `loss` tensor; since the dual objective's gradient with
    respect to its own dual variable vanishes at that optimum, the gradient
    of the re-evaluated objective with respect to `x` is exactly the
    gradient one would get by differentiating through the full inner solve,
    without actually needing to do so.

    `MinimaxSolver` therefore reduces to wiring that composed objective into
    `solver`, the outer solver minimizing over `x`. This deliberately does not
    reimplement a
    primal-dual ascent-descent directly over the full candidate
    distribution `q` (as `SaddlePointSolver` does generically): every
    concrete `AmbiguitySet` already reduces that potentially high-
    dimensional inner maximization to a low-dimensional convex dual (or an
    exact closed form), which is more numerically direct than a generic
    ascent-descent over the simplex.

    Attributes:
        solver: Solver minimizing the composed worst-case-expectation
            objective over the decision variable.
    """

    def __init__(
        self,
        solver: Solver[MinimizationProblem, MinimizationResult],
    ) -> None:
        """Initialize the minimax solver.

        Args:
            solver: Solver minimizing the composed worst-case-expectation
                objective over the decision variable.
        """
        self.solver = solver

    def solve(self, problem: MinimaxProblem) -> MinimaxResult:
        """Solve the DRO minimax problem described by `problem`.

        Args:
            problem: Ambiguity set, per-scenario loss function, and initial
                decision-variable point to solve from.

        Returns:
            A `MinimaxResult` holding the robust-optimal decision variable
            and convergence diagnostics from `solver`.
        """

        def objective(point: torch.Tensor) -> torch.Tensor:
            return problem.ambiguity_set.worst_case_expectation(problem.loss_fn(point))

        inner_problem = MinimizationProblem(
            objective=objective, initial_point=problem.initial_point
        )
        result = self.solver.solve(inner_problem)
        return MinimaxResult(
            point=result.point,
            value=result.value,
            converged=result.converged,
            num_iterations=result.num_iterations,
        )
