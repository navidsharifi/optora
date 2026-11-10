"""Minimax solver wiring a divergence-based ambiguity set and a solver together."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from optora.core.convergence import ConvergenceDiagnostics, ConvergenceStatus
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

    Every `AmbiguitySet` subclass in `optora.dro` evaluates its inner
    supremum over candidate distributions `q` in `worst_case_expectation`,
    either in closed form or by a dedicated inner solve. `MinimaxProblem`
    therefore only describes the remaining outer minimization over the
    decision variable `x`.

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
class MinimaxResult(ConvergenceDiagnostics):
    """Outcome of a `MinimaxSolver` solve.

    Attributes:
        point: Final decision-variable iterate.
        value: Worst-case expected loss at `point`, i.e.
            `ambiguity_set.worst_case_expectation(loss_fn(point))`.
        status: Convergence diagnostics of the outer solve, exposed on the
            host as `converged` and `num_iterations` by
            `ConvergenceDiagnostics` and read back from the device only
            when one of those is accessed.
    """

    point: torch.Tensor
    value: torch.Tensor
    status: ConvergenceStatus


class MinimaxSolver(Solver[MinimaxProblem, MinimaxResult]):
    r"""Solves the DRO minimax problem by delegating to an ambiguity set's dual.

    Wires together a divergence-based `AmbiguitySet` and an (outer) `Solver`
    to solve

    $$
    \min_x \;
    \sup_{q:\, \mathrm{divergence}(q, \mathrm{nominal}) \,\le\, \mathrm{radius}}
        \mathbb{E}_q[\mathrm{loss\_fn}(x)]
    $$

    Every `AmbiguitySet` subclass evaluates its inner supremum in
    `worst_case_expectation(loss)`, so the composed function
    `x -> ambiguity_set.worst_case_expectation(loss_fn(x))` is an ordinary
    differentiable scalar objective of `x`, and the remaining work is a
    plain minimization over `x`. The composition stays differentiable in
    `x` without differentiating through the inner solve. Each ambiguity
    set finds its maximizer, or its dual optimum, in a detached solve and
    attaches only the final value to the graph. By Danskin's theorem the
    derivative of the supremum with respect to `loss` is the maximizing
    distribution, so the chain rule through `loss_fn` gives the exact
    gradient in `x`.

    `MinimaxSolver` therefore passes that composed objective to `solver`,
    the outer solver minimizing over `x`. It does not run a primal-dual
    ascent-descent over the full candidate distribution `q`, as the
    generic `SaddlePointSolver` does. Each concrete `AmbiguitySet` already
    replaces that $n$-dimensional inner maximization with a closed form, a
    scalar root search, a low-dimensional convex dual or, for
    `MMDAmbiguitySet`, an interior-point method. None of them iterates
    over the simplex.

    One outer iteration therefore costs an entire inner solve, which is
    orders of magnitude more expensive than the host synchronization of a
    convergence check. Configure `solver` with `check_interval=1`. The
    library-wide default `optora.core.convergence.DEFAULT_CHECK_INTERVAL`
    is tuned for cheap inner loops, and with it the outer loop would keep
    running frozen, and therefore wasted, inner solves after it has
    converged.

    The solve composes with `torch.func` transforms of its caller whenever
    `solver` does, as `optora.solvers.GradientDescent` does. Every
    ambiguity set's `worst_case_expectation` supports them, so
    `torch.func.vmap` over a tensor radius, for example, solves one robust
    problem per radius in one call.

    Attributes:
        solver: Solver minimizing the composed worst-case-expectation
            objective over the decision variable. Should use
            `check_interval=1`.
    """

    def __init__(
        self,
        solver: Solver[MinimizationProblem, MinimizationResult],
    ) -> None:
        """Initialize the minimax solver.

        Args:
            solver: Solver minimizing the composed worst-case-expectation
                objective over the decision variable. Should use
                `check_interval=1`, since one of its iterations costs a full
                inner dual solve.
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
            status=result.status,
        )
