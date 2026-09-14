"""Shared contract for ambiguity sets used by DRO formulations."""

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from torch import nn

from optora.core.divergence_base import Divergence
from optora.core.solver_base import (
    MinimizationProblem,
    MinimizationResult,
    Solver,
)


class AmbiguitySet(nn.Module, ABC):
    """Set of distributions within a bounded divergence of a nominal distribution.

    An ambiguity set pairs a `Divergence` with a radius: every distribution
    `q` inside the set satisfies `divergence(q, nominal) <= radius`. Modules
    in `optora.dro` subclass `AmbiguitySet` to implement the inner
    maximization of the DRO minimax problem for a specific divergence
    geometry (for example KL, a general phi-divergence, or Wasserstein).

    Inherits from `torch.nn.Module` (rather than a plain ABC) so `nominal`
    is registered as a buffer and `divergence` as a submodule: a single
    `.to(device)`/`.cuda()` call then moves the nominal distribution and any
    tensor state the divergence holds (for example `SinkhornDivergence`'s
    ground-cost matrix) together, and both surface through `state_dict()`.

    Attributes:
        nominal: Reference distribution the ambiguity set is centered on, a
            nonnegative tensor that sums to one along its last dimension.
        divergence: Divergence used to measure distance from `nominal`.
        radius: Nonnegative scalar bounding the divergence of any
            distribution inside the ambiguity set from `nominal`.
    """

    nominal: torch.Tensor

    def __init__(
        self,
        nominal: torch.Tensor,
        divergence: Divergence,
        radius: float,
    ) -> None:
        """Initialize the ambiguity set.

        Args:
            nominal: Reference distribution the ambiguity set is centered
                on.
            divergence: Divergence used to measure distance from `nominal`.
            radius: Nonnegative scalar bounding the divergence of any
                distribution inside the ambiguity set from `nominal`.

        Raises:
            ValueError: If `radius` is negative.
        """
        super().__init__()
        if radius < 0:
            raise ValueError(f"radius must be nonnegative, got {radius}.")
        self.register_buffer("nominal", nominal)
        self.divergence = divergence
        self.radius = radius

    def contains(self, candidate: torch.Tensor) -> torch.Tensor:
        """Check whether a candidate distribution lies inside the ambiguity set.

        The answer is returned as a boolean tensor on `candidate`'s device
        rather than as a Python `bool`, so membership can be used as a mask
        or composed with further tensor work without forcing a
        device-to-host synchronization. Call `bool(...)` on the result only
        where a host-side branch is genuinely needed.

        Args:
            candidate: Candidate distribution with the same shape as
                `nominal`.

        Returns:
            A boolean tensor that is `True` where the divergence of
            `candidate` from `nominal` does not exceed `radius`.
        """
        divergence: torch.Tensor = self.divergence(candidate, self.nominal)
        return divergence <= self.radius

    @abstractmethod
    def worst_case_expectation(self, loss: torch.Tensor) -> torch.Tensor:
        """Compute the worst-case expected loss over the ambiguity set.

        Args:
            loss: Per-scenario loss values, one entry per element of
                `nominal`'s support.

        Returns:
            A scalar tensor holding the worst-case expected loss attainable
            by any distribution inside the ambiguity set.
        """
        raise NotImplementedError


class DualAmbiguitySet(AmbiguitySet):
    r"""Ambiguity set whose worst-case expectation is a low-dimensional dual solve.

    Every divergence-based ambiguity set that reformulates its inner
    supremum as a convex dual minimization over a handful of dual variables
    (`optora.dro.KLAmbiguitySet` over $\log(\eta)$,
    `optora.dro.PhiAmbiguitySet` over $(\log(\eta), \lambda)$,
    `optora.dro.WassersteinAmbiguitySet` over $\gamma$) runs the same
    machinery around a formulation-specific dual objective, so that
    machinery lives here: hold an injected `dual_solver`, start it from a
    dual point, and re-evaluate the dual objective at the returned optimum
    so the result stays differentiable with respect to `loss` by the
    envelope theorem.

    Repeated calls are warm-started. The dual optimum moves only slightly
    between consecutive `worst_case_expectation` calls on a slowly changing
    loss — exactly what an `optora.dro.MinimaxSolver` outer iteration
    produces — so `_solve_dual` caches the detached optimum of each solve
    and starts the next solve from it instead of from `initial_dual_point`.
    That makes the second and later solves of such a sequence converge in
    far fewer inner iterations at the same optimum, since each dual is
    convex and its solution does not depend on where the iteration started.
    Call `reset_warm_start()` to discard the cache and go back to
    `initial_dual_point`, for example before evaluating an unrelated loss or
    after a diverged solve.

    Attributes:
        nominal: Reference distribution the ambiguity set is centered on.
        divergence: Divergence used to measure distance from `nominal`.
        radius: Nonnegative scalar bounding the divergence of any
            distribution inside the ambiguity set from `nominal`.
        dual_solver: Solver minimizing the dual objective. Required to
            evaluate a positive-radius set.
        initial_dual_point: Dual point the first solve starts from,
            registered as a buffer so it is built once and follows
            `.to(device)` with the rest of the module.
    """

    initial_dual_point: torch.Tensor
    _dual_warm_start: torch.Tensor | None

    def __init__(
        self,
        nominal: torch.Tensor,
        divergence: Divergence,
        radius: float,
        dual_solver: Solver[MinimizationProblem, MinimizationResult] | None,
        initial_dual_point: torch.Tensor,
    ) -> None:
        """Initialize the dual-solved ambiguity set.

        Args:
            nominal: Reference distribution the ambiguity set is centered
                on.
            divergence: Divergence used to measure distance from `nominal`.
            radius: Nonnegative scalar bounding the divergence of any
                distribution inside the ambiguity set from `nominal`.
            dual_solver: Solver minimizing the dual objective. Required to
                evaluate a positive-radius set.
            initial_dual_point: Dual point the first solve starts from.

        Raises:
            ValueError: If `radius` is negative.
        """
        super().__init__(nominal=nominal, divergence=divergence, radius=radius)
        self.dual_solver = dual_solver
        self.register_buffer("initial_dual_point", initial_dual_point)
        self.register_buffer("_dual_warm_start", None, persistent=False)

    def reset_warm_start(self) -> None:
        """Discard the cached dual optimum so the next solve starts cold.

        After this call the next `worst_case_expectation` starts from
        `initial_dual_point` again, as the first one did.
        """
        self._dual_warm_start = None

    def _solve_dual(
        self, dual_objective: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """Minimize a dual objective and return its value at the optimum.

        Args:
            dual_objective: Formulation-specific convex dual objective of
                the dual variables, closing over the `loss` it was built
                for.

        Returns:
            A scalar tensor holding `dual_objective` re-evaluated at the
            dual optimum found by `dual_solver`. The re-evaluation is what
            keeps the result differentiable with respect to `loss` even
            though the dual variables themselves are detached.

        Raises:
            RuntimeError: If `dual_solver` is `None`.
        """
        if self.dual_solver is None:
            raise RuntimeError(
                "dual_solver is required to evaluate a positive-radius "
                f"{type(self).__name__}."
            )
        warm_start = self._dual_warm_start
        problem = MinimizationProblem(
            objective=dual_objective,
            initial_point=(
                self.initial_dual_point if warm_start is None else warm_start
            ),
        )
        result = self.dual_solver.solve(problem)
        self._dual_warm_start = result.point.detach()
        return dual_objective(result.point)
