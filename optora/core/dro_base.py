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

_MASS_TOLERANCE_FLOOR = 1e-6


def _mass_tolerance(nominal: torch.Tensor) -> float:
    """Return the tolerance allowed on a nominal distribution's total mass.

    Summing `n` entries accumulates on the order of `n` roundings of
    relative size `eps`, so the tolerance scales with the support size and
    the dtype instead of being one constant that is slack in float64 and
    unreachable in float32 on a large support.

    Args:
        nominal: Reference distribution whose mass is being checked.

    Returns:
        The largest absolute deviation from unit mass that still counts as a
        normalized distribution.
    """
    support_size = nominal.shape[-1]
    return max(_MASS_TOLERANCE_FLOOR, support_size * torch.finfo(nominal.dtype).eps)


def _validate_nominal(nominal: torch.Tensor) -> None:
    """Check that `nominal` is a valid probability distribution.

    Both value checks are plain reductions read back as a single boolean,
    rather than elementwise predicates such as
    `torch.all(torch.isfinite(nominal) & (nominal >= 0))`, so validation
    costs two host synchronizations and no tensor temporaries the size of
    `nominal`. Each test is written as `not (reduction <= bound)` so that a
    `nan` reduction, which compares false against everything, fails the
    check instead of passing it.

    Args:
        nominal: Candidate reference distribution.

    Raises:
        ValueError: If `nominal` is not a floating-point tensor, holds a
            negative or non-finite entry, or does not sum to one along its
            last dimension within `_mass_tolerance(nominal)`.
    """
    if not nominal.is_floating_point():
        raise ValueError(
            f"nominal must be a floating-point tensor, got dtype {nominal.dtype}."
        )
    if not bool(torch.amin(nominal) >= 0.0):
        raise ValueError("nominal must be nonnegative and finite.")
    tolerance = _mass_tolerance(nominal)
    mass_error = torch.amax(torch.abs(torch.sum(nominal, dim=-1) - 1.0))
    if not bool(mass_error <= tolerance):
        raise ValueError(
            f"nominal must sum to one along its last dimension within {tolerance}."
        )


def _broadcast_batch_shapes(**shapes: torch.Size) -> torch.Size:
    """Broadcast named batch shapes together under NumPy broadcasting rules.

    Args:
        **shapes: Batch shapes to broadcast, keyed by the name of the
            argument each came from so a failure can point at it.

    Returns:
        The common shape every input broadcasts to.

    Raises:
        ValueError: If the shapes are not mutually broadcastable.
    """
    rank = max((len(shape) for shape in shapes.values()), default=0)
    aligned = [(1,) * (rank - len(shape)) + tuple(shape) for shape in shapes.values()]
    broadcast: list[int] = []
    for dimensions in zip(*aligned, strict=True):
        size = max(dimensions)
        if any(dimension not in (1, size) for dimension in dimensions):
            named = ", ".join(
                f"{name}={tuple(shape)}" for name, shape in shapes.items()
            )
            raise ValueError(f"batch shapes do not broadcast: {named}.")
        broadcast.append(size)
    return torch.Size(broadcast)


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
        radius: Nonnegative bound on the divergence of any distribution
            inside the ambiguity set from `nominal`, either a Python float
            or a tensor of radii broadcastable against the batch shape of
            `worst_case_expectation`'s `loss`.
    """

    nominal: torch.Tensor
    radius: float | torch.Tensor

    def __init__(
        self,
        nominal: torch.Tensor,
        divergence: Divergence,
        radius: float | torch.Tensor,
        validate: bool = False,
    ) -> None:
        """Initialize the ambiguity set.

        Args:
            nominal: Reference distribution the ambiguity set is centered
                on.
            divergence: Divergence used to measure distance from `nominal`.
            radius: Nonnegative bound on the divergence of any distribution
                inside the ambiguity set from `nominal`. A tensor radius is
                registered as a buffer and broadcast against the batch
                shape of `worst_case_expectation`'s `loss`, which evaluates
                a whole sweep of radii in one call; its entries are not
                checked for nonnegativity, since reading them on the host
                would block on the device (the same opt-in policy the
                divergences apply to their tensor arguments), and a tensor
                radius never takes the zero-radius shortcut.
            validate: Whether to check that `nominal` is nonnegative and
                sums to one along its last dimension. The check reads two
                reductions over `nominal` on the host, which blocks until
                the device has produced them, so it is opt-in and off by
                default to keep construction asynchronous. The shape and
                hyperparameter checks are metadata-only and always run.

        Raises:
            ValueError: If `nominal` is a scalar tensor, if `radius` is a
                negative float, or if `validate` is set and `nominal` is not
                a valid probability distribution.
        """
        super().__init__()
        if nominal.ndim == 0:
            raise ValueError(
                "nominal must have shape (..., n) with a trailing support "
                "dimension, got a scalar tensor."
            )
        if validate:
            _validate_nominal(nominal)
        self.register_buffer("nominal", nominal)
        self.divergence = divergence
        if isinstance(radius, torch.Tensor):
            self.register_buffer("radius", radius)
            self._radius_is_zero = False
        else:
            if radius < 0:
                raise ValueError(f"radius must be nonnegative, got {radius}.")
            self.radius = radius
            self._radius_is_zero = radius == 0.0

    def _batch_shape(self, loss: torch.Tensor) -> torch.Size:
        """Validate a loss tensor and return the batch shape it induces.

        Args:
            loss: Per-scenario loss values of shape `(..., n)`, where `n` is
                `nominal`'s support size.

        Returns:
            The broadcast of `loss`'s leading dimensions against those of
            `nominal` and against `radius`'s shape: the shape every
            `worst_case_expectation` returns, and the shape of the per-batch
            dual variables solved for along the way.

        Raises:
            ValueError: If `loss` is zero-dimensional, if its trailing
                dimension does not match `nominal`'s support size, or if its
                leading dimensions do not broadcast against `nominal` and
                `radius`.
        """
        nominal = self.nominal
        if loss.ndim == 0 or loss.shape[-1] != nominal.shape[-1]:
            raise ValueError(
                "loss must have shape (..., n) matching nominal's support size "
                f"{nominal.shape[-1]}, got {tuple(loss.shape)}."
            )
        return _broadcast_batch_shapes(
            loss=loss.shape[:-1],
            nominal=nominal.shape[:-1],
            radius=self._radius_shape,
        )

    @property
    def _radius_shape(self) -> torch.Size:
        """Shape a tensor radius contributes to the batch shape, empty for a float."""
        radius = self.radius
        return radius.shape if isinstance(radius, torch.Tensor) else torch.Size()

    def _nominal_expectation(
        self, loss: torch.Tensor, batch_shape: torch.Size
    ) -> torch.Tensor:
        """Return the nominal expectation of `loss`, broadcast to `batch_shape`.

        Args:
            loss: Per-scenario loss values of shape `(..., n)`.
            batch_shape: Shape returned by `_batch_shape` for this `loss`.

        Returns:
            `sum(nominal * loss)` over the support, expanded to
            `batch_shape`. This is the exact worst-case expectation of a
            zero-radius ambiguity set, which contains only `nominal`.
        """
        return torch.sum(self.nominal * loss, dim=-1).expand(batch_shape)

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
            loss: Per-scenario loss values of shape `(..., n)`, one trailing
                entry per element of `nominal`'s support. Leading dimensions
                are a batch of independent loss vectors, solved in one call.

        Returns:
            A tensor of shape `(...)` holding, for each batch element, the
            worst-case expected loss attainable by any distribution inside
            the ambiguity set. An unbatched `(n,)` loss gives a scalar.
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

    A batched `loss` of shape `(..., n)` is solved with one set of dual
    variables per batch element, since the duals decouple across batch
    elements. `_solve_dual` therefore hands `dual_solver` the *sum* of the
    per-element dual objectives: the sum's gradient with respect to any one
    element's dual variables is that element's own gradient, so a single
    joint solve reproduces independent per-element solves exactly.
    Convergence is judged on the joint gradient norm over the whole batch,
    so every element keeps iterating until the batch as a whole is
    stationary (see `progress/decisions.md`).

    Attributes:
        nominal: Reference distribution the ambiguity set is centered on.
        divergence: Divergence used to measure distance from `nominal`.
        radius: Nonnegative bound on the divergence of any distribution
            inside the ambiguity set from `nominal`.
        dual_solver: Solver minimizing the dual objective. Required to
            evaluate a positive-radius set.
        initial_dual_point: Dual point of a *single* batch element that the
            first solve starts from, registered as a buffer so it is built
            once and follows `.to(device)` with the rest of the module. It
            is expanded over the batch shape of the loss being evaluated.
    """

    initial_dual_point: torch.Tensor
    _dual_warm_start: torch.Tensor | None

    def __init__(
        self,
        nominal: torch.Tensor,
        divergence: Divergence,
        radius: float | torch.Tensor,
        dual_solver: Solver[MinimizationProblem, MinimizationResult] | None,
        initial_dual_point: torch.Tensor,
        validate: bool = False,
    ) -> None:
        """Initialize the dual-solved ambiguity set.

        Args:
            nominal: Reference distribution the ambiguity set is centered
                on.
            divergence: Divergence used to measure distance from `nominal`.
            radius: Nonnegative bound on the divergence of any distribution
                inside the ambiguity set from `nominal`.
            dual_solver: Solver minimizing the dual objective. Required to
                evaluate a positive-radius set.
            initial_dual_point: Dual point of a single batch element that
                the first solve starts from.
            validate: Whether to check that `nominal` is a valid
                probability distribution, off by default because the check
                synchronizes with the device.

        Raises:
            ValueError: If `nominal` is a scalar tensor, if `radius` is a
                negative float, or if `validate` is set and `nominal` is not
                a valid probability distribution.
        """
        super().__init__(
            nominal=nominal,
            divergence=divergence,
            radius=radius,
            validate=validate,
        )
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
        self,
        dual_objective: Callable[[torch.Tensor], torch.Tensor],
        batch_shape: torch.Size,
    ) -> torch.Tensor:
        """Minimize a dual objective and return its value at the optimum.

        Args:
            dual_objective: Formulation-specific convex dual objective,
                closing over the `loss` it was built for. It maps dual
                variables of shape `batch_shape + initial_dual_point.shape`
                to per-batch-element dual values of shape `batch_shape`.
            batch_shape: Batch shape of the loss being evaluated, as
                returned by `AmbiguitySet._batch_shape`.

        Returns:
            A tensor of shape `batch_shape` holding `dual_objective`
            re-evaluated at the dual optimum found by `dual_solver`. The
            re-evaluation is what keeps the result differentiable with
            respect to `loss` even though the dual variables themselves are
            detached. The cached warm start is reused only when its shape
            still matches the batch being solved.

        Raises:
            RuntimeError: If `dual_solver` is `None`.
        """
        if self.dual_solver is None:
            raise RuntimeError(
                "dual_solver is required to evaluate a positive-radius "
                f"{type(self).__name__}."
            )
        initial = self.initial_dual_point
        start_shape = batch_shape + initial.shape
        warm_start = self._dual_warm_start
        problem = MinimizationProblem(
            objective=lambda point: torch.sum(dual_objective(point)),
            initial_point=(
                warm_start
                if warm_start is not None and warm_start.shape == start_shape
                else initial.expand(start_shape)
            ),
        )
        result = self.dual_solver.solve(problem)
        self._dual_warm_start = result.point.detach()
        return dual_objective(result.point)
