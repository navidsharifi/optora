"""Shared contract for ambiguity sets used by DRO formulations."""

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.func import jvp

from optora.core.divergence_base import Divergence
from optora.core.precision import working_dtype
from optora.core.solver_base import (
    MinimizationProblem,
    MinimizationResult,
    Solver,
)

_MASS_TOLERANCE_FLOOR = 1e-6


def _bisection_steps(dtype: torch.dtype) -> int:
    """Return the number of bisection steps that exhaust `dtype`'s precision.

    Bisecting the unit interval `k` times narrows the bracket to `2 ** -k`,
    so there is nothing left to gain once that width drops below the
    spacing of the floating-point numbers near one. The count is therefore
    read off the dtype rather than exposed as a tolerance: 52 steps in
    float64 and 23 in float32.

    Args:
        dtype: Floating-point dtype the bisection iterates in.

    Returns:
        The number of halvings needed to resolve a point of the unit
        interval to `dtype`'s precision.
    """
    return int(-math.log2(torch.finfo(dtype).eps))


# Entries of one batched tilt evaluation below which a solve is bound by
# per-kernel launch overhead rather than by arithmetic: up to this size,
# evaluating several tilts per step in one batched call costs about as much
# as evaluating one, and cuts the number of steps.
_LAUNCH_BOUND_ENTRIES = 2**14

# Largest number of bits a search step resolves, 15 tilts per step.
_MAX_SEARCH_BITS = 4


def _search_bits(entries: int) -> int:
    """Return the bits of the tilt each search step resolves.

    A step that evaluates $2^b - 1$ evenly spaced tilts narrows the bracket
    by $2^b$, so it replaces $b$ bisection steps at $2^b - 1$ times the
    arithmetic per step. The largest $b$ whose batched evaluation stays
    within `_LAUNCH_BOUND_ENTRIES` is chosen: small problems, whose steps
    are dominated by kernel launches, take few wide steps, and large ones
    bisect. The choice reads only shapes, so it never synchronizes.

    Args:
        entries: Number of entries of one tilt evaluation, the batch size
            times the support size.

    Returns:
        The number of bits $b$, from one (bisection) to `_MAX_SEARCH_BITS`.
    """
    for bits in range(_MAX_SEARCH_BITS, 1, -1):
        if (2**bits - 1) * entries <= _LAUNCH_BOUND_ENTRIES:
            return bits
    return 1


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


def _validate_radius(radius: torch.Tensor) -> None:
    """Check that every entry of a tensor radius is nonnegative.

    Written as `not (minimum >= 0)` so that a `nan` entry, which `torch.amin`
    propagates and which compares false against everything, fails the check.

    Args:
        radius: Candidate tensor of radii.

    Raises:
        ValueError: If `radius` holds a negative or `nan` entry.
    """
    if radius.numel() > 0 and not bool(torch.amin(radius) >= 0.0):
        raise ValueError("radius must be nonnegative and not nan.")


def _validate_loss(loss: torch.Tensor) -> None:
    """Check that every entry of a loss tensor is finite.

    Args:
        loss: Per-scenario loss values.

    Raises:
        ValueError: If `loss` holds a `nan` or infinite entry.
    """
    if not bool(torch.all(torch.isfinite(loss))):
        raise ValueError("loss must be finite.")


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

    Numerical policy, shared by every subclass through `_prepare_loss` and
    `_as_result`:

    - **Precision.** The worst case is computed in
      `optora.core.working_dtype(loss, nominal)`, so half-precision losses
      or a half-precision set are solved in float32, and the result is
      returned in `torch.promote_types(loss.dtype, nominal.dtype)`.
    - **Non-finite losses.** A `nan` or infinite loss entry makes the
      corresponding batch element of the result non-finite (`nan`, or the
      infinite value itself), never a finite number; other batch elements
      are unaffected.
    - **Validation.** Metadata checks (shapes, a negative or `nan` float
      radius) always run. Checks that read tensor values back to the host
      run only with `validate=True`, in the spirit of
      `torch.distributions`' `validate_args`.
    - **Gradients.** The worst case is differentiable in `loss` by
      Danskin's theorem, and to first order in `nominal` and a tensor
      `radius` by the envelope theorem (`_attach_parameter_gradients`);
      only the derivative along directions that keep `nominal` summing to
      one is defined.

    Attributes:
        nominal: Reference distribution the ambiguity set is centered on, a
            nonnegative tensor that sums to one along its last dimension.
        divergence: Divergence used to measure distance from `nominal`.
        radius: Nonnegative bound on the divergence of any distribution
            inside the ambiguity set from `nominal`, either a Python float
            or a tensor of radii broadcastable against the batch shape of
            `worst_case_expectation`'s `loss`.
        validate: Whether value checks that synchronize with the device run:
            on `nominal` and a tensor `radius` at construction, and on
            `loss` at every `worst_case_expectation` call.
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
                a whole sweep of radii in one call; its entries are checked
                for nonnegativity only when `validate` is set, since reading
                them on the host would block on the device, and a tensor
                radius never takes the zero-radius shortcut. An infinite
                radius is allowed and admits every distribution the
                divergence can reach.
            validate: Whether to run the value checks that synchronize with
                the device: that `nominal` is nonnegative and sums to one
                along its last dimension, that a tensor `radius` is
                nonnegative and not `nan`, and, on every
                `worst_case_expectation` call, that `loss` is finite. They
                are opt-in and off by default to keep construction and
                evaluation asynchronous. The shape and hyperparameter
                checks are metadata-only and always run.

        Raises:
            ValueError: If `nominal` is a scalar tensor, if `radius` is a
                negative or `nan` float, or if `validate` is set and
                `nominal` is not a valid probability distribution or a
                tensor `radius` holds a negative or `nan` entry.
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
        self.validate = validate
        if isinstance(radius, torch.Tensor):
            if validate:
                _validate_radius(radius)
            self.register_buffer("radius", radius)
            self._radius_is_zero = False
        else:
            if not radius >= 0:
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

    def _prepare_loss(self, loss: torch.Tensor) -> tuple[torch.Size, torch.Tensor]:
        """Check a loss tensor and cast it to the precision the solve runs in.

        Every `worst_case_expectation` starts here and returns through
        `_as_result`. The cast is differentiable and a no-op for a float32
        or float64 loss on a set of the same dtype.

        Args:
            loss: Per-scenario loss values of shape `(..., n)`.

        Returns:
            The batch shape from `_batch_shape`, and `loss` cast to
            `optora.core.working_dtype(loss, nominal)`. Buffers of a
            half-precision set must be cast to that dtype wherever they
            enter a computation that `loss` does not already promote.

        Raises:
            ValueError: If `loss`'s shape does not match `nominal` and
                `radius`, or if `validate` is set and `loss` is not finite.
        """
        batch_shape = self._batch_shape(loss)
        if self.validate:
            _validate_loss(loss)
        return batch_shape, loss.to(working_dtype(loss, self.nominal))

    def _as_result(self, value: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        """Cast a working-precision worst case back to the caller's dtype.

        Args:
            value: Worst-case expectation computed in working precision.
            loss: The loss as the caller passed it, before `_prepare_loss`.

        Returns:
            `value` in `torch.promote_types(loss.dtype, nominal.dtype)`, the
            dtype a plain PyTorch operation on the two would return.
        """
        return value.to(torch.promote_types(loss.dtype, self.nominal.dtype))

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

    @property
    def _constant_nominal(self) -> torch.Tensor:
        """`nominal` cut from the autograd graph, for inner solves.

        A subclass whose worst case is not an explicit function of `nominal`
        reads it through this property while solving, so that its only
        gradient path to `nominal` is the one `_attach_parameter_gradients`
        installs.
        """
        return self.nominal.detach()

    @property
    def _radius_parameter(self) -> torch.Tensor | None:
        """The tensor the radius derivative is taken with respect to, if any.

        A tensor `radius` by default, `None` for a float. A subclass whose
        radius is derived from another parameter overrides this.
        """
        return self.radius if isinstance(self.radius, torch.Tensor) else None

    def _attach_parameter_gradients(
        self,
        value: torch.Tensor,
        sensitivities: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        r"""Make a worst-case value differentiable in `nominal` and `radius`.

        By the envelope theorem, the derivative of
        $V(\ell, p, \rho) = \sup \{\langle q, \ell \rangle : D(q \,\|\, p)
        \le \rho\}$ with respect to the radius is the optimal Lagrange
        multiplier $\lambda^\star$ of the divergence constraint, and with
        respect to the nominal it is the partial derivative of the
        Lagrangian at the optimum. Subclasses that compute these in closed
        form from their own solve pass them in through `sensitivities`,
        which is only called when `nominal` or `_radius_parameter` requires
        grad, so an ordinary training step pays nothing.

        Args:
            value: Worst-case expectation of shape `batch_shape`, in working
                precision.
            sensitivities: Callable returning the derivative with respect
                to `nominal`, of shape `batch_shape + (n,)`, and with
                respect to `_radius_parameter`, of shape `batch_shape`.

        Returns:
            `value`, unchanged, whose backward pass additionally routes the
            cotangent to `nominal` and to `_radius_parameter`.
        """
        radius = self._radius_parameter
        needs_gradient = self.nominal.requires_grad or (
            radius is not None and radius.requires_grad
        )
        if not (needs_gradient and torch.is_grad_enabled()):
            return value
        nominal_gradient, radius_gradient = sensitivities()
        result: torch.Tensor = _ParameterSensitivity.apply(
            value, self.nominal, radius, nominal_gradient, radius_gradient
        )
        return result

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


class _ParameterSensitivity(torch.autograd.Function):
    """Identity on a worst-case value that carries its parameter derivatives.

    The forward pass returns the value unchanged, so attaching the
    derivatives never perturbs it. The backward pass passes the cotangent
    through to the value and routes it, scaled by the precomputed
    sensitivities, to `nominal` and to the tensor the radius is set by. The
    sensitivities are detached constants, so second derivatives with
    respect to either are not available.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(
        value: torch.Tensor,
        nominal: torch.Tensor,
        radius: torch.Tensor | None,
        nominal_gradient: torch.Tensor,
        radius_gradient: torch.Tensor,
    ) -> torch.Tensor:
        """Return a copy of `value`.

        Args:
            value: Worst-case expectation of shape `batch_shape`.
            nominal: The ambiguity set's nominal distribution.
            radius: The ambiguity set's `_radius_parameter`, or `None`.
            nominal_gradient: Derivative of `value` with respect to
                `nominal`, of shape `batch_shape + (n,)`.
            radius_gradient: Derivative of `value` with respect to
                `radius`, of shape `batch_shape`.

        Returns:
            `value`.
        """
        return value.clone()

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        """Save the sensitivities and the parameters' shapes and dtypes."""
        _, nominal, radius, nominal_gradient, radius_gradient = inputs
        ctx.save_for_backward(nominal_gradient, radius_gradient)
        ctx.nominal_meta = (nominal.shape, nominal.dtype)
        ctx.radius_meta = None if radius is None else (radius.shape, radius.dtype)

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None, None]:
        """Scale the cotangent by each sensitivity and reduce to each shape."""
        nominal_gradient, radius_gradient = ctx.saved_tensors
        shape, dtype = ctx.nominal_meta
        grad_nominal = (grad_output.unsqueeze(-1) * nominal_gradient).sum_to_size(shape)
        grad_radius = None
        if ctx.radius_meta is not None:
            radius_shape, radius_dtype = ctx.radius_meta
            grad_radius = (
                (grad_output * radius_gradient)
                .sum_to_size(radius_shape)
                .to(radius_dtype)
            )
        return grad_output, grad_nominal.to(dtype), grad_radius, None, None


class _MaximizerExpectation(torch.autograd.Function):
    r"""Expectation of `loss` under a detached maximizer, twice differentiable.

    The forward pass returns $\max \ell + \sum_i q^\star_i (\ell_i - \max
    \ell)$ for the detached maximizer $q^\star$, and the first-order
    reverse-mode derivative is $q^\star$ itself, the Danskin gradient, at no
    cost beyond the forward pass. Only when the backward pass is itself
    recorded (`create_graph=True`, as in a Hessian-vector product), and in
    forward mode, is $q^\star$ rebuilt as a differentiable function of
    `loss` by `maximizer`, so that differentiating the gradient yields the
    Jacobian $\partial q^\star / \partial \ell$, the Hessian of the
    worst-case expectation, instead of zero.

    Every tensor `maximizer` reads is passed in as an input rather than
    captured, which is what lets `torch.func` generate a `vmap` rule for
    this function.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(
        loss: torch.Tensor,
        distribution: torch.Tensor,
        max_loss: torch.Tensor,
        scale: torch.Tensor,
        tilt: torch.Tensor,
        maximizer: Callable[..., torch.Tensor],
    ) -> torch.Tensor:
        """Evaluate the expectation as a correction to the maximum loss.

        Args:
            loss: Per-scenario loss values of shape `(..., n)`.
            distribution: Detached maximizer of shape `batch_shape + (n,)`.
            max_loss: Detached maximum of `loss` over the nominal's support,
                with a trailing singleton dimension.
            scale: Detached positive loss spread, shaped like `max_loss`.
            tilt: Solved tilt of shape `batch_shape`.
            maximizer: Map from `(loss, max_loss, scale, tilt)` to the
                maximizer, differentiable with respect to `loss`, used only
                by a recorded backward.

        Returns:
            The expectation of `loss` under `distribution`, of shape
            `batch_shape`.
        """
        shortfall = torch.sum(distribution * (loss - max_loss), dim=-1)
        return max_loss.squeeze(-1) + shortfall

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        """Save the tensors the backward and forward-mode rules read."""
        loss, distribution, max_loss, scale, tilt, maximizer = inputs
        ctx.save_for_backward(loss, distribution, max_loss, scale, tilt)
        ctx.save_for_forward(loss, max_loss, scale, tilt)
        ctx.maximizer = maximizer

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None, None]:
        """Return the maximizer, rebuilt differentiably if the graph is recorded."""
        loss, distribution, max_loss, scale, tilt = ctx.saved_tensors
        if torch.is_grad_enabled():
            distribution = ctx.maximizer(loss, max_loss, scale, tilt)
        grad_loss = (grad_output.unsqueeze(-1) * distribution).sum_to_size(loss.shape)
        return grad_loss, None, None, None, None, None

    @staticmethod
    def jvp(ctx: Any, loss_tangent: torch.Tensor, *_: Any) -> torch.Tensor:
        """Return the directional derivative along `loss_tangent`.

        Forward mode is opt-in, so the maximizer is always rebuilt
        differentiably here: reverse mode over forward mode, as in
        `torch.func.jacrev(torch.func.jacfwd(...))`, then sees its Jacobian
        rather than a constant. Forward mode over forward mode does not
        differentiate a custom `jvp` rule in PyTorch, so it is unsupported.
        """
        distribution = ctx.maximizer(*ctx.saved_tensors)
        return torch.sum(distribution * loss_tangent, dim=-1)


class TiltedAmbiguitySet(AmbiguitySet):
    r"""Ambiguity set whose worst case is a tilt of the nominal, located by bisection.

    For the divergence balls whose inner supremum has a known maximizer
    shape, the worst-case distribution is a one-parameter *tilt* of the
    nominal: a family $q_t$ with $q_0 = \mathrm{nominal}$ along which the
    divergence $D(q_t \,\|\, \mathrm{nominal})$ increases monotonically
    from $0$ to the divergence of the distribution concentrated on the
    highest-loss scenarios. KL-DRO tilts exponentially,
    $q \propto \mathrm{nominal} \cdot e^{\beta\,\mathrm{loss}}$, and
    chi-square-DRO tilts linearly,
    $q \propto \mathrm{nominal} \cdot (\mathrm{loss} - c)_+$; both read
    directly off the primal KKT conditions
    (Hu and Hong 2013; Ben-Tal et al. 2013; Duchi and Namkoong 2021).

    Because the divergence along the path is monotone, the constraint
    $D(q_t \,\|\, \mathrm{nominal}) \le \mathrm{radius}$ pins the tilt by a
    *bracketed* scalar root solve rather than by minimizing a convex dual.
    That is what this class implements, once, for every such formulation:

    - **A guaranteed bracket.** The tilt is parameterized by
      $t \in [0, 1)$, a bounded reparameterization of the unbounded
      natural parameter (the inverse multiplier $1/\eta$). The bracket
      $[0, 1]$ is valid for every loss and every radius by construction, so
      bisection cannot fail to make progress the way an unconstrained
      first-order dual solve can when the minimizer runs off to infinity.
    - **A fixed, synchronization-free trip count.** The resolution is read
      off the dtype by `_bisection_steps`, so the solve is a fixed number
      of device-side kernels with no host read of a convergence flag. A
      small problem, whose cost is kernel launches rather than arithmetic,
      evaluates up to 15 tilts per step in one batched call and so needs a
      quarter of the steps (`_search_bits`); a large one bisects.
    - **Feasibility by construction.** Each step keeps the endpoint that
      tested feasible, so the returned tilt always satisfies the radius
      constraint. The reported value is therefore a primal objective value
      at a feasible distribution: it can never exceed $\max \mathrm{loss}$,
      and it is `nan`-free whatever the support size or loss scale.
    - **Saturation without a special case.** Once `radius` reaches the
      divergence of the distribution concentrated on the highest-loss
      scenarios, every tilt is feasible, bisection drives $t$ to the top of
      the bracket, and $q_t$ *is* that distribution, so the value is
      exactly $\max \mathrm{loss}$.

    The tilt is computed on the loss standardized to unit spread,
    $(\max \ell - \ell) / (\max \ell - \min \ell)$. An ambiguity set does
    not depend on `loss`, so the maximizer of an affinely rescaled loss is
    the same distribution; standardizing only keeps the tilt parameter and
    the divergence well conditioned on a loss of any scale or offset.

    The maximum and minimum are taken over the reachable scenarios only
    (`_reachable`): the support of the nominal by default, or of whatever
    base distribution a subclass tilts, such as the reference measure of
    `optora.dro.SinkhornAmbiguitySet`, which also moves the path's start
    off the nominal (`_center_expectation`). With the default, every tilt
    is a reweighting $q \propto \mathrm{nominal} \cdot w$, so a
    scenario with zero nominal mass is unreachable whatever its loss, and
    letting its loss set the scale would both misreport the saturated value
    as that unreachable loss and leave no zero gap on the support for the
    tilt to concentrate on. Such scenarios are given the largest gap, `1`,
    which keeps every intermediate of the tilt finite while their weight
    stays exactly zero.

    The worst-case expectation is reported as
    $\sum_i q^\star_i \,\mathrm{loss}_i$ with $q^\star$ detached. By
    Danskin's theorem the gradient of $\sup_{q \in \mathcal{U}}
    \mathbb{E}_q[\mathrm{loss}]$ with respect to `loss` is exactly the
    maximizer $q^\star$, so detaching the tilt costs no accuracy in the
    backward pass while keeping the bisection itself out of the autograd
    graph.

    Second derivatives are exact as well. A recorded backward pass
    (`create_graph=True`, `torch.func.hessian`) rebuilds $q^\star$ from the
    solved tilt by one implicit-function step (`_implicit_maximizer`), so
    differentiating the gradient yields $\partial q^\star / \partial
    \mathrm{loss}$, the Hessian of the worst case, rather than the zero a
    detached maximizer would give. A plain `.backward()` never pays for
    that step. Where the worst case is not differentiable (tied highest
    losses once the radius saturates the set) the gradient is the
    maximizer that splits the tied mass in proportion to `nominal`, a
    subgradient of the convex worst case.

    The tilt path is solved on a detached `nominal`, and the first
    derivatives with respect to `nominal` and a tensor `radius` are
    attached afterwards by the envelope theorem, from the multiplier
    $\lambda^\star$ that each subclass reads off its solved tilt in closed
    form (`_standardized_sensitivities`).

    Attributes:
        nominal: Reference distribution the ambiguity set is centered on.
        divergence: Divergence used to measure distance from `nominal`.
        radius: Nonnegative bound on the divergence of any distribution
            inside the ambiguity set from `nominal`.
    """

    @abstractmethod
    def _tilted_distribution(
        self, gap: torch.Tensor, tilt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Evaluate the tilt path at `tilt` and its divergence from `nominal`.

        Both are returned together because every formulation computes the
        divergence from the same intermediates as the distribution, and the
        bisection needs the divergence at each step while only the final
        step needs the distribution.

        Args:
            gap: Shortfall of each scenario below the maximum loss on the
                nominal's support, standardized to unit spread, of shape
                `(..., n)` and therefore lying in $[0, 1]$ with at least one
                zero entry on the support; scenarios off the support carry
                `1`. Parameterizing by the gap rather than by the loss keeps
                the tilt of the highest-loss scenarios exactly `1` however
                extreme the tilt becomes.
            tilt: Path parameter in $[0, 1)$ of shape `batch_shape`, where
                `0` is `nominal` and the upper end is the distribution
                concentrated on the highest-loss scenarios.

        Returns:
            The tilted distribution, of shape `batch_shape + (n,)`, and its
            divergence from `nominal`, of shape `batch_shape`.
        """
        raise NotImplementedError

    def _standardized_sensitivities(
        self,
        gap: torch.Tensor,
        tilt: torch.Tensor,
        distribution: torch.Tensor,
        divergence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Return the parameter derivatives of the worst case of `-gap`.

        On the standardized loss $-\mathrm{gap}$, the derivative with
        respect to `radius` is the multiplier $\lambda^\star$ of the radius
        constraint. Only the part of the derivative with respect to
        `nominal` along directions that keep it summing to one is defined;
        it is returned as the representative $g$ with
        $\sum_i \mathrm{nominal}_i \, g_i = 0$, which stays finite as the
        radius vanishes, where $\lambda^\star$ grows without bound.

        Args:
            gap: Standardized shortfall below the maximum reachable loss, of
                shape `(..., n)`, taken from the loss itself on every
                scenario, including unreachable ones.
            tilt: Solved tilt of shape `batch_shape`.
            distribution: Maximizer at `tilt`, of shape
                `batch_shape + (n,)`.
            divergence: Divergence of `distribution`, of shape
                `batch_shape`.

        Returns:
            The derivative $g$ with respect to `nominal`, of shape
            `batch_shape + (n,)`, and $\lambda^\star$, of shape
            `batch_shape`.

        Raises:
            NotImplementedError: Unless a subclass implements it, so that a
                formulation without parameter derivatives fails loudly
                instead of returning none.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement gradients with respect "
            "to nominal or radius."
        )

    def _parameter_sensitivities(
        self,
        loss: torch.Tensor,
        value: torch.Tensor,
        max_loss: torch.Tensor,
        scale: torch.Tensor,
        tilt: torch.Tensor,
        distribution: torch.Tensor,
        divergence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Return the parameter derivatives of the worst case in loss units.

        The worst case is $\max \ell - s \, V(-\mathrm{gap})$ for the
        spread $s$, so the radius derivative is $s \lambda^\star$, and the
        nominal derivative is $s g$ shifted by the worst case itself. The
        shift lies along the all-ones direction, where the derivative is
        not defined, and makes $\sum_i \mathrm{nominal}_i \, \partial V /
        \partial \mathrm{nominal}_i = V$, matching the derivative $\ell$ of
        the zero-radius worst case $\langle \mathrm{nominal}, \ell \rangle$.

        Args:
            loss: Detached working-precision loss of shape `(..., n)`.
            value: Detached worst case of shape `batch_shape`.
            max_loss: Maximum reachable loss, with a trailing singleton
                dimension.
            scale: Positive loss spread, shaped like `max_loss`.
            tilt: Solved tilt of shape `batch_shape`.
            distribution: Maximizer of shape `batch_shape + (n,)`.
            divergence: Divergence of `distribution`, of shape
                `batch_shape`.

        Returns:
            The derivatives with respect to `nominal`, of shape
            `batch_shape + (n,)`, and `radius`, of shape `batch_shape`.
        """
        gap, multiplier = self._standardized_sensitivities(
            (max_loss - loss) / scale, tilt, distribution, divergence
        )
        return value.unsqueeze(-1) + scale * gap, scale.squeeze(-1) * multiplier

    @property
    def _reachable(self) -> torch.Tensor:
        """Boolean mask of the scenarios a tilted distribution can put mass on.

        Every tilt reweights the nominal, so by default these are the
        scenarios with positive nominal mass. A formulation whose tilt
        reweights a different base distribution overrides this.
        """
        return self.nominal > 0.0

    def _center_expectation(
        self, loss: torch.Tensor, batch_shape: torch.Size
    ) -> torch.Tensor:
        """Return the expectation under the tilt path's start, `tilt = 0`.

        This is the exact worst case of a zero-radius set. It is the nominal
        expectation by default; a formulation whose path starts elsewhere
        overrides this.

        Args:
            loss: Per-scenario loss values of shape `(..., n)`.
            batch_shape: Shape returned by `_batch_shape` for this `loss`.

        Returns:
            The expectation, broadcast to `batch_shape`.
        """
        return self._nominal_expectation(loss, batch_shape)

    @staticmethod
    def _tilt_rate(tilt: torch.Tensor) -> torch.Tensor:
        r"""Map a tilt in $[0, 1)$ to the unbounded natural parameter $t / (1 - t)$.

        The natural parameter of both tilt paths (the exponential rate for
        KL, the reciprocal cut depth for chi-square) is unbounded above,
        which is exactly what makes an unbracketed solve on it fragile.
        This bijection moves the solve onto the unit interval instead. The
        complement is floored at the dtype's smallest normal number so that
        a tilt that rounds to exactly `1` yields a large finite rate rather
        than an infinity, keeping the product with `gap` finite.

        Args:
            tilt: Path parameter in $[0, 1)$.

        Returns:
            The corresponding nonnegative natural parameter, unsqueezed
            along a trailing support dimension so it broadcasts against a
            `(..., n)` gap.
        """
        complement = torch.clamp(1.0 - tilt, min=torch.finfo(tilt.dtype).tiny)
        return (tilt / complement).unsqueeze(-1)

    def _solve_tilt(self, gap: torch.Tensor, batch_shape: torch.Size) -> torch.Tensor:
        """Search the unit interval for the largest tilt inside the radius.

        Each step evaluates $2^b - 1$ evenly spaced tilts inside the current
        bracket, of width $2^{-bk}$ after $k$ steps, in one call batched
        along a new leading dimension, and keeps the subinterval above the
        largest feasible one, so the bracket ends on the same dyadic grid
        bisection ($b = 1$) would reach. Every endpoint is a dyadic rational
        and the width a power of two, so the candidates are exact.

        Args:
            gap: Standardized shortfall below the maximum loss, of shape
                `(..., n)`.
            batch_shape: Batch shape of the loss being evaluated, as
                returned by `AmbiguitySet._batch_shape`.

        Returns:
            A tensor of shape `batch_shape` holding, per batch element, the
            largest tilt the search found whose distribution lies inside
            the ambiguity set. Elements are searched independently through
            elementwise reductions, so a saturated or degenerate element
            cannot disturb the rest of its batch.
        """
        bits = _search_bits(math.prod(batch_shape) * gap.shape[-1])
        points = 2**bits - 1
        steps = -(-_bisection_steps(gap.dtype) // bits)
        index = torch.arange(1, points + 1, dtype=gap.dtype, device=gap.device)
        index = index.reshape((points,) + (1,) * len(batch_shape))
        lower = torch.zeros(batch_shape, dtype=gap.dtype, device=gap.device)
        spacing = 1.0
        for _ in range(steps):
            spacing /= points + 1
            _, divergence = self._tilted_distribution(gap, lower + spacing * index)
            # A non-finite divergence compares false and so counts as
            # infeasible, which is the correct reading: it means the tilt
            # left the support of the nominal. The largest feasible
            # candidate is kept rather than the count of feasible ones, so
            # the returned tilt was itself tested feasible even where
            # rounding makes feasibility non-monotone.
            feasible = divergence <= self.radius
            lower = lower + spacing * torch.amax(
                torch.where(feasible, index, 0.0), dim=0
            )
        return lower

    def worst_case_expectation(self, loss: torch.Tensor) -> torch.Tensor:
        """Compute the worst-case expected loss at the tilt pinned by `radius`.

        Args:
            loss: Per-scenario loss values of shape `(..., n)`, one trailing
                entry per element of `nominal`'s support. Leading dimensions
                are a batch of independent loss vectors, bisected together
                in one vectorized solve.

        Returns:
            A tensor of shape `(...)` holding the worst-case expected loss:
            the exact `sum(nominal * loss)` when `radius` is a zero float
            (the set then contains only `nominal`), and otherwise the
            expectation of `loss` under the feasible tilted distribution
            the bisection returned, which is exactly the largest loss on
            the nominal's support once the radius saturates the set.

            The expectation is accumulated as a correction to `max(loss)`
            rather than directly. Every term of
            `sum(q * (loss - max(loss)))` is nonpositive, so the result can
            never round to above `max(loss)`, and a loss with a small
            spread around a large offset keeps the precision of its spread
            instead of losing it to the offset.

        Raises:
            ValueError: If `loss`'s trailing dimension does not match
                `nominal`'s support size, its batch shape does not
                broadcast against `nominal` and `radius`, or `validate` is
                set and `loss` is not finite.
        """
        batch_shape, working = self._prepare_loss(loss)
        if self._radius_is_zero:
            return self._as_result(self._center_expectation(working, batch_shape), loss)

        detached = working.detach()
        on_support = self._reachable
        max_loss = torch.amax(
            torch.where(on_support, detached, -torch.inf), dim=-1, keepdim=True
        )
        spread = max_loss - torch.amin(
            torch.where(on_support, detached, torch.inf), dim=-1, keepdim=True
        )
        scale = torch.where(spread > 0, spread, torch.ones_like(spread))
        gap = torch.where(on_support, (max_loss - detached) / scale, 1.0)
        tilt = self._solve_tilt(gap, batch_shape)
        distribution, divergence = self._tilted_distribution(gap, tilt)
        value: torch.Tensor = _MaximizerExpectation.apply(
            working, distribution, max_loss, scale, tilt, self._implicit_maximizer
        )
        expectation = value.expand(batch_shape)
        result = self._attach_parameter_gradients(
            expectation,
            lambda: self._parameter_sensitivities(
                detached,
                expectation.detach(),
                max_loss,
                scale,
                tilt,
                distribution,
                divergence,
            ),
        )
        return self._as_result(result, loss)

    def _implicit_maximizer(
        self,
        loss: torch.Tensor,
        max_loss: torch.Tensor,
        scale: torch.Tensor,
        tilt: torch.Tensor,
    ) -> torch.Tensor:
        r"""Rebuild the maximizer as a function of `loss` with an exact Jacobian.

        The solved tilt $t^\star$ is pinned by $D(t^\star, \mathrm{gap}) =
        \mathrm{radius}$, so by the implicit function theorem it moves with
        the loss at the rate $\partial t / \partial \ell = -(\partial D /
        \partial \ell) / (\partial D / \partial t)$. One Newton step on that
        constraint from $t^\star$, with the slope held fixed,

        $$
        t(\ell) = t^\star
            - \frac{D(t^\star, \mathrm{gap}(\ell)) - D(t^\star,
              \mathrm{gap}(\ell^\star))}{\partial D / \partial t},
        $$

        has exactly the value $t^\star$ and exactly that derivative, so the
        tilted distribution at $t(\ell)$ is the maximizer with its exact
        Jacobian, which is all a second derivative of the worst case needs.
        The standardizing shift and scale stay fixed, which is harmless:
        the tilt path for any fixed positive affine image of the loss is the
        same family of distributions.

        Where the slope vanishes the constraint is inactive (a saturated
        radius, a constant loss, or a zero tilt at a zero radius). The
        maximizer is then locally constant away from ties, so it is
        returned detached, with a zero Jacobian. Differentiating the tilted
        distribution there would instead multiply rounding errors by the
        astronomically large rate of a tilt at the top of its bracket.

        Args:
            loss: Per-scenario loss values of shape `(..., n)`, attached to
                the graph the Jacobian is taken in.
            max_loss: Detached maximum loss over the nominal's support used
                by the forward pass.
            scale: Detached positive loss spread used by the forward pass.
            tilt: Solved tilt of shape `batch_shape`.

        Returns:
            The maximizer of shape `batch_shape + (n,)`, equal in value to
            the forward pass's and differentiable with respect to `loss`.
        """
        gap = torch.where(self._reachable, (max_loss - loss) / scale, 1.0)
        slope: torch.Tensor = jvp(
            lambda path: self._tilted_distribution(gap.detach(), path)[1],
            (tilt,),
            (torch.ones_like(tilt),),
        )[1]
        _, divergence = self._tilted_distribution(gap, tilt)
        active = slope > 0.0
        step = (divergence - divergence.detach()) / torch.where(
            active, slope, torch.ones_like(slope)
        )
        distribution, _ = self._tilted_distribution(gap, tilt - step)
        return torch.where(active.unsqueeze(-1), distribution, distribution.detach())


def _implicit_curvature(
    dual_objective: Callable[[torch.Tensor], torch.Tensor],
    point: torch.Tensor,
    batch_shape: torch.Size,
) -> torch.Tensor:
    r"""Return a zero term that completes the Hessian of a dual re-evaluation.

    The worst case is $V(\theta) = F(x^\star(\theta), \theta)$ for the dual
    objective $F$, its minimizer $x^\star$, and the inputs $\theta$
    (`loss`, `nominal`, `radius`). Re-evaluating $F$ at a detached optimum
    $\bar x$ gives the exact first derivative $F_\theta$ by the envelope
    theorem, but differentiating it again yields only $F_{\theta\theta}$
    and misses the implicit-function term from $\partial x^\star /
    \partial \theta = -H^{-1} F_{x\theta}$, where $H = F_{xx}(\bar x)$.
    With the residual $r(\theta) = F_x(\bar x, \theta) - F_x(\bar x,
    \bar\theta)$, this returns

    $$
    c(\theta) = -\tfrac{1}{2}\, r(\theta)^\top H^{-1} r(\theta),
    $$

    whose value and first derivative vanish at $\bar\theta$, since
    $r(\bar\theta) = 0$ exactly, and whose second derivative is exactly
    $-F_{\theta x} H^{-1} F_{x\theta}$. Adding it to the re-evaluation
    therefore leaves the value and gradient untouched and makes every
    second derivative exact once the dual solve has converged. $H$ is held
    fixed, which costs nothing at second order. Where $H$ is not finite
    and positive definite, the term is zero.

    Args:
        dual_objective: Map from dual points of shape `batch_shape + d` to
            per-element dual values of shape `batch_shape`, attached to the
            inputs being differentiated.
        point: Detached dual optimum of shape `batch_shape + d`.
        batch_shape: Batch shape of the solve.

    Returns:
        The zero-valued correction, of shape `batch_shape`.
    """
    with torch.enable_grad():
        variable = point.detach().requires_grad_(True)
        (slope,) = torch.autograd.grad(
            dual_objective(variable).sum(), variable, create_graph=True
        )
        slope = slope.reshape(batch_shape + (-1,))
        size = slope.shape[-1]
        rows = [
            torch.autograd.grad(
                slope[..., row].sum(), variable, retain_graph=True, allow_unused=True
            )[0]
            for row in range(size)
        ]
    hessian = torch.stack(
        [
            torch.zeros_like(slope) if row is None else row.reshape(slope.shape)
            for row in rows
        ],
        dim=-2,
    )
    identity = torch.eye(size, dtype=hessian.dtype, device=hessian.device)
    finite = torch.isfinite(hessian).all(dim=-1).all(dim=-1)
    factor, info = torch.linalg.cholesky_ex(
        torch.where(finite[..., None, None], hessian, identity)
    )
    usable = finite & (info == 0)
    factor = torch.where(usable[..., None, None], factor, identity)
    residual = slope - slope.detach()
    solved = torch.cholesky_solve(residual.unsqueeze(-1), factor).squeeze(-1)
    correction = -0.5 * torch.sum(residual * solved, dim=-1)
    return torch.where(usable, correction, torch.zeros_like(correction))


class DualAmbiguitySet(AmbiguitySet):
    r"""Ambiguity set whose worst-case expectation is a low-dimensional dual solve.

    Every divergence-based ambiguity set that reformulates its inner
    supremum as a convex dual minimization over a handful of dual variables
    (`optora.dro.PhiAmbiguitySet` over $(\log(\eta), \lambda)$) runs the
    same machinery around a formulation-specific dual objective, so that
    machinery lives here: hold an injected `dual_solver`, start it from a
    dual point, and re-evaluate the dual objective at the returned optimum
    so the result stays differentiable with respect to `loss`, `nominal`,
    and `radius` by the envelope theorem. A zero-valued implicit-function
    term (`_implicit_curvature`) built from the dual Hessian at the optimum
    makes second derivatives exact as well, once the solve has converged.

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

    The loss scale is handled here once rather than in each formulation.
    The set of candidate distributions does not depend on `loss`, so
    $\sup_q \mathbb{E}_q[a\,\ell + b] = a \sup_q \mathbb{E}_q[\ell] + b$
    for $a > 0$. The dual's curvature grows with the squared loss spread,
    so a step size tuned for an order-one loss diverges on a wide one. The
    dual is therefore solved on the loss standardized to unit spread,
    $(\ell - \min \ell) / (\max \ell - \min \ell)$, with a detached shift
    and scale: for any fixed constants this is the same function of `loss`,
    so values and every derivative are unchanged. `initial_dual_point` and
    the warm start stay in the units of the raw loss and are converted at
    the boundary by `_to_standard_units` and `_from_standard_units`.

    What standardization cannot fix is the shape of the dual itself. The
    minimizer runs off to infinity as `radius` vanishes, and it ceases to
    exist at all once `radius` reaches the divergence of the distribution
    concentrated on the highest-loss scenarios, where the infimum is
    approached only in the limit. A first-order solver returns a stale
    iterate in the first case and can diverge outright in the second, so
    this class is the fallback for a formulation whose maximizer optora
    does not know in closed form. A formulation that does know it should
    use `TiltedAmbiguitySet`, whose bracketed bisection has neither
    failure mode.

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
            It is read in the units of the raw loss unless the set was
            built with `initial_in_raw_units=False`, in which case it is
            read in standardized units and so means the same thing on a
            loss of any scale.
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
        initial_in_raw_units: bool = True,
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
            initial_in_raw_units: Whether `initial_dual_point` is in the
                units of the raw loss (the default, for a start the caller
                chose) or already in standardized units (for a default
                start that must not depend on the loss scale).
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
        self._initial_in_raw_units = initial_in_raw_units
        self.register_buffer("_dual_warm_start", None, persistent=False)

    def reset_warm_start(self) -> None:
        """Discard the cached dual optimum so the next solve starts cold.

        After this call the next `worst_case_expectation` starts from
        `initial_dual_point` again, as the first one did.
        """
        self._dual_warm_start = None

    @abstractmethod
    def _dual_objective(
        self,
        point: torch.Tensor,
        loss: torch.Tensor,
        radius: float | torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the formulation's convex dual objective.

        Args:
            point: Dual variables in standardized-loss units, of shape
                `batch_shape + initial_dual_point.shape`.
            loss: Loss standardized to unit spread, of shape `(..., n)`.
            radius: Radius to evaluate the dual at, broadcastable against
                the batch shape.

        Returns:
            Per-batch-element dual values of shape `batch_shape`.
        """
        raise NotImplementedError

    @abstractmethod
    def _to_standard_units(
        self, point: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """Express raw-loss dual variables in standardized-loss units.

        Args:
            point: Dual variables of shape `batch_shape +
                initial_dual_point.shape`, in the units of the raw loss.
            shift: Per-element shift subtracted from the loss, broadcastable
                against `batch_shape`.
            scale: Per-element positive scale the loss was divided by,
                broadcastable against `batch_shape`.

        Returns:
            The same dual point for the standardized loss. A multiplier
            with the units of a loss (such as `lam`) maps to
            `(lam - shift) / scale`; a log-scale variable (such as
            `log(eta)`) maps to `log(eta) - log(scale)`.
        """
        raise NotImplementedError

    @abstractmethod
    def _from_standard_units(
        self, point: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """Express standardized-loss dual variables in the units of the raw loss.

        Args:
            point: Dual variables in standardized-loss units.
            shift: Per-element shift subtracted from the loss.
            scale: Per-element positive scale the loss was divided by.

        Returns:
            The inverse of `_to_standard_units`.
        """
        raise NotImplementedError

    def worst_case_expectation(self, loss: torch.Tensor) -> torch.Tensor:
        """Compute the worst-case expected loss from the convex dual.

        Args:
            loss: Per-scenario loss values of shape `(..., n)`, one trailing
                entry per element of `nominal`'s support. Leading dimensions
                are a batch of independent loss vectors, each solved with
                its own dual variables in a single joint solve.

        Returns:
            A tensor of shape `(...)` holding the worst-case expected loss:
            the exact `sum(nominal * loss)` when `radius` is a zero float
            (the set then contains only `nominal`), and otherwise the dual
            objective evaluated at the optimum found by `dual_solver`. The
            dual value bounds the worst case from above for any dual point,
            so an unconverged solve reports a conservative number rather
            than a wrong direction.

        Raises:
            ValueError: If `loss`'s trailing dimension does not match
                `nominal`'s support size, its batch shape does not
                broadcast against `nominal` and `radius`, or `validate` is
                set and `loss` is not finite.
            RuntimeError: If `radius` is positive and `dual_solver` is
                `None`.
        """
        batch_shape, working = self._prepare_loss(loss)
        if self._radius_is_zero:
            return self._as_result(
                self._nominal_expectation(working, batch_shape), loss
            )

        shift = torch.amin(working, dim=-1).detach()
        spread = torch.amax(working, dim=-1).detach() - shift
        scale = torch.where(spread > 0, spread, torch.ones_like(spread))
        standardized = (working - shift.unsqueeze(-1)) / scale.unsqueeze(-1)

        dual_value = self._solve_dual(
            lambda point: self._dual_objective(point, standardized, self.radius),
            batch_shape,
            shift,
            scale,
        )
        return self._as_result(scale * dual_value + shift, loss)

    def _solve_dual(
        self,
        dual_objective: Callable[[torch.Tensor], torch.Tensor],
        batch_shape: torch.Size,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Minimize a dual objective and return its value at the optimum.

        Args:
            dual_objective: Formulation-specific convex dual objective on
                the standardized loss. It maps dual variables of shape
                `batch_shape + initial_dual_point.shape` to
                per-batch-element dual values of shape `batch_shape`.
            batch_shape: Batch shape of the loss being evaluated, as
                returned by `AmbiguitySet._batch_shape`.
            shift: Per-element shift the loss was standardized with.
            scale: Per-element positive scale the loss was standardized
                with.

        Returns:
            A tensor of shape `batch_shape` holding `dual_objective`
            re-evaluated at the dual optimum found by `dual_solver`. The
            re-evaluation is what keeps the result differentiable with
            respect to `loss` even though the dual variables themselves are
            detached, and `_implicit_curvature` adds, at zero value and zero
            first derivative, the second-order effect of the optimum moving
            with `loss`, `nominal`, and `radius`. It is built only when the
            re-evaluated objective requires grad. The cached warm start is
            reused only when its shape
            still matches the batch being solved. Both the initial point and
            the cache are kept in raw-loss units and converted to and from
            the standardized units the solver iterates in.

        Raises:
            RuntimeError: If `dual_solver` is `None`.
        """
        if self.dual_solver is None:
            raise RuntimeError(
                "dual_solver is required to evaluate a positive-radius "
                f"{type(self).__name__}."
            )
        initial = self.initial_dual_point.to(scale.dtype)
        start_shape = batch_shape + initial.shape
        warm_start = self._dual_warm_start
        if warm_start is not None and warm_start.shape == start_shape:
            start = self._to_standard_units(warm_start, shift, scale)
        elif self._initial_in_raw_units:
            start = self._to_standard_units(initial.expand(start_shape), shift, scale)
        else:
            start = initial.expand(start_shape)
        problem = MinimizationProblem(
            objective=lambda point: torch.sum(dual_objective(point)),
            initial_point=start,
        )
        result = self.dual_solver.solve(problem)
        point = result.point.detach()
        self._dual_warm_start = self._from_standard_units(point, shift, scale)
        value = dual_objective(point)
        if not value.requires_grad:
            return value
        return value + _implicit_curvature(dual_objective, point, batch_shape)
