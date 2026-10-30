"""Entropic optimal transport cost relative to a kernel-smoothed reference."""

import warnings

import torch

from optora.core.convergence import (
    DEFAULT_CHECK_INTERVAL,
    ConvergenceTracker,
    validate_check_interval,
)
from optora.core.divergence_base import Divergence
from optora.core.precision import working_dtype

# Bounds and update factor of the Levenberg-Marquardt damping: a rejected
# step multiplies it, an accepted one divides it, so the solve moves
# between gradient-like and pure Newton steps as the model earns trust.
_DAMPING_FACTOR = 4.0
_DAMPING_BOUNDS = (1e-12, 1e12)


def _warn_deprecated_eps(stacklevel: int) -> None:
    """Warn that the `eps` marginal clamp is deprecated and has no effect.

    Args:
        stacklevel: Stack level of the warning, counted from this helper's
            caller, so it points at the line that passed `eps`.
    """
    warnings.warn(
        "eps is deprecated and has no effect: zero-mass marginal entries are "
        "excluded from the solve exactly. Drop the argument.",
        DeprecationWarning,
        stacklevel=stacklevel + 1,
    )


def _log_on_support(weights: torch.Tensor) -> torch.Tensor:
    """Return `log(weights)`, exactly `-inf` where `weights` is zero."""
    positive = weights > 0.0
    return torch.where(
        positive,
        torch.log(torch.where(positive, weights, torch.ones_like(weights))),
        -torch.inf,
    )


def _semi_dual(
    g: torch.Tensor,
    rows: torch.Tensor,
    columns: torch.Tensor,
    log_kernel: torch.Tensor,
) -> torch.Tensor:
    r"""Evaluate the semi-dual $\sum_j p_j g_j - \sum_i q_i \log \sum_j Q_{ij} e^{g_j}$.

    A column outside the solve carries the potential $-\infty$, which drops
    it from every row's normalizer; its term $p_j g_j$ is the limit
    $0 \cdot (-\infty) = 0$ and is read as zero, so it neither turns the
    value into `nan` nor sends a gradient into $p_j$.

    Args:
        g: Column potentials of shape `(..., n)`, `-inf` on the columns
            outside the solve.
        rows: Row marginal $q$ of shape `(..., n)`.
        columns: Column marginal $p$ of shape `(..., n)`, zero wherever `g`
            is `-inf`.
        log_kernel: Row-stochastic $\log Q$ of shape `(n, n)`.

    Returns:
        The semi-dual value over the batch, of shape `(...)`.
    """
    log_normalizer = torch.logsumexp(g.unsqueeze(-2) + log_kernel, dim=-1)
    finite_g = torch.where(torch.isneginf(g), 0.0, g)
    return torch.sum(columns * finite_g, dim=-1) - torch.sum(
        rows * log_normalizer, dim=-1
    )


def _marginal_violation(
    columns: torch.Tensor, marginal: torch.Tensor, active: torch.Tensor
) -> torch.Tensor:
    """Return the largest L1 column-marginal violation over the batch.

    It is the L1 norm of the semi-dual's gradient, the residual both the
    Sinkhorn and the Newton phase stop on.

    Args:
        columns: Target column marginal of shape `(..., n)`.
        marginal: Column marginal of the current plan, of shape `(..., n)`.
        active: Boolean mask of shape `(..., n)` of the columns in the
            solve.

    Returns:
        A zero-dimensional tensor.
    """
    violation = torch.where(active, torch.abs(columns - marginal), 0.0)
    return torch.amax(torch.sum(violation, dim=-1))


class EntropicTransportDivergence(Divergence):
    r"""Excess entropic transport cost of a candidate over its smoothed reference.

    For a reference distribution `q` and a candidate `p` on a shared finite
    support with ground cost $C$, a regularization strength $\epsilon > 0$,
    and a reference measure $\nu$ on the support, the Sinkhorn distance of
    Wang, Gao and Xie (2025) is

    $$
    \mathcal{W}_\epsilon(q, p) = \min_{\gamma \in \Gamma(q, p)}
        \mathbb{E}_\gamma[C] + \epsilon\, H(\gamma \mid q \otimes \nu),
    $$

    with $H$ the relative entropy. Its minimum over `p` is attained at the
    kernel-smoothed reference $P^0 = q^\top Q$, where
    $Q_{ij} \propto \nu_j e^{-C_{ij}/\epsilon}$ is the row-stochastic Gibbs
    kernel, and subtracting that minimum leaves

    $$
    D(p \,\|\, q) = \epsilon \min_{\gamma \in \Gamma(q, p)}
        \mathrm{KL}\big(\gamma \,\big\|\, \mathrm{diag}(q)\, Q\big)
    = \epsilon \max_g \; \sum_j p_j g_j
        - \sum_i q_i \log \sum_j Q_{ij} e^{g_j},
    $$

    the second form being the concave semi-dual over column potentials
    $g$. It is nonnegative and vanishes exactly at its centre $P^0$, not at
    `p == q`: entropic smoothing moves the centre of the ball off the
    reference. `optora.dro.SinkhornAmbiguitySet` bounds this quantity by
    its radius.

    The semi-dual is maximized in two phases, both stopping once the L1
    column-marginal violation (the norm of the semi-dual's gradient) falls
    below `tol`:

    1. Sinkhorn scaling in the log domain, $O(n^2)$ per iteration. It
       converges linearly at a rate that degrades as $\max C/\epsilon$
       grows, and at a sharp kernel it stalls far from the maximum.
    2. If Sinkhorn stops short, a Levenberg-Marquardt-damped Newton method
       on the semi-dual with its exact Hessian (Brauer et al. 2017),
       $O(n^3)$ per iteration, warm-started from the Sinkhorn potentials.
       A step is kept only if it increases the semi-dual, so no step can
       move away from the maximum, and the damping falls on success so
       the method turns into a pure Newton method near the maximum, where
       it converges quadratically however sharp the kernel is.

    The value returned is $\epsilon$ times the semi-dual at the final
    potentials. Every dual point bounds the divergence from below, so an
    inexact solve can only understate it, and near the maximum the
    understatement is second order in the marginal violation, where the
    transport plan's primal objective would be off to first order. The
    potentials are held fixed under autograd, so by the envelope theorem
    the gradient is exact once the solve has converged.

    Zero-mass entries of either marginal are excluded from the solve
    exactly, never clamped: a row with $q_i = 0$ transports nothing, and a
    column with $p_j = 0$ takes the potential $-\infty$, the semi-dual's
    maximizer in that coordinate, which removes it from every row's
    normalizer. The value is then that of the problem restricted to the two
    supports, with no accuracy floor. The derivative with respect to the
    mass of an empty column is $-\infty$, the slope of $t \log t$ at zero,
    and is not representable; autograd reports zero for it.

    Attributes:
        cost: Square ground cost matrix of shape `(n, n)`.
        reference: Nonnegative reference measure $\nu$ of shape `(n,)`. Only
            its normalization-free shape matters: $Q$ normalizes each row.
        epsilon: Positive entropic regularization strength.
        max_iter: Maximum number of Sinkhorn scaling iterations.
        newton_max_iter: Maximum number of damped Newton iterations run
            after Sinkhorn stops short of `tol`; zero disables the phase.
        tol: Convergence tolerance on the L1 column-marginal violation,
            floored at $n$ times the machine epsilon of the working dtype,
            the rounding floor of the violation itself.
        eps: Deprecated and unused; `None` unless passed explicitly.
        check_interval: Sinkhorn iterations between host reads of the
            convergence flag; the potentials freeze once converged. The
            Newton phase, whose iterations are expensive, reads it after
            every iteration.
    """

    cost: torch.Tensor
    reference: torch.Tensor

    def __init__(
        self,
        cost: torch.Tensor,
        epsilon: float,
        reference: torch.Tensor | None = None,
        max_iter: int = 1000,
        newton_max_iter: int = 50,
        tol: float = 1e-9,
        eps: float | None = None,
        check_interval: int = DEFAULT_CHECK_INTERVAL,
        validate: bool = False,
    ) -> None:
        """Initialize the entropic transport divergence.

        Args:
            cost: Square ground cost matrix of shape `(n, n)`.
            epsilon: Positive entropic regularization strength.
            reference: Nonnegative reference measure of shape `(n,)`. Defaults
                to the counting measure, `ones(n)` in `cost`'s dtype.
            max_iter: Maximum number of Sinkhorn scaling iterations.
            newton_max_iter: Maximum number of damped Newton iterations
                run after Sinkhorn stops short of `tol`. Each solves one
                `(n, n)` linear system per batch element; zero disables
                the phase.
            tol: Convergence tolerance on the L1 column-marginal violation,
                over the whole batched problem. It is floored at `n` times
                the machine epsilon of the working dtype, below which the
                violation is rounding error.
            eps: Deprecated and ignored. Zero-mass marginal entries are
                excluded from the solve exactly, so there is nothing left to
                clamp; passing it emits a `DeprecationWarning`, and it will
                be removed in a future minor release.
            check_interval: Sinkhorn iterations between host reads of the
                convergence flag.
            validate: Whether to check that `cost` is finite and `reference`
                is nonnegative with positive mass. The check reads
                reductions on the host, so it is opt-in.

        Raises:
            ValueError: If `cost` is not a square 2D tensor, `reference` has
                the wrong shape, a hyperparameter is out of range, or
                `validate` is set and `cost` or `reference` is invalid.
        """
        super().__init__()
        if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
            raise ValueError(
                f"cost must be a square 2D tensor, got shape {tuple(cost.shape)}."
            )
        if reference is None:
            reference = torch.ones(cost.shape[0], dtype=cost.dtype, device=cost.device)
        if reference.shape != (cost.shape[0],):
            raise ValueError(
                f"reference must have shape ({cost.shape[0]},), "
                f"got {tuple(reference.shape)}."
            )
        for name, value in (("epsilon", epsilon), ("tol", tol)):
            if not value > 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if eps is not None:
            _warn_deprecated_eps(stacklevel=2)
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {max_iter}.")
        if newton_max_iter < 0:
            raise ValueError(
                f"newton_max_iter must be nonnegative, got {newton_max_iter}."
            )
        if validate:
            finite_cost, valid_reference = torch.stack(
                [
                    torch.all(torch.isfinite(cost)),
                    (torch.amin(reference) >= 0.0) & (torch.sum(reference) > 0),
                ]
            ).tolist()
            if not finite_cost:
                raise ValueError("cost must be finite.")
            if not valid_reference:
                raise ValueError("reference must be nonnegative with positive mass.")
        self.register_buffer("cost", cost)
        self.register_buffer("reference", reference)
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.newton_max_iter = newton_max_iter
        self.tol = tol
        self.eps = eps
        self.check_interval = validate_check_interval(check_interval)

    def log_kernel(self, dtype: torch.dtype) -> torch.Tensor:
        r"""Return the row-stochastic Gibbs kernel $\log Q$ in `dtype`.

        $\log Q_{ij} = \log \nu_j - C_{ij}/\epsilon - \log \sum_k \nu_k
        e^{-C_{ik}/\epsilon}$, normalized by `logsumexp` so that a large
        cost over a small $\epsilon$ never underflows a whole row. Columns
        with $\nu_j = 0$ are exactly `-inf`.

        Args:
            dtype: Floating-point dtype to compute in.

        Returns:
            A tensor of shape `(n, n)` whose rows exponentiate to
            distributions.
        """
        log_reference = _log_on_support(self.reference.to(dtype))
        unnormalized = log_reference - self.cost.to(dtype) / self.epsilon
        return unnormalized - torch.logsumexp(unnormalized, dim=-1, keepdim=True)

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        r"""Compute the excess entropic transport cost of `p` over `q`.

        Args:
            p: Candidate distribution of shape `(..., n)`.
            q: Reference distribution of shape `(..., n)`, broadcasting
                against `p`.

        Returns:
            $D(p \,\|\, q)$ over the broadcast batch, clamped to be
            nonnegative, computed in `optora.core.working_dtype(p, q, cost)`
            and returned in the promoted dtype of `p` and `q`. A candidate
            with mass where $\nu$ has none admits no coupling and returns
            `inf`.

        Raises:
            ValueError: If the last dimension of `p` or `q` does not match
                `cost`.
        """
        size = self.cost.shape[0]
        for name, tensor in (("p", p), ("q", q)):
            if tensor.shape[-1] != size:
                raise ValueError(
                    f"{name} must have shape (..., {size}) to index cost, "
                    f"got {tuple(tensor.shape)}."
                )
        result_dtype = torch.promote_types(p.dtype, q.dtype)
        dtype = working_dtype(p, q, self.cost)
        log_kernel = self.log_kernel(dtype)
        reachable = self.reference > 0.0
        tol = max(self.tol, size * torch.finfo(dtype).eps)
        candidate, reference = p.to(dtype), q.to(dtype)
        columns = candidate + torch.zeros_like(reference)
        rows = reference + torch.zeros_like(candidate)
        with torch.no_grad():
            # Mass off the reachable columns makes the value infinite below;
            # dropping it from the solve keeps the batch's other elements
            # well posed.
            column_mass = torch.where(reachable, columns, 0.0)
            active = column_mass > 0.0
            # No coupling meets marginals of different total mass, so a
            # rounding-level mismatch would floor the marginal violation
            # above `tol`; the solve matches the totals instead.
            column_mass = column_mass * (
                torch.sum(rows, dim=-1, keepdim=True)
                / torch.sum(column_mass, dim=-1, keepdim=True)
            )
            g, converged = self._sinkhorn(rows, column_mass, log_kernel, active, tol)
            if not converged and self.newton_max_iter > 0:
                g = self._newton(g, rows, column_mass, log_kernel, active, tol)
        divergence = torch.clamp(
            self.epsilon * _semi_dual(g, rows, columns, log_kernel), min=0.0
        )
        infeasible = torch.any((columns > 0.0) & ~reachable, dim=-1)
        return torch.where(infeasible, torch.inf, divergence).to(result_dtype)

    def _sinkhorn(
        self,
        row_mass: torch.Tensor,
        column_mass: torch.Tensor,
        log_kernel: torch.Tensor,
        active: torch.Tensor,
        tol: float,
    ) -> tuple[torch.Tensor, bool]:
        r"""Run log-domain Sinkhorn scaling on the column potentials.

        Args:
            row_mass: Nonnegative row marginal of shape `(..., n)`.
            column_mass: Column marginal of shape `(..., n)`, positive
                exactly on `active`.
            log_kernel: Row-stochastic $\log Q$ of shape `(n, n)`.
            active: Boolean mask of shape `(..., n)` of the columns in the
                solve; the potential stays `-inf` elsewhere.
            tol: Floored tolerance on the marginal violation.

        Returns:
            The column potentials, and whether the marginal violation fell
            below `tol` at a checkpoint.
        """
        log_rows = _log_on_support(row_mass)
        log_columns = _log_on_support(column_mass)
        g = torch.where(active, 0.0, -torch.inf).to(log_columns.dtype)
        tracker = ConvergenceTracker(tol, self.check_interval, g)
        for iteration in range(self.max_iter):
            f = log_rows - torch.logsumexp(g.unsqueeze(-2) + log_kernel, dim=-1)
            log_scale = torch.logsumexp(f.unsqueeze(-1) + log_kernel, dim=-2)
            residual = _marginal_violation(
                column_mass, torch.exp(g + log_scale), active
            )
            converged = tracker.update(residual)
            g = torch.where(converged | ~active, g, log_columns - log_scale)
            if tracker.should_stop(iteration):
                return g, True
        return g, False

    def _newton(
        self,
        g: torch.Tensor,
        row_mass: torch.Tensor,
        column_mass: torch.Tensor,
        log_kernel: torch.Tensor,
        active: torch.Tensor,
        tol: float,
    ) -> torch.Tensor:
        r"""Refine column potentials by damped Newton ascent on the semi-dual.

        The semi-dual's negative Hessian is
        $H = \mathrm{diag}(m) - \gamma^\top \mathrm{diag}(q)^{-1} \gamma$,
        with $\gamma$ the plan and $m$ its column marginal. It is positive
        semidefinite with the constant vector in its kernel, since a common
        shift of $g$ leaves the semi-dual unchanged; the step solves
        $(H + \mathbf{1}\mathbf{1}^\top / n + \mu\,\mathrm{diag}(m))\,d =
        \nabla$ for a per-element damping $\mu$, then drops the step's mean.
        A step is accepted unless it lowers the semi-dual by more than the
        rounding error of its evaluation: the final steps of a converging
        Newton method improve it by less than that, so a strict increase
        test would reject them. The solve does not check for singularity: a
        failed solve yields a non-finite trial, which is rejected like any
        other non-improving step. Columns outside the solve are pinned by an
        identity block in the system and keep the potential `-inf`; rows of
        zero mass carry no plan and drop out of the Hessian.

        Args:
            g: Warm-start column potentials of shape `(..., n)`.
            row_mass: Nonnegative row marginal of shape `(..., n)`.
            column_mass: Column marginal of shape `(..., n)`, positive
                exactly on `active`.
            log_kernel: Row-stochastic $\log Q$ of shape `(n, n)`.
            active: Boolean mask of shape `(..., n)` of the columns in the
                solve; the potential stays `-inf` elsewhere.
            tol: Floored tolerance on the marginal violation.

        Returns:
            The refined column potentials.
        """
        size = g.shape[-1]
        rounding = size * torch.finfo(g.dtype).eps
        log_rows = _log_on_support(row_mass)
        safe_rows = torch.where(row_mass > 0.0, row_mass, 1.0)
        inactive = torch.diag_embed((~active).to(g.dtype))
        damping = torch.ones(g.shape[:-1], dtype=g.dtype, device=g.device)
        value = _semi_dual(g, row_mass, column_mass, log_kernel)
        tracker = ConvergenceTracker(tol, 1, g)
        for iteration in range(self.newton_max_iter):
            log_normalizer = torch.logsumexp(g.unsqueeze(-2) + log_kernel, dim=-1)
            f = log_rows - log_normalizer
            plan = torch.exp(f.unsqueeze(-1) + g.unsqueeze(-2) + log_kernel)
            marginal = torch.sum(plan, dim=-2)
            tracker.update(_marginal_violation(column_mass, marginal, active))
            if tracker.should_stop(iteration):
                break
            gradient = torch.where(active, column_mass - marginal, 0.0)
            hessian = torch.diag_embed(marginal) - plan.mT @ (
                plan / safe_rows.unsqueeze(-1)
            )
            system = (
                hessian
                + 1.0 / size
                + damping[..., None, None] * torch.diag_embed(marginal)
                + inactive
            )
            step = torch.linalg.solve_ex(system, gradient.unsqueeze(-1))[0]
            step = step.squeeze(-1)
            trial = torch.where(
                active, g + step - step.mean(dim=-1, keepdim=True), -torch.inf
            )
            trial_value = _semi_dual(trial, row_mass, column_mass, log_kernel)
            finite_g = torch.where(active, g, 0.0)
            magnitude = torch.sum(
                torch.abs(column_mass * finite_g), dim=-1
            ) + torch.sum(row_mass * torch.abs(log_normalizer), dim=-1)
            accept = trial_value >= value - rounding * magnitude
            g = torch.where(accept.unsqueeze(-1), trial, g)
            value = torch.where(accept, trial_value, value)
            damping = torch.clamp(
                torch.where(
                    accept, damping / _DAMPING_FACTOR, damping * _DAMPING_FACTOR
                ),
                *_DAMPING_BOUNDS,
            )
        return g
