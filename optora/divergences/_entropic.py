r"""Shared solver for the semi-dual of entropic optimal transport.

Both entropic divergences reduce to maximizing the concave semi-dual

$$
\Psi(g) = \sum_j c_j g_j - \sum_i r_i \log \sum_j Q_{ij} e^{g_j}
$$

over column potentials $g$, for a row marginal $r$, a column marginal $c$,
and a row-stochastic Gibbs kernel $Q$. Its maximizer defines the entropic
transport plan $\gamma_{ij} = r_i Q_{ij} e^{g_j} / \sum_k Q_{ik} e^{g_k}$,
whose rows meet $r$ exactly and whose columns meet $c$ at the maximum.
`solve_semi_dual` runs log-domain Sinkhorn scaling and, if that stops
short, a damped Newton polish, both stopping on the L1 column-marginal
violation, which is the norm of the semi-dual's gradient.
"""

import warnings

import torch

from optora.core.convergence import ConvergenceTracker

# Bounds and update factor of the Levenberg-Marquardt damping: a rejected
# step multiplies it, an accepted one divides it, so the solve moves
# between gradient-like and pure Newton steps as the model earns trust.
_DAMPING_FACTOR = 4.0
_DAMPING_BOUNDS = (1e-12, 1e12)


def warn_deprecated_eps(stacklevel: int) -> None:
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


def log_on_support(weights: torch.Tensor) -> torch.Tensor:
    """Return `log(weights)`, exactly `-inf` where `weights` is zero."""
    positive = weights > 0.0
    return torch.where(
        positive,
        torch.log(torch.where(positive, weights, torch.ones_like(weights))),
        -torch.inf,
    )


def semi_dual(
    g: torch.Tensor,
    rows: torch.Tensor,
    columns: torch.Tensor,
    log_kernel: torch.Tensor,
) -> torch.Tensor:
    r"""Evaluate the semi-dual $\sum_j c_j g_j - \sum_i r_i \log \sum_j Q_{ij} e^{g_j}$.

    A column outside the solve carries the potential $-\infty$, which drops
    it from every row's normalizer; its term $c_j g_j$ is the limit
    $0 \cdot (-\infty) = 0$ and is read as zero, so it neither turns the
    value into `nan` nor sends a gradient into $c_j$.

    Args:
        g: Column potentials of shape `(..., n)`, `-inf` on the columns
            outside the solve.
        rows: Row marginal $r$ of shape `(..., n)`.
        columns: Column marginal $c$ of shape `(..., n)`, zero wherever `g`
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


def solve_semi_dual(
    rows: torch.Tensor,
    columns: torch.Tensor,
    log_kernel: torch.Tensor,
    tol: float,
    max_iter: int,
    newton_max_iter: int,
    check_interval: int,
) -> torch.Tensor:
    r"""Maximize the semi-dual, excluding zero-mass marginal entries exactly.

    A row with $r_i = 0$ transports nothing, and a column with $c_j = 0$
    takes the potential $-\infty$, the semi-dual's maximizer in that
    coordinate, so the solve is that of the problem restricted to the two
    supports. No coupling meets marginals of different total mass, so a
    rounding-level mismatch would floor the marginal violation above `tol`;
    the solve rescales the columns to the rows' total instead.

    Args:
        rows: Nonnegative row marginal of shape `(..., n)`.
        columns: Nonnegative column marginal of shape `(..., n)`, zero on
            any column the kernel cannot reach, with positive total mass.
        log_kernel: Row-stochastic $\log Q$ of shape `(n, n)`.
        tol: Tolerance on the L1 column-marginal violation over the whole
            batch, floored at `n` times the machine epsilon of the dtype,
            below which the violation is rounding error.
        max_iter: Maximum number of Sinkhorn scaling iterations.
        newton_max_iter: Maximum number of damped Newton iterations run
            after Sinkhorn stops short of `tol`; zero disables the phase.
        check_interval: Sinkhorn iterations between host reads of the
            convergence flag. The Newton phase reads it every iteration.

    Returns:
        The column potentials of shape `(..., n)`, `-inf` on zero-mass
        columns. The solve runs without autograd.
    """
    tol = max(tol, rows.shape[-1] * torch.finfo(rows.dtype).eps)
    with torch.no_grad():
        active = columns > 0.0
        column_mass = columns * (
            torch.sum(rows, dim=-1, keepdim=True)
            / torch.sum(columns, dim=-1, keepdim=True)
        )
        g, converged = _sinkhorn(
            rows, column_mass, log_kernel, active, tol, max_iter, check_interval
        )
        if not converged and newton_max_iter > 0:
            g = _newton(g, rows, column_mass, log_kernel, active, tol, newton_max_iter)
    return g


def _sinkhorn(
    row_mass: torch.Tensor,
    column_mass: torch.Tensor,
    log_kernel: torch.Tensor,
    active: torch.Tensor,
    tol: float,
    max_iter: int,
    check_interval: int,
) -> tuple[torch.Tensor, bool]:
    r"""Run log-domain Sinkhorn scaling on the column potentials.

    It converges linearly at a rate that degrades as the kernel sharpens,
    and at a sharp kernel it stalls far from the maximum.

    Args:
        row_mass: Nonnegative row marginal of shape `(..., n)`.
        column_mass: Column marginal of shape `(..., n)`, positive exactly
            on `active`.
        log_kernel: Row-stochastic $\log Q$ of shape `(n, n)`.
        active: Boolean mask of shape `(..., n)` of the columns in the
            solve; the potential stays `-inf` elsewhere.
        tol: Floored tolerance on the marginal violation.
        max_iter: Maximum number of iterations.
        check_interval: Iterations between host reads of the convergence
            flag; the potentials freeze once converged.

    Returns:
        The column potentials, and whether the marginal violation fell below
        `tol` at a checkpoint.
    """
    log_rows = log_on_support(row_mass)
    log_columns = log_on_support(column_mass)
    g = torch.where(active, 0.0, -torch.inf).to(log_columns.dtype)
    tracker = ConvergenceTracker(tol, check_interval, g)
    for iteration in range(max_iter):
        f = log_rows - torch.logsumexp(g.unsqueeze(-2) + log_kernel, dim=-1)
        log_scale = torch.logsumexp(f.unsqueeze(-1) + log_kernel, dim=-2)
        residual = _marginal_violation(column_mass, torch.exp(g + log_scale), active)
        converged = tracker.update(residual)
        g = torch.where(converged | ~active, g, log_columns - log_scale)
        if tracker.should_stop(iteration):
            return g, True
    return g, False


def _newton(
    g: torch.Tensor,
    row_mass: torch.Tensor,
    column_mass: torch.Tensor,
    log_kernel: torch.Tensor,
    active: torch.Tensor,
    tol: float,
    max_iter: int,
) -> torch.Tensor:
    r"""Refine column potentials by damped Newton ascent on the semi-dual.

    The semi-dual's negative Hessian is
    $H = \mathrm{diag}(m) - \gamma^\top \mathrm{diag}(r)^{-1} \gamma$, with
    $\gamma$ the plan and $m$ its column marginal (Brauer et al. 2017). It
    is positive semidefinite with the constant vector in its kernel, since
    a common shift of $g$ leaves the semi-dual unchanged; the step solves
    $(H + \mathbf{1}\mathbf{1}^\top / n + \mu\,\mathrm{diag}(m))\,d =
    \nabla$ for a per-element Levenberg-Marquardt damping $\mu$, then drops
    the step's mean. A step is accepted unless it lowers the semi-dual by
    more than the rounding error of its evaluation: the final steps of a
    converging Newton method improve it by less than that, so a strict
    increase test would reject them. The damping falls on success, so the
    method turns into a pure Newton method near the maximum, where it
    converges quadratically however sharp the kernel is. The solve does not
    check for singularity: a failed solve yields a non-finite trial, which
    is rejected like any other non-improving step. Columns outside the
    solve are pinned by an identity block in the system and keep the
    potential `-inf`; rows of zero mass carry no plan and drop out of the
    Hessian.

    Args:
        g: Warm-start column potentials of shape `(..., n)`.
        row_mass: Nonnegative row marginal of shape `(..., n)`.
        column_mass: Column marginal of shape `(..., n)`, positive exactly
            on `active`.
        log_kernel: Row-stochastic $\log Q$ of shape `(n, n)`.
        active: Boolean mask of shape `(..., n)` of the columns in the
            solve; the potential stays `-inf` elsewhere.
        tol: Floored tolerance on the marginal violation.
        max_iter: Maximum number of iterations.

    Returns:
        The refined column potentials.
    """
    size = g.shape[-1]
    rounding = size * torch.finfo(g.dtype).eps
    log_rows = log_on_support(row_mass)
    safe_rows = torch.where(row_mass > 0.0, row_mass, 1.0)
    inactive = torch.diag_embed((~active).to(g.dtype))
    damping = torch.ones(g.shape[:-1], dtype=g.dtype, device=g.device)
    value = semi_dual(g, row_mass, column_mass, log_kernel)
    tracker = ConvergenceTracker(tol, 1, g)
    for iteration in range(max_iter):
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
        step = torch.linalg.solve_ex(system, gradient.unsqueeze(-1))[0].squeeze(-1)
        trial = torch.where(
            active, g + step - step.mean(dim=-1, keepdim=True), -torch.inf
        )
        trial_value = semi_dual(trial, row_mass, column_mass, log_kernel)
        finite_g = torch.where(active, g, 0.0)
        magnitude = torch.sum(torch.abs(column_mass * finite_g), dim=-1) + torch.sum(
            row_mass * torch.abs(log_normalizer), dim=-1
        )
        accept = trial_value >= value - rounding * magnitude
        g = torch.where(accept.unsqueeze(-1), trial, g)
        value = torch.where(accept, trial_value, value)
        damping = torch.clamp(
            torch.where(accept, damping / _DAMPING_FACTOR, damping * _DAMPING_FACTOR),
            *_DAMPING_BOUNDS,
        )
    return g
