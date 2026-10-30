"""Entropic optimal transport cost relative to a kernel-smoothed reference."""

import torch

from optora.core.convergence import DEFAULT_CHECK_INTERVAL, validate_check_interval
from optora.core.divergence_base import Divergence
from optora.core.precision import working_dtype
from optora.divergences._entropic import (
    log_on_support,
    semi_dual,
    solve_semi_dual,
    warn_deprecated_eps,
)


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
            warn_deprecated_eps(stacklevel=2)
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
        log_reference = log_on_support(self.reference.to(dtype))
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
        candidate, reference = p.to(dtype), q.to(dtype)
        columns = candidate + torch.zeros_like(reference)
        rows = reference + torch.zeros_like(candidate)
        # Mass off the reachable columns makes the value infinite below;
        # dropping it from the solve keeps the batch's other elements well
        # posed.
        g = solve_semi_dual(
            rows,
            torch.where(reachable, columns, 0.0),
            log_kernel,
            self.tol,
            self.max_iter,
            self.newton_max_iter,
            self.check_interval,
        )
        divergence = torch.clamp(
            self.epsilon * semi_dual(g, rows, columns, log_kernel), min=0.0
        )
        infeasible = torch.any((columns > 0.0) & ~reachable, dim=-1)
        return torch.where(infeasible, torch.inf, divergence).to(result_dtype)
