"""Entropy-regularized Wasserstein (Sinkhorn) divergence between distributions."""

import torch

from optora.core.convergence import DEFAULT_CHECK_INTERVAL, validate_check_interval
from optora.core.divergence_base import Divergence
from optora.core.precision import working_dtype
from optora.divergences._entropic import (
    log_on_support,
    solve_semi_dual,
    warn_deprecated_eps,
)


class SinkhornDivergence(Divergence):
    r"""Debiased entropic Wasserstein divergence between discrete distributions.

    For discrete distributions `p` and `q` (nonnegative tensors that sum to
    one along their last dimension) sharing a common support with pairwise
    ground cost `cost`, the entropic optimal transport cost is

    $$
    \mathrm{OT}_\epsilon(p, q)
        = \min_{\pi \in U(p, q)} \langle \mathrm{cost}, \pi \rangle
        + \epsilon \sum_{ij} \pi_{ij} \big(\log \pi_{ij} - 1\big)
    $$

    where $U(p, q)$ is the set of transport plans (joint distributions) with
    marginals `p` and `q`, and $\epsilon$ is the entropic regularization
    strength. It is evaluated through its dual (Feydy et al. 2019),

    $$
    \mathrm{OT}_\epsilon(p, q) = \max_{f, g} \; \langle f, p \rangle
        + \langle g, q \rangle
        - \epsilon \big\langle p \otimes q,\,
        e^{(f \oplus g - \mathrm{cost}) / \epsilon} - 1 \big\rangle
        + c(p, q),
    $$

    where $c(p, q) = -\epsilon \sum_i p_i \log p_i - \epsilon \sum_j q_j
    \log q_j - \epsilon$ collects terms that cancel in the debiased
    divergence below and are therefore dropped. The potentials stay finite
    even where `p` or `q` has no mass, and zero-mass entries are excluded
    from the solve exactly rather than clamped. The dual is maximized by
    log-domain Sinkhorn scaling and, if that stops short, a damped Newton
    polish (the solver `optora.divergences.EntropicTransportDivergence`
    uses), both stopping on the L1 marginal violation of the plan.
    Updates go through `torch.logsumexp` rather than the Gibbs kernel
    $\exp(-\mathrm{cost} / \epsilon)$, which underflows to exact zero for
    small $\epsilon$ or large `cost`.

    The value is the dual objective at the final potentials, with $f$ the
    exact $c$-transform of $g$, rather than the transport cost
    $\langle \mathrm{cost}, \pi \rangle$ of the final plan: every dual point
    bounds $\mathrm{OT}_\epsilon$ from below and is off by an amount
    second order in the marginal violation, where the plan, which does not
    meet its marginals until the solve converges, would be off to first
    order. The potentials are held fixed under autograd, so by the
    envelope theorem the gradients, $f$ in `p` and $g$ in `q`, are exact
    once the solve has converged.

    Plain entropic OT cost is biased: $\mathrm{OT}_\epsilon(p, p)$ is not
    exactly zero for `eps > 0`. This class instead computes the debiased
    Sinkhorn divergence

    $$
    S_\epsilon(p, q) = \mathrm{OT}_\epsilon(p, q)
        - \tfrac{1}{2}\mathrm{OT}_\epsilon(p, p)
        - \tfrac{1}{2}\mathrm{OT}_\epsilon(q, q),
    $$

    which removes that self-transport bias so $S_\epsilon(p, p) = 0$ exactly,
    as required by the `Divergence` contract, while still converging to the
    Wasserstein distance induced by `cost` as $\epsilon \to 0$. The three
    transport costs the debiasing needs share one ground cost, so they are
    solved as a single batched Sinkhorn problem rather than three
    sequential loops. `optora.dro.wasserstein_dro` uses this divergence to
    define Wasserstein-based ambiguity sets.

    Attributes:
        cost: Square, nonnegative pairwise ground cost matrix between the
            shared support points of `p` and `q`, shape `(n, n)`.
        epsilon: Positive entropic regularization strength; smaller values
            approximate the exact Wasserstein distance more closely at the
            cost of more Sinkhorn iterations to converge.
        max_iter: Maximum number of Sinkhorn scaling iterations.
        tol: Convergence tolerance on the L1 marginal violation of the
            transport plan, taken over the whole batched problem so it stops
            when its slowest element does, and floored at $n$ times the
            machine epsilon of the working dtype.
        eps: Deprecated and unused; `None` unless passed explicitly.
        check_interval: Number of Sinkhorn iterations between host reads of
            the convergence flag. The potentials are frozen once converged,
            so the divergence does not depend on this interval.
        newton_max_iter: Maximum number of damped Newton iterations run
            after Sinkhorn stops short of `tol`; zero disables the phase.
    """

    cost: torch.Tensor

    def __init__(
        self,
        cost: torch.Tensor,
        epsilon: float = 0.1,
        max_iter: int = 100,
        tol: float = 1e-6,
        eps: float | None = None,
        check_interval: int = DEFAULT_CHECK_INTERVAL,
        validate: bool = False,
        newton_max_iter: int = 50,
    ) -> None:
        """Initialize the Sinkhorn divergence.

        Args:
            cost: Square, nonnegative pairwise ground cost matrix between
                the shared support points of `p` and `q`, shape `(n, n)`.
            epsilon: Positive entropic regularization strength.
            max_iter: Maximum number of Sinkhorn scaling iterations.
            tol: Convergence tolerance on the L1 marginal violation of the
                transport plan, over the whole batched problem. It is floored
                at `n` times the machine epsilon of the working dtype, below
                which the violation is rounding error.
            eps: Deprecated and ignored. Zero-mass entries of `p` and `q`
                are excluded from the solve exactly, so there is nothing
                left to clamp; passing it emits a `DeprecationWarning`, and
                it will be removed in a future minor release.
            check_interval: Number of Sinkhorn iterations between host
                reads of the convergence flag. Raise it to trade redundant
                frozen iterations for fewer device synchronizations.
            validate: Whether to check that `cost` is nonnegative. The
                check reads a reduction over `cost` on the host, which
                blocks until the device has produced it, so it is opt-in
                and off by default to keep construction asynchronous. The
                shape and hyperparameter checks are metadata-only and
                always run.
            newton_max_iter: Maximum number of damped Newton iterations run
                after Sinkhorn stops short of `tol`. Each solves one
                `(n, n)` linear system per transport problem; zero disables
                the phase.

        Raises:
            ValueError: If `cost` is not a square 2D tensor, if `validate`
                is set and `cost` contains negative entries, if `epsilon`,
                `max_iter`, `tol`, or `check_interval` are not positive, or
                if `newton_max_iter` is negative.
        """
        super().__init__()
        if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
            raise ValueError(
                f"cost must be a square 2D tensor, got shape {tuple(cost.shape)}."
            )
        if validate and bool(torch.amin(cost) < 0):
            raise ValueError("cost must be nonnegative.")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}.")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {max_iter}.")
        if tol <= 0:
            raise ValueError(f"tol must be positive, got {tol}.")
        if newton_max_iter < 0:
            raise ValueError(
                f"newton_max_iter must be nonnegative, got {newton_max_iter}."
            )
        if eps is not None:
            warn_deprecated_eps(stacklevel=2)
        self.register_buffer("cost", cost)
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.newton_max_iter = newton_max_iter
        self.tol = tol
        self.eps = eps
        self.check_interval = validate_check_interval(check_interval)

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        r"""Compute the debiased Sinkhorn divergence of `p` from `q`.

        The three entropic transport costs the debiasing needs are solved
        as one batched Sinkhorn problem rather than three sequential ones:
        the marginal pairs $(p, q)$, $(p, p)$ and $(q, q)$ stack along a
        new leading dimension against the shared ground cost, so the three
        problems share one loop, one convergence tracker, and one third of
        the periodic host reads.

        Args:
            p: Candidate distribution, a nonnegative tensor of shape
                `(..., n)` whose last dimension sums to one and indexes
                `cost`.
            q: Reference distribution, a nonnegative tensor of shape
                `(..., n)` whose last dimension sums to one and indexes
                `cost`. Its batch shape must broadcast against `p`'s.

        Returns:
            A tensor of shape `batch_shape` holding $S_\epsilon(p, q)$ over
            the broadcast batch, clamped to be nonnegative to absorb
            floating-point error near zero, computed in
            `optora.core.working_dtype(p, q, cost)` and returned in the
            promoted dtype of `p` and `q`.

        Raises:
            ValueError: If the shape of `p` or `q` does not match `cost`.
        """
        if p.shape[-1] != self.cost.shape[0]:
            raise ValueError(
                f"p must have shape (..., {self.cost.shape[0]}) to index "
                f"cost, got {tuple(p.shape)}."
            )
        if q.shape[-1] != self.cost.shape[0]:
            raise ValueError(
                f"q must have shape (..., {self.cost.shape[0]}) to index "
                f"cost, got {tuple(q.shape)}."
            )
        result_dtype = torch.promote_types(p.dtype, q.dtype)
        dtype = working_dtype(p, q, self.cost)
        p, q = p.to(dtype), q.to(dtype)
        # Stacking the three marginal pairs into one batched problem needs
        # `p` and `q` at a common shape; adding a zero of the other's shape
        # broadcasts each to it.
        p, q = p + torch.zeros_like(q), q + torch.zeros_like(p)
        cost_pq, cost_pp, cost_qq = torch.unbind(
            self._entropic_transport_cost(
                torch.stack((p, p, q)), torch.stack((q, p, q))
            )
        )
        divergence = cost_pq - 0.5 * cost_pp - 0.5 * cost_qq
        return torch.clamp(divergence, min=0.0).to(result_dtype)

    def _entropic_transport_cost(
        self, p: torch.Tensor, q: torch.Tensor
    ) -> torch.Tensor:
        r"""Compute $\mathrm{OT}_\epsilon(p, q)$ up to the terms debiasing cancels.

        With the scaled potentials $\phi = f / \epsilon$ and $\psi = g /
        \epsilon$, the dual over $g$ alone is the semi-dual the shared
        solver maximizes, for the row-stochastic kernel $\log Q_{ij} =
        -C_{ij}/\epsilon - r_i$ with $r_i = \log \sum_k e^{-C_{ik}/\epsilon}$,
        in the column potentials $h_j = \psi_j + \log q_j$. From its
        maximizer the row potential is the exact $c$-transform

        $$
        \phi_i = -\log \sum_j q_j\, e^{\psi_j - C_{ij}/\epsilon}
        = -\log \sum_j Q_{ij} e^{h_j} - r_i,
        $$

        under which the plan's rows meet `p` exactly and its total mass is
        one, so the dual objective reduces to $\epsilon (\langle \phi, p
        \rangle + \langle \psi, q \rangle)$. A column with $q_j = 0$ is
        outside the solve; its potential, which only its gradient needs, is
        the $c$-transform $\psi_j = -\log \sum_i p_i e^{\phi_i -
        C_{ij}/\epsilon}$.

        Args:
            p: Row marginals, a nonnegative tensor of shape `(..., n)` whose
                last dimension sums to one.
            q: Column marginals, a nonnegative tensor of shape `(..., n)`
                whose last dimension sums to one.

        Returns:
            A tensor of shape `batch_shape` holding the dual value of each
            transport problem.
        """
        scaled_cost = self.cost.to(p.dtype) / self.epsilon
        row_normalizer = torch.logsumexp(-scaled_cost, dim=-1)
        log_kernel = -scaled_cost - row_normalizer.unsqueeze(-1)
        column_potential = solve_semi_dual(
            p,
            q,
            log_kernel,
            self.tol,
            self.max_iter,
            self.newton_max_iter,
            self.check_interval,
        )
        with torch.no_grad():
            phi = -(
                torch.logsumexp(column_potential.unsqueeze(-2) + log_kernel, dim=-1)
                + row_normalizer
            )
            transform = -torch.logsumexp(
                (log_on_support(p) + phi).unsqueeze(-1) - scaled_cost, dim=-2
            )
            psi = torch.where(q > 0.0, column_potential - log_on_support(q), transform)
        return self.epsilon * (torch.sum(p * phi, dim=-1) + torch.sum(q * psi, dim=-1))
