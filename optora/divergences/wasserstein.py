"""Entropy-regularized Wasserstein (Sinkhorn) divergence between distributions."""

import torch

from optora.core.convergence import (
    DEFAULT_CHECK_INTERVAL,
    ConvergenceTracker,
    validate_check_interval,
)
from optora.core.divergence_base import Divergence


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
    strength. Sinkhorn's algorithm computes the minimizing plan $\pi$ by
    alternately updating dual potentials `f` and `g` until both marginal
    constraints hold, and $\mathrm{OT}_\epsilon$ is then read off as
    $\langle \mathrm{cost}, \pi \rangle$ for the converged plan
    $\pi = \exp\!\big((f \oplus g - \mathrm{cost}) / \epsilon\big)$. The
    potential updates are computed with `torch.logsumexp` (the log-sum-exp
    trick) rather than by forming the Gibbs kernel
    $\exp(-\mathrm{cost} / \epsilon)$ directly: that kernel underflows to
    exact zero for small $\epsilon$ or large `cost`, which silently collapses
    the whole divergence to zero instead of raising an error, so it is
    avoided rather than merely guarded against with a larger $\epsilon$.

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
    Wasserstein distance induced by `cost` as $\epsilon \to 0$.
    `optora.dro.wasserstein_dro` uses this divergence to define
    Wasserstein-based ambiguity sets.

    Attributes:
        cost: Square, nonnegative pairwise ground cost matrix between the
            shared support points of `p` and `q`, shape `(n, n)`.
        epsilon: Positive entropic regularization strength; smaller values
            approximate the exact Wasserstein distance more closely at the
            cost of more Sinkhorn iterations to converge.
        max_iter: Maximum number of Sinkhorn scaling iterations.
        tol: Convergence tolerance on the change in the row dual potential
            between iterations.
        eps: Small positive constant used to clamp `p` and `q` away from
            zero before taking the logarithm, avoiding `log(0)` without
            branching.
        check_interval: Number of Sinkhorn iterations between host reads of
            the convergence flag. The potentials are frozen once converged,
            so the divergence does not depend on this interval.
    """

    cost: torch.Tensor

    def __init__(
        self,
        cost: torch.Tensor,
        epsilon: float = 0.1,
        max_iter: int = 100,
        tol: float = 1e-6,
        eps: float = 1e-12,
        check_interval: int = DEFAULT_CHECK_INTERVAL,
    ) -> None:
        """Initialize the Sinkhorn divergence.

        Args:
            cost: Square, nonnegative pairwise ground cost matrix between
                the shared support points of `p` and `q`, shape `(n, n)`.
            epsilon: Positive entropic regularization strength.
            max_iter: Maximum number of Sinkhorn scaling iterations.
            tol: Convergence tolerance on the change in the row dual
                potential between iterations.
            eps: Small positive constant used to clamp `p` and `q` away
                from zero before taking the logarithm.
            check_interval: Number of Sinkhorn iterations between host
                reads of the convergence flag. Raise it to trade redundant
                frozen iterations for fewer device synchronizations.

        Raises:
            ValueError: If `cost` is not a square 2D tensor, contains
                negative entries, or if `epsilon`, `max_iter`, `tol`,
                `eps`, or `check_interval` are not positive.
        """
        super().__init__()
        if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
            raise ValueError(
                f"cost must be a square 2D tensor, got shape {tuple(cost.shape)}."
            )
        if torch.any(cost < 0):
            raise ValueError("cost must be nonnegative.")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}.")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {max_iter}.")
        if tol <= 0:
            raise ValueError(f"tol must be positive, got {tol}.")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")
        self.register_buffer("cost", cost)
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps
        self.check_interval = validate_check_interval(check_interval)

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        r"""Compute the debiased Sinkhorn divergence of `p` from `q`.

        Args:
            p: Candidate distribution, a nonnegative tensor of shape `(n,)`
                that sums to one, indexing `cost`.
            q: Reference distribution, a nonnegative tensor of shape `(n,)`
                that sums to one, indexing `cost`.

        Returns:
            A scalar tensor holding $S_\epsilon(p, q)$, clamped to be
            nonnegative to absorb floating-point error near zero.

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
        cost_pq = self._entropic_transport_cost(p, q)
        cost_pp = self._entropic_transport_cost(p, p)
        cost_qq = self._entropic_transport_cost(q, q)
        divergence = cost_pq - 0.5 * cost_pp - 0.5 * cost_qq
        return torch.clamp(divergence, min=0.0)

    def _entropic_transport_cost(
        self, p: torch.Tensor, q: torch.Tensor
    ) -> torch.Tensor:
        r"""Compute the entropic optimal transport cost $\mathrm{OT}_\epsilon(p, q)$.

        Args:
            p: Row marginal, a nonnegative tensor of shape `(n,)` that sums
                to one.
            q: Column marginal, a nonnegative tensor of shape `(n,)` that
                sums to one.

        Returns:
            A scalar tensor holding $\langle \mathrm{cost}, \pi \rangle$ for
            the transport plan `pi` produced by Sinkhorn's algorithm.
        """
        transport_plan = self._sinkhorn(p, q)
        return torch.sum(transport_plan * self.cost)

    def _sinkhorn(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        r"""Run Sinkhorn's algorithm to compute an entropic transport plan.

        Updates the dual potentials `f` and `g` in the log domain via
        `torch.logsumexp` rather than rescaling `u = p / (kernel @ v)`
        against the raw Gibbs kernel $\exp(-\mathrm{cost} / \epsilon)$: the
        two are algebraically equivalent ($f = \epsilon \log u$,
        $g = \epsilon \log v$), but the raw kernel underflows to exact zero
        for small $\epsilon$ or large `cost`, while `logsumexp` stays
        accurate in that regime by construction.

        The stopping test on `f` runs on the potentials' device and is read
        back to the host only every `check_interval` iterations; the
        potentials are frozen once converged, so the returned plan is the
        same one a per-iteration test would produce.

        Args:
            p: Row marginal, a nonnegative tensor of shape `(n,)` that sums
                to one.
            q: Column marginal, a nonnegative tensor of shape `(n,)` that
                sums to one.

        Returns:
            The converged transport plan
            $\pi = \exp\!\big((f \oplus g - \mathrm{cost}) / \epsilon\big)$
            (computed as `exp((f.unsqueeze(-1) + g.unsqueeze(-2) - cost) /
            eps)`), a tensor of shape `(n, n)` with row sums approximating
            `p` and column sums approximating `q`.
        """
        log_p = torch.log(torch.clamp(p, min=self.eps))
        log_q = torch.log(torch.clamp(q, min=self.eps))
        f = torch.zeros_like(p)
        g = torch.zeros_like(q)
        tracker = ConvergenceTracker(self.tol, self.check_interval, f)
        for iteration in range(self.max_iter):
            f_prev = f
            f = torch.where(
                tracker.converged,
                f,
                self.epsilon
                * (
                    log_p
                    - torch.logsumexp(
                        (g.unsqueeze(-2) - self.cost) / self.epsilon, dim=-1
                    )
                ),
            )
            g = torch.where(
                tracker.converged,
                g,
                self.epsilon
                * (
                    log_q
                    - torch.logsumexp(
                        (f.unsqueeze(-1) - self.cost) / self.epsilon, dim=-2
                    )
                ),
            )
            tracker.update(torch.max(torch.abs(f - f_prev)))
            if tracker.should_stop(iteration):
                break
        return torch.exp((f.unsqueeze(-1) + g.unsqueeze(-2) - self.cost) / self.epsilon)
