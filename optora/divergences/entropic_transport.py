"""Entropic optimal transport cost relative to a kernel-smoothed reference."""

import torch

from optora.core.convergence import (
    DEFAULT_CHECK_INTERVAL,
    ConvergenceTracker,
    validate_check_interval,
)
from optora.core.divergence_base import Divergence
from optora.core.precision import working_dtype


def _log_on_support(weights: torch.Tensor) -> torch.Tensor:
    """Return `log(weights)`, exactly `-inf` where `weights` is zero."""
    positive = weights > 0.0
    return torch.where(
        positive,
        torch.log(torch.where(positive, weights, torch.ones_like(weights))),
        -torch.inf,
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
        \mathrm{KL}\big(\gamma \,\big\|\, \mathrm{diag}(q)\, Q\big),
    $$

    which this class computes by Sinkhorn scaling in the log domain. It is
    nonnegative and vanishes exactly at its centre $P^0$, not at `p == q`:
    entropic smoothing moves the centre of the ball off the reference.
    `optora.dro.SinkhornAmbiguitySet` bounds this quantity by its radius.

    Attributes:
        cost: Square ground cost matrix of shape `(n, n)`.
        reference: Nonnegative reference measure $\nu$ of shape `(n,)`. Only
            its normalization-free shape matters: $Q$ normalizes each row.
        epsilon: Positive entropic regularization strength.
        max_iter: Maximum number of Sinkhorn scaling iterations.
        tol: Convergence tolerance on the largest change in a row potential.
        eps: Small positive constant clamping the marginals away from zero
            before their logarithm.
        check_interval: Iterations between host reads of the convergence
            flag; the potentials freeze once converged.
    """

    cost: torch.Tensor
    reference: torch.Tensor

    def __init__(
        self,
        cost: torch.Tensor,
        epsilon: float,
        reference: torch.Tensor | None = None,
        max_iter: int = 1000,
        tol: float = 1e-9,
        eps: float = 1e-12,
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
            tol: Convergence tolerance on the largest change in a row
                potential, over the whole batched problem.
            eps: Small positive constant clamping the marginals away from
                zero before their logarithm.
            check_interval: Iterations between host reads of the
                convergence flag.
            validate: Whether to check that `cost` is finite and `reference`
                is nonnegative with positive mass. The check reads
                reductions on the host, so it is opt-in.

        Raises:
            ValueError: If `cost` is not a square 2D tensor, `reference` has
                the wrong shape, a hyperparameter is not positive, or
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
        for name, value in (("epsilon", epsilon), ("tol", tol), ("eps", eps)):
            if not value > 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {max_iter}.")
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
            with mass where $\nu$ has none is infeasible and returns a value
            of order $\epsilon \log(1/\mathrm{eps})$ or larger.

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
        log_rows = torch.log(torch.clamp(q.to(dtype), min=self.eps))
        log_columns = torch.log(torch.clamp(p.to(dtype), min=self.eps))
        log_rows, log_columns = (
            log_rows + torch.zeros_like(log_columns),
            log_columns + torch.zeros_like(log_rows),
        )
        f = torch.zeros_like(log_rows)
        g = torch.zeros_like(log_columns)
        tracker = ConvergenceTracker(self.tol, self.check_interval, f)
        for iteration in range(self.max_iter):
            f_prev = f
            f = torch.where(
                tracker.converged,
                f,
                log_rows - torch.logsumexp(g.unsqueeze(-2) + log_kernel, dim=-1),
            )
            g = torch.where(
                tracker.converged,
                g,
                log_columns - torch.logsumexp(f.unsqueeze(-1) + log_kernel, dim=-2),
            )
            tracker.update(torch.max(torch.abs(f - f_prev)))
            if tracker.should_stop(iteration):
                break
        log_ratio = f.unsqueeze(-1) + g.unsqueeze(-2) - log_rows.unsqueeze(-1)
        plan = torch.exp(f.unsqueeze(-1) + g.unsqueeze(-2) + log_kernel)
        divergence = self.epsilon * torch.sum(
            torch.where(plan > 0, plan * log_ratio, 0.0), dim=(-2, -1)
        )
        return torch.clamp(divergence, min=0.0).to(result_dtype)
