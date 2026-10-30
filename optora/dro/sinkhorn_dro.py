"""Sinkhorn-distance-constrained ambiguity set (Sinkhorn DRO)."""

import math

import torch

from optora.core.convergence import DEFAULT_CHECK_INTERVAL
from optora.core.dro_base import TiltedAmbiguitySet
from optora.divergences._entropic import warn_deprecated_eps
from optora.divergences.entropic_transport import EntropicTransportDivergence

# Divergence above which a row's KL divergence is read from its direct form
# rather than its centered form, as in `optora.dro.kl_dro`.
_LARGE_DIVERGENCE = 1.0
_NEAR_EXCESS_FLOOR = math.expm1(-2.0 * _LARGE_DIVERGENCE)


class SinkhornAmbiguitySet(TiltedAmbiguitySet):
    r"""Sinkhorn-distance ball for Sinkhorn DRO (Wang, Gao and Xie 2025).

    The Sinkhorn ball around the nominal $\hat P$ with entropic strength
    $\epsilon$, reference measure $\nu$, and ground cost $C$ is
    $\{P : \mathcal{W}_\epsilon(\hat P, P) \le \rho\}$ (see
    `optora.divergences.EntropicTransportDivergence`). On a finite support
    it is a shared-budget mixture of KL balls (their Lemma 3.11):

    $$
    \sup_{P:\, \mathcal{W}_\epsilon(\hat P, P) \le \rho} \mathbb{E}_P[\ell]
    = \sup_{\gamma_1, \ldots, \gamma_n}
        \Big\{ \sum_i \hat p_i\, \mathbb{E}_{\gamma_i}[\ell]
        \;:\; \epsilon \sum_i \hat p_i\, \mathrm{KL}(\gamma_i \,\|\, Q_i)
        \le \bar\rho \Big\},
    $$

    where $Q_i \propto \nu\, e^{-C_{i\cdot}/\epsilon}$ is the Gibbs kernel
    row of support point $i$ and $\bar\rho = \rho - \min_P
    \mathcal{W}_\epsilon(\hat P, P)$ is the budget in excess of the
    entropic bias. `radius` is $\bar\rho$, so every nonnegative radius is
    feasible and radius zero is the smallest non-empty set: the single
    kernel-smoothed nominal $P^0 = \hat p^\top Q$, not $\hat P$ itself.

    Every row's maximizer is an exponential tilt
    $\gamma_i \propto Q_i\, e^{\ell/(\lambda \epsilon)}$ at one shared
    temperature (their Remark 3.3), so the worst case is pinned by the
    bracketed tilt bisection of `optora.core.TiltedAmbiguitySet`, exactly
    as for `optora.dro.KLAmbiguitySet`: solver-free, synchronization-free,
    exact first and second derivatives. The worst-case distribution
    $\sum_i \hat p_i \gamma_i$ is supported on the support of $\nu$, not
    only of $\hat P$, and the worst case saturates at the largest loss on
    that support once
    $\bar\rho \ge \epsilon \sum_i \hat p_i \log(1 / Q_i(A))$, with $A$ the
    highest-loss scenarios. With a zero cost every row is $\nu$ and the set
    is the KL ball of radius $\bar\rho/\epsilon$ around $\nu$; as
    $\epsilon \to 0$ the dual tends to the Wasserstein-DRO dual (their
    Remark 3.2).

    Attributes:
        nominal: Nominal distribution $\hat P$ the transport starts from.
        divergence: `EntropicTransportDivergence` holding `cost`,
            `reference`, and `epsilon`; `contains` uses it.
        radius: Nonnegative budget $\bar\rho$ in excess of the entropic
            bias, a float or a tensor of radii evaluated as one batch.
    """

    divergence: EntropicTransportDivergence
    log_kernel: torch.Tensor

    def __init__(
        self,
        nominal: torch.Tensor,
        cost: torch.Tensor,
        epsilon: float,
        radius: float | torch.Tensor,
        reference: torch.Tensor | None = None,
        max_iter: int = 1000,
        tol: float = 1e-9,
        eps: float | None = None,
        check_interval: int = DEFAULT_CHECK_INTERVAL,
        validate: bool = False,
    ) -> None:
        """Initialize the Sinkhorn DRO ambiguity set.

        Args:
            nominal: Nominal distribution, a nonnegative tensor that sums to
                one along its last dimension.
            cost: Square ground cost matrix of shape `(n, n)` between the
                support points.
            epsilon: Positive entropic regularization strength.
            radius: Nonnegative budget in excess of the entropic bias. A
                tensor radius is broadcast against the batch shape of
                `worst_case_expectation`'s `loss`.
            reference: Nonnegative reference measure $\\nu$ of shape `(n,)`
                that the worst case may move mass onto. Defaults to the
                counting measure on the whole support.
            max_iter: Sinkhorn iterations for `contains` only; the worst
                case does not iterate.
            tol: Sinkhorn tolerance for `contains` only.
            eps: Deprecated and ignored, as in
                `EntropicTransportDivergence`; passing it emits a
                `DeprecationWarning`.
            check_interval: Sinkhorn host-read interval for `contains` only.
            validate: Whether to run the value checks that synchronize with
                the device on `nominal`, a tensor `radius`, `cost`, and
                `reference`.

        Raises:
            ValueError: If `cost` does not match `nominal`'s support, a
                hyperparameter is invalid, `radius` is a negative float, or
                `validate` is set and an input is invalid.
        """
        if nominal.ndim > 0 and (cost.ndim != 2 or cost.shape[0] != nominal.shape[-1]):
            raise ValueError(
                f"cost must have shape ({nominal.shape[-1]}, {nominal.shape[-1]}) "
                f"to match nominal, got {tuple(cost.shape)}."
            )
        if eps is not None:
            warn_deprecated_eps(stacklevel=2)
        super().__init__(
            nominal=nominal,
            divergence=EntropicTransportDivergence(
                cost,
                epsilon,
                reference=reference,
                max_iter=max_iter,
                tol=tol,
                check_interval=check_interval,
                validate=validate,
            ),
            radius=radius,
            validate=validate,
        )
        self.register_buffer(
            "log_kernel", self.divergence.log_kernel(cost.dtype), persistent=False
        )

    @property
    def cost(self) -> torch.Tensor:
        """Ground cost matrix, aliasing `divergence.cost`."""
        return self.divergence.cost

    @property
    def epsilon(self) -> float:
        """Entropic regularization strength, aliasing `divergence.epsilon`."""
        return self.divergence.epsilon

    @property
    def _reachable(self) -> torch.Tensor:
        """Scenarios the worst case can reach: the support of the reference."""
        return self.divergence.reference > 0.0

    def _kernel(self, dtype: torch.dtype) -> torch.Tensor:
        """Return `log_kernel` in `dtype`, recomputing it rather than rounding it."""
        if self.log_kernel.dtype == dtype:
            return self.log_kernel
        return self.divergence.log_kernel(dtype)

    def smoothed_nominal(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        r"""Return the kernel-smoothed nominal $P^0 = \hat p^\top Q$, the ball's centre.

        Args:
            dtype: Dtype to compute in; defaults to `nominal`'s.

        Returns:
            The distribution attained at radius zero, of `nominal`'s shape.
        """
        dtype = self.nominal.dtype if dtype is None else dtype
        kernel = torch.exp(self._kernel(dtype))
        return torch.sum(self.nominal.to(dtype).unsqueeze(-1) * kernel, dim=-2)

    def _center_expectation(
        self, loss: torch.Tensor, batch_shape: torch.Size
    ) -> torch.Tensor:
        """Return the expectation under the smoothed nominal, the zero-radius value."""
        center = self.smoothed_nominal(loss.dtype)
        return torch.sum(center * loss, dim=-1).expand(batch_shape)

    def _tilt_entries(self, gap: torch.Tensor, batch_shape: torch.Size) -> int:
        """Return the entries of one tilt: every kernel row, `n` times the default."""
        return super()._tilt_entries(gap, batch_shape) * gap.shape[-1]

    def _tilted_distribution(
        self, gap: torch.Tensor, tilt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Tilt every kernel row at one rate and return the mixture and its budget.

        Row $i$ becomes $\gamma_i \propto Q_i e^{-\beta\,\mathrm{gap}}$ and
        the returned distribution is $\sum_i \hat p_i \gamma_i$. The budget
        spent is $\epsilon \sum_i \hat p_i \mathrm{KL}(\gamma_i \,\|\, Q_i)$,
        each row's divergence read from the centered or the direct form
        exactly as in `optora.dro.KLAmbiguitySet`, so a vanishing radius
        keeps full precision and a large one never reports a spurious
        infinity. Rows with zero nominal mass are excluded from the budget
        explicitly, since their divergence is irrelevant and may be
        infinite.

        Args:
            gap: Standardized shortfall below the maximum reachable loss, of
                shape `(..., n)`.
            tilt: Path parameter in $[0, 1)$ of shape `batch_shape`.

        Returns:
            The mixture distribution, of shape `batch_shape + (n,)`, and the
            budget it spends, of shape `batch_shape`.
        """
        rows, row_divergence = self._tilted_rows(gap, tilt)
        nominal = self._constant_nominal.to(gap.dtype)
        weighted = torch.where(nominal > 0.0, nominal * row_divergence, 0.0)
        budget = self.epsilon * torch.sum(weighted, dim=-1)
        distribution = torch.sum(nominal.unsqueeze(-1) * rows, dim=-2)
        return distribution, budget

    def _standardized_sensitivities(
        self,
        gap: torch.Tensor,
        tilt: torch.Tensor,
        distribution: torch.Tensor,
        divergence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Return the envelope derivatives of the row-wise tilt.

        The radius constraint
        $\epsilon \sum_i \hat p_i \mathrm{KL}(\gamma_i \,\|\, Q_i) \le
        \rho$ has the multiplier $\lambda^\star = 1 / (\epsilon \beta)$, and
        the Lagrangian is linear in $\hat p$, so the derivative with respect
        to $\hat p_i$ is row $i$'s own contribution,
        $\mathbb{E}_{\gamma_i}[-\mathrm{gap}] - \lambda^\star \epsilon\,
        \mathrm{KL}(\gamma_i \,\|\, Q_i)$, centered on its nominal mean.

        Args:
            gap: Standardized shortfall of every scenario, of shape
                `(..., n)`.
            tilt: Solved tilt of shape `batch_shape`.
            distribution: Maximizer at `tilt`.
            divergence: Budget spent at `tilt`.

        Returns:
            The derivative with respect to `nominal` and the multiplier
            $\lambda^\star$, infinite at a zero tilt.
        """
        # Unreachable columns carry no mass whatever their gap, and an
        # extreme gap there would meet their infinite log-kernel as inf - inf.
        gap = torch.where(self._reachable, gap, 1.0)
        rows, row_divergence = self._tilted_rows(gap, tilt)
        rate = self._tilt_rate(tilt)
        row_gap = torch.sum(rows * gap.unsqueeze(-2), dim=-1)
        mean_gap = torch.sum(distribution * gap, dim=-1, keepdim=True)
        excess_divergence = row_divergence - (divergence / self.epsilon).unsqueeze(-1)
        positive = rate > 0.0
        divergence_term = torch.where(
            positive, excess_divergence / torch.where(positive, rate, 1.0), 0.0
        )
        multiplier = torch.reciprocal(self.epsilon * rate.squeeze(-1))
        return mean_gap - row_gap - divergence_term, multiplier

    def _tilted_rows(
        self, gap: torch.Tensor, tilt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tilt every kernel row at one rate.

        Args:
            gap: Standardized shortfall below the maximum reachable loss, of
                shape `(..., n)`.
            tilt: Path parameter in $[0, 1)$ of shape `batch_shape`.

        Returns:
            The tilted rows, of shape `batch_shape + (n, n)`, and each row's
            KL divergence from its kernel row, of shape `batch_shape + (n,)`.
        """
        log_kernel = self._kernel(gap.dtype)
        rate = self._tilt_rate(tilt).unsqueeze(-1)
        column_gap = gap.unsqueeze(-2)
        log_weight = log_kernel - rate * column_gap
        log_normalizer = torch.logsumexp(log_weight, dim=-1, keepdim=True)
        rows = torch.exp(log_weight - log_normalizer)
        mean_gap = torch.sum(rows * column_gap, dim=-1, keepdim=True)
        largest_exponent = rate * mean_gap
        exponent_limit = 0.5 * math.log(torch.finfo(gap.dtype).max)
        centered = torch.clamp(largest_exponent - rate * column_gap, max=exponent_limit)
        excess = torch.sum(torch.exp(log_kernel) * torch.expm1(centered), dim=-1)
        near = -torch.log1p(torch.clamp(excess, min=_NEAR_EXCESS_FLOOR))
        far = -(largest_exponent + log_normalizer).squeeze(-1)
        use_far = (far > _LARGE_DIVERGENCE) | (
            largest_exponent.squeeze(-1) > exponent_limit
        )
        row_divergence = torch.clamp(torch.where(use_far, far, near), min=0.0)
        return rows, row_divergence
