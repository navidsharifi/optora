"""KL-divergence-constrained ambiguity set (KL-DRO)."""

import torch

from optora.core.dro_base import TiltedAmbiguitySet
from optora.divergences.kl import KLDivergence


class KLAmbiguitySet(TiltedAmbiguitySet):
    r"""KL-divergence-constrained ambiguity set for KL-DRO.

    Bounds every candidate distribution `q` by
    $D_{\mathrm{KL}}(q \,\|\, \mathrm{nominal}) \le \mathrm{radius}$. The
    KKT conditions of

    $$
    \sup_{q:\, D_{\mathrm{KL}}(q \,\|\, \mathrm{nominal}) \,\le\, \mathrm{radius}}
        \mathbb{E}_q[\mathrm{loss}]
    $$

    make the maximizer an exponential (Gibbs) tilt of the nominal
    (Hu and Hong 2013; Ben-Tal et al. 2013),

    $$
    q_\beta \propto \mathrm{nominal} \cdot e^{\beta\,\mathrm{loss}},
        \qquad \beta = 1/\eta \ge 0,
    $$

    whose divergence $D_{\mathrm{KL}}(q_\beta \,\|\, \mathrm{nominal})$
    rises monotonically in $\beta$ from $0$ to $-\log P^\star$, the
    divergence of the nominal restricted to the highest-loss scenarios,
    where $P^\star$ is the nominal mass those scenarios carry. The radius
    constraint is therefore tight at the optimum below that threshold and
    slack at or above it, and either way the tilt is pinned by one monotone
    scalar root solve rather than by an unconstrained minimization of the
    convex dual

    $$
    \inf_{\eta > 0} \; \eta \cdot \mathrm{radius}
        + \eta \log \mathbb{E}_{\mathrm{nominal}}\!\left[
            e^{\mathrm{loss}/\eta}\right].
    $$

    See `optora.core.dro_base.TiltedAmbiguitySet` for the bracketed
    bisection that runs the root solve, and `docs/formulations.md` for the
    equivalence between the two. This formulation needs no
    optimal-transport machinery, making it the simplest DRO formulation to
    build (see `progress/architecture.md`).

    Attributes:
        nominal: Reference distribution the ambiguity set is centered on.
        divergence: `KLDivergence` instance measuring distance from
            `nominal`.
        radius: Nonnegative bound on the KL divergence of any distribution
            inside the ambiguity set from `nominal`, either a float or a
            tensor of radii evaluated as one batch.
        eps: Small positive constant used to clamp `nominal` away from zero
            before taking the logarithm inside the tilt.
        log_nominal: Elementwise logarithm of the clamped `nominal`, a
            constant of the tilt cached once rather than recomputed on
            every call.
    """

    log_nominal: torch.Tensor

    def __init__(
        self,
        nominal: torch.Tensor,
        radius: float | torch.Tensor,
        eps: float = 1e-12,
        validate: bool = False,
    ) -> None:
        """Initialize the KL-DRO ambiguity set.

        Args:
            nominal: Reference distribution the ambiguity set is centered
                on, a nonnegative tensor that sums to one along its last
                dimension.
            radius: Nonnegative bound on the KL divergence of any
                distribution inside the ambiguity set from `nominal`. A
                tensor radius is broadcast against the batch shape of
                `worst_case_expectation`'s `loss`, evaluating a sweep of
                radii in one solve.
            eps: Small positive constant used to clamp `nominal` away from
                zero before taking the logarithm inside the tilt, and
                passed through to the underlying `KLDivergence`.
            validate: Whether to check that `nominal` is nonnegative and
                sums to one. The check synchronizes with the device, so it
                is opt-in and off by default.

        Raises:
            ValueError: If `radius` is a negative float, if `eps` is not
                positive, or if `validate` is set and `nominal` is not a
                valid probability distribution.
        """
        super().__init__(
            nominal=nominal,
            divergence=KLDivergence(eps=eps),
            radius=radius,
            validate=validate,
        )
        self.eps = eps
        self.register_buffer(
            "log_nominal",
            torch.log(torch.clamp(nominal, min=eps)),
            persistent=False,
        )

    def _tilted_distribution(
        self, gap: torch.Tensor, tilt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Evaluate the Gibbs tilt and its KL divergence from `nominal`.

        The tilt is applied to the gap below the maximum loss rather than
        to the loss itself. The two differ only by the constant factor
        $e^{\beta \max \ell}$, which the normalization cancels, but the gap
        form keeps the largest exponent at exactly zero, so no intermediate
        overflows however extreme $\beta$ becomes.

        The divergence is reported as $-\log
        \mathbb{E}_{\mathrm{nominal}}[e^{-\beta(\mathrm{gap} -
        \mathbb{E}_{q_\beta}[\mathrm{gap}])}]$, algebraically the same as
        $-\beta\,\mathbb{E}_{q_\beta}[\mathrm{gap}] - \log Z$ but with the
        exponent centered on the tilted mean first. The expectation then
        sits within $O(\beta^2)$ of one and is read through `expm1` and
        `log1p`, so a vanishing radius is resolved to full precision
        instead of being lost to cancellation between two nearly equal
        $O(\beta)$ terms.

        Args:
            gap: Standardized shortfall below the maximum loss, of shape
                `(..., n)`.
            tilt: Path parameter in $[0, 1)$ of shape `batch_shape`.

        Returns:
            The tilted distribution and its KL divergence from `nominal`.
        """
        rate = self._tilt_rate(tilt)
        log_weight = self.log_nominal - rate * gap
        log_normalizer = torch.logsumexp(log_weight, dim=-1, keepdim=True)
        distribution = torch.exp(log_weight - log_normalizer)
        mean_gap = torch.sum(distribution * gap, dim=-1, keepdim=True)
        centered = -rate * (gap - mean_gap)
        divergence = -torch.log1p(
            torch.sum(self.nominal * torch.expm1(centered), dim=-1)
        )
        return distribution, torch.clamp(divergence, min=0.0)
