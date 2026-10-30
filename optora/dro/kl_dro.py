"""KL-divergence-constrained ambiguity set (KL-DRO)."""

import math

import torch

from optora.core.dro_base import TiltedAmbiguitySet
from optora.divergences.kl import KLDivergence

# Divergence above which the tilt's KL divergence is read from its direct
# form rather than its centered form. Both are accurate to a few ulps here.
_LARGE_DIVERGENCE = 1.0

# Lower bound on the centered form's excess, `exp(-D) - 1`, at twice the
# switching divergence: below `_LARGE_DIVERGENCE` it never binds.
_NEAR_EXCESS_FLOOR = math.expm1(-2.0 * _LARGE_DIVERGENCE)


def _log_on_support(nominal: torch.Tensor) -> torch.Tensor:
    """Return `log(nominal)`, exactly `-inf` where `nominal` is zero.

    Args:
        nominal: Reference distribution.

    Returns:
        The elementwise logarithm of `nominal`, taken without ever
        evaluating `log(0)`, so the graph holds no infinite derivative.
    """
    on_support = nominal > 0.0
    return torch.where(
        on_support,
        torch.log(torch.where(on_support, nominal, torch.ones_like(nominal))),
        -torch.inf,
    )


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
        eps: Small positive constant passed through to the underlying
            `KLDivergence`, which clamps its arguments away from zero.
        log_nominal: Elementwise logarithm of `nominal`, a constant of the
            tilt cached once rather than recomputed on every call. It is
            exactly $-\infty$ on scenarios with zero nominal mass, so the
            tilt never moves mass off the nominal's support, where the KL
            divergence is infinite.
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
            eps: Small positive constant passed through to the underlying
                `KLDivergence`, used by `contains(...)`. The tilt itself
                takes the exact logarithm of `nominal` and needs no
                clamping.
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
            "log_nominal", _log_on_support(nominal.detach()), persistent=False
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

        The divergence has two algebraically equal forms, each accurate in
        one regime, and the one suited to the regime is returned:

        - **Small divergence.** $-\log
          \mathbb{E}_{\mathrm{nominal}}[e^{c}]$ with the exponent
          $c = -\beta(\mathrm{gap} - \mathbb{E}_{q_\beta}[\mathrm{gap}])$
          centered on the tilted mean. The expectation then sits within
          $O(\beta^2)$ of one and is read through `expm1` and `log1p`, so a
          vanishing radius is resolved to full precision instead of being
          lost to cancellation between two nearly equal $O(\beta)$ terms.
        - **Large divergence.** $-\beta\,\mathbb{E}_{q_\beta}[\mathrm{gap}]
          - \log Z$, read off the `logsumexp` normalizer at no extra cost.
          The centered form represents $e^{-D}$ as one plus a sum near
          $-1$, which rounds to exactly $-1$ once $D$ exceeds about
          $-\log \varepsilon$ (16 in float32, 36 in float64) and would
          report an infinite divergence for a feasible tilt.

        The direct form also takes over when the largest centered exponent,
        $\beta\,\mathbb{E}_{q_\beta}[\mathrm{gap}]$ (the gap is zero on the
        highest loss), passes half the dtype's logarithmic range. Every
        product $\mathrm{nominal}_i e^{c_i} = q_i e^{-D}$ is at most one,
        but $e^{c_i}$ alone would overflow. That needs a highest-loss mass
        below $e^{-44}$ in float32, where the direct form is still accurate
        to the tilt's own resolution.

        The tilt stays in the log domain, normalized by `logsumexp`, so a
        scenario whose nominal mass and tilt weight would underflow as a
        product still gets its correct, representable tilted mass. The
        cached `log_nominal` is used when it is already in `gap`'s working
        precision; a half-precision set instead takes the logarithm of its
        nominal in that precision, since a logarithm rounded to a
        half-precision significand would perturb every tilted weight.

        Args:
            gap: Standardized shortfall below the maximum loss, of shape
                `(..., n)`.
            tilt: Path parameter in $[0, 1)$ of shape `batch_shape`.

        Returns:
            The tilted distribution and its KL divergence from `nominal`.
        """
        nominal = self._constant_nominal
        log_nominal = self.log_nominal
        if log_nominal.dtype != gap.dtype or self._nominal_is_attached:
            log_nominal = _log_on_support(nominal.to(gap.dtype))
        rate = self._tilt_rate(tilt)
        log_weight = log_nominal - rate * gap
        log_normalizer = torch.logsumexp(log_weight, dim=-1, keepdim=True)
        distribution = torch.exp(log_weight - log_normalizer)
        mean_gap = torch.sum(distribution * gap, dim=-1, keepdim=True)
        largest_exponent = rate * mean_gap
        exponent_limit = 0.5 * math.log(torch.finfo(gap.dtype).max)
        # Both clamps only touch the regime where `near` is discarded; they
        # keep that branch and its derivative finite.
        centered = torch.clamp(largest_exponent - rate * gap, max=exponent_limit)
        excess = torch.sum(nominal * torch.expm1(centered), dim=-1)
        near = -torch.log1p(torch.clamp(excess, min=_NEAR_EXCESS_FLOOR))
        far = -(largest_exponent + log_normalizer).squeeze(-1)
        use_far = (far > _LARGE_DIVERGENCE) | (
            largest_exponent.squeeze(-1) > exponent_limit
        )
        divergence = torch.where(use_far, far, near)
        return distribution, torch.clamp(divergence, min=0.0)

    def _standardized_sensitivities(
        self,
        gap: torch.Tensor,
        tilt: torch.Tensor,
        distribution: torch.Tensor,
        divergence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Return the envelope derivatives of the Gibbs tilt.

        The multiplier of the radius constraint is the temperature
        $\eta^\star = 1 / \beta$, and the derivative with respect to
        $\mathrm{nominal}_i$ is $\eta^\star (r_i - 1)$ for the likelihood
        ratio $r_i = q^\star_i / \mathrm{nominal}_i$. The ratio is read off
        the tilt as $\log r_i = \beta (\mathbb{E}_{q^\star}[\mathrm{gap}]
        - \mathrm{gap}_i) + D$, which also covers scenarios with zero
        nominal mass, and $(r_i - 1) / \beta$ is evaluated through `expm1`
        so a vanishing rate keeps its limit,
        $\mathbb{E}_{\mathrm{nominal}}[\mathrm{gap}] - \mathrm{gap}_i$.

        Args:
            gap: Standardized shortfall of every scenario, of shape
                `(..., n)`.
            tilt: Solved tilt of shape `batch_shape`.
            distribution: Maximizer at `tilt`.
            divergence: KL divergence of `distribution` from `nominal`.

        Returns:
            The derivative with respect to `nominal` and the multiplier
            $\eta^\star$, infinite at a zero tilt.
        """
        rate = self._tilt_rate(tilt)
        mean_gap = torch.sum(distribution * gap, dim=-1, keepdim=True)
        log_ratio = rate * (mean_gap - gap) + divergence.unsqueeze(-1)
        positive = rate > 0.0
        nominal_gradient = torch.where(
            positive,
            torch.expm1(log_ratio) / torch.where(positive, rate, 1.0),
            mean_gap - gap,
        )
        return nominal_gradient, torch.reciprocal(rate.squeeze(-1))
