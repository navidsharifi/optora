"""Wasserstein-distance-constrained ambiguity set (Wasserstein-DRO)."""

import math
from typing import cast

import torch

from optora.core.dro_base import AmbiguitySet
from optora.divergences.wasserstein import SinkhornDivergence

_BISECTION_HEADROOM = 2


def _bisection_steps(dtype: torch.dtype) -> int:
    """Return the number of bisection steps that exhaust a dtype's precision.

    Each step halves the bracket, so `-log2(eps)` steps shrink it from its
    initial width to that width times the dtype's unit roundoff, after which
    further steps cannot refine the minimizer.

    Args:
        dtype: Floating-point dtype the bisection runs in.

    Returns:
        The number of halvings to perform, with a small headroom.
    """
    return math.ceil(-math.log2(torch.finfo(dtype).eps)) + _BISECTION_HEADROOM


class WassersteinAmbiguitySet(AmbiguitySet):
    r"""Wasserstein-distance-constrained ambiguity set for Wasserstein-DRO.

    Bounds every candidate distribution `q`, sharing `nominal`'s finite
    support with pairwise ground cost `cost`, by the type-1 Wasserstein
    distance $W_c(q, \mathrm{nominal}) \le \mathrm{radius}$, where $W_c$ is
    the optimal-transport cost of moving `nominal` to `q` under `cost`. The
    worst-case expected loss over this set admits an exact strong-duality
    reformulation (Mohajerin Esfahani and Kuhn 2018; Blanchet and Murthy
    2019; Gao and Kleywegt 2022), specialized to a finite shared support:

    $$
    \sup_{q:\, W_c(q, \mathrm{nominal}) \,\le\, \mathrm{radius}}
        \mathbb{E}_q[\mathrm{loss}]
    = \inf_{\gamma \ge 0} \;
        \gamma \cdot \mathrm{radius}
        + \mathbb{E}_{\mathrm{nominal}}\!\left[
            \max_j \big(\mathrm{loss}_j - \gamma \cdot \mathrm{cost}(\cdot, j)\big)
        \right]
    $$

    This follows from LP duality on the transportation polytope: the
    primal is

    $$
    \max_{\pi \ge 0} \sum_{ij} \pi_{ij}\, \mathrm{loss}_j
    \quad \text{s.t.} \quad
    \sum_j \pi_{ij} = \mathrm{nominal}_i, \qquad
    \sum_{ij} \pi_{ij}\, \mathrm{cost}_{ij} \le \mathrm{radius}
    $$

    (the column marginal, which defines the candidate `q`, is left free).
    Dualizing the budget constraint with multiplier $\gamma \ge 0$ and the
    row constraints with free multipliers, and maximizing out $\pi$
    pointwise, yields exactly the one-dimensional convex dual above.

    Unlike `optora.dro.KLAmbiguitySet` and `optora.dro.PhiAmbiguitySet`,
    this dual is not handed to an injected first-order solver. It is a
    nonnegative combination of maxima of affine functions of the scalar
    $\gamma$, hence convex **piecewise linear**: its derivative is
    piecewise constant and jumps across the minimizer, so a fixed-step
    gradient method oscillates around the kink and never satisfies a
    gradient-norm stopping test. What the derivative does do is increase
    monotonically,

    $$
    \frac{\partial}{\partial \gamma}\Big(
        \gamma \cdot \mathrm{radius}
        + \sum_i \mathrm{nominal}_i \max_j
            (\mathrm{loss}_j - \gamma\,\mathrm{cost}_{ij})\Big)
    = \mathrm{radius}
        - \sum_i \mathrm{nominal}_i\, \mathrm{cost}_{i, j^\star(i, \gamma)},
    $$

    where $j^\star(i, \gamma)$ is the maximizing column of row $i$, so the
    minimizer is located by bisecting that derivative's sign change. The
    search starts from the exact bracket $[0, \Gamma]$, where

    $$
    \Gamma = \max_i \; \max_{k:\, \mathrm{cost}_{ik} > m_i}
        \frac{\mathrm{loss}_k - \mathrm{loss}_{j_0(i)}}
             {\mathrm{cost}_{ik} - m_i},
    \qquad m_i = \min_j \mathrm{cost}_{ij},
    $$

    with $j_0(i)$ the highest-loss column attaining $m_i$: beyond $\Gamma$
    every row is maximized at a cheapest column, the derivative is frozen
    at $\mathrm{radius} - \sum_i \mathrm{nominal}_i m_i \ge 0$, and the
    minimizer cannot lie further right. Bisection therefore reaches the
    dual optimum to working precision in a fixed, dtype-determined number
    of steps, with no step size, iteration budget, or tolerance to tune --
    the same treatment `optora.dro.TotalVariationAmbiguitySet` gets for its
    own exactly solvable inner problem. It also removes the need to
    reparameterize $\gamma$: the constraint $\gamma \ge 0$ is the left end
    of the bracket, and the optimum is genuinely attained at $\gamma = 0$
    once `radius` is large enough to move all nominal mass onto the single
    highest-loss support point.

    The bracket assumes the ambiguity set is nonempty, that is
    $\mathrm{radius} \ge \sum_i \mathrm{nominal}_i m_i$, the smallest
    transport cost at which any candidate distribution can be reached. That
    holds automatically for a ground cost with a zero diagonal, the
    standard choice, for which $m_i = 0$.

    Because $\gamma^\star$ is a kink, the worst-case expectation is then
    read off the *primal* solution rather than the dual value. At a kink
    several columns tie for a row's inner maximum, and re-evaluating the
    dual objective there would differentiate through that tie with
    arbitrary weights, giving a transport plan that does not spend the
    radius exactly and therefore a wrong gradient with respect to `loss`.
    The two ends of the final bracket instead name the two extreme
    tie-breaks explicitly: the plan that is still worth paying for just
    below $\gamma^\star$ overspends the radius, the one that takes over
    just above it underspends, and the unique convex combination of the two
    that spends exactly `radius` is the worst-case distribution
    $q^\star$. Complementary slackness then makes
    $\mathbb{E}_{q^\star}[\mathrm{loss}]$ equal to the dual minimum, and
    since $q^\star$ carries no autograd graph, differentiating that
    expectation gives exactly the envelope-theorem gradient with respect to
    `loss`. A minimizer at $\gamma^\star = 0$ is the degenerate case where
    both ends agree and $q^\star$ simply moves all mass to the highest-loss
    support point.

    `divergence` is fixed to a `SinkhornDivergence` over `cost` (the only
    Wasserstein-type divergence implemented in `optora.divergences`), so
    `contains(...)` checks the entropy-regularized Sinkhorn divergence as a
    fast, differentiable approximation of exact Wasserstein-ball
    membership. `worst_case_expectation` does not depend on this
    approximation: it solves the exact (non-regularized) dual above
    directly from `cost`, since that dual has a clean closed form and does
    not need entropic relaxation for tractability.

    Attributes:
        nominal: Reference distribution the ambiguity set is centered on.
        divergence: `SinkhornDivergence` instance approximating distance
            from `nominal` for `contains(...)`.
        radius: Nonnegative bound on the Wasserstein distance of any
            distribution inside the ambiguity set from `nominal`, either a
            float or a tensor of radii evaluated as one batch.
        cost: Square, nonnegative pairwise ground cost matrix between the
            shared support points of `nominal` and any candidate
            distribution. Aliases `divergence.cost`; not a separate buffer.
    """

    def __init__(
        self,
        nominal: torch.Tensor,
        cost: torch.Tensor,
        radius: float | torch.Tensor,
        epsilon: float = 0.1,
        sinkhorn_max_iter: int = 100,
        sinkhorn_tol: float = 1e-6,
        eps: float = 1e-12,
        validate: bool = False,
    ) -> None:
        """Initialize the Wasserstein-DRO ambiguity set.

        Args:
            nominal: Reference distribution the ambiguity set is centered
                on, a nonnegative tensor that sums to one along its last
                dimension.
            cost: Square, nonnegative pairwise ground cost matrix between
                the shared support points of `nominal` and any candidate
                distribution, shape `(n, n)` where `n = nominal.shape[-1]`.
            radius: Nonnegative bound on the Wasserstein distance of any
                distribution inside the ambiguity set from `nominal`. A
                tensor radius is broadcast against the batch shape of
                `worst_case_expectation`'s `loss`, evaluating a sweep of
                radii in one solve.
            epsilon: Positive entropic regularization strength for the
                `SinkhornDivergence` used by `contains(...)`.
            sinkhorn_max_iter: Maximum number of Sinkhorn scaling
                iterations, passed through to `SinkhornDivergence`.
            sinkhorn_tol: Sinkhorn scaling convergence tolerance, passed
                through to `SinkhornDivergence`.
            eps: Small positive constant used to clamp Sinkhorn scaling
                denominators away from zero, passed through to
                `SinkhornDivergence`.
            validate: Whether to check that `cost` is nonnegative, passed
                through to `SinkhornDivergence`, and that `nominal` is a
                valid probability distribution. The checks read reductions
                over `cost` and `nominal` on the host, which blocks until
                the device has produced them, so they are opt-in and off by
                default to keep construction asynchronous.

        Raises:
            ValueError: If `radius` is a negative float, if `cost` is not a
                square 2D tensor, if `cost`'s size does not match
                `nominal`'s support size, or if `validate` is set and
                `cost` contains negative entries or `nominal` is not a
                valid probability distribution.
        """
        # A scalar nominal has no support size to compare against; let
        # AmbiguitySet reject it with its own message.
        if nominal.ndim > 0 and cost.shape[-1] != nominal.shape[-1]:
            raise ValueError(
                "cost must have shape (n, n) matching nominal's support "
                f"size {nominal.shape[-1]}, got cost shape {tuple(cost.shape)}."
            )
        super().__init__(
            nominal=nominal,
            divergence=SinkhornDivergence(
                cost=cost,
                epsilon=epsilon,
                max_iter=sinkhorn_max_iter,
                tol=sinkhorn_tol,
                eps=eps,
                validate=validate,
            ),
            radius=radius,
            validate=validate,
        )

    @property
    def cost(self) -> torch.Tensor:
        """Ground cost matrix, aliasing `divergence.cost` (no separate buffer)."""
        # `divergence` is declared as the base `Divergence` type, but this
        # class always constructs it as a `SinkhornDivergence` above.
        return cast(SinkhornDivergence, self.divergence).cost

    def _maximizing_columns(
        self, loss: torch.Tensor, gamma: torch.Tensor, prefer_costly: bool = False
    ) -> torch.Tensor:
        """Return the column maximizing each support point's inner max.

        Args:
            loss: Per-scenario loss values of shape `(..., n)`.
            gamma: Nonnegative dual multiplier of shape `batch_shape`.
            prefer_costly: Whether to settle ties on the costliest tied
                column instead of the cheapest. Ties are exactly the
                multipliers at which the transport plan changes, so the two
                settings recover the plan just below and just above such a
                multiplier; picking arbitrarily among them would leave the
                plan undefined precisely where the dual is minimized.

        Returns:
            `argmax_j (loss_j - gamma * cost_ij)` for every row `i`, of shape
            `batch_shape + (n, 1)`: the support point each unit of nominal
            mass is transported to at this multiplier.
        """
        shifted = loss.unsqueeze(-2) - gamma[..., None, None] * self.cost
        tied = shifted == torch.amax(shifted, dim=-1, keepdim=True)
        preference = self.cost if prefer_costly else -self.cost
        return torch.argmax(
            torch.where(tied, preference, torch.full_like(shifted, -torch.inf)),
            dim=-1,
            keepdim=True,
        )

    def _transported(self, values: torch.Tensor, columns: torch.Tensor) -> torch.Tensor:
        """Average `values[i, columns_i]` over `nominal`.

        Args:
            values: Tensor broadcastable to `batch_shape + (n, n)`, either
                the ground cost or `loss.unsqueeze(-2)`.
            columns: Destination column per row, as returned by
                `_maximizing_columns`.

        Returns:
            `sum_i nominal_i * values[i, columns_i]`, of shape `batch_shape`:
            the transport cost of the implied plan when `values` is `cost`,
            and the expected loss under the distribution it induces when
            `values` is the loss.
        """
        expanded = values.expand(columns.shape[:-1] + values.shape[-1:])
        selected = torch.gather(expanded, -1, columns).squeeze(-1)
        return torch.sum(self.nominal * selected, dim=-1)

    def _dual_derivative(self, loss: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """Evaluate a subderivative of the dual objective at a multiplier.

        Args:
            loss: Per-scenario loss values of shape `(..., n)`.
            gamma: Nonnegative dual multiplier of shape `batch_shape`.

        Returns:
            `radius - sum_i nominal_i * cost[i, argmax_j(loss_j - gamma *
            cost_ij)]`, of shape `batch_shape`: the radius left unspent by
            the cheapest transport plan this multiplier prices. Settling
            ties on the cheapest column makes this the dual's right
            derivative, which is negative exactly to the left of the
            minimizer -- the sign test the bisection needs.
        """
        return self.radius - self._transported(
            self.cost, self._maximizing_columns(loss, gamma)
        )

    def _dual_upper_bound(self, loss: torch.Tensor) -> torch.Tensor:
        """Return a multiplier at or beyond which the dual stops decreasing.

        Past this value every row of the dual's inner maximization is
        attained at one of that row's cheapest columns, so the derivative is
        frozen at its largest value and the minimizer lies to the left.

        Args:
            loss: Per-scenario loss values of shape `(..., n)`.

        Returns:
            The right end of a bracket containing the dual minimizer, of
            shape `loss.shape[:-1]`.
        """
        cost = self.cost
        excess_cost = cost - torch.amin(cost, dim=-1, keepdim=True)
        is_costlier = excess_cost > 0.0
        losses = loss.unsqueeze(-2)
        # Highest-loss column among each row's cheapest ones: the column that
        # eventually wins that row, so the one every other column must cross.
        cheapest_loss = torch.amax(
            torch.where(is_costlier, torch.full_like(excess_cost, -torch.inf), losses),
            dim=-1,
            keepdim=True,
        )
        crossing = (losses - cheapest_loss) / torch.where(
            is_costlier, excess_cost, torch.ones_like(excess_cost)
        )
        return torch.clamp(
            torch.amax(
                torch.where(is_costlier, crossing, torch.zeros_like(crossing)),
                dim=(-2, -1),
            ),
            min=0.0,
        )

    def _dual_bracket(
        self, loss: torch.Tensor, batch_shape: torch.Size
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bisect the monotone dual derivative down onto the dual minimizer.

        Args:
            loss: Per-scenario loss values of shape `(..., n)`, detached.
            batch_shape: Batch shape of the loss being evaluated, as
                returned by `AmbiguitySet._batch_shape`.

        Returns:
            The two ends of the final bracket, each of shape `batch_shape`
            and carrying no autograd graph: the largest multiplier found at
            which the dual is still decreasing and the smallest at which it
            is not. They straddle the minimizer to the working precision,
            and their transport plans are the two that the worst-case
            distribution mixes. The left end stays exactly zero whenever the
            minimizer sits on the constraint boundary.
        """
        with torch.no_grad():
            upper = torch.broadcast_to(self._dual_upper_bound(loss), batch_shape)
            lower = torch.zeros_like(upper)
            for _ in range(_bisection_steps(upper.dtype)):
                midpoint = 0.5 * (lower + upper)
                decreasing = self._dual_derivative(loss, midpoint) < 0.0
                lower = torch.where(decreasing, midpoint, lower)
                upper = torch.where(decreasing, upper, midpoint)
            return lower, upper

    def worst_case_expectation(self, loss: torch.Tensor) -> torch.Tensor:
        """Compute the worst-case expected loss over the Wasserstein ambiguity set.

        Args:
            loss: Per-scenario loss values of shape `(..., n)`, one trailing
                entry per element of `nominal`'s support. Leading dimensions
                are a batch of independent loss vectors, each bisected to its
                own dual multiplier `gamma` in one vectorized search.

        Returns:
            A tensor of shape `(...)` holding the worst-case expected loss:
            the exact `sum(nominal * loss)` when `radius` is zero (the
            ambiguity set then contains only `nominal`), otherwise the
            expectation of `loss` under the worst-case distribution
            recovered from the dual optimum.

        Raises:
            ValueError: If `loss`'s trailing dimension does not match
                `nominal`'s support size, or its batch shape does not
                broadcast against `nominal` and `radius`.
        """
        batch_shape = self._batch_shape(loss)
        if self._radius_is_zero:
            return self._nominal_expectation(loss, batch_shape)

        detached = loss.detach()
        lower, upper = self._dual_bracket(detached, batch_shape)
        with torch.no_grad():
            eager = self._maximizing_columns(detached, lower, prefer_costly=True)
            thrifty = self._maximizing_columns(detached, upper)
            underspend = self.radius - self._transported(self.cost, thrifty)
            spread = self._transported(self.cost, eager) - self._transported(
                self.cost, thrifty
            )
            mixes = spread > 0.0
            weight = torch.clamp(
                torch.where(
                    mixes,
                    underspend / torch.where(mixes, spread, torch.ones_like(spread)),
                    torch.zeros_like(spread),
                ),
                min=0.0,
                max=1.0,
            )
        losses = loss.unsqueeze(-2)
        return weight * self._transported(losses, eager) + (
            1.0 - weight
        ) * self._transported(losses, thrifty)
