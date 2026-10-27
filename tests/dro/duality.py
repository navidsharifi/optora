r"""Dual objectives and primal divergences for checking strong duality.

Every ambiguity set in `optora.dro` evaluates

$$
\sup_{q \in \mathcal{U}} \; \mathbb{E}_q[\mathrm{loss}]
$$

as the expectation of `loss` under a feasible distribution, so its value
is a *primal* objective value. This module provides the other side of the
duality gap, written in NumPy and SciPy without importing optora:

- a convex dual objective per set, evaluated at an arbitrary dual point,
  which by weak duality bounds the worst case from above wherever it is
  evaluated;
- the minimum of that dual, which by strong duality equals the worst case;
- the primal divergence of a candidate distribution from the nominal, so a
  reported maximizer can be checked for feasibility.

The total-variation, CVaR, and Wasserstein duals are convex and piecewise
linear,
so their minima are found exactly by evaluating every breakpoint. The KL
and chi-square duals are smooth and convex in one scalar, and are minimized
with a bounded Brent search over a logarithmic reparameterization that
reaches both the vanishing-radius regime, where the minimizer runs off to
infinity, and the saturated regime, where it is approached only in the
limit.
"""

from collections.abc import Callable
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog, minimize_scalar
from scipy.special import logsumexp

Array = NDArray[np.float64]

# Half-width, in natural-log units of the loss spread, of the interval the
# smooth duals are searched over. `exp(-40)` is far below every resolvable
# scale and `exp(40)` far above every minimizer a radius down to `1e-14`
# produces, so the boundary of the search never truncates the infimum by
# more than roundoff.
_LOG_SEARCH_HALF_WIDTH: Final = 40.0
_SEARCH_TOLERANCE: Final = 1e-12

_LP_OPTIONS: Final = {
    "primal_feasibility_tolerance": 1e-10,
    "dual_feasibility_tolerance": 1e-10,
}


def _positive_support(nominal: Array, loss: Array) -> tuple[Array, Array]:
    """Restrict `nominal` and `loss` to the scenarios with positive nominal mass.

    Args:
        nominal: Reference distribution of shape `(n,)`.
        loss: Per-scenario losses of shape `(n,)`.

    Returns:
        The nominal and loss on the scenarios with `nominal_i > 0`, the only
        ones a KL or chi-square ball can reach.
    """
    support = nominal > 0.0
    return nominal[support], loss[support]


def _minimize_on_log_scale(objective: Callable[[float], float], spread: float) -> float:
    """Minimize a unimodal function of a positive scale over a log-spaced range.

    Args:
        objective: Function of a positive scale, unimodal on the half line.
        spread: Natural unit of that scale, the loss spread.

    Returns:
        The smallest value found, including the two ends of the range so a
        monotone objective returns its boundary infimum.
    """
    lower = -_LOG_SEARCH_HALF_WIDTH
    upper = _LOG_SEARCH_HALF_WIDTH

    def on_log_scale(log_scale: float) -> float:
        return objective(spread * float(np.exp(log_scale)))

    solution = minimize_scalar(
        on_log_scale,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": _SEARCH_TOLERANCE},
    )
    return min(float(solution.fun), on_log_scale(lower), on_log_scale(upper))


def kl_dual(nominal: Array, loss: Array, radius: float, eta: float) -> float:
    r"""Evaluate the KL dual at a multiplier $\eta > 0$.

    The dual objective is
    $\eta\,\mathrm{radius} + \eta \log \mathbb{E}_p[e^{\ell/\eta}]$. The
    log-moment generating function is shifted by $\max \ell$ and read
    as $\log(1 + \mathbb{E}_p[e^{x} - 1])$ through `log1p` and `expm1`,
    using that `nominal` has unit mass, so a large $\eta$ (the
    vanishing-radius regime) keeps full precision instead of cancelling
    two nearly equal terms. Writing the unit mass as `sum(nominal)`
    instead would multiply its rounding error by $\eta$.

    Args:
        nominal: Reference distribution of shape `(n,)`.
        loss: Per-scenario losses of shape `(n,)`.
        radius: Nonnegative KL budget.
        eta: Positive dual multiplier.

    Returns:
        The dual objective, an upper bound on the KL worst case.
    """
    nominal, loss = _positive_support(nominal, loss)
    maximum = float(loss.max())
    moment = np.log1p(np.sum(nominal * np.expm1((loss - maximum) / eta)))
    return float(eta * radius + maximum + eta * moment)


def kl_dual_minimum(nominal: Array, loss: Array, radius: float) -> float:
    """Minimize `kl_dual` over the multiplier.

    Args:
        nominal: Reference distribution of shape `(n,)`.
        loss: Per-scenario losses of shape `(n,)`.
        radius: Nonnegative KL budget.

    Returns:
        The dual minimum, equal to the KL worst case by strong duality.
    """
    positive_nominal, positive_loss = _positive_support(nominal, loss)
    spread = float(positive_loss.max() - positive_loss.min())
    if spread == 0.0:
        return float(positive_loss.max())
    return _minimize_on_log_scale(
        lambda eta: kl_dual(positive_nominal, positive_loss, radius, eta), spread
    )


def chi_square_dual(nominal: Array, loss: Array, radius: float, lam: float) -> float:
    r"""Evaluate the chi-square dual at a multiplier $\lambda$.

    The dual of Duchi and Namkoong (2021) for the generator
    $\phi(t) = (t - 1)^2$, with the scale multiplier minimized out in closed
    form, is

    $$
    \lambda + \sqrt{(1 + \mathrm{radius})\,
        \mathbb{E}_p[(\ell - \lambda)_+^2]}.
    $$

    Writing $w = (\ell - \lambda)_+$, $T = \mathbb{E}_p[w]$ and
    $S = \mathbb{E}_p[w^2]$, it is evaluated as

    $$
    \mathbb{E}_p[\max(\ell, \lambda)]
        + \frac{\mathrm{Var}_p(w) + \mathrm{radius}\,S}
               {\sqrt{(1 + \mathrm{radius})\,S} + T},
    $$

    which is algebraically the same but does not cancel $\lambda$ against
    $T$ when $\lambda$ lies far below the losses, as it does once the
    radius vanishes.

    Args:
        nominal: Reference distribution of shape `(n,)`.
        loss: Per-scenario losses of shape `(n,)`.
        radius: Nonnegative chi-square budget.
        lam: Real dual multiplier.

    Returns:
        The dual objective, an upper bound on the chi-square worst case.
    """
    nominal, loss = _positive_support(nominal, loss)
    clipped = np.maximum(loss, lam)
    surplus = clipped - lam
    first_moment = float(np.sum(nominal * surplus))
    if first_moment == 0.0:
        return float(lam)
    second_moment = float(np.sum(nominal * surplus**2))
    variance = float(np.sum(nominal * (surplus - first_moment) ** 2))
    correction = (variance + radius * second_moment) / (
        np.sqrt((1.0 + radius) * second_moment) + first_moment
    )
    return float(np.sum(nominal * clipped) + correction)


def chi_square_dual_minimum(nominal: Array, loss: Array, radius: float) -> float:
    """Minimize `chi_square_dual` over the multiplier.

    The multiplier is searched as its depth below the maximum loss, on which
    the dual is unimodal: it is convex in the multiplier and increasing
    above the maximum loss.

    Args:
        nominal: Reference distribution of shape `(n,)`.
        loss: Per-scenario losses of shape `(n,)`.
        radius: Nonnegative chi-square budget.

    Returns:
        The dual minimum, equal to the chi-square worst case by strong
        duality.
    """
    positive_nominal, positive_loss = _positive_support(nominal, loss)
    maximum = float(positive_loss.max())
    spread = maximum - float(positive_loss.min())
    if spread == 0.0:
        return maximum
    return _minimize_on_log_scale(
        lambda depth: chi_square_dual(
            positive_nominal, positive_loss, radius, maximum - depth
        ),
        spread,
    )


def total_variation_dual(
    nominal: Array, loss: Array, radius: float, level: float
) -> float:
    r"""Evaluate the total-variation dual at a quantile level $c$.

    Moving mass $r = \min(\mathrm{radius}, 1)$ off the lowest losses onto
    the highest one leaves the upper $(1 - r)$-tail of the nominal in place,
    whose mass-weighted mean has the Rockafellar-Uryasev representation, so

    $$
    \sup_{q:\, D_{\mathrm{TV}}(q,\, p) \,\le\, \mathrm{radius}} \mathbb{E}_q[\ell]
    = r \max \ell + \inf_{c \in \mathbb{R}}
        \big((1 - r)\,c + \mathbb{E}_p[(\ell - c)_+]\big).
    $$

    The radius is capped at one because no two distributions are further
    than one apart in total variation. Above every loss the tail term
    vanishes and the objective is $r \max \ell + (1 - r)\,c$ as written;
    otherwise, using that `nominal` has unit mass, it is evaluated as
    $r (\max \ell - c) + \mathbb{E}_p[\max(\ell, c)]$. Each form avoids
    cancelling a large $c$ on its own side of the losses.

    Args:
        nominal: Reference distribution of shape `(n,)`.
        loss: Per-scenario losses of shape `(n,)`.
        radius: Nonnegative total-variation budget.
        level: Real quantile level $c$.

    Returns:
        The dual objective, an upper bound on the total-variation worst
        case.
    """
    moved = min(radius, 1.0)
    maximum = float(loss.max())
    if level >= maximum:
        return moved * maximum + (1.0 - moved) * level
    clipped = float(np.sum(nominal * np.maximum(loss, level)))
    return moved * (maximum - level) + clipped


def total_variation_dual_minimum(nominal: Array, loss: Array, radius: float) -> float:
    r"""Minimize `total_variation_dual` exactly over its breakpoints.

    The dual is convex and piecewise linear in the level, with slope
    $-\min(\mathrm{radius}, 1) \le 0$ below every loss and
    $1 - \min(\mathrm{radius}, 1) \ge 0$ above every loss, so its
    minimum is attained at one of the losses.

    Args:
        nominal: Reference distribution of shape `(n,)`.
        loss: Per-scenario losses of shape `(n,)`.
        radius: Nonnegative total-variation budget.

    Returns:
        The exact dual minimum.
    """
    return min(
        total_variation_dual(nominal, loss, radius, float(level)) for level in loss
    )


def cvar_dual(nominal: Array, loss: Array, radius: float, level: float) -> float:
    r"""Evaluate the Rockafellar-Uryasev CVaR dual at a level $c$.

    The CVaR set of tail probability $\alpha = e^{-\mathrm{radius}}$ has
    the dual objective $c + \mathbb{E}_p[(\ell - c)_+] / \alpha$
    (Rockafellar and Uryasev 2000). Using that `nominal` has unit mass, it
    is evaluated as

    $$
    \mathbb{E}_p[\max(\ell, c)]
        + \left(e^{\mathrm{radius}} - 1\right) \mathbb{E}_p[(\ell - c)_+],
    $$

    a sum of two terms that never cancel a large $|c|$, with the factor
    $1/\alpha - 1$ read through `expm1` so a vanishing radius keeps full
    precision. Past $\mathrm{radius} \approx 709$ that factor overflows
    to infinity, which is its correct float64 value: the objective is then
    infinite below the largest reachable loss, as at $\alpha = 0$.

    Args:
        nominal: Reference distribution of shape `(n,)`.
        loss: Per-scenario losses of shape `(n,)`.
        radius: Nonnegative max-divergence radius $\log(1/\alpha)$.
        level: Real quantile level $c$.

    Returns:
        The dual objective, an upper bound on the CVaR worst case.
    """
    nominal, loss = _positive_support(nominal, loss)
    clipped = float(np.sum(nominal * np.maximum(loss, level)))
    tail = float(np.sum(nominal * np.maximum(loss - level, 0.0)))
    if tail == 0.0:
        return clipped
    with np.errstate(over="ignore"):
        excess_ratio = float(np.expm1(radius))
    return clipped + excess_ratio * tail


def cvar_dual_minimum(nominal: Array, loss: Array, radius: float) -> float:
    r"""Minimize `cvar_dual` exactly over its breakpoints.

    The dual is convex and piecewise linear in the level with slope
    $1 - 1/\alpha \le 0$ below every loss and $1$ above every loss, so its
    minimum is attained at one of the losses.

    Args:
        nominal: Reference distribution of shape `(n,)`.
        loss: Per-scenario losses of shape `(n,)`.
        radius: Nonnegative max-divergence radius.

    Returns:
        The exact dual minimum.
    """
    return min(cvar_dual(nominal, loss, radius, float(level)) for level in loss)


def wasserstein_dual(
    nominal: Array, cost: Array, loss: Array, radius: float, gamma: float
) -> float:
    r"""Evaluate the Wasserstein dual at a multiplier $\gamma \ge 0$.

    The dual objective is
    $\gamma\,\mathrm{radius}
    + \mathbb{E}_{i \sim p}[\max_j (\ell_j - \gamma\, c_{ij})]$
    (Mohajerin Esfahani and Kuhn 2018), specialized to a finite support.

    Args:
        nominal: Reference distribution of shape `(n,)`.
        cost: Nonnegative ground cost of shape `(n, n)` with a zero diagonal.
        loss: Per-scenario losses of shape `(n,)`.
        radius: Nonnegative transport budget.
        gamma: Nonnegative dual multiplier.

    Returns:
        The dual objective, an upper bound on the Wasserstein worst case.
    """
    inner = np.max(loss[None, :] - gamma * cost, axis=1)
    return float(gamma * radius + np.sum(nominal * inner))


def wasserstein_dual_minimum(
    nominal: Array, cost: Array, loss: Array, radius: float
) -> float:
    r"""Minimize `wasserstein_dual` exactly over its breakpoints.

    The dual is convex and piecewise linear in the multiplier, and with a
    zero-diagonal cost its slope tends to `radius >= 0`, so its minimum is
    attained at zero or where two columns of some row swap as that row's
    maximizer: at $\gamma = (\ell_j - \ell_k) / (c_{ij} - c_{ik})$.

    Args:
        nominal: Reference distribution of shape `(n,)`.
        cost: Nonnegative ground cost of shape `(n, n)` with a zero diagonal.
        loss: Per-scenario losses of shape `(n,)`.
        radius: Nonnegative transport budget.

    Returns:
        The exact dual minimum.
    """
    support_size = loss.shape[0]
    shape = (support_size, support_size, support_size)
    loss_difference = np.broadcast_to(loss[None, :, None] - loss[None, None, :], shape)
    cost_difference = cost[:, :, None] - cost[:, None, :]
    distinct = cost_difference != 0.0
    crossings = loss_difference[distinct] / cost_difference[distinct]
    candidates = np.concatenate([[0.0], crossings[crossings > 0.0]])
    return min(
        wasserstein_dual(nominal, cost, loss, radius, float(gamma))
        for gamma in candidates
    )


def sinkhorn_kernel(cost: Array, epsilon: float, reference: Array) -> Array:
    r"""Return the row-stochastic Gibbs kernel $Q \propto \nu e^{-C/\epsilon}$.

    Args:
        cost: Ground cost of shape `(n, n)`.
        epsilon: Positive entropic regularization strength.
        reference: Nonnegative reference measure of shape `(n,)`.

    Returns:
        The kernel, each row shifted by its smallest cost before
        exponentiating so no row underflows entirely.
    """
    shifted = cost - cost.min(axis=1, keepdims=True)
    weights = reference[None, :] * np.exp(-shifted / epsilon)
    return weights / weights.sum(axis=1, keepdims=True)


def sinkhorn_dual(
    nominal: Array, kernel: Array, loss: Array, radius: float, eta: float
) -> float:
    r"""Evaluate the Sinkhorn-DRO dual at a temperature $\eta = \lambda\epsilon > 0$.

    Wang, Gao and Xie's dual, in its form shifted by the entropic bias, is
    $\inf_{\lambda \ge 0} \lambda \bar\rho + \lambda \epsilon \sum_i p_i
    \log \sum_j Q_{ij} e^{\ell_j / (\lambda\epsilon)}$. In terms of $\eta$
    the first term is $\eta \bar\rho / \epsilon$, and the caller passes
    `radius` $= \bar\rho / \epsilon$. Each row's log-moment generating
    function is shifted by the largest loss the kernel reaches and read
    through `log1p` and `expm1`, as in `kl_dual`, while that sum stays
    away from $-1$, and through `logsumexp` otherwise: when the largest
    loss carries a tiny kernel mass the sum rounds to $-1$.

    Args:
        nominal: Nominal distribution of shape `(n,)`.
        kernel: Row-stochastic kernel of shape `(n, n)`.
        loss: Per-scenario losses of shape `(n,)`.
        radius: Budget divided by the regularization strength.
        eta: Positive temperature.

    Returns:
        The dual objective, an upper bound on the Sinkhorn worst case.
    """
    rows = nominal > 0.0
    reachable = kernel[rows].sum(axis=0) > 0.0
    maximum = float(loss[reachable].max())
    shifted = np.where(reachable, loss - maximum, 0.0)
    excess = np.sum(kernel[rows] * np.expm1(shifted / eta), axis=1)
    with np.errstate(divide="ignore"):
        log_kernel = np.log(kernel[rows])
    direct = logsumexp(log_kernel + shifted / eta, axis=1)
    moments = np.where(excess > -0.5, np.log1p(np.maximum(excess, -0.5)), direct)
    return float(eta * radius + maximum + eta * np.sum(nominal[rows] * moments))


def sinkhorn_dual_minimum(
    nominal: Array, kernel: Array, loss: Array, radius: float
) -> float:
    """Minimize `sinkhorn_dual` over the temperature.

    Args:
        nominal: Nominal distribution of shape `(n,)`.
        kernel: Row-stochastic kernel of shape `(n, n)`.
        loss: Per-scenario losses of shape `(n,)`.
        radius: Budget divided by the regularization strength.

    Returns:
        The dual minimum, equal to the Sinkhorn worst case by strong
        duality (Wang, Gao and Xie 2025, Theorem 3.1).
    """
    reachable = kernel[nominal > 0.0].sum(axis=0) > 0.0
    spread = float(loss[reachable].max() - loss[reachable].min())
    if spread == 0.0:
        return float(loss[reachable].max())
    return _minimize_on_log_scale(
        lambda eta: sinkhorn_dual(nominal, kernel, loss, radius, eta), spread
    )


def kl_divergence(candidate: Array, nominal: Array) -> float:
    r"""Return $D_{\mathrm{KL}}(q \,\|\, p)$, infinite if $q$ leaves the support of $p$.

    Args:
        candidate: Candidate distribution of shape `(n,)`.
        nominal: Reference distribution of shape `(n,)`.

    Returns:
        The Kullback-Leibler divergence in nats.
    """
    support = candidate > 0.0
    if np.any(nominal[support] == 0.0):
        return float(np.inf)
    ratio = candidate[support] / nominal[support]
    return float(np.sum(candidate[support] * np.log(ratio)))


def chi_square_divergence(candidate: Array, nominal: Array) -> float:
    r"""Return $\sum_i (q_i - p_i)^2 / p_i$, infinite if $q$ leaves the support of $p$.

    Args:
        candidate: Candidate distribution of shape `(n,)`.
        nominal: Reference distribution of shape `(n,)`.

    Returns:
        The chi-square divergence.
    """
    support = nominal > 0.0
    if np.any(candidate[~support] > 0.0):
        return float(np.inf)
    deviation = candidate[support] - nominal[support]
    return float(np.sum(deviation**2 / nominal[support]))


def max_divergence(candidate: Array, nominal: Array) -> float:
    r"""Return $\log \max_i q_i / p_i$, infinite if $q$ leaves the support of $p$.

    Args:
        candidate: Candidate distribution of shape `(n,)`.
        nominal: Reference distribution of shape `(n,)`.

    Returns:
        The max-divergence (Renyi divergence of order infinity) in nats.
    """
    support = candidate > 0.0
    if np.any(nominal[support] == 0.0):
        return float(np.inf)
    return float(np.log(np.max(candidate[support] / nominal[support])))


def total_variation_distance(candidate: Array, nominal: Array) -> float:
    r"""Return $\frac{1}{2} \sum_i |q_i - p_i|$.

    Args:
        candidate: Candidate distribution of shape `(n,)`.
        nominal: Reference distribution of shape `(n,)`.

    Returns:
        The total variation distance.
    """
    return float(0.5 * np.sum(np.abs(candidate - nominal)))


def transport_cost(candidate: Array, nominal: Array, cost: Array) -> float:
    r"""Return the optimal transport cost $W_c(q, p)$ as an exact linear program.

    Args:
        candidate: Candidate distribution of shape `(n,)`.
        nominal: Reference distribution of shape `(n,)`.
        cost: Nonnegative ground cost of shape `(n, n)`.

    Returns:
        The minimal $\sum_{ij} \pi_{ij} c_{ij}$ over couplings $\pi \ge 0$
        with row marginal `nominal` and column marginal `candidate`.

    Raises:
        RuntimeError: If the linear program does not solve to optimality.
    """
    support_size = nominal.shape[0]
    identity = np.eye(support_size)
    marginals = np.vstack(
        [
            np.kron(identity, np.ones((1, support_size))),
            np.kron(np.ones((1, support_size)), identity),
        ]
    )
    # The two marginals each sum to one, so one equality is redundant;
    # dropping it keeps the system full rank for HiGHS.
    solution = linprog(
        c=cost.reshape(-1),
        A_eq=marginals[:-1],
        b_eq=np.concatenate([nominal, candidate])[:-1],
        bounds=(0.0, None),
        method="highs",
        options=_LP_OPTIONS,
    )
    if not solution.success:
        raise RuntimeError(f"transport LP failed: {solution.message}")
    return float(solution.fun)
