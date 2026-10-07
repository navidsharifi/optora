r"""Independent references for the worst-case expectation of each ambiguity set.

Every function here solves the *primal* inner maximization

$$
\sup_{q \in \mathcal{U}} \; \mathbb{E}_q[\mathrm{loss}],
$$

of `optora.dro`'s ambiguity sets without importing, reusing, or restating
any optora code: the arrays are NumPy, the solvers are SciPy's, and the
derivations are taken from the primal problem rather than from the duals
optora evaluates. A test that compares optora against one of these
functions therefore fails on a bug in either side independently, which a
cross-check written against the same dual formula cannot do.

Two families of reference are used:

- Exact linear programs (`total_variation_worst_case`,
  `wasserstein_worst_case`) solved by `scipy.optimize.linprog`'s HiGHS
  simplex. The optimum is a vertex, so the value is exact up to the LP
  solver's own residual tolerance.
- Closed forms derived from the KKT conditions (`kl_worst_case`,
  `chi_square_worst_case`). The chi-square form is fully explicit; the KL
  form needs one monotone scalar root solve, done with
  `scipy.optimize.brentq`.

Each reference handles the saturated regime, where the radius is large
enough that the ambiguity set contains a distribution supported entirely
on the highest-loss scenarios and the worst case is simply `max(loss)`,
as a separate branch rather than as a numerical limit.
"""

from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, linprog

Array = NDArray[np.float64]

_LP_METHOD: Final = "highs"

# HiGHS defaults to a 1e-7 primal feasibility tolerance, which is larger
# than the whole optimal reallocation once the radius is small, and lets it
# report a value well above the true optimum. 1e-10 is the tightest setting
# HiGHS accepts, and is enough to make these small linear programs exact to
# roughly 1e-15.
_LP_OPTIONS: Final = {
    "primal_feasibility_tolerance": 1e-10,
    "dual_feasibility_tolerance": 1e-10,
}


def _as_float64(values: object) -> Array:
    """Return `values` as a contiguous float64 array.

    Args:
        values: Anything `numpy.asarray` accepts, typically a detached
            `torch.Tensor` or a Python sequence.

    Returns:
        The same data as a float64 NumPy array.
    """
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64))


def _saturation_mass(nominal: Array, loss: Array) -> float:
    """Return the nominal mass sitting on the highest-loss scenarios.

    Args:
        nominal: Reference distribution of shape `(n,)`.
        loss: Per-scenario losses of shape `(n,)`.

    Returns:
        $\\sum_{i:\\, \\mathrm{loss}_i = \\max_j \\mathrm{loss}_j}
        \\mathrm{nominal}_i$, the largest mass any candidate distribution
        can keep in place while moving everything else onto the maximum.
    """
    return float(nominal[loss >= loss.max()].sum())


def total_variation_worst_case(nominal: Array, loss: Array, radius: float) -> float:
    r"""Solve the total-variation worst case as an exact linear program.

    The primal is linear in $q$ and the total-variation constraint
    $\frac{1}{2}\sum_i |q_i - p_i| \le \mathrm{radius}$ is linearized with
    one slack $u_i \ge |q_i - p_i|$ per scenario:

    $$
    \max_{q,\,u \ge 0} \; \mathrm{loss}^\top q
    \quad \text{s.t.} \quad
    \mathbf{1}^\top q = 1, \;
    q_i - u_i \le p_i, \;
    -q_i - u_i \le -p_i, \;
    \mathbf{1}^\top u \le 2\,\mathrm{radius}.
    $$

    Args:
        nominal: Reference distribution of shape `(n,)`.
        loss: Per-scenario losses of shape `(n,)`.
        radius: Nonnegative total-variation budget.

    Returns:
        The worst-case expected loss.

    Raises:
        RuntimeError: If the linear program does not solve to optimality.
    """
    nominal = _as_float64(nominal)
    loss = _as_float64(loss)
    support_size = loss.shape[0]

    objective = np.concatenate([-loss, np.zeros(support_size)])
    identity = np.eye(support_size)
    inequality = np.block(
        [
            [identity, -identity],
            [-identity, -identity],
            [np.zeros((1, support_size)), np.ones((1, support_size))],
        ]
    )
    inequality_bound = np.concatenate([nominal, -nominal, [2.0 * radius]])
    equality = np.concatenate([np.ones(support_size), np.zeros(support_size)])

    solution = linprog(
        c=objective,
        A_ub=inequality,
        b_ub=inequality_bound,
        A_eq=equality.reshape(1, -1),
        b_eq=np.array([1.0]),
        bounds=[(0.0, None)] * support_size + [(0.0, None)] * support_size,
        method=_LP_METHOD,
        options=_LP_OPTIONS,
    )
    if not solution.success:
        raise RuntimeError(f"total-variation LP failed: {solution.message}")
    return float(-solution.fun)


def wasserstein_worst_case(
    nominal: Array, cost: Array, loss: Array, radius: float
) -> float:
    r"""Solve the Wasserstein worst case as an exact transport linear program.

    Optimizes directly over the transport plan $\pi \ge 0$ on the shared
    support, leaving the column marginal (the candidate distribution) free:

    $$
    \max_{\pi \ge 0} \sum_{ij} \pi_{ij}\, \mathrm{loss}_j
    \quad \text{s.t.} \quad
    \sum_j \pi_{ij} = p_i, \qquad
    \sum_{ij} \pi_{ij}\, \mathrm{cost}_{ij} \le \mathrm{radius}.
    $$

    This is the formulation optora's `WassersteinAmbiguitySet` dualizes; it
    is solved here in the primal, so an error in that dualization or in its
    bisection shows up as a disagreement.

    Args:
        nominal: Reference distribution of shape `(n,)`.
        cost: Nonnegative ground cost matrix of shape `(n, n)`.
        loss: Per-scenario losses of shape `(n,)`.
        radius: Nonnegative transport budget.

    Returns:
        The worst-case expected loss.

    Raises:
        RuntimeError: If the linear program does not solve to optimality.
    """
    nominal = _as_float64(nominal)
    cost = _as_float64(cost)
    loss = _as_float64(loss)
    support_size = loss.shape[0]

    objective = -np.broadcast_to(loss, (support_size, support_size)).reshape(-1)
    budget_row = cost.reshape(1, -1)
    row_marginal = np.zeros((support_size, support_size * support_size))
    for row in range(support_size):
        row_marginal[row, row * support_size : (row + 1) * support_size] = 1.0

    solution = linprog(
        c=objective,
        A_ub=budget_row,
        b_ub=np.array([radius]),
        A_eq=row_marginal,
        b_eq=nominal,
        bounds=(0.0, None),
        method=_LP_METHOD,
        options=_LP_OPTIONS,
    )
    if not solution.success:
        raise RuntimeError(f"Wasserstein LP failed: {solution.message}")
    return float(-solution.fun)


def _positive_support(nominal: Array, loss: Array) -> tuple[Array, Array]:
    """Drop the scenarios carrying no nominal mass.

    A scenario with `nominal_i = 0` is unreachable under any divergence
    that is infinite off the nominal's support, which both KL and
    chi-square are, so the exact worst case is the one computed on the
    restricted support.

    Args:
        nominal: Reference distribution of shape `(n,)`.
        loss: Per-scenario losses of shape `(n,)`.

    Returns:
        The nominal and loss restricted to the scenarios with positive
        nominal mass.
    """
    support = nominal > 0.0
    return nominal[support], loss[support]


def _tilted_distribution(nominal: Array, loss: Array, inverse_eta: float) -> Array:
    r"""Return the exponentially tilted distribution $q \propto p\,e^{\beta \ell}$.

    Args:
        nominal: Reference distribution of shape `(n,)`, strictly positive.
        loss: Per-scenario losses of shape `(n,)`.
        inverse_eta: Tilt strength $\beta = 1/\eta \ge 0$.

    Returns:
        The normalized tilted distribution, computed through a shifted
        exponential so a large tilt does not overflow.
    """
    exponent = np.log(nominal) + inverse_eta * loss
    tilted = np.exp(exponent - exponent.max())
    return tilted / tilted.sum()


def _kl_divergence(candidate: Array, nominal: Array) -> float:
    r"""Return $D_{\mathrm{KL}}(q \,\|\, p)$, treating $q_i = 0$ as contributing zero.

    Args:
        candidate: Candidate distribution of shape `(n,)`.
        nominal: Reference distribution of shape `(n,)`.

    Returns:
        The Kullback-Leibler divergence in nats.
    """
    support = candidate > 0.0
    return float(
        np.sum(candidate[support] * np.log(candidate[support] / nominal[support]))
    )


def kl_worst_case(nominal: Array, loss: Array, radius: float) -> float:
    r"""Evaluate the KL worst case from its exponential-tilt closed form.

    The KKT conditions of

    $$
    \max_q \; \mathrm{loss}^\top q
    \quad \text{s.t.} \quad
    \mathbf{1}^\top q = 1, \; q \ge 0, \;
    D_{\mathrm{KL}}(q \,\|\, p) \le \mathrm{radius}
    $$

    make the maximizer a Gibbs tilt of the nominal,
    $q_\beta \propto p \, e^{\beta\,\mathrm{loss}}$ with $\beta = 1/\eta \ge 0$.
    $D_{\mathrm{KL}}(q_\beta \,\|\, p)$ increases monotonically in $\beta$
    from $0$ to $-\log P^\star$, where $P^\star$ is the nominal mass on the
    highest-loss scenarios, so the constraint is tight at the optimum for
    every $\mathrm{radius} < -\log P^\star$ and the tilt is pinned by a
    one-dimensional monotone root solve. At or beyond that radius the set
    contains a distribution supported on the maximizers and the worst case
    saturates at $\max_i \mathrm{loss}_i$.

    No part of optora's dual
    $\inf_{\eta>0} \eta\,\mathrm{radius} + \eta \log
    \mathbb{E}_p[e^{\mathrm{loss}/\eta}]$ is used: the tilt is read off the
    primal optimality conditions and the reported value is
    $\mathbb{E}_{q_\beta}[\mathrm{loss}]$, a primal objective value.

    Args:
        nominal: Reference distribution of shape `(n,)`.
        loss: Per-scenario losses of shape `(n,)`.
        radius: Nonnegative KL budget in nats.

    Returns:
        The worst-case expected loss.
    """
    nominal = _as_float64(nominal)
    loss = _as_float64(loss)
    if radius == 0.0:
        return float(np.sum(nominal * loss))

    nominal, loss = _positive_support(nominal, loss)
    if radius >= -np.log(_saturation_mass(nominal, loss)):
        return float(loss.max())

    def excess_divergence(inverse_eta: float) -> float:
        return (
            _kl_divergence(_tilted_distribution(nominal, loss, inverse_eta), nominal)
            - radius
        )

    upper = 1.0
    while excess_divergence(upper) < 0.0:
        upper *= 2.0
        if upper > 1e12:
            return float(loss.max())
    inverse_eta = brentq(excess_divergence, 0.0, upper, xtol=1e-15, rtol=8.9e-16)
    return float(np.sum(_tilted_distribution(nominal, loss, inverse_eta) * loss))


def chi_square_worst_case(nominal: Array, loss: Array, radius: float) -> float:
    r"""Evaluate the chi-square worst case from its active-set closed form.

    The KKT conditions of

    $$
    \max_q \; \mathrm{loss}^\top q
    \quad \text{s.t.} \quad
    \mathbf{1}^\top q = 1, \; q \ge 0, \;
    \sum_i \frac{(q_i - p_i)^2}{p_i} \le \mathrm{radius}
    $$

    give $q_i = p_i\,(1 + (\mathrm{loss}_i - \lambda)/\eta)_+$, so the
    support of the maximizer is a *top-$k$ set* of scenarios ordered by
    loss. Fixing that active set $A$ and eliminating $\lambda$ and $\eta$
    from the two tight constraints leaves a fully explicit value:

    $$
    \mathbb{E}_{q^\star}[\mathrm{loss}]
    = \mu_A + \sqrt{V_A \left(\mathrm{radius} - \frac{1 - P_A}{P_A}\right)},
    $$

    where $P_A = \sum_{i \in A} p_i$, $\mu_A$ is the $p$-weighted mean of
    the losses on $A$, and $V_A$ their $p$-weighted variance. Taking $A$ to
    be the whole support recovers the textbook
    $\mathbb{E}_p[\mathrm{loss}] + \sqrt{\mathrm{radius}\,
    \mathrm{Var}_p(\mathrm{loss})}$. The correct $k$ is the one whose
    implied distribution is nonnegative on $A$ and nonpositive just outside
    it, found by scanning the sorted losses.

    Args:
        nominal: Reference distribution of shape `(n,)`.
        loss: Per-scenario losses of shape `(n,)`.
        radius: Nonnegative chi-square budget.

    Returns:
        The worst-case expected loss.

    Raises:
        RuntimeError: If no top-$k$ active set satisfies the optimality
            conditions, which would mean the closed form above is wrong.
    """
    nominal = _as_float64(nominal)
    loss = _as_float64(loss)
    if radius == 0.0:
        return float(np.sum(nominal * loss))

    nominal, loss = _positive_support(nominal, loss)
    if radius >= 1.0 / _saturation_mass(nominal, loss) - 1.0:
        return float(loss.max())

    order = np.argsort(-loss, kind="stable")
    sorted_loss = loss[order]
    sorted_nominal = nominal[order]

    for active_size in range(loss.shape[0], 0, -1):
        active_loss = sorted_loss[:active_size]
        active_nominal = sorted_nominal[:active_size]
        active_mass = float(active_nominal.sum())
        mean = float(active_nominal @ active_loss) / active_mass
        variance = float(active_nominal @ (active_loss - mean) ** 2)
        budget = radius - (1.0 - active_mass) / active_mass
        if budget < 0.0 or variance <= 0.0:
            continue
        inverse_eta = np.sqrt(budget / variance)
        ratios = 1.0 / active_mass + inverse_eta * (active_loss - mean)
        if ratios.min() < -1e-12:
            continue
        if active_size < loss.shape[0]:
            excluded = 1.0 / active_mass + inverse_eta * (
                sorted_loss[active_size] - mean
            )
            if excluded > 1e-12:
                continue
        return mean + float(np.sqrt(variance * budget))

    raise RuntimeError("no chi-square active set satisfied the optimality conditions.")
